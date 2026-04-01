import tensorflow as tf
import numpy as np
from .layers import InferenceLayer, HybridThresholdL2Regularizer, ClipMaxValue


class WindowSpatialAttention(InferenceLayer, tf.keras.layers.Layer):
    """
    Swin-style window attention with forced overlap and averaging.

    Key features:
    - Fully graph-compatible (no Python loops)
    - Separate Q, K, V projections
    - Edge clipping (no cyclic padding)
    - Forced window overlap with averaging
    - Adaptive shift based on image size (targets window_size/2)
    - Optimized memory layout
    - Supports non-square windows (window_size can be tuple)

    Extended feature:
    - multi_query mode: query tensor can have an additional leading axis Q
      (shape: (Q, B, Y, X, C)) so several queries attend to the same K/V
      without tiling K/V in memory.
    """

    def __init__(self, num_heads: int, attention_filters: int, window_size:tuple,
                 use_bias:bool = True, dropout: float = 0.1, skip_connection: bool = True, layer_normalization:bool=True,
                 add_distance_embedding:bool=True,
                 window_processing:str="auto", # "batch", "row", "col", "sequential", "auto"="batch" at train and "sequential" otherwise
                 overlap_reduction: str = 'geometrical',  # 'mean', 'attention_weighted', 'geometrical'
                 l2_reg: float = 0., position_encoding_l2_reg: float = 1e-5, name="WindowSpatialAttention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        # Support both int and tuple for window_size
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = tuple(window_size)
        self.use_bias=use_bias
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.position_encoding_l2_reg=position_encoding_l2_reg
        self.skip_connection = skip_connection
        self.layer_normalization=layer_normalization
        self.overlap_reduction = overlap_reduction
        self.window_processing=window_processing
        assert overlap_reduction in ['mean', 'attention_weighted', 'geometrical'], \
            f"overlap_reduction must be 'mean' or 'attention_weighted', got {overlap_reduction}"
        self.add_distance_embedding=add_distance_embedding
        # Multi-query mode settings (populated in build)
        self.multi_query = False
        self.q_count = 1
        self.filters = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "attention_filters": self.attention_filters,
            "window_size": self.window_size,
            "use_bias":self.use_bias,
            "dropout": self.dropout,
            "l2_reg": self.l2_reg,
            "position_encoding_l2_reg":self.position_encoding_l2_reg,
            "skip_connection": self.skip_connection,
            "layer_normalization": self.layer_normalization,
            "window_processing":self.window_processing,
            "overlap_reduction": self.overlap_reduction,
            "add_distance_embedding": self.add_distance_embedding
        })
        return config

    def build(self, input_shapes):
        if not isinstance(input_shapes, list):  # single tensor : self attention
            input_shapes = [input_shapes]
        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass
        input_shape = input_shapes[0]
        tridim_mode = len(input_shapes[-1]) == 5
        if tridim_mode and len(self.window_size) == 2:
            self.window_size = (1, self.window_size[0], self.window_size[1])
        # Detect multi-query mode statically from input shape if provided
        # Expected query shapes:
        # - Single query: (B, Y, X, C) or (B, Z, Y, X, C)
        # - Multi-query: (Q, B, Y, X, C) or (Q, B, Z Y, X, C) and K/V must be provided
        if len(input_shape) == 5 + (1 if tridim_mode else 0):
            assert len(input_shapes) >= 2, "in multi-query mode, K/V must be provided"
            # static multi-query
            self.multi_query = True
            self.q_count = input_shape[0]
            input_shape = input_shape[1:]
        elif len(input_shape) == 4 + (1 if tridim_mode else 0):
            self.multi_query = False
            self.q_count = 1
        else:
            raise ValueError(f"Invalid query shape: {input_shape}")
        for s in input_shapes[1:min(3, len(input_shapes))]:
            assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape[:-1], s[:-1])), \
                f"all tensors must have same input shape: {input_shape} != {s}"

        if self.layer_normalization:
            # LayerNorm will be applied per (batch-like) sample. For multi_query we apply it
            # after reshaping Q to (Q*B, Y, X, C) in call(), so the same ln_q can be used.
            self.ln_q = tf.keras.layers.LayerNormalization(dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
            if len(input_shapes) >= 3 : # Q, K, V provided
                self.ln_v = tf.keras.layers.LayerNormalization(dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
                self.ln_k = tf.keras.layers.LayerNormalization(dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
            elif len(input_shapes) == 2: # Q, K provided
                self.ln_k = tf.keras.layers.LayerNormalization(dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
                self.ln_v = None
            else:
                self.ln_v = None
                self.ln_k = None

        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)

        HF = self.num_heads * self.attention_filters

        # Separate Q, K, V projections
        conv_op = tf.keras.layers.Conv3D if tridim_mode else tf.keras.layers.Conv2D
        self.qproj = conv_op(HF, 1, padding='same',
                                            use_bias=self.use_bias, name="qproj",
                                            dtype=self.dtype_policy,
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                            bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                            kernel_constraint=ClipMaxValue(),
                                            bias_constraint=ClipMaxValue(),
                                            )
        self.kproj = conv_op(HF, 1, padding='same',
                                            use_bias=self.use_bias, name="kproj",
                                            dtype=self.dtype_policy,
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                            bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                            kernel_constraint=ClipMaxValue(),
                                            bias_constraint=ClipMaxValue(),
                                            )
        self.vproj = conv_op(HF, 1, padding='same',
                                            use_bias=self.use_bias, name="vproj",
                                            dtype=self.dtype_policy,
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                            bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                            kernel_constraint=ClipMaxValue(),
                                            bias_constraint=ClipMaxValue(),
                                            )
        self.outproj = conv_op(self.filters, 1, padding='same',
                                              use_bias=self.use_bias, name="outproj",
                                              dtype=self.dtype_policy,
                                              bias_initializer=tf.keras.initializers.Zeros(),
                                              kernel_regularizer=HybridThresholdL2Regularizer( directional_strength=self.l2_reg * 10,  elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                              bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                                              kernel_constraint=ClipMaxValue(),
                                              bias_constraint=ClipMaxValue(),
                                              )

        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        # Relative position bias table
        if tridim_mode:
            WSZ, WSY, WSX = self.window_size
        else:
            WSY, WSX = self.window_size
            WSZ = 1
        embedding_size = (2 * WSZ - 1) * (2 * WSY - 1) * (2 * WSX - 1)
        self.relative_position_bias_table = self.add_weight(
            name="rpb",
            shape=(embedding_size, self.num_heads),
            initializer=tf.initializers.Zeros(),
            constraint=tf.keras.constraints.MaxNorm(max_value=100.0, axis=0),
            regularizer=HybridThresholdL2Regularizer(directional_threshold=1, elementwise_threshold=5, directional_strength=self.position_encoding_l2_reg * 10, elementwise_strength=self.position_encoding_l2_reg) if self.position_encoding_l2_reg > 0 else None,
            trainable=True
        )
        # Pre-compute relative position indices for window
        coords_y = tf.range(WSY)
        coords_x = tf.range(WSX)
        if WSZ > 1:
            coords_z = tf.range(WSZ)
            coords = tf.stack(tf.meshgrid(coords_z, coords_y, coords_x, indexing='ij'))
            coords_flatten = tf.reshape(coords, [3, -1])
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = tf.transpose(relative_coords, [1, 2, 0])
            relative_coords = relative_coords + [WSZ - 1, WSY - 1, WSX - 1]  # shift to start from 0
            relative_position_index = (
                    relative_coords[:, :, 0] * (2 * WSY - 1) * (2 * WSX - 1)  # z contribution
                    + relative_coords[:, :, 1] * (2 * WSX - 1)  # y contribution
                    + relative_coords[:, :, 2]  # x contribution
            )
        else:
            coords = tf.stack(tf.meshgrid(coords_y, coords_x, indexing='ij'))
            coords_flatten = tf.reshape(coords, [2, -1])
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = tf.transpose(relative_coords, [1, 2, 0])
            relative_coords = relative_coords + [WSY - 1, WSX - 1]
            relative_position_index = (relative_coords[:, :, 0] * (2 * WSX - 1) + relative_coords[:, :, 1])

        self.relative_position_index = tf.constant(relative_position_index, dtype=tf.int32)
        if self.add_distance_embedding: # embedding added to V
            if WSZ > 1:
                hz, hy, hx = self._get_distance_encoding_output_dim()
                self.distance_encoding_z = tf.keras.layers.Embedding(
                    name="distance_encoding_z",
                    input_dim=WSZ, output_dim=hz, dtype=self.dtype_policy,
                    embeddings_regularizer=HybridThresholdL2Regularizer( directional_strength=self.position_encoding_l2_reg * 10,  elementwise_strength=self.position_encoding_l2_reg, axis=1) if self.position_encoding_l2_reg > 0 else None,
                    embeddings_constraint=ClipMaxValue()
                )
            else:
                self.distance_encoding_z = None
                hy, hx = self._get_distance_encoding_output_dim()
            self.distance_encoding_y = tf.keras.layers.Embedding(
                name = "distance_encoding_y",
                input_dim=WSY, output_dim=hy, dtype=self.dtype_policy,
                embeddings_regularizer=HybridThresholdL2Regularizer(directional_strength=self.position_encoding_l2_reg * 10, elementwise_strength=self.position_encoding_l2_reg, axis=1) if self.position_encoding_l2_reg > 0 else None,
                embeddings_constraint=ClipMaxValue()
            )
            self.distance_encoding_x = tf.keras.layers.Embedding(
                name="distance_encoding_x",
                input_dim=WSX, output_dim=hx, dtype=self.dtype_policy,
                embeddings_regularizer=HybridThresholdL2Regularizer(directional_strength=self.position_encoding_l2_reg * 10, elementwise_strength=self.position_encoding_l2_reg, axis=1) if self.position_encoding_l2_reg > 0 else None,
                embeddings_constraint=ClipMaxValue()
            )
        if self.overlap_reduction == "geometrical":
            self.geometrical_confidence = self._geometrical_confidence_3d() if tridim_mode else self._geometrical_confidence_2d()

        super().build(input_shape)

    def _get_distance_encoding_output_dim(self):
        HF = self.num_heads * self.attention_filters
        if len(self.window_size) == 3:
            WSZ, WSY, WSX = self.window_size
        else:
            WSY, WSX = self.window_size
            WSZ = 1
        if WSZ > 1:
            WS = float(WSZ + WSY + WSX)
            hz = max(1, int(float(HF * WSZ) / WS + 0.5))
            hy = max(1, int(float(HF * WSY) / WS + 0.5))
            hx = HF - hz - hy
            assert hx >= 1, f"invalid hidden dimension: {HF}"
            return hz, hy, hx
        else:
            WS = float(WSY + WSX)
            hy = max(1, int(float(HF * WSY) / WS + 0.5))
            hx = HF - hy
            assert hx >= 1, f"invalid hidden dimension: {HF}"
            return hy, hx

    @staticmethod
    def _compute_axis_coords_with_min_overlap_2d(size, tile_size):
        """
        Compute window grid ensuring minimum overlap of tile_size // 2.

        Args:
            size: spatial dimension (Y or X)
            tile_size: window size in that dimension (WSY or WSX)

        Returns:
            coords: (n_tiles,) - window start coordinates
        """
        min_overlap = tile_size // 2

        # Number of tiles needed
        n_tiles = 1 + tf.cast( tf.math.ceil(tf.cast(size - tile_size, tf.float32) /  tf.cast(tile_size - min_overlap, tf.float32)), tf.int32 )

        # Handle edge case where only 1 tile is needed
        n_tiles = tf.maximum(n_tiles, 1)

        # Compute total stride sum
        sum_stride = tf.abs(n_tiles * tile_size - size)

        # Build stride array
        def build_stride():
            # Base stride for all positions except first
            base_stride = sum_stride // (n_tiles - 1)
            stride = tf.concat([
                tf.zeros([1], dtype=tf.int32),
                tf.fill([n_tiles - 1], base_stride)
            ], axis=0)

            # Distribute remainder
            remains = sum_stride % (n_tiles - 1)

            # Add 1 to positions [1:remains+1]
            mask = tf.concat([
                tf.zeros([1], dtype=tf.int32),
                tf.ones([remains], dtype=tf.int32),
                tf.zeros([n_tiles - 1 - remains], dtype=tf.int32)
            ], axis=0)
            stride = stride + mask

            # Apply sign
            sign = tf.sign(n_tiles * tile_size - size)
            stride = tf.cond(
                sign > 0,
                lambda: -stride,
                lambda: stride
            )

            return stride

        # Handle single tile case (avoid division by zero)
        stride = tf.cond(
            n_tiles > 1,
            build_stride,
            lambda: tf.zeros([1], dtype=tf.int32)
        )

        # Cumulative sum
        stride = tf.cumsum(stride)

        # Compute coordinates
        indices = tf.range(n_tiles)
        coords = tile_size * indices + stride

        # Ensure coords are within valid range
        coords = tf.clip_by_value(coords, 0, tf.maximum(size - tile_size, 0))

        return coords

    def _compute_window_grid_2d(self, Y, X):
        """Now uses minimum overlap strategy instead of fixed shifts."""
        WSY, WSX = self.window_size
        y_starts = self._compute_axis_coords_with_min_overlap_2d(Y, WSY)
        x_starts = self._compute_axis_coords_with_min_overlap_2d(X, WSX)
        return y_starts, x_starts

    def _compute_window_grid_3d(self, Z, Y, X):
        """Now uses minimum overlap strategy instead of fixed shifts."""
        WSZ, WSY, WSX = self.window_size
        z_starts = self._compute_axis_coords_with_min_overlap_2d(Z, WSZ)
        y_starts = self._compute_axis_coords_with_min_overlap_2d(Y, WSY)
        x_starts = self._compute_axis_coords_with_min_overlap_2d(X, WSX)
        return z_starts, y_starts, x_starts

    @tf.function(jit_compile=True)
    def _extract_windows_vectorized_2d(self, x, y_starts, x_starts):
        """
        Extract windows using vectorized operations.

        Args:
            x: (B, Y, X, C)  -- NOTE: in multi-query mode x can be (Q*B, Y, X, C)
            y_starts: (num_y,) - window start positions in Y dimension
            x_starts: (num_x,) - window start positions in X dimension

        Returns:
            windows: (num_y*num_x*B, WSY, WSX, C)
        """
        B, Y, X, _ = tf.unstack(tf.shape(x))
        HF = self.num_heads * self.attention_filters
        WSY, WSX = self.window_size

        # Create grid of window coordinates
        y_grid, x_grid = tf.meshgrid(y_starts, x_starts, indexing='ij')  # (num_y, num_x)
        window_coords = tf.stack([y_grid, x_grid], axis=-1)  # (num_y, num_x, 2)
        window_coords_flat = tf.reshape(window_coords, [-1, 2])  # (num_y*num_x, 2)

        # Create offsets within window
        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        y_off_grid, x_off_grid = tf.meshgrid(y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.stack([y_off_grid, x_off_grid], axis=-1)  # (WSY, WSX, 2)

        # Broadcast to get all coordinates: (num_windows, WSY, WSX, 2)
        all_coords = window_coords_flat[:, None, None, :] + window_offsets[None, :, :, :]

        # Clip to valid range
        all_coords = tf.clip_by_value(all_coords, 0, [Y - 1, X - 1])

        # Convert to flat indices
        flat_indices = all_coords[..., 0] * X + all_coords[..., 1]  # (num_windows, WSY, WSX)

        # Reshape x for gathering: (B, Y*X, C)
        x_flat = tf.reshape(x, [B, Y * X, HF])

        # Use tf.gather with batch_dims for efficient batched gathering
        num_windows = tf.shape(flat_indices)[0]
        flat_indices_batched = tf.tile(flat_indices[None, :, :, :], [B, 1, 1, 1])  # (B, num_windows, WSY, WSX)

        # Reshape for batch gather
        flat_indices_batched = tf.reshape(flat_indices_batched, [B, num_windows * WSY * WSX])

        # Gather: (B, num_windows*WSY*WSX, C)
        gathered = tf.gather(x_flat, flat_indices_batched, axis=1, batch_dims=1)

        # Reshape to (B, num_windows, WSY, WSX, C)
        gathered = tf.reshape(gathered, [B, num_windows, WSY, WSX, HF])

        # Transpose and reshape to (num_windows*B, WSY, WSX, C)
        windows = tf.transpose(gathered, [1, 0, 2, 3, 4])
        windows = tf.reshape(windows, [num_windows * B, WSY, WSX, HF])

        return windows

    @tf.function(jit_compile=True)
    def _extract_windows_vectorized_3d(self, x, z_starts, y_starts, x_starts):
        """
        Extract windows using vectorized operations.

        Args:
            x: (B, Z, Y, X, C)  -- NOTE: in multi-query mode x can be (Q*B, Z, Y, X, C)
            z_starts: (num_z,) - window start positions in Z dimension
            y_starts: (num_y,) - window start positions in Y dimension
            x_starts: (num_x,) - window start positions in X dimension

        Returns:
            windows: (num_z*num_y*num_x*B, WSZ, WSY, WSX, C)
        """
        B, Z, Y, X, _ = tf.unstack(tf.shape(x))
        HF = self.num_heads * self.attention_filters
        WSZ, WSY, WSX = self.window_size

        # Create grid of window coordinates
        z_grid, y_grid, x_grid = tf.meshgrid(z_starts, y_starts, x_starts, indexing='ij')  # (num_z, num_y, num_x)
        window_coords = tf.stack([z_grid, y_grid, x_grid], axis=-1)  # (num_z, num_y, num_x, 3)
        window_coords_flat = tf.reshape(window_coords, [-1, 3])  # (num_z*num_y*num_x, 3)

        # Create offsets within window
        z_offsets = tf.range(WSZ)
        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        z_off_grid, y_off_grid, x_off_grid = tf.meshgrid(z_offsets, y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.stack([z_off_grid, y_off_grid, x_off_grid], axis=-1)  # (WSZ, WSY, WSX, 3)

        # Broadcast to get all coordinates: (num_windows, WSZ, WSY, WSX, 3)
        all_coords = window_coords_flat[:, None, None, None, :] + window_offsets[None, :, :, :, :]

        # Clip to valid range
        all_coords = tf.clip_by_value(all_coords, 0, [Z - 1, Y - 1, X - 1])

        # Convert to flat indices
        flat_indices = all_coords[..., 0] * Y * X + all_coords[..., 1] * X + all_coords[..., 2]  # (num_windows, WSZ, WSY, WSX)

        # Reshape x for gathering: (B, Z*Y*X, C)
        x_flat = tf.reshape(x, [B, Z * Y * X, HF])

        # Use tf.gather with batch_dims for efficient batched gathering
        num_windows = tf.shape(flat_indices)[0]
        flat_indices_batched = tf.tile(flat_indices[None, :, :, :, :], [B, 1, 1, 1, 1])  # (B, num_windows, WSZ, WSY, WSX)

        # Reshape for batch gather
        flat_indices_batched = tf.reshape(flat_indices_batched, [B, num_windows * WSZ * WSY * WSX])

        # Gather: (B, num_windows*WSZ*WSY*WSX, C)
        gathered = tf.gather(x_flat, flat_indices_batched, axis=1, batch_dims=1)

        # Reshape to (B, num_windows, WSZ, WSY, WSX, C)
        gathered = tf.reshape(gathered, [B, num_windows, WSZ, WSY, WSX, HF])

        # Transpose and reshape to (num_windows*B, WSZ, WSY, WSX, C)
        windows = tf.transpose(gathered, [1, 0, 2, 3, 4, 5])
        windows = tf.reshape(windows, [num_windows * B, WSZ, WSY, WSX, HF])

        return windows

    @tf.function(jit_compile=True)
    def _scatter_windows_mean_2d(self, windows, y_starts, x_starts, Y, X):
        """
        Scatter windows back with simple averaging (optimized for mean reduction).

        Args:
            windows: (num_y*num_x*B, WSY, WSX, C)
            y_starts: (num_y,)
            x_starts: (num_x,)
            Y, X: original spatial dimensions

        Returns:
            output: (B, Y, X, C)
        """
        C = self.num_heads * self.attention_filters
        WSY, WSX = self.window_size
        num_y = tf.shape(y_starts)[0]
        num_x = tf.shape(x_starts)[0]
        num_windows = num_y * num_x
        B = tf.shape(windows)[0] // num_windows

        # Reshape: (num_windows, B, WSY, WSX, C)
        windows = tf.reshape(windows, [num_windows, B, WSY, WSX, C])

        # Initialize output and counts
        output = tf.zeros([B, Y, X, C], dtype=windows.dtype)
        counts = tf.zeros([B, Y, X, 1], dtype=windows.dtype)

        # Get window coordinates
        y_grid, x_grid = tf.meshgrid(y_starts, x_starts, indexing='ij')
        window_coords = tf.stack([y_grid, x_grid], axis=-1)
        window_coords_flat = tf.reshape(window_coords, [-1, 2])

        # Create offsets within window
        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        y_off_grid, x_off_grid = tf.meshgrid(y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.stack([y_off_grid, x_off_grid], axis=-1)

        # All coordinates: (num_windows, WSY, WSX, 2)
        all_coords = window_coords_flat[:, None, None, :] + window_offsets[None, :, :, :]
        all_coords = tf.clip_by_value(all_coords, 0, [Y - 1, X - 1])

        # Flatten: (num_windows*WSY*WSX, 2)
        scatter_coords = tf.reshape(all_coords, [-1, 2])

        # Create batch indices: (B, num_windows*WSY*WSX, 3) where last dim is [b, y, x]
        batch_indices = tf.range(B)
        b_indices = tf.tile(batch_indices[:, None], [1, num_windows * WSY * WSX])

        full_indices = tf.concat([
            b_indices[:, :, None],
            tf.tile(scatter_coords[None, :, :], [B, 1, 1])
        ], axis=-1)
        full_indices = tf.reshape(full_indices, [-1, 3])

        # Flatten windows: (B, num_windows, WSY, WSX, C) -> (B, num_windows*WSY*WSX, C)
        windows_transposed = tf.transpose(windows, [1, 0, 2, 3, 4])
        windows_flat = tf.reshape(windows_transposed, [B, -1, C])
        updates = tf.reshape(windows_flat, [-1, C])

        # Scatter add
        output = tf.tensor_scatter_nd_add(output, full_indices, updates)

        # Count occurrences
        count_updates = tf.ones([B * num_windows * WSY * WSX, 1], dtype=windows.dtype)
        counts = tf.tensor_scatter_nd_add(counts, full_indices, count_updates)

        # Average at overlaps
        output = output / tf.maximum(counts, 1.0)

        return output

    @tf.function(jit_compile=True)
    def _scatter_windows_with_averaging_2d(self, windows, window_weights, y_starts, x_starts, Y, X):
        """
        Scatter windows back to spatial dimensions with weighted averaging at overlaps.
        """
        C = self.num_heads * self.attention_filters
        WSY, WSX = self.window_size
        num_y = tf.shape(y_starts)[0]
        num_x = tf.shape(x_starts)[0]
        num_windows = num_y * num_x
        B = tf.shape(windows)[0] // num_windows

        # Reshape: (num_windows, B, WSY, WSX, C)
        windows = tf.reshape(windows, [num_windows, B, WSY, WSX, C])
        window_weights = tf.reshape(window_weights, [num_windows, B, WSY, WSX, 1])

        # Initialize output and weight sums
        output = tf.zeros([B, Y, X, C], dtype=windows.dtype)
        weight_sums = tf.zeros([B, Y, X, 1], dtype=windows.dtype)

        # Get window coordinates
        y_grid, x_grid = tf.meshgrid(y_starts, x_starts, indexing='ij')
        window_coords = tf.stack([y_grid, x_grid], axis=-1)
        window_coords_flat = tf.reshape(window_coords, [-1, 2])

        # Create offsets within window
        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        y_off_grid, x_off_grid = tf.meshgrid(y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.stack([y_off_grid, x_off_grid], axis=-1)

        # All coordinates: (num_windows, WSY, WSX, 2)
        all_coords = window_coords_flat[:, None, None, :] + window_offsets[None, :, :, :]
        all_coords = tf.clip_by_value(all_coords, 0, [Y - 1, X - 1])

        # Flatten: (num_windows*WSY*WSX, 2)
        scatter_coords = tf.reshape(all_coords, [-1, 2])

        # Create batch indices: (B, num_windows*WSY*WSX, 3)
        batch_indices = tf.range(B)
        b_indices = tf.tile(batch_indices[:, None], [1, num_windows * WSY * WSX])

        full_indices = tf.concat([
            b_indices[:, :, None],
            tf.tile(scatter_coords[None, :, :], [B, 1, 1])
        ], axis=-1)
        full_indices = tf.reshape(full_indices, [-1, 3])

        # Flatten windows and weights
        windows_transposed = tf.transpose(windows, [1, 0, 2, 3, 4])
        windows_flat = tf.reshape(windows_transposed, [B, -1, C])

        weights_transposed = tf.transpose(window_weights, [1, 0, 2, 3, 4])
        weights_flat = tf.reshape(weights_transposed, [B, -1, 1])

        # Weight the windows
        weighted_windows = windows_flat * weights_flat

        # Flatten for scatter
        updates = tf.reshape(weighted_windows, [-1, C])
        weight_updates = tf.reshape(weights_flat, [-1, 1])

        # Scatter add weighted values
        output = tf.tensor_scatter_nd_add(output, full_indices, updates)

        # Scatter add weights
        weight_sums = tf.tensor_scatter_nd_add(weight_sums, full_indices, weight_updates)

        epsilon = tf.cast(1e-3, output.dtype)
        output = output / tf.maximum(weight_sums, epsilon)

        return output

    @tf.function(jit_compile=True)
    def _scatter_windows_accumulate_2d(self, windows, window_weights, y_starts, x_starts, output, weight_sums, Y, X):
        """
        Directly scatter and accumulate windows into existing output tensors.
        This avoids creating intermediate full-size tensors.

        Args:
            windows: (num_windows*B, WSY, WSX, C)
            window_weights: (num_windows*B, WSY, WSX, 1)
            y_starts, x_starts: window positions
            output: (B, Y, X, C) - accumulator for weighted outputs
            weight_sums: (B, Y, X, 1) - accumulator for weights
            Y, X: spatial dimensions

        Returns:
            updated output, updated weight_sums
        """
        C = self.num_heads * self.attention_filters
        WSY, WSX = self.window_size
        num_y = tf.shape(y_starts)[0]
        num_x = tf.shape(x_starts)[0]
        num_windows = num_y * num_x
        B = tf.shape(windows)[0] // num_windows

        # Reshape to (num_windows, B, WSY, WSX, C/1)
        windows = tf.reshape(windows, [num_windows, B, WSY, WSX, C])
        window_weights = tf.reshape(window_weights, [num_windows, B, WSY, WSX, 1])

        # Compute scatter indices
        y_grid, x_grid = tf.meshgrid(y_starts, x_starts, indexing='ij')
        window_coords_flat = tf.reshape(tf.stack([y_grid, x_grid], -1), [-1, 2])

        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        y_off_grid, x_off_grid = tf.meshgrid(y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.reshape(tf.stack([y_off_grid, x_off_grid], -1), [-1, 2])

        all_coords = window_coords_flat[:, None, :] + window_offsets[None, :, :]
        all_coords = tf.clip_by_value(all_coords, 0, [Y - 1, X - 1])
        scatter_coords = tf.reshape(all_coords, [-1, 2])

        # Batch indices
        batch_indices = tf.range(B)
        b_indices = tf.tile(batch_indices[:, None], [1, num_windows * WSY * WSX])
        full_indices = tf.concat([
            b_indices[:, :, None],
            tf.tile(scatter_coords[None, :, :], [B, 1, 1])
        ], axis=-1)
        full_indices = tf.reshape(full_indices, [-1, 3])

        # Transpose and flatten: (num_windows, B, ...) -> (B, num_windows, ...)
        windows_transposed = tf.transpose(windows, [1, 0, 2, 3, 4])
        weights_transposed = tf.transpose(window_weights, [1, 0, 2, 3, 4])

        windows_flat = tf.reshape(windows_transposed, [B, -1, C])
        weights_flat = tf.reshape(weights_transposed, [B, -1, 1])

        # Weight the outputs
        weighted_windows = windows_flat * weights_flat

        # Flatten for scatter
        out_updates = tf.reshape(weighted_windows, [-1, C])
        weight_updates = tf.reshape(weights_flat, [-1, 1])

        # Accumulate into output tensors
        output = tf.tensor_scatter_nd_add(output, full_indices, out_updates)
        weight_sums = tf.tensor_scatter_nd_add(weight_sums, full_indices, weight_updates)

        return output, weight_sums

    def _geometrical_confidence_2d(self, min_confidence:float = 0.1):
        WSY, WSX = self.window_size
        y_center, x_center = (WSY - 1) / 2 ,  (WSX -1) / 2

        # Create distance map from center
        y_coords = tf.range(WSY, dtype=tf.float32) - y_center
        x_coords = tf.range(WSX, dtype=tf.float32) - x_center
        y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing='ij')
        dist_from_center = tf.sqrt(y_grid ** 2 + x_grid ** 2)

        # Invert: center = 1.0, edges → 0
        max_dist = tf.cast( tf.sqrt(float(y_center ** 2 + x_center ** 2)), tf.float32)
        confidence = tf.cast(1.0, tf.float32) - (dist_from_center / max_dist)
        max_confidence = tf.reduce_max(confidence)
        confidence = (confidence + min_confidence) / (max_confidence + min_confidence)
        return tf.reshape(confidence, [1, WSY, WSX, 1])

    @tf.function(jit_compile=True)
    def _scatter_windows_mean_3d(self, windows, z_starts, y_starts, x_starts, Z, Y, X):
        """
        Scatter 3D windows back with simple averaging.

        Args:
            windows: (num_z*num_y*num_x*B, WSZ, WSY, WSX, C)
            z_starts: (num_z,)
            y_starts: (num_y,)
            x_starts: (num_x,)
            Z, Y, X: original spatial dimensions

        Returns:
            output: (B, Z, Y, X, C)
        """
        C = self.num_heads * self.attention_filters
        WSZ, WSY, WSX = self.window_size  # unpack 3D window size
        num_z = tf.shape(z_starts)[0]
        num_y = tf.shape(y_starts)[0]
        num_x = tf.shape(x_starts)[0]
        num_windows = num_z * num_y * num_x
        B = tf.shape(windows)[0] // num_windows

        # Reshape: (num_windows, B, WSZ, WSY, WSX, C)
        windows = tf.reshape(windows, [num_windows, B, WSZ, WSY, WSX, C])

        # Initialize output and counts
        output = tf.zeros([B, Z, Y, X, C], dtype=windows.dtype)
        counts = tf.zeros([B, Z, Y, X, 1], dtype=windows.dtype)

        # Get window coordinates: each (num_z, num_y, num_x)
        z_grid, y_grid, x_grid = tf.meshgrid(z_starts, y_starts, x_starts, indexing='ij')
        window_coords = tf.stack([z_grid, y_grid, x_grid], axis=-1)
        window_coords_flat = tf.reshape(window_coords, [-1, 3])  # (num_windows, 3)

        # Create offsets within window
        z_offsets = tf.range(WSZ)
        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        z_off_grid, y_off_grid, x_off_grid = tf.meshgrid(z_offsets, y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.stack([z_off_grid, y_off_grid, x_off_grid], axis=-1)  # (WSZ, WSY, WSX, 3)

        # All coordinates: (num_windows, WSZ, WSY, WSX, 3)
        all_coords = window_coords_flat[:, None, None, None, :] + window_offsets[None, :, :, :, :]
        all_coords = tf.clip_by_value(all_coords, 0, [Z - 1, Y - 1, X - 1])

        # Flatten: (num_windows*WSZ*WSY*WSX, 3)
        scatter_coords = tf.reshape(all_coords, [-1, 3])

        # Create batch indices: (B, num_windows*WSZ*WSY*WSX, 4) where last dim is [b, z, y, x]
        batch_indices = tf.range(B)
        b_indices = tf.tile(batch_indices[:, None], [1, num_windows * WSZ * WSY * WSX])

        full_indices = tf.concat([
            b_indices[:, :, None],
            tf.tile(scatter_coords[None, :, :], [B, 1, 1])
        ], axis=-1)
        full_indices = tf.reshape(full_indices, [-1, 4])

        # Flatten windows: (B, num_windows*WSZ*WSY*WSX, C)
        windows_transposed = tf.transpose(windows, [1, 0, 2, 3, 4, 5])
        windows_flat = tf.reshape(windows_transposed, [B, -1, C])
        updates = tf.reshape(windows_flat, [-1, C])

        # Scatter add
        output = tf.tensor_scatter_nd_add(output, full_indices, updates)

        # Count occurrences
        count_updates = tf.ones([B * num_windows * WSZ * WSY * WSX, 1], dtype=windows.dtype)
        counts = tf.tensor_scatter_nd_add(counts, full_indices, count_updates)

        # Average at overlaps
        output = output / tf.maximum(counts, 1.0)

        return output

    @tf.function(jit_compile=True)
    def _scatter_windows_with_averaging_3d(self, windows, window_weights, z_starts, y_starts, x_starts, Z, Y, X):
        """
        Scatter 3D windows back to spatial dimensions with weighted averaging at overlaps.

        Args:
            windows: (num_z*num_y*num_x*B, WSZ, WSY, WSX, C)
            window_weights: (num_z*num_y*num_x*B, WSZ, WSY, WSX, 1)
            z_starts, y_starts, x_starts: window start positions
            Z, Y, X: original spatial dimensions

        Returns:
            output: (B, Z, Y, X, C)
        """
        C = self.num_heads * self.attention_filters
        WSZ, WSY, WSX = self.window_size
        num_z = tf.shape(z_starts)[0]
        num_y = tf.shape(y_starts)[0]
        num_x = tf.shape(x_starts)[0]
        num_windows = num_z * num_y * num_x
        B = tf.shape(windows)[0] // num_windows

        # Reshape: (num_windows, B, WSZ, WSY, WSX, C/1)
        windows = tf.reshape(windows, [num_windows, B, WSZ, WSY, WSX, C])
        window_weights = tf.reshape(window_weights, [num_windows, B, WSZ, WSY, WSX, 1])

        # Initialize output and weight sums
        output = tf.zeros([B, Z, Y, X, C], dtype=windows.dtype)
        weight_sums = tf.zeros([B, Z, Y, X, 1], dtype=windows.dtype)

        # Get window coordinates
        z_grid, y_grid, x_grid = tf.meshgrid(z_starts, y_starts, x_starts, indexing='ij')
        window_coords = tf.stack([z_grid, y_grid, x_grid], axis=-1)
        window_coords_flat = tf.reshape(window_coords, [-1, 3])

        # Create offsets within window
        z_offsets = tf.range(WSZ)
        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        z_off_grid, y_off_grid, x_off_grid = tf.meshgrid(z_offsets, y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.stack([z_off_grid, y_off_grid, x_off_grid], axis=-1)

        # All coordinates: (num_windows, WSZ, WSY, WSX, 3)
        all_coords = window_coords_flat[:, None, None, None, :] + window_offsets[None, :, :, :, :]
        all_coords = tf.clip_by_value(all_coords, 0, [Z - 1, Y - 1, X - 1])

        # Flatten: (num_windows*WSZ*WSY*WSX, 3)
        scatter_coords = tf.reshape(all_coords, [-1, 3])

        # Create batch indices: (B, num_windows*WSZ*WSY*WSX, 4)
        batch_indices = tf.range(B)
        b_indices = tf.tile(batch_indices[:, None], [1, num_windows * WSZ * WSY * WSX])

        full_indices = tf.concat([
            b_indices[:, :, None],
            tf.tile(scatter_coords[None, :, :], [B, 1, 1])
        ], axis=-1)
        full_indices = tf.reshape(full_indices, [-1, 4])

        # Flatten windows and weights
        windows_transposed = tf.transpose(windows, [1, 0, 2, 3, 4, 5])
        windows_flat = tf.reshape(windows_transposed, [B, -1, C])

        weights_transposed = tf.transpose(window_weights, [1, 0, 2, 3, 4, 5])
        weights_flat = tf.reshape(weights_transposed, [B, -1, 1])

        # Weight the windows
        weighted_windows = windows_flat * weights_flat

        # Flatten for scatter
        updates = tf.reshape(weighted_windows, [-1, C])
        weight_updates = tf.reshape(weights_flat, [-1, 1])

        # Scatter add weighted values and weights
        output = tf.tensor_scatter_nd_add(output, full_indices, updates)
        weight_sums = tf.tensor_scatter_nd_add(weight_sums, full_indices, weight_updates)

        epsilon = tf.cast(1e-3, output.dtype)
        output = output / tf.maximum(weight_sums, epsilon)

        return output

    @tf.function(jit_compile=True)
    def _scatter_windows_accumulate_3d(self, windows, window_weights, z_starts, y_starts, x_starts, output, weight_sums, Z, Y, X):
        """
        Directly scatter and accumulate 3D windows into existing output tensors.

        Args:
            windows: (num_windows*B, WSZ, WSY, WSX, C)
            window_weights: (num_windows*B, WSZ, WSY, WSX, 1)
            z_starts, y_starts, x_starts: window positions
            output: (B, Z, Y, X, C) - accumulator for weighted outputs
            weight_sums: (B, Z, Y, X, 1) - accumulator for weights
            Z, Y, X: spatial dimensions

        Returns:
            updated output: (B, Z, Y, X, C)
            updated weight_sums: (B, Z, Y, X, 1)
        """
        C = self.num_heads * self.attention_filters
        WSZ, WSY, WSX = self.window_size
        num_z = tf.shape(z_starts)[0]
        num_y = tf.shape(y_starts)[0]
        num_x = tf.shape(x_starts)[0]
        num_windows = num_z * num_y * num_x
        B = tf.shape(windows)[0] // num_windows

        # Reshape to (num_windows, B, WSZ, WSY, WSX, C/1)
        windows = tf.reshape(windows, [num_windows, B, WSZ, WSY, WSX, C])
        window_weights = tf.reshape(window_weights, [num_windows, B, WSZ, WSY, WSX, 1])

        # Compute scatter indices
        z_grid, y_grid, x_grid = tf.meshgrid(z_starts, y_starts, x_starts, indexing='ij')
        window_coords_flat = tf.reshape(tf.stack([z_grid, y_grid, x_grid], -1), [-1, 3])

        z_offsets = tf.range(WSZ)
        y_offsets = tf.range(WSY)
        x_offsets = tf.range(WSX)
        z_off_grid, y_off_grid, x_off_grid = tf.meshgrid(z_offsets, y_offsets, x_offsets, indexing='ij')
        window_offsets = tf.reshape(tf.stack([z_off_grid, y_off_grid, x_off_grid], -1), [-1, 3])

        # (num_windows, WSZ*WSY*WSX, 3)
        all_coords = window_coords_flat[:, None, :] + window_offsets[None, :, :]
        all_coords = tf.clip_by_value(all_coords, 0, [Z - 1, Y - 1, X - 1])
        scatter_coords = tf.reshape(all_coords, [-1, 3])

        # Batch indices: (B, num_windows*WSZ*WSY*WSX, 4)
        batch_indices = tf.range(B)
        b_indices = tf.tile(batch_indices[:, None], [1, num_windows * WSZ * WSY * WSX])
        full_indices = tf.concat([
            b_indices[:, :, None],
            tf.tile(scatter_coords[None, :, :], [B, 1, 1])
        ], axis=-1)
        full_indices = tf.reshape(full_indices, [-1, 4])

        # Transpose and flatten: (num_windows, B, ...) -> (B, num_windows, ...)
        windows_transposed = tf.transpose(windows, [1, 0, 2, 3, 4, 5])
        weights_transposed = tf.transpose(window_weights, [1, 0, 2, 3, 4, 5])

        windows_flat = tf.reshape(windows_transposed, [B, -1, C])
        weights_flat = tf.reshape(weights_transposed, [B, -1, 1])

        # Weight the outputs
        weighted_windows = windows_flat * weights_flat

        # Flatten for scatter
        out_updates = tf.reshape(weighted_windows, [-1, C])
        weight_updates = tf.reshape(weights_flat, [-1, 1])

        # Accumulate into output tensors
        output = tf.tensor_scatter_nd_add(output, full_indices, out_updates)
        weight_sums = tf.tensor_scatter_nd_add(weight_sums, full_indices, weight_updates)

        return output, weight_sums

    def _geometrical_confidence_3d(self, min_confidence: float = 0.1):
        """
        Compute a 3D spatial confidence map, highest at window center, decaying radially outward.

        Returns:
            confidence: (1, WSZ, WSY, WSX, 1)
        """
        WSZ, WSY, WSX = self.window_size
        z_center = (WSZ - 1) / 2
        y_center = (WSY - 1) / 2
        x_center = (WSX - 1) / 2

        # Distance from center along each axis
        z_coords = tf.range(WSZ, dtype=tf.float32) - z_center
        y_coords = tf.range(WSY, dtype=tf.float32) - y_center
        x_coords = tf.range(WSX, dtype=tf.float32) - x_center

        z_grid, y_grid, x_grid = tf.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        dist_from_center = tf.sqrt(z_grid ** 2 + y_grid ** 2 + x_grid ** 2)

        # Invert: center = 1.0, corners → 0
        max_dist = tf.cast(tf.sqrt(z_center ** 2 + y_center ** 2 + x_center ** 2), tf.float32)
        confidence = 1.0 - (dist_from_center / max_dist)

        max_confidence = tf.reduce_max(confidence)
        confidence = (confidence + min_confidence) / (max_confidence + min_confidence)

        return tf.reshape(confidence, [1, WSZ, WSY, WSX, 1])

    def _window_attention(self, q_windows, k_windows, v_windows, training=None, num_windows=None):
        """
        Compute attention within windows. Works for both 2D (WSY, WSX) and 3D (WSZ, WSY, WSX)
        window sizes, detected statically from len(self.window_size).

        Args (2D):
            q_windows: (num_windows * QB, WSY, WSX, HF)
            k_windows: (num_windows *  B, WSY, WSX, HF)
            v_windows: (num_windows *  B, WSY, WSX, HF)

        Args (3D):
            q_windows: (num_windows * QB, WSZ, WSY, WSX, HF)
            k_windows: (num_windows *  B, WSZ, WSY, WSX, HF)
            v_windows: (num_windows *  B, WSZ, WSY, WSX, HF)

        Returns:
            output:             same shape as q_windows
            attention_weights:  same spatial shape + trailing 1, or None
        """
        is_3d = len(self.window_size) == 3

        HF = self.num_heads * self.attention_filters
        H = self.num_heads
        F = self.attention_filters

        if is_3d:
            WSZ, WSY, WSX = self.window_size
            N = WSZ * WSY * WSX
            spatial_shape = [WSZ, WSY, WSX]
            spatial_axes = [1, 2, 3]  # axes to reduce over for normalisation
        else:
            WSY, WSX = self.window_size
            WSZ = 1
            N = WSY * WSX
            spatial_shape = [WSY, WSX]
            spatial_axes = [1, 2]

        # ------------------------------------------------------------------ #
        #  Distance embedding                                                  #
        # ------------------------------------------------------------------ #
        if self.add_distance_embedding:
            if WSZ > 1:
                hz, hy, hx = self._get_distance_encoding_output_dim()
                emb_z = tf.reshape(self.distance_encoding_z(tf.range(WSZ)), (1, WSZ, 1, 1, hz))
                emb_y = tf.reshape(self.distance_encoding_y(tf.range(WSY)), (1, 1, WSY, 1, hy))
                emb_x = tf.reshape(self.distance_encoding_x(tf.range(WSX)), (1, 1, 1, WSX, hx))
                distance_emb = tf.concat([
                    tf.broadcast_to(emb_z, [1, WSZ, WSY, WSX, hz]),
                    tf.broadcast_to(emb_y, [1, WSZ, WSY, WSX, hy]),
                    tf.broadcast_to(emb_x, [1, WSZ, WSY, WSX, hx]),
                ], axis=-1)  # (1, WSZ, WSY, WSX, HF)
            else:
                hy, hx = self._get_distance_encoding_output_dim()
                emb_y = tf.reshape(self.distance_encoding_y(tf.range(WSY)), (1, WSY, 1, hy))
                emb_x = tf.reshape(self.distance_encoding_x(tf.range(WSX)), (1, 1, WSX, hx))
                distance_emb = tf.concat([
                    tf.broadcast_to(emb_y, [1, WSY, WSX, hy]),
                    tf.broadcast_to(emb_x, [1, WSY, WSX, hx]),
                ], axis=-1)  # (1, WSY, WSX, HF)
                if len(self.window_size) == 3:
                    distance_emb = tf.expand_dims(distance_emb, axis=1) # WSZ = 1 but still a 5D tensor
            v_windows = v_windows - distance_emb

        # ------------------------------------------------------------------ #
        #  Relative position bias (shared between single/multi-query)          #
        # ------------------------------------------------------------------ #
        rpb = tf.gather(self.relative_position_bias_table,
                        tf.reshape(self.relative_position_index, [-1]))
        rpb = tf.reshape(rpb, [N, N, H])
        rpb = tf.transpose(rpb, [2, 0, 1])  # (H, N, N)

        # ------------------------------------------------------------------ #
        #  Single-query path                                                   #
        # ------------------------------------------------------------------ #
        if not self.multi_query:
            B_win = tf.shape(q_windows)[0]

            q = tf.reshape(q_windows, [B_win, N, H, F])
            k = tf.reshape(k_windows, [B_win, N, H, F])
            v = tf.reshape(v_windows, [B_win, N, H, F])

            q = tf.transpose(q, [0, 2, 1, 3])  # (B_win, H, N, F)
            v = tf.transpose(v, [0, 2, 1, 3])  # (B_win, H, N, F)
            k = tf.transpose(k, [0, 2, 3, 1])  # (B_win, H, F, N)

            scale = tf.math.rsqrt(tf.cast(F, q.dtype))
            attn = tf.matmul(q, k) * scale  # (B_win, H, N, N)
            attn = attn + rpb[None, :, :, :]

            attn_probs = tf.nn.softmax(attn, axis=-1)
            if self.dropout > 0 and training:
                attn_probs = self.dropout_layer(attn_probs, training=training)

            if self.overlap_reduction == "attention_weighted":
                max_attn = tf.reduce_max(attn_probs, axis=-1)  # (B_win, H, N)
                confidence = tf.reduce_mean(max_attn, axis=1)  # (B_win, N)
                confidence_spatial = tf.reshape(confidence, [B_win] + spatial_shape + [1])
                confidence_spatial = confidence_spatial / tf.reduce_sum(
                    confidence_spatial, axis=spatial_axes, keepdims=True)
                confidence_spatial = tf.maximum(confidence_spatial,
                                                tf.cast(1e-3, confidence_spatial.dtype))
                confidence_spatial = tf.stop_gradient(confidence_spatial)
            else:
                confidence_spatial = None

            out = tf.matmul(attn_probs, tf.cast(v, attn_probs.dtype))  # (B_win, H, N, F)
            out = tf.cast(out, v.dtype)
            out = tf.transpose(out, [0, 2, 1, 3])  # (B_win, N, H, F)
            out = tf.reshape(out, [B_win] + spatial_shape + [HF])

            if self.add_distance_embedding:
                out = out + distance_emb
            return out, confidence_spatial

        # ------------------------------------------------------------------ #
        #  Multi-query path                                                    #
        # ------------------------------------------------------------------ #
        else:
            assert num_windows is not None, "num_windows must be provided in multi-query mode"

            # N is the only thing that differs between 2D/3D here; reshapes are identical
            q = tf.reshape(q_windows, [num_windows, self.q_count, -1, N, H, F])  # (W, Q, B, N, H, F)
            k = tf.reshape(k_windows, [num_windows, -1, N, H, F])  # (W,    B, N, H, F)
            v = tf.reshape(v_windows, [num_windows, -1, N, H, F])  # (W,    B, N, H, F)

            scale = tf.math.rsqrt(tf.cast(F, q.dtype))
            attn = tf.einsum('mqbihf,mbkhf->mqbhik', q, k, optimize='optimal') * scale  # (W,Q,B,H,N,N)
            attn = attn + rpb[None, None, None, :, :, :]

            attn_probs = tf.nn.softmax(attn, axis=-1)
            if self.dropout > 0 and training:
                attn_probs = self.dropout_layer(attn_probs, training=training)

            confidence_spatial = None
            if self.overlap_reduction == "attention_weighted":
                max_attn = tf.reduce_max(attn_probs, axis=-1)  # (W, Q, B, H, N)
                confidence = tf.reduce_mean(max_attn, axis=3)  # (W, Q, B, N)
                confidence_spatial = tf.reshape(confidence, [-1] + spatial_shape + [1])
                confidence_spatial = confidence_spatial / tf.reduce_sum(
                    confidence_spatial, axis=spatial_axes, keepdims=True)
                confidence_spatial = tf.maximum(confidence_spatial,
                                                tf.cast(1e-3, confidence_spatial.dtype))
                confidence_spatial = tf.stop_gradient(confidence_spatial)

            out = tf.einsum('mqbhik,mbkhf->mqbihf', attn_probs, v, optimize='optimal')
            out = tf.reshape(out, [-1] + spatial_shape + [HF])

            if self.add_distance_embedding:
                out = out + distance_emb
            return out, confidence_spatial

    def call(self, x, training: bool = None):

        # ------------------------------------------------------------------ #
        #  Static 2D / 3D switch                                              #
        # ------------------------------------------------------------------ #
        is_3d = len(self.window_size) == 3

        if is_3d:
            WSZ, WSY, WSX = self.window_size
            _extract = self._extract_windows_vectorized_3d
            _mean = self._scatter_windows_mean_3d
            _weighted = self._scatter_windows_with_averaging_3d
            _accum = self._scatter_windows_accumulate_3d
        else:
            WSY, WSX = self.window_size
            _extract = self._extract_windows_vectorized_2d
            _mean = self._scatter_windows_mean_2d
            _weighted = self._scatter_windows_with_averaging_2d
            _accum = self._scatter_windows_accumulate_2d

        # ------------------------------------------------------------------ #
        #  Input handling                                                      #
        # ------------------------------------------------------------------ #
        if isinstance(x, list):
            if len(x) == 1:
                assert not self.multi_query
                source_query = x[0]
                x = self.ln_q(x[0]) if self.layer_normalization else x[0]
                key = value = query = x

            elif len(x) == 2:
                source_query, key = x
                query = source_query
                if self.multi_query:
                    if is_3d:
                        Q, B, Z, Y, X, _ = tf.unstack(tf.shape(query))
                        query = tf.reshape(query, [Q * B, Z, Y, X, self.filters])
                    else:
                        Q, B, Y, X, _ = tf.unstack(tf.shape(query))
                        query = tf.reshape(query, [Q * B, Y, X, self.filters])
                if self.layer_normalization:
                    query = self.ln_q(query)
                    key = self.ln_k(key)
                value = key

            elif len(x) == 3:
                source_query, key, value = x
                query = source_query
                if self.multi_query:
                    if is_3d:
                        Q, B, Z, Y, X, _ = tf.unstack(tf.shape(query))
                        query = tf.reshape(query, [Q * B, Z, Y, X, self.filters])
                    else:
                        Q, B, Y, X, _ = tf.unstack(tf.shape(query))
                        query = tf.reshape(query, [Q * B, Y, X, self.filters])
                if self.layer_normalization:
                    query = self.ln_q(query)
                    key = self.ln_k(key)
                    value = self.ln_v(value)

            elif len(x) == 4:
                source_query, key, value, (emb_q, emb_k) = x
                if isinstance(emb_q, tuple):
                    emb_mul, emb_add = emb_q
                    query = source_query * emb_mul + emb_add
                else:
                    query = source_query + emb_q
                if isinstance(emb_k, tuple):
                    emb_mul, emb_add = emb_k
                    key = key * emb_mul + emb_add
                else:
                    key = key + emb_k
                if self.multi_query:
                    if is_3d:
                        Q, B, Z, Y, X, _ = tf.unstack(tf.shape(query))
                        query = tf.reshape(query, [Q * B, Z, Y, X, self.filters])
                    else:
                        Q, B, Y, X, _ = tf.unstack(tf.shape(query))
                        query = tf.reshape(query, [Q * B, Y, X, self.filters])
                if self.layer_normalization:
                    query = self.ln_q(query)
                    key = self.ln_k(key)
                    value = self.ln_v(value)
            else:
                raise ValueError("Invalid input length, should be <= 4")

        else:
            assert not self.multi_query
            source_query = x
            if self.layer_normalization:
                x = self.ln_q(x)
            key = value = query = x

        # ------------------------------------------------------------------ #
        #  Spatial shape                                                       #
        # ------------------------------------------------------------------ #
        if is_3d:
            B, Z, Y, X, _ = tf.unstack(tf.shape(query))
        else:
            B, Y, X, _ = tf.unstack(tf.shape(query))

        # ------------------------------------------------------------------ #
        #  Q / K / V projections                                               #
        # ------------------------------------------------------------------ #
        Q_proj = self.qproj(query)
        K_proj = self.kproj(key)
        V_proj = self.vproj(value)

        window_processing = (self.window_processing if self.window_processing != "auto"
                             else ("sequential" if self.inference_mode else "batch"))

        # ------------------------------------------------------------------ #
        #  Single-window branch (image fits in one window)                     #
        # ------------------------------------------------------------------ #
        def single():
            if is_3d:
                pad_z = tf.maximum(0, WSZ - Z)
                pad_y = tf.maximum(0, WSY - Y)
                pad_x = tf.maximum(0, WSX - X)
                need_padding = tf.reduce_any(tf.greater([pad_z, pad_y, pad_x], 0))

                def pad():
                    Qp = tf.pad(Q_proj, [[0, 0], [0, pad_z], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                    Kp = tf.pad(K_proj, [[0, 0], [0, pad_z], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                    Vp = tf.pad(V_proj, [[0, 0], [0, pad_z], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                    return Qp, Kp, Vp

                Qp, Kp, Vp = tf.cond(need_padding, pad, lambda: (Q_proj, K_proj, V_proj))
                out_windows, _ = self._window_attention(Qp, Kp, Vp, training, num_windows=1)
                return out_windows[:, :Z, :Y, :X, :]
            else:
                pad_y = tf.maximum(0, WSY - Y)
                pad_x = tf.maximum(0, WSX - X)
                need_padding = tf.reduce_any(tf.greater([pad_y, pad_x], 0))

                def pad():
                    Qp = tf.pad(Q_proj, [[0, 0], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                    Kp = tf.pad(K_proj, [[0, 0], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                    Vp = tf.pad(V_proj, [[0, 0], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                    return Qp, Kp, Vp

                Qp, Kp, Vp = tf.cond(need_padding, pad, lambda: (Q_proj, K_proj, V_proj))
                out_windows, _ = self._window_attention(Qp, Kp, Vp, training, num_windows=1)
                return out_windows[:, :Y, :X, :]

        # ------------------------------------------------------------------ #
        #  Multi-window branch                                                 #
        # ------------------------------------------------------------------ #
        def multiple():
            C = self.num_heads * self.attention_filters
            B_size = tf.shape(Q_proj)[0]

            if is_3d:
                z_starts, y_starts, x_starts = self._compute_window_grid_3d(Z, Y, X)
                num_z = tf.shape(z_starts)[0]
                num_y = tf.shape(y_starts)[0]
                num_x = tf.shape(x_starts)[0]

                if window_processing == 'batch':
                    num_windows = num_z * num_y * num_x
                    Q_windows = _extract(Q_proj, z_starts, y_starts, x_starts)
                    K_windows = _extract(K_proj, z_starts, y_starts, x_starts)
                    V_windows = _extract(V_proj, z_starts, y_starts, x_starts)
                    out_windows, attention_confidence = self._window_attention(
                        Q_windows, K_windows, V_windows, training, num_windows=num_windows)

                    if self.overlap_reduction == 'attention_weighted':
                        return _weighted(out_windows, attention_confidence,
                                         z_starts, y_starts, x_starts, Z, Y, X)
                    elif self.overlap_reduction == 'geometrical':
                        geo = tf.tile(tf.cast(self.geometrical_confidence, out_windows.dtype),
                                      [tf.shape(out_windows)[0], 1, 1, 1, 1])
                        return _weighted(out_windows, geo,
                                         z_starts, y_starts, x_starts, Z, Y, X)
                    else:
                        return _mean(out_windows, z_starts, y_starts, x_starts, Z, Y, X)

                else:
                    # Chunked / sequential processing
                    if window_processing == 'slice':  # one Z-slice at a time
                        num_chunks = num_z
                        get_coords = lambda i: (z_starts[i:i + 1], y_starts, x_starts)
                    elif window_processing == 'row':
                        num_chunks = num_z * num_y
                        get_coords = lambda i: (z_starts[i // num_y:i // num_y + 1],
                                                y_starts[i % num_y:i % num_y + 1],
                                                x_starts)
                    else:  # sequential
                        total_windows = num_z * num_y * num_x
                        num_chunks = total_windows
                        get_coords = lambda idx: (
                            z_starts[idx // (num_y * num_x): idx // (num_y * num_x) + 1],
                            y_starts[(idx % (num_y * num_x)) // num_x: (idx % (num_y * num_x)) // num_x + 1],
                            x_starts[idx % num_x: idx % num_x + 1],
                        )

                    def process_chunk(idx, output_acc, weights_acc):
                        z_sub, y_sub, x_sub = get_coords(idx)
                        num_wins = tf.shape(z_sub)[0] * tf.shape(y_sub)[0] * tf.shape(x_sub)[0]

                        Q_wins = _extract(Q_proj, z_sub, y_sub, x_sub)
                        K_wins = _extract(K_proj, z_sub, y_sub, x_sub)
                        V_wins = _extract(V_proj, z_sub, y_sub, x_sub)

                        out_wins, attn_conf = self._window_attention(
                            Q_wins, K_wins, V_wins, training, num_wins)

                        if self.overlap_reduction == 'attention_weighted':
                            conf = attn_conf
                        elif self.overlap_reduction == 'geometrical':
                            conf = tf.tile(tf.cast(self.geometrical_confidence, out_wins.dtype),
                                           [tf.shape(out_wins)[0], 1, 1, 1, 1])
                        else:
                            conf = tf.ones([tf.shape(out_wins)[0], WSZ, WSY, WSX, 1],
                                           dtype=out_wins.dtype)

                        output_acc, weights_acc = _accum(
                            out_wins, conf, z_sub, y_sub, x_sub,
                            output_acc, weights_acc, Z, Y, X)
                        return idx + 1, output_acc, weights_acc

                    initial_output = tf.zeros([B_size, Z, Y, X, C], dtype=Q_proj.dtype)
                    initial_weights = tf.zeros([B_size, Z, Y, X, 1], dtype=Q_proj.dtype)

                    _, output, weight_sums = tf.while_loop(
                        cond=lambda i, o, w: i < num_chunks,
                        body=process_chunk,
                        loop_vars=[0, initial_output, initial_weights],
                        parallel_iterations=1,
                        swap_memory=False,
                        shape_invariants=[
                            tf.TensorShape([]),
                            tf.TensorShape([None, None, None, None, C]),
                            tf.TensorShape([None, None, None, None, 1]),
                        ]
                    )
                    epsilon = tf.cast(1e-3 if self.overlap_reduction != 'mean' else 1.0, output.dtype)
                    return output / tf.maximum(weight_sums, epsilon)

            else:
                # ---- 2D (unchanged) ----------------------------------------
                y_starts, x_starts = self._compute_window_grid_2d(Y, X)
                num_y = tf.shape(y_starts)[0]
                num_x = tf.shape(x_starts)[0]

                if window_processing == 'batch':
                    num_windows = num_y * num_x
                    Q_windows = _extract(Q_proj, y_starts, x_starts)
                    K_windows = _extract(K_proj, y_starts, x_starts)
                    V_windows = _extract(V_proj, y_starts, x_starts)
                    out_windows, attention_confidence = self._window_attention(
                        Q_windows, K_windows, V_windows, training, num_windows=num_windows)

                    if self.overlap_reduction == 'attention_weighted':
                        return _weighted(out_windows, attention_confidence, y_starts, x_starts, Y, X)
                    elif self.overlap_reduction == 'geometrical':
                        geo = tf.tile(tf.cast(self.geometrical_confidence, out_windows.dtype),
                                      [tf.shape(out_windows)[0], 1, 1, 1])
                        return _weighted(out_windows, geo, y_starts, x_starts, Y, X)
                    else:
                        return _mean(out_windows, y_starts, x_starts, Y, X)

                else:
                    if window_processing == 'row':
                        num_chunks = num_y
                        get_coords = lambda i: (y_starts[i:i + 1], x_starts)
                    elif window_processing == 'col':
                        num_chunks = num_x
                        get_coords = lambda j: (y_starts, x_starts[j:j + 1])
                    else:  # sequential
                        num_chunks = num_y * num_x
                        get_coords = lambda idx: (y_starts[idx // num_x:idx // num_x + 1],
                                                  x_starts[idx % num_x:idx % num_x + 1])

                    def process_chunk(idx, output_acc, weights_acc):
                        y_sub, x_sub = get_coords(idx)
                        num_wins = tf.shape(y_sub)[0] * tf.shape(x_sub)[0]

                        Q_wins = _extract(Q_proj, y_sub, x_sub)
                        K_wins = _extract(K_proj, y_sub, x_sub)
                        V_wins = _extract(V_proj, y_sub, x_sub)

                        out_wins, attn_conf = self._window_attention(
                            Q_wins, K_wins, V_wins, training, num_wins)

                        if self.overlap_reduction == 'attention_weighted':
                            conf = attn_conf
                        elif self.overlap_reduction == 'geometrical':
                            conf = tf.tile(tf.cast(self.geometrical_confidence, out_wins.dtype),
                                           [tf.shape(out_wins)[0], 1, 1, 1])
                        else:
                            conf = tf.ones([tf.shape(out_wins)[0], WSY, WSX, 1], dtype=out_wins.dtype)

                        output_acc, weights_acc = _accum(
                            out_wins, conf, y_sub, x_sub, output_acc, weights_acc, Y, X)
                        return idx + 1, output_acc, weights_acc

                    initial_output = tf.zeros([B_size, Y, X, C], dtype=Q_proj.dtype)
                    initial_weights = tf.zeros([B_size, Y, X, 1], dtype=Q_proj.dtype)

                    _, output, weight_sums = tf.while_loop(
                        cond=lambda i, o, w: i < num_chunks,
                        body=process_chunk,
                        loop_vars=[0, initial_output, initial_weights],
                        parallel_iterations=1,
                        swap_memory=False,
                        shape_invariants=[
                            tf.TensorShape([]),
                            tf.TensorShape([None, None, None, C]),
                            tf.TensorShape([None, None, None, 1]),
                        ]
                    )
                    epsilon = tf.cast(1e-3 if self.overlap_reduction != 'mean' else 1.0, output.dtype)
                    return output / tf.maximum(weight_sums, epsilon)

        # ------------------------------------------------------------------ #
        #  Dispatch                                                            #
        # ------------------------------------------------------------------ #
        if is_3d:
            single_window = tf.logical_and(tf.logical_and(tf.less_equal(Z, WSZ), tf.less_equal(Y, WSY)), tf.less_equal(X, WSX))
        else:
            single_window = tf.logical_and(tf.less_equal(Y, WSY), tf.less_equal(X, WSX))

        output = tf.cond(single_window, single, multiple)

        # ------------------------------------------------------------------ #
        #  Output projection + skip connection                                 #
        # ------------------------------------------------------------------ #
        output = self.outproj(output)
        if self.multi_query:
            output = tf.reshape(output, tf.shape(source_query))
        if self.skip_connection:
            output = output + source_query

        return output
# ============================================================================
# TEST CODE  –  2D + 3D
# ============================================================================

def test_coordinate_handling(multi_query: bool = False, mode: str = '2d'):
    """
    Test that window extraction and scattering preserves data correctly.
    mode: '2d' | '3d'
    """
    assert mode in ('2d', '3d'), "mode must be '2d' or '3d'"
    is_3d = mode == '3d'

    print("=" * 60)
    print(f"Testing Window Coordinate Handling  [{mode.upper()}]")
    print("=" * 60)

    for overlap_method in ['mean', 'attention_weighted', 'geometrical']:
        print(f"\n{'=' * 60}")
        print(f"Testing with overlap_reduction='{overlap_method}'")
        print(f"{'=' * 60}")

        if is_3d:
            window_size = (4, 7, 4)
            B, Z, H, W, C = 2, 8, 20, 4, 16
            k = tf.random.normal([B, Z, H, W, C])
            layer = WindowSpatialAttention(
                num_heads=4, attention_filters=4,
                window_size=window_size, overlap_reduction=overlap_method)

            if multi_query:
                Q = 2
                q = tf.random.normal([Q, B, Z, H, W, C])
                q_resh = tf.reshape(q, [-1, Z, H, W, C])
            else:
                Q = 1
                q = tf.random.normal([B, Z, H, W, C])
                q_resh = q

            layer.build([q.shape, k.shape])

            print(f"\nVolume size: {Z}x{H}x{W}")
            print(f"Window size: {layer.window_size}")

            z_starts, h_starts, w_starts = layer._compute_window_grid_3d(Z, H, W)
            print(f"\nWindow grid:")
            print(f"  z_starts: {z_starts.numpy()}")
            print(f"  h_starts: {h_starts.numpy()}")
            print(f"  w_starts: {w_starts.numpy()}")
            n_wins = len(z_starts) * len(h_starts) * len(w_starts)
            print(f"  Number of windows: {len(z_starts)}x{len(h_starts)}x{len(w_starts)} = {n_wins}")

            max_z_covered = tf.reduce_max(z_starts) + layer.window_size[0]
            max_h_covered = tf.reduce_max(h_starts) + layer.window_size[1]
            max_w_covered = tf.reduce_max(w_starts) + layer.window_size[2]
            print(f"\nCoverage check:")
            print(f"  Max Z covered: {max_z_covered.numpy()} (volume Z: {Z})")
            print(f"  Max H covered: {max_h_covered.numpy()} (volume H: {H})")
            print(f"  Max W covered: {max_w_covered.numpy()} (volume W: {W})")
            print(f"  Full coverage: {max_z_covered >= Z and max_h_covered >= H and max_w_covered >= W}")

            print(f"\n{'-' * 60}")
            print("Testing Extract -> Scatter Round-trip")
            print(f"{'-' * 60}")

            windows = layer._extract_windows_vectorized_3d(q_resh, z_starts, h_starts, w_starts)
            print(f"Windows shape: {windows.shape}")

            WSZ, WSY, WSX = layer.window_size
            if overlap_method == 'mean':
                q_reconstructed = layer._scatter_windows_mean_3d(
                    windows, z_starts, h_starts, w_starts, Z, H, W)
            else:
                weights = tf.random.uniform(
                    [tf.shape(windows)[0], WSZ, WSY, WSX, 1], 0.3, 1.0)
                q_reconstructed = layer._scatter_windows_with_averaging_3d(
                    windows, weights, z_starts, h_starts, w_starts, Z, H, W)
            print(f"Reconstructed shape: {q_reconstructed.shape}")

            diff = tf.abs(q_resh - q_reconstructed)
            print(f"\nReconstruction quality:")
            print(f"  Max absolute error:  {tf.reduce_max(diff).numpy():.6f}")
            print(f"  Mean absolute error: {tf.reduce_mean(diff).numpy():.6f}")

            # Overlap statistics
            counts = tf.zeros([Q * B, Z, H, W, 1], dtype=q.dtype)
            for z_s in z_starts.numpy():
                for h_s in h_starts.numpy():
                    for w_s in w_starts.numpy():
                        z_e = min(z_s + WSZ, Z)
                        h_e = min(h_s + WSY, H)
                        w_e = min(w_s + WSX, W)
                        c = counts.numpy()
                        c[:, z_s:z_e, h_s:h_e, w_s:w_e, :] += 1
                        counts = tf.constant(c)

            overlap_mask = counts > 1
            print(f"\nOverlap statistics:")
            print(f"  Pixels with overlaps: {tf.reduce_sum(tf.cast(overlap_mask, tf.int32)).numpy()}")
            print(f"  Max overlap count:    {tf.reduce_max(counts).numpy():.0f}")

            print(f"\n{'-' * 60}")
            print("Testing Full Forward Pass")
            print(f"{'-' * 60}")

            output = layer([q, k], training=False)
            print(f"Output shape: {output.shape}")
            print(f"Input shape:  {q.shape}")
            assert output.shape == q.shape, "Output shape should match input shape!"
            print("✓ Shape check passed")

        else:
            # ---- Original 2D path ----------------------------------------
            window_size = (7, 4)
            B, H, W, C = 2, 20, 4, 16
            k = tf.random.normal([B, H, W, C])
            layer = WindowSpatialAttention(
                num_heads=4, attention_filters=4,
                window_size=window_size, overlap_reduction=overlap_method)

            if multi_query:
                Q = 2
                q = tf.random.normal([Q, B, H, W, C])
                q_resh = tf.reshape(q, [-1, H, W, C])
            else:
                Q = 1
                q = tf.random.normal([B, H, W, C])
                q_resh = q

            layer.build([q.shape, k.shape])

            print(f"\nImage size: {H}x{W}")
            print(f"Window size: {layer.window_size}")

            h_starts, w_starts = layer._compute_window_grid_2d(H, W)
            print(f"\nWindow grid:")
            print(f"  h_starts: {h_starts.numpy()}")
            print(f"  w_starts: {w_starts.numpy()}")
            print(f"  Number of windows: {len(h_starts)} x {len(w_starts)} = {len(h_starts) * len(w_starts)}")

            max_h_covered = tf.reduce_max(h_starts) + layer.window_size[0]
            max_w_covered = tf.reduce_max(w_starts) + layer.window_size[1]
            print(f"\nCoverage check:")
            print(f"  Max H covered: {max_h_covered.numpy()} (image H: {H})")
            print(f"  Max W covered: {max_w_covered.numpy()} (image W: {W})")
            print(f"  Full coverage: {max_h_covered >= H and max_w_covered >= W}")

            print(f"\n{'-' * 60}")
            print("Testing Extract -> Scatter Round-trip")
            print(f"{'-' * 60}")

            windows = layer._extract_windows_vectorized_2d(q_resh, h_starts, w_starts)
            print(f"Windows shape: {windows.shape}")

            WSY, WSX = layer.window_size
            if overlap_method == 'mean':
                q_reconstructed = layer._scatter_windows_mean_2d(
                    windows, h_starts, w_starts, H, W)
            else:
                weights = tf.random.uniform(
                    [tf.shape(windows)[0], WSY, WSX, 1], 0.3, 1.0)
                q_reconstructed = layer._scatter_windows_with_averaging_2d(
                    windows, weights, h_starts, w_starts, H, W)
            print(f"Reconstructed shape: {q_reconstructed.shape}")

            diff = tf.abs(q_resh - q_reconstructed)
            print(f"\nReconstruction quality:")
            print(f"  Max absolute error:  {tf.reduce_max(diff).numpy():.6f}")
            print(f"  Mean absolute error: {tf.reduce_mean(diff).numpy():.6f}")

            counts = tf.zeros([Q * B, H, W, 1], dtype=q.dtype)
            for h_s in h_starts.numpy():
                for w_s in w_starts.numpy():
                    h_e = min(h_s + WSY, H)
                    w_e = min(w_s + WSX, W)
                    c = counts.numpy()
                    c[:, h_s:h_e, w_s:w_e, :] += 1
                    counts = tf.constant(c)

            overlap_mask = counts > 1
            print(f"\nOverlap statistics:")
            print(f"  Pixels with overlaps: {tf.reduce_sum(tf.cast(overlap_mask, tf.int32)).numpy()}")
            print(f"  Max overlap count:    {tf.reduce_max(counts).numpy():.0f}")

            print(f"\n{'-' * 60}")
            print("Testing Full Forward Pass")
            print(f"{'-' * 60}")

            output = layer([q, k], training=False)
            print(f"Output shape: {output.shape}")
            print(f"Input shape:  {q.shape}")
            assert output.shape == q.shape, "Output shape should match input shape!"
            print("✓ Shape check passed")

    # ---- Multiple-size sweep ------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Testing Multiple {'Volume' if is_3d else 'Image'} Sizes (mean reduction)")
    print("=" * 60)

    C = 16
    if is_3d:
        layer = WindowSpatialAttention(
            num_heads=1, attention_filters=C,
            window_size=(4, 7, 7), overlap_reduction='mean')
        test_sizes = [(6, 14, 14), (8, 20, 30), (12, 50, 50), (4, 7, 7)]

        for Z_t, H_t, W_t in test_sizes:
            k_t = tf.random.normal([1, Z_t, H_t, W_t, C])
            if multi_query:
                Q = 2
                q_t = tf.random.normal([Q, 1, Z_t, H_t, W_t, C])
                q_t_resh = tf.reshape(q_t, [-1, Z_t, H_t, W_t, C])
            else:
                Q = 1
                q_t = tf.random.normal([1, Z_t, H_t, W_t, C])
                q_t_resh = q_t

            layer.build([q_t.shape, k_t.shape])
            z_s, h_s, w_s = layer._compute_window_grid_3d(Z_t, H_t, W_t)
            windows = layer._extract_windows_vectorized_3d(q_t_resh, z_s, h_s, w_s)
            q_recon = layer._scatter_windows_mean_3d(windows, z_s, h_s, w_s, Z_t, H_t, W_t)
            error = tf.reduce_max(tf.abs(q_t_resh - q_recon))
            n_wins = len(z_s) * len(h_s) * len(w_s)
            print(f"Size {Z_t:3d}x{H_t:3d}x{W_t:3d}: "
                  f"windows={len(z_s):2d}x{len(h_s):2d}x{len(w_s):2d}={n_wins:4d}, "
                  f"max_error={error.numpy():.6f}")
    else:
        layer = WindowSpatialAttention(
            num_heads=1, attention_filters=C,
            window_size=7, overlap_reduction='mean')
        test_sizes = [(14, 14), (20, 30), (50, 50), (7, 7)]

        for H_t, W_t in test_sizes:
            k_t = tf.random.normal([1, H_t, W_t, C])
            if multi_query:
                Q = 2
                q_t = tf.random.normal([Q, 1, H_t, W_t, C])
                q_t_resh = tf.reshape(q_t, [-1, H_t, W_t, C])
            else:
                Q = 1
                q_t = tf.random.normal([1, H_t, W_t, C])
                q_t_resh = q_t

            layer.build([q_t.shape, k_t.shape])
            h_s, w_s = layer._compute_window_grid_2d(H_t, W_t)
            windows = layer._extract_windows_vectorized_2d(q_t_resh, h_s, w_s)
            q_recon = layer._scatter_windows_mean_2d(windows, h_s, w_s, H_t, W_t)
            error = tf.reduce_max(tf.abs(q_t_resh - q_recon))
            print(f"Size {H_t:3d}x{W_t:3d}: "
                  f"windows={len(h_s):2d}x{len(w_s):2d}, "
                  f"max_error={error.numpy():.6f}")

    print("\n" + "=" * 60)
    print("All coordinate handling tests passed!")
    print("=" * 60)


# ---------------------------------------------------------------------------

def test_window_coverage_with_geometrical(multi_query: bool, mode: str = '2d'):
    """
    mode: '2d' | '3d'
    In 3D we use a small (2, 3, 3) volume with Z=2, Y=4, X=4 for readability.
    """
    assert mode in ('2d', '3d'), "mode must be '2d' or '3d'"
    is_3d = mode == '3d'

    if is_3d:
        # Build a small (Z=2, Y=4, X=4) volume with distinct voxel values
        vol = np.arange(1, 2 * 4 * 4 + 1, dtype=np.float32).reshape(2, 4, 4)
        vol_3ch = np.stack([vol, vol * 2, vol * 3], axis=-1)          # (Z, Y, X, 3)
        test_image = tf.constant(vol_3ch[None])               # (1, Z, Y, X, 3)
        test_q = (tf.stack([test_image, test_image * 2, test_image * 3], axis=0)
                  if multi_query else test_image)

        window_sizes = [(2, 3, 3), (2, 4, 4), (1, 2, 2)]
        overlap_modes = ['mean', 'geometrical']

        for window_size in window_sizes:
            for overlap_mode in overlap_modes:
                print(f"\n{'=' * 50}")
                print(f"[3D] Window {window_size}, Overlap {overlap_mode}")
                print(f"{'=' * 50}")

                wa = WindowSpatialAttention(
                    num_heads=1, attention_filters=3,
                    window_size=window_size, overlap_reduction=overlap_mode,
                    layer_normalization=False, skip_connection=False)
                wa.build([test_q.shape, test_image.shape])

                Z, Y, X, C_ch = (test_image.shape[1], test_image.shape[2],
                                  test_image.shape[3], test_image.shape[4])
                z_starts, y_starts, x_starts = wa._compute_window_grid_3d(Z, Y, X)

                print(f"Z starts: {z_starts.numpy()}")
                print(f"Y starts: {y_starts.numpy()}")
                print(f"X starts: {x_starts.numpy()}")

                Q_in = tf.reshape(test_q, [-1, Z, Y, X, C_ch])
                windows = wa._extract_windows_vectorized_3d(
                    Q_in, z_starts, y_starts, x_starts)

                WSZ, WSY, WSX = wa.window_size
                num_windows = len(z_starts) * len(y_starts) * len(x_starts)
                print(f"Extracted windows shape: {windows.shape}  (n_wins={num_windows})")

                if overlap_mode == 'mean':
                    reconstructed = wa._scatter_windows_mean_3d(
                        windows, z_starts, y_starts, x_starts, Z, Y, X)
                else:
                    geo = wa._geometrical_confidence_3d()
                    print(f"Geometrical confidence shape: {geo.shape}")
                    tiled = tf.tile(tf.cast(geo, windows.dtype),
                                    [tf.shape(windows)[0], 1, 1, 1, 1])
                    reconstructed = wa._scatter_windows_with_averaging_3d(
                        windows, tiled, z_starts, y_starts, x_starts, Z, Y, X)

                diff = tf.abs(Q_in - reconstructed)
                print(f"Max difference:  {tf.reduce_max(diff).numpy():.4f}")
                print(f"Mean difference: {tf.reduce_mean(diff).numpy():.4f}")

                # Coverage mask
                coverage = np.zeros((Z, Y, X), dtype=np.float32)
                for z_s in z_starts.numpy():
                    for y_s in y_starts.numpy():
                        for x_s in x_starts.numpy():
                            coverage[z_s:z_s + WSZ, y_s:y_s + WSY, x_s:x_s + WSX] += 1

                min_cov = coverage.min()
                print(f"Coverage  min={min_cov:.0f}  max={coverage.max():.0f}  "
                      f"all_covered={min_cov >= 1}")

    else:
        # ---- Original 2D path -----------------------------------------------
        test_image = tf.constant([
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
            [[13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18]],
            [[19, 19], [20, 20], [21, 21], [22, 22], [23, 23], [24, 24]],
            [[25, 25], [26, 26], [27, 27], [28, 28], [29, 29], [30, 30]],
            [[31, 31], [32, 32], [33, 33], [34, 34], [35, 35], [36, 36]]
        ], dtype=tf.float32)
        test_image = tf.expand_dims(test_image, axis=0)
        test_q = tf.stack([test_image, test_image * 2, test_image * 3], 0) if multi_query else test_image

        window_sizes = [(3, 3), (4, 4), (6, 6)]
        overlap_modes = ['mean', 'geometrical']

        for window_size in window_sizes:
            for overlap_mode in overlap_modes:
                print(f"\n{'=' * 50}")
                print(f"[2D] Window {window_size}, Overlap {overlap_mode}")
                print(f"{'=' * 50}")

                wa = WindowSpatialAttention(
                    num_heads=1, attention_filters=2,
                    window_size=window_size, overlap_reduction=overlap_mode,
                    layer_normalization=False, skip_connection=False)
                wa.build([test_q.shape, test_image.shape])

                Y, X, C_ch = test_image.shape[1], test_image.shape[2], test_image.shape[3]
                y_starts, x_starts = wa._compute_window_grid_2d(Y, X)

                print(f"Y starts: {y_starts.numpy()}")
                print(f"X starts: {x_starts.numpy()}")

                Q_in = tf.reshape(test_q, [-1, Y, X, C_ch])
                windows = wa._extract_windows_vectorized_2d(Q_in, y_starts, x_starts)

                WSY, WSX = wa.window_size
                num_windows = len(y_starts) * len(x_starts)
                print(f"Extracted windows shape: {windows.shape}  (n_wins={num_windows})")

                if overlap_mode == 'mean':
                    reconstructed = wa._scatter_windows_mean_2d(
                        windows, y_starts, x_starts, Y, X)
                else:
                    geo = wa._geometrical_confidence_2d()
                    print(f"Geometrical confidence shape: {geo.shape}")
                    print(geo[0, :, :, 0].numpy())
                    tiled = tf.tile(tf.cast(geo, windows.dtype),
                                    [tf.shape(windows)[0], 1, 1, 1])
                    reconstructed = wa._scatter_windows_with_averaging_2d(
                        windows, tiled, y_starts, x_starts, Y, X)

                print(f"\nReconstructed (channel 0, batch 0):")
                for y in range(Y):
                    print(" ".join(f"{reconstructed[0, y, x, 0].numpy():5.1f}" for x in range(X)))

                diff = tf.abs(Q_in - reconstructed)
                print(f"Max difference:  {tf.reduce_max(diff).numpy():.4f}")
                print(f"Mean difference: {tf.reduce_mean(diff).numpy():.4f}")

                coverage = np.zeros((Y, X), dtype=np.float32)
                for y_s in y_starts.numpy():
                    for x_s in x_starts.numpy():
                        coverage[y_s:y_s + WSY, x_s:x_s + WSX] += 1

                print(f"Coverage  min={coverage.min():.0f}  max={coverage.max():.0f}  "
                      f"all_covered={coverage.min() >= 1}")


# ---------------------------------------------------------------------------

def test_multiquery_equivalence(mode: str = '2d'):
    """
    Verify multi_query=True gives the same result as sequential single-query.
    mode: '2d' | '3d'
    """
    assert mode in ('2d', '3d'), "mode must be '2d' or '3d'"
    is_3d = mode == '3d'

    print("=" * 70)
    print(f"Testing Multi-Query vs Sequential Single-Query Equivalence  [{mode.upper()}]")
    print("=" * 70)

    Q, B = 3, 2
    C, num_heads, attention_filters = 16, 4, 4

    if is_3d:
        Z, H, W   = 8, 14, 16
        window_size = (4, 7, 5)
        queries = tf.random.normal([Q, B, Z, H, W, C], seed=42)
        key     = tf.random.normal([B, Z, H, W, C], seed=43)
        value   = tf.random.normal([B, Z, H, W, C], seed=44)
    else:
        H, W      = 14, 16
        window_size = (7, 5)
        queries = tf.random.normal([Q, B, H, W, C], seed=42)
        key     = tf.random.normal([B, H, W, C], seed=43)
        value   = tf.random.normal([B, H, W, C], seed=44)

    for overlap_method in ['mean', 'attention_weighted', 'geometrical']:
        print(f"\n{'=' * 70}")
        print(f"Testing overlap_reduction='{overlap_method}'")
        print(f"{'=' * 70}")

        tf.random.set_seed(100)
        layer_multi = WindowSpatialAttention(
            num_heads=num_heads, attention_filters=attention_filters,
            window_size=window_size, overlap_reduction=overlap_method,
            skip_connection=False, layer_normalization=False,
            dropout=0.0, add_distance_embedding=False)

        output_multi = layer_multi([queries, key, value], training=False)
        print(f"Multi-query output shape: {output_multi.shape}")

        tf.random.set_seed(100)
        layer_single = WindowSpatialAttention(
            num_heads=num_heads, attention_filters=attention_filters,
            window_size=window_size, overlap_reduction=overlap_method,
            skip_connection=False, layer_normalization=False,
            dropout=0.0, add_distance_embedding=False)

        _ = layer_single([queries[0], key, value], training=False)

        # Copy weights
        for attr in ('qproj', 'kproj', 'vproj', 'outproj'):
            getattr(layer_single, attr).set_weights(
                getattr(layer_multi, attr).get_weights())
        layer_single.relative_position_bias_table.assign(
            layer_multi.relative_position_bias_table)

        outputs_seq = []
        for q_idx in range(Q):
            outputs_seq.append(layer_single([queries[q_idx], key, value], training=False))
        output_seq = tf.stack(outputs_seq, axis=0)
        print(f"Stacked sequential output shape: {output_seq.shape}")

        diff = tf.abs(output_multi - output_seq)
        max_diff  = tf.reduce_max(diff).numpy()
        mean_diff = tf.reduce_mean(diff).numpy()
        print(f"Max difference:  {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")
        tolerance = 1e-4
        print(f"{'✓ PASS' if max_diff < tolerance else '✗ FAIL'}  (tol={tolerance})")


# ---------------------------------------------------------------------------

def test_edge_cases(mode: str = '2d'):
    """
    Edge cases: single-window volume, single query in multi-query format.
    mode: '2d' | '3d'
    """
    assert mode in ('2d', '3d'), "mode must be '2d' or '3d'"
    is_3d = mode == '3d'

    print("\n" + "=" * 70)
    print(f"Testing Edge Cases  [{mode.upper()}]")
    print("=" * 70)

    Q, B, C = 2, 1, 8

    if is_3d:
        Z, H, W     = 4, 7, 7
        window_size = (4, 7, 7)
        queries = tf.random.normal([Q, B, Z, H, W, C])
        key     = tf.random.normal([B, Z, H, W, C])
    else:
        H, W        = 7, 7
        window_size = (7, 7)
        queries = tf.random.normal([Q, B, H, W, C])
        key     = tf.random.normal([B, H, W, C])

    print("\n--- Single Window (no overlap) ---")
    layer = WindowSpatialAttention(
        num_heads=2, attention_filters=4,
        window_size=window_size, overlap_reduction='mean',
        skip_connection=False, layer_normalization=False,
        add_distance_embedding=False)
    layer.build([queries.shape, key.shape])
    output = layer([queries, key], training=False)
    print(f"Output shape:   {output.shape}")
    print(f"Expected shape: {queries.shape}")
    assert output.shape == queries.shape
    print("✓ Single window case passed")

    print("\n--- Single Query in Multi-Query Format ---")
    Q_single = 1
    if is_3d:
        queries_single = tf.random.normal([Q_single, B, Z, H, W, C])
    else:
        queries_single = tf.random.normal([Q_single, B, H, W, C])

    layer_sq = WindowSpatialAttention(
        num_heads=2, attention_filters=4,
        window_size=window_size, overlap_reduction='mean',
        skip_connection=False, add_distance_embedding=False)
    layer_sq.build([queries_single.shape, key.shape])
    output_sq = layer_sq([queries_single, key], training=False)
    print(f"Output shape:   {output_sq.shape}")
    print(f"Expected shape: {queries_single.shape}")
    assert output_sq.shape == queries_single.shape
    print("✓ Single query in multi-query format passed")


# ---------------------------------------------------------------------------

def test_attention_computation_details(mode: str = '2d'):
    """
    Verify einsum multi-query == sequential matmul single-query at the raw
    attention-score level.
    mode: '2d' | '3d'
    """
    assert mode in ('2d', '3d'), "mode must be '2d' or '3d'"
    is_3d = mode == '3d'

    print("\n" + "=" * 70)
    print(f"Testing Attention Computation Details  [{mode.upper()}]")
    print("=" * 70)

    Q, B, C = 2, 1, 8
    num_heads, attention_filters = 2, 4
    HF = num_heads * attention_filters

    if is_3d:
        Z, H, W     = 4, 7, 7
        window_size = (4, 7, 7)
        N           = Z * H * W
        queries     = tf.random.normal([Q, B, Z, H, W, C])
        key         = tf.random.normal([B, Z, H, W, C])
        value       = tf.random.normal([B, Z, H, W, C])
    else:
        H, W        = 7, 7
        window_size = (7, 7)
        N           = H * W
        queries     = tf.random.normal([Q, B, H, W, C])
        key         = tf.random.normal([B, H, W, C])
        value       = tf.random.normal([B, H, W, C])

    layer = WindowSpatialAttention(
        num_heads=num_heads, attention_filters=attention_filters,
        window_size=window_size, overlap_reduction='mean',
        skip_connection=False, layer_normalization=False,
        dropout=0.0, add_distance_embedding=False)
    layer.build([queries.shape, key.shape, value.shape])

    queries_flat = tf.reshape(queries, [Q * B, *queries.shape[2:]])
    Q_proj = layer.qproj(queries_flat)
    K_proj = layer.kproj(key)

    scale = tf.math.rsqrt(tf.cast(attention_filters, Q_proj.dtype))

    # Multi-query einsum path
    print("\n--- Multi-Query Einsum Path ---")
    q_mq = tf.reshape(Q_proj, [1, Q, B, N, num_heads, attention_filters])
    k_mq = tf.reshape(K_proj, [1, B, N, num_heads, attention_filters])
    attn_mq = tf.einsum('mqbihf,mbkhf->mqbhik', q_mq, k_mq) * scale
    print(f"Attention scores shape (multi-query): {attn_mq.shape}")

    # Sequential single-query matmul path
    print("\n--- Sequential Single-Query Matmul Path ---")
    attn_scores_seq = []
    for q_idx in range(Q):
        q_sq = tf.reshape(Q_proj[q_idx:q_idx + 1], [B, N, num_heads, attention_filters])
        k_sq = tf.reshape(K_proj, [B, N, num_heads, attention_filters])
        q_sq = tf.transpose(q_sq, [0, 2, 1, 3])   # (B, H, N, F)
        k_sq = tf.transpose(k_sq, [0, 2, 3, 1])   # (B, H, F, N)
        attn_sq = tf.matmul(q_sq, k_sq) * scale    # (B, H, N, N)
        attn_scores_seq.append(attn_sq)
    attn_scores_seq = tf.stack(attn_scores_seq, axis=0)   # (Q, B, H, N, N)
    print(f"Attention scores shape (sequential): {attn_scores_seq.shape}")

    attn_mq_rs = tf.reshape(attn_mq, [Q, B, num_heads, N, N])
    diff = tf.abs(attn_mq_rs - attn_scores_seq)
    print(f"\nAttention score differences:")
    print(f"  Max:  {tf.reduce_max(diff).numpy():.2e}")
    print(f"  Mean: {tf.reduce_mean(diff).numpy():.2e}")
    if tf.reduce_max(diff) < 1e-5:
        print("✓ Attention computation equivalent")
    else:
        print("✗ Attention computation differs!")


# ---------------------------------------------------------------------------

def test_window_processing_modes_equivalence(mode: str = '2d'):
    """
    Test that batch / row / col / sequential (/ slice for 3D) modes all match.
    mode: '2d' | '3d'
    """
    assert mode in ('2d', '3d'), "mode must be '2d' or '3d'"
    is_3d = mode == '3d'

    print("\n" + "=" * 70)
    print(f"Testing Window Processing Modes Equivalence  [{mode.upper()}]")
    print("=" * 70)

    B, C = 2, 16
    num_heads, attention_filters = 4, 4

    if is_3d:
        Z, H, W     = 8, 20, 16
        window_size = (4, 7, 5)
        chunked_modes = ['slice', 'sequential']
    else:
        H, W        = 20, 16
        window_size = (7, 5)
        chunked_modes = ['row', 'col', 'sequential']

    np.random.seed(42)
    tf.random.set_seed(42)

    for multi_query in [False, True]:
        print(f"\n{'=' * 70}")
        print(f"Testing {'Multi-Query' if multi_query else 'Single-Query'} Mode")
        print(f"{'=' * 70}")

        if multi_query:
            Q = 3
            if is_3d:
                queries = tf.random.normal([Q, B, Z, H, W, C])
            else:
                queries = tf.random.normal([Q, B, H, W, C])
        else:
            if is_3d:
                queries = tf.random.normal([B, Z, H, W, C])
            else:
                queries = tf.random.normal([B, H, W, C])

        if is_3d:
            key   = tf.random.normal([B, Z, H, W, C])
            value = tf.random.normal([B, Z, H, W, C])
        else:
            key   = tf.random.normal([B, H, W, C])
            value = tf.random.normal([B, H, W, C])

        for overlap_method in ['mean', 'attention_weighted', 'geometrical']:
            print(f"\n{'-' * 70}")
            print(f"Testing overlap_reduction='{overlap_method}'")
            print(f"{'-' * 70}")

            # Reference: batch mode
            layer_ref = WindowSpatialAttention(
                num_heads=num_heads, attention_filters=attention_filters,
                window_size=window_size, overlap_reduction=overlap_method,
                window_processing='batch', skip_connection=False,
                layer_normalization=False, dropout=0.0,
                add_distance_embedding=False)
            _ = layer_ref([queries, key, value], training=False)
            output_ref = layer_ref([queries, key, value], training=False)

            def copy_weights(src, dst):
                for attr in ('qproj', 'kproj', 'vproj', 'outproj'):
                    getattr(dst, attr).set_weights(getattr(src, attr).get_weights())
                dst.relative_position_bias_table.assign(src.relative_position_bias_table)
                if hasattr(src, 'distance_encoding_y'):
                    dst.distance_encoding_y.set_weights(src.distance_encoding_y.get_weights())
                    dst.distance_encoding_x.set_weights(src.distance_encoding_x.get_weights())
                if is_3d and hasattr(src, 'distance_encoding_z'):
                    dst.distance_encoding_z.set_weights(src.distance_encoding_z.get_weights())

            for proc_mode in chunked_modes:
                layer_m = WindowSpatialAttention(
                    num_heads=num_heads, attention_filters=attention_filters,
                    window_size=window_size, overlap_reduction=overlap_method,
                    window_processing=proc_mode, skip_connection=False,
                    layer_normalization=False, dropout=0.0,
                    add_distance_embedding=False)
                _ = layer_m([queries, key, value], training=False)
                copy_weights(layer_ref, layer_m)
                output_m = layer_m([queries, key, value], training=False)

                diff = tf.abs(output_ref - output_m)
                max_diff  = tf.reduce_max(diff).numpy()
                mean_diff = tf.reduce_mean(diff).numpy()
                rel_err   = mean_diff / (tf.reduce_mean(tf.abs(output_ref)).numpy() + 1e-8)
                tolerance = 1e-4
                status = "✓ PASS" if max_diff < tolerance else "✗ FAIL"
                print(f"  Mode '{proc_mode:12s}': "
                      f"max={max_diff:.2e}  mean={mean_diff:.2e}  "
                      f"rel={rel_err:.2e}  {status}")

    print("\n" + "=" * 70)
    print("Window Processing Modes Equivalence Test Complete")
    print("=" * 70)


# ---------------------------------------------------------------------------

def test_window_processing_memory_profile(mode: str = '2d'):
    """
    Demonstrate memory trade-offs between processing modes.
    mode: '2d' | '3d'
    """
    assert mode in ('2d', '3d'), "mode must be '2d' or '3d'"
    is_3d = mode == '3d'

    print("\n" + "=" * 70)
    print(f"Window Processing Memory Profile Demo  [{mode.upper()}]")
    print("=" * 70)

    if is_3d:
        print("\n  'batch':      Fastest, highest memory (all windows at once)")
        print("  'slice':      Medium  (one Z-slice at a time)")
        print("  'sequential': Slowest, lowest memory (one window at a time)")

        B, Z, H, W, C = 1, 16, 50, 50, 16
        window_size   = (4, 7, 7)
        x = tf.random.normal([B, Z, H, W, C])

        layer = WindowSpatialAttention(
            num_heads=4, attention_filters=4,
            window_size=window_size, window_processing='batch',
            dropout=0.0, add_distance_embedding=False)
        layer.build(x.shape)

        z_s, y_s, x_s = layer._compute_window_grid_3d(Z, H, W)
        n_wins = len(z_s) * len(y_s) * len(x_s)

        print(f"\nExample: {Z}x{H}x{W} volume, window {window_size}")
        print(f"  Total windows: {n_wins}")
        print(f"  'batch'  mode: all {n_wins} windows at once")
        print(f"  'slice'  mode: {len(y_s) * len(x_s)} windows/iter "
              f"({len(z_s)} iters)")
        print(f"  'sequential': 1 window/iter ({n_wins} iters)")
    else:
        print("\n  'batch':      Fastest, highest memory")
        print("  'row':        Medium speed/memory")
        print("  'col':        Medium speed/memory")
        print("  'sequential': Slowest, lowest memory")

        B, H, W, C  = 1, 50, 50, 16
        window_size = (7, 7)
        x = tf.random.normal([B, H, W, C])

        layer = WindowSpatialAttention(
            num_heads=4, attention_filters=4,
            window_size=window_size, window_processing='batch',
            dropout=0.0, add_distance_embedding=False)
        layer.build(x.shape)

        y_s, x_s = layer._compute_window_grid_2d(H, W)
        n_wins = len(y_s) * len(x_s)

        print(f"\nExample: {H}x{W} image, window {window_size}")
        print(f"  Total windows: {n_wins}")
        print(f"  'batch' mode: {n_wins} windows at once")
        print(f"  'row'   mode: {len(x_s)} windows/iter ({len(y_s)} iters)")
        print(f"  'col'   mode: {len(y_s)} windows/iter ({len(x_s)} iters)")
        print(f"  'sequential': 1 window/iter ({n_wins} iters)")

    print("\n" + "=" * 70)

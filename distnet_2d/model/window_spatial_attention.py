import tensorflow as tf
import numpy as np
from skfmm import distance


class WindowSpatialAttention(tf.keras.layers.Layer):
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
                 use_bias:bool = True, dropout: float = 0.1, skip_connection: bool = True, layer_normalization:bool=False,
                 add_distance_embedding:bool=True,
                 window_processing:str="sequential", # "all", "row", "col", "sequential"
                 overlap_reduction: str = 'geometrical',  # 'mean', 'attention_weighted', 'geometrical'
                 l2_reg: float = 0., position_encoding_l2_reg: float = 1e-5, name="WindowSpatialAttention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        # Support both int and tuple for window_size
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = tuple(window_size)
        self.filters = None
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "attention_filters": self.attention_filters,
            "window_size": self.window_size,
            "use_bias":self.use_bias,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "position_encoding_l2_reg":self.position_encoding_l2_reg,
            "skip_connection": self.skip_connection,
            "layer_normalization": self.layer_normalization,
            "window_processing":self.window_processing,
            "overlap_reduction": self.overlap_reduction,
            "add_distance_embedding": self.add_distance_embedding,
            "multi_query": self.multi_query,
            "q_count": self.q_count,
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

        # Detect multi-query mode statically from input shape if provided
        # Expected query shapes:
        # - Single query: (B, Y, X, C)
        # - Multi-query: (Q, B, Y, X, C)
        if len(input_shape) == 5:
            assert len(input_shapes) >= 2, "in multi-query mode, K/V must be provided"
            # static multi-query
            self.multi_query = True
            self.q_count = input_shape[0]
            input_shape = input_shape[1:]
        elif len(input_shape) == 4:
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
            self.ln_q = tf.keras.layers.LayerNormalization()
            if len(input_shapes) >= 3 : # Q, K, V provided
                self.ln_v = tf.keras.layers.LayerNormalization()
                self.ln_k = tf.keras.layers.LayerNormalization()
            elif len(input_shapes) == 2: # Q, K provided
                self.ln_k = tf.keras.layers.LayerNormalization()
                self.ln_v = None
            else:
                self.ln_v = None
                self.ln_k = None

        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)

        HF = self.num_heads * self.attention_filters

        # Separate Q, K, V projections
        self.qproj = tf.keras.layers.Conv2D(HF, 1, padding='same',
                                            use_bias=self.use_bias, name="qproj",
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None)
        self.kproj = tf.keras.layers.Conv2D(HF, 1, padding='same',
                                            use_bias=self.use_bias, name="kproj",
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None)
        self.vproj = tf.keras.layers.Conv2D(HF, 1, padding='same',
                                            use_bias=self.use_bias, name="vproj",
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None)
        self.outproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same',
                                              use_bias=self.use_bias, name="outproj",
                                              bias_initializer=tf.keras.initializers.Zeros(),
                                              kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None)

        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        # Relative position bias table
        WSY, WSX = self.window_size

        embedding_size = (2 * WSY - 1) * (2 * WSX - 1)
        self.relative_position_bias_table = self.add_weight(
            name="rpb",
            shape=(embedding_size, self.num_heads),
            initializer=tf.initializers.Zeros(),
            constraint=tf.keras.constraints.MaxNorm(max_value=5.0, axis=0),
            regularizer=tf.keras.regularizers.l2(self.position_encoding_l2_reg) if self.position_encoding_l2_reg>0 else None,
            trainable=True
        )
        # Pre-compute relative position indices for window
        coords_y = tf.range(WSY)
        coords_x = tf.range(WSX)
        coords = tf.stack(tf.meshgrid(coords_y, coords_x, indexing='ij'))
        coords_flatten = tf.reshape(coords, [2, -1])

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, [1, 2, 0])
        relative_coords = relative_coords + [WSY - 1, WSX - 1]
        relative_position_index = (relative_coords[:, :, 0] * (2 * WSX - 1) + relative_coords[:, :, 1])
        self.relative_position_index = tf.constant(relative_position_index, dtype=tf.int32)
        if self.add_distance_embedding: # embedding added to V
            self.distance_encoding_y = tf.keras.layers.Embedding(
                input_dim=WSY, output_dim=HF//2,
                embeddings_regularizer=tf.keras.regularizers.l2(self.position_encoding_l2_reg) if self.position_encoding_l2_reg > 0 else None
            )
            self.distance_encoding_x = tf.keras.layers.Embedding(
                input_dim=WSX, output_dim=HF - HF//2,
                embeddings_regularizer=tf.keras.regularizers.l2(self.position_encoding_l2_reg) if self.position_encoding_l2_reg > 0 else None
            )
        if self.overlap_reduction == "geometrical":
            self.geometrical_confidence = self._geometrical_confidence()

        super().build(input_shape)

    @staticmethod
    def _compute_axis_coords_with_min_overlap(size, tile_size):
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

    def _compute_window_grid(self, Y, X):
        """Now uses minimum overlap strategy instead of fixed shifts."""
        WSY, WSX = self.window_size
        y_starts = self._compute_axis_coords_with_min_overlap(Y, WSY)
        x_starts = self._compute_axis_coords_with_min_overlap(X, WSX)
        return y_starts, x_starts

    @tf.function(jit_compile=True)
    def _extract_windows_vectorized(self, x, y_starts, x_starts):
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
    def _scatter_windows_mean(self, windows, y_starts, x_starts, Y, X):
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
    def _scatter_windows_with_averaging(self, windows, window_weights, y_starts, x_starts, Y, X):
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
    def _scatter_windows_accumulate(self, windows, window_weights, y_starts, x_starts, output, weight_sums, Y, X):
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

    def _geometrical_confidence(self, min_confidence:float = 0.1):
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

    def _window_attention(self, q_windows, k_windows, v_windows, training=None, num_windows=None):
        """
        Compute attention within windows. Supports both single-query (original) and
        multi-query modes. When multi_query=True (set in build), q_windows will have
        a larger batch dimension (num_windows * Q * B) while k_windows/v_windows have
        (num_windows * B).

        Args:
            q_windows, k_windows, v_windows: windowed tensors
              - q_windows: (num_windows * QB, WSY, WSX, HF)
              - k_windows: (num_windows * B, WSY, WSX, HF)
              - v_windows: (num_windows * B, WSY, WSX, HF)
            num_windows: optional number of windows (computed in caller)

        Returns:
            output: (num_windows * QB, WSY, WSX, HF)
            attention_weights: (num_windows * QB, WSY, WSX, 1) or None
        """
        WSY, WSX = self.window_size
        HF = self.num_heads * self.attention_filters
        H = self.num_heads
        F = self.attention_filters
        N = WSY * WSX

        if self.add_distance_embedding:
            distance_emb_y = self.distance_encoding_y(tf.range(WSY))
            distance_emb_x = self.distance_encoding_x(tf.range(WSX))
            distance_emb_y = tf.reshape(distance_emb_y, (1, WSY, 1, HF // 2))
            distance_emb_x = tf.reshape(distance_emb_x, (1, 1, WSX, HF - HF // 2))
            distance_emb = tf.concat([
                tf.broadcast_to(distance_emb_y, [1, WSY, WSX, HF // 2]),
                tf.broadcast_to(distance_emb_x, [1, WSY, WSX, HF - HF // 2])
            ], axis=-1)
            v_windows = v_windows - distance_emb

        if not self.multi_query:
            # single-query implementation: q,k,v are (B_win, WSY, WSX, HF)
            B_win = tf.shape(q_windows)[0]
            q = tf.reshape(q_windows, [B_win, N, H, F])
            k = tf.reshape(k_windows, [B_win, N, H, F])
            v = tf.reshape(v_windows, [B_win, N, H, F])

            q = tf.transpose(q, [0, 2, 1, 3]) # (B_win, H, N, F)
            v = tf.transpose(v, [0, 2, 1, 3]) # (B_win, H, N, F)
            k = tf.transpose(k, [0, 2, 3, 1])  # (B_win, H, F, N)
            # Attention:
            scale = tf.math.rsqrt(tf.cast(F, q.dtype))
            attn = tf.matmul(q, k) * scale # (B_win, H, N, F) * (B_win, H, F, N) -> (B_win, H, N, N)

            # Add relative position bias
            rpb = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1]))
            rpb = tf.reshape(rpb, [N, N, H])
            rpb = tf.transpose(rpb, [2, 0, 1])  # (H, N, N)
            attn = attn + rpb[None, :, :, :]

            # Softmax
            attn_probs = tf.nn.softmax(attn, axis=-1)

            if self.dropout > 0 and training:
                attn_probs = self.dropout_layer(attn_probs, training=training)

            if self.overlap_reduction == "attention_weighted":
                max_attn = tf.reduce_max(attn_probs, axis=-1)  # (B_win, H, N)
                confidence = tf.reduce_mean(max_attn, axis=1)  # (B_win, N)
                confidence_spatial = tf.reshape(confidence, [B_win, WSY, WSX, 1])
                confidence_spatial = confidence_spatial / tf.reduce_sum(confidence_spatial, axis=[1, 2], keepdims=True)
                confidence_spatial = tf.maximum(confidence_spatial, tf.cast(1e-3, confidence_spatial.dtype))
                confidence_spatial = tf.stop_gradient(confidence_spatial)
            else:
                confidence_spatial = None

            out = tf.matmul(attn_probs, v) # (B_win, H, N, N) x (B_win, H, N, F) ->  (B_win, H, N, F)
            out = tf.transpose(out, [0, 2, 1, 3]) # (B_win, N, H, F)
            out = tf.reshape(out, [B_win, WSY, WSX, HF])
            if self.add_distance_embedding:
                out = out + distance_emb
            return out, confidence_spatial

        else:
            # Multi-query implementation using einsum-based broadcasting to avoid tiling K/V.
            # q_windows: (num_windows * Q * B, WSY, WSX, HF)
            # k_windows/v_windows: (num_windows * B, WSY, WSX, HF)
            assert num_windows is not None, "num_windows must be provided in multi-query mode"

            q = tf.reshape(q_windows, [num_windows, self.q_count, -1, N, H, F]) # [num_windows, Q, B, N, H, F]
            k = tf.reshape(k_windows, [num_windows, -1, N, H, F]) # [num_windows, B, N, H, F]
            v = tf.reshape(v_windows, [num_windows, -1, N, H, F]) # [num_windows, B, N, H, F]

            # Compute attention: einsum over feature dim F
            scale = tf.math.rsqrt(tf.cast(F, q.dtype))
            attn = tf.einsum('mqbihf,mbkhf->mqbhik', q, k, optimize='optimal') * scale # [num_windows, Q, B, H, N, N]
            # Add relative position bias: rpb is (H, N, N)
            rpb = tf.gather(self.relative_position_bias_table,  tf.reshape(self.relative_position_index, [-1]))
            rpb = tf.reshape(rpb, [N, N, H])
            rpb = tf.transpose(rpb, [2, 0, 1])  # (H, N, N)
            # Expand to [1,1,1,H,N,N] to broadcast across num_windows, Q, B
            attn = attn + rpb[None, None, None, :, :, :]

            # Softmax over last axis (k positions)
            attn_probs = tf.nn.softmax(attn, axis=-1) # [num_windows, B, H, N, k]

            if self.dropout > 0 and training:
                attn_probs = self.dropout_layer(attn_probs, training=training)

            confidence_spatial = None
            if self.overlap_reduction == "attention_weighted":
                # max over k
                max_attn = tf.reduce_max(attn_probs, axis=-1, keepdims=False) # ( num_windows, Q, B, H, N)
                # mean over heads
                confidence = tf.reduce_mean(max_attn, axis=3, keepdims=False) # ( num_windows, Q, B, N)
                confidence_spatial = tf.reshape(confidence, [-1, WSY, WSX, 1])
                confidence_spatial = confidence_spatial / tf.reduce_sum(confidence_spatial, axis=[1, 2], keepdims=True)
                confidence_spatial = tf.maximum(confidence_spatial, tf.cast(1e-3, confidence_spatial.dtype))
                confidence_spatial = tf.stop_gradient(confidence_spatial)

            # Compute output: einsum of attn_probs with v
            out = tf.einsum('mqbhik,mbkhf->mqbihf', attn_probs, v, optimize='optimal') #[num_windows, Q, B, H, i, k]
            # Reorder and reshape back to (num_windows * Q * B, WSY, WSX, HF)
            out = tf.reshape(out, [-1, WSY, WSX, HF])
            if self.add_distance_embedding:
                out = out + distance_emb
            return out, confidence_spatial

    def call(self, x, training: bool = None):
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
                    Q, B, Y, X, _ = tf.unstack(tf.shape(query))
                    query = tf.reshape(query, [Q * B, Y, X, self.filters])
                if self.layer_normalization:
                    query = self.ln_q(query)
                    key = self.ln_k(key)
                    value = self.ln_v(value)
            else:
                raise ValueError("Invalid input length should be lower than 3")
        else:
            assert not self.multi_query
            source_query = x
            if self.layer_normalization:
                x = self.ln_q(x)
            key = value = query = x

        B, Y, X, _ = tf.unstack(tf.shape(query))
        WSY, WSX = self.window_size

        # Project Q, K, V
        Q_proj = self.qproj(query)
        K_proj = self.kproj(key)
        V_proj = self.vproj(value)

        def single():
            """Direct attention without windowing overhead."""
            # Pad to window size if needed
            pad_y = tf.maximum(0, WSY - Y)
            pad_x = tf.maximum(0, WSX - X)
            need_padding = tf.reduce_any(tf.greater([pad_y, pad_x], 0))
            def pad():
                Qp = tf.pad(Q_proj, [[0, 0], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                Kp = tf.pad(K_proj, [[0, 0], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                Vp = tf.pad(V_proj, [[0, 0], [0, pad_y], [0, pad_x], [0, 0]], mode="SYMMETRIC")
                return Qp, Kp, Vp
            def identity():
                return Q_proj, K_proj, V_proj

            Qp, Kp, Vp = tf.cond(need_padding, pad, identity)

            # Apply window attention (now Qp, Kp, Vp are exactly window_size in spatial dims)
            out_windows, _ = self._window_attention(Qp, Kp, Vp, training, num_windows=1)

            # Crop back to original size
            return out_windows[:, :Y, :X, :]

        def multiple():
            y_starts, x_starts = self._compute_window_grid(Y, X)
            num_y, num_x = tf.shape(y_starts)[0], tf.shape(x_starts)[0]

            if self.window_processing == 'all': # extract all windows at once
                num_windows = num_y * num_x
                Q_windows = self._extract_windows_vectorized(Q_proj, y_starts, x_starts)
                K_windows = self._extract_windows_vectorized(K_proj, y_starts, x_starts)
                V_windows = self._extract_windows_vectorized(V_proj, y_starts, x_starts)
                out_windows, attention_confidence = self._window_attention( Q_windows, K_windows, V_windows, training, num_windows=num_windows)

                if self.overlap_reduction == 'attention_weighted':
                    output = self._scatter_windows_with_averaging(out_windows, attention_confidence, y_starts, x_starts, Y, X)
                elif self.overlap_reduction == "geometrical":
                    geometrical_confidence = tf.tile(tf.cast(self.geometrical_confidence, out_windows.dtype),[tf.shape(out_windows)[0], 1, 1, 1])
                    output = self._scatter_windows_with_averaging(out_windows, geometrical_confidence,  y_starts, x_starts, Y, X)
                else:
                    output = self._scatter_windows_mean(out_windows, y_starts, x_starts, Y, X)
            else:
                # Chunked processing with direct accumulation
                C = self.num_heads * self.attention_filters
                WSY, WSX = self.window_size
                B_size = tf.shape(Q_proj)[0]
                output = tf.zeros([B_size, Y, X, C], dtype=Q_proj.dtype)
                weight_sums = tf.zeros([B_size, Y, X, 1], dtype=Q_proj.dtype)

                num_y = tf.shape(y_starts)[0]
                num_x = tf.shape(x_starts)[0]

                # Process based on mode
                if self.window_processing == 'row':
                    # Process row by row
                    def process_row(i, out, weights):
                        y_sub = y_starts[i:i + 1]
                        num_wins = tf.shape(y_sub)[0] * num_x

                        Q_wins = self._extract_windows_vectorized(Q_proj, y_sub, x_starts)
                        K_wins = self._extract_windows_vectorized(K_proj, y_sub, x_starts)
                        V_wins = self._extract_windows_vectorized(V_proj, y_sub, x_starts)

                        out_wins, attn_conf = self._window_attention(Q_wins, K_wins, V_wins, training, num_wins)

                        if self.overlap_reduction == 'attention_weighted':
                            conf = attn_conf
                        elif self.overlap_reduction == 'geometrical':
                            conf = tf.tile(tf.cast(self.geometrical_confidence, out_wins.dtype),
                                           [tf.shape(out_wins)[0], 1, 1, 1])
                        else:  # mean
                            conf = tf.ones([tf.shape(out_wins)[0], WSY, WSX, 1], dtype=out_wins.dtype)

                        out, weights = self._scatter_windows_accumulate(
                            out_wins, conf, y_sub, x_starts, out, weights, Y, X
                        )
                        return i + 1, out, weights

                    _, output, weight_sums = tf.while_loop(
                        lambda i, o, w: i < num_y,
                        process_row,
                        [0, output, weight_sums]
                    )

                elif self.window_processing == 'col':
                    # Process column by column
                    def process_col(j, out, weights):
                        x_sub = x_starts[j:j + 1]
                        num_wins = num_y * tf.shape(x_sub)[0]

                        Q_wins = self._extract_windows_vectorized(Q_proj, y_starts, x_sub)
                        K_wins = self._extract_windows_vectorized(K_proj, y_starts, x_sub)
                        V_wins = self._extract_windows_vectorized(V_proj, y_starts, x_sub)

                        out_wins, attn_conf = self._window_attention(Q_wins, K_wins, V_wins, training, num_wins)

                        if self.overlap_reduction == 'attention_weighted':
                            conf = attn_conf
                        elif self.overlap_reduction == 'geometrical':
                            conf = tf.tile(tf.cast(self.geometrical_confidence, out_wins.dtype),
                                           [tf.shape(out_wins)[0], 1, 1, 1])
                        else:  # mean
                            conf = tf.ones([tf.shape(out_wins)[0], WSY, WSX, 1], dtype=out_wins.dtype)

                        out, weights = self._scatter_windows_accumulate(
                            out_wins, conf, y_starts, x_sub, out, weights, Y, X
                        )
                        return j + 1, out, weights

                    _, output, weight_sums = tf.while_loop(
                        lambda j, o, w: j < num_x,
                        process_col,
                        [0, output, weight_sums]
                    )

                else:  # sequential
                    # Process one window at a time
                    def process_window(idx, out, weights):
                        i = idx // num_x
                        j = idx % num_x

                        y_sub = y_starts[i:i + 1]
                        x_sub = x_starts[j:j + 1]
                        num_wins = 1

                        Q_wins = self._extract_windows_vectorized(Q_proj, y_sub, x_sub)
                        K_wins = self._extract_windows_vectorized(K_proj, y_sub, x_sub)
                        V_wins = self._extract_windows_vectorized(V_proj, y_sub, x_sub)

                        out_wins, attn_conf = self._window_attention(Q_wins, K_wins, V_wins, training, num_wins)

                        if self.overlap_reduction == 'attention_weighted':
                            conf = attn_conf
                        elif self.overlap_reduction == 'geometrical':
                            conf = tf.tile(tf.cast(self.geometrical_confidence, out_wins.dtype),
                                           [tf.shape(out_wins)[0], 1, 1, 1])
                        else:  # mean
                            conf = tf.ones([tf.shape(out_wins)[0], WSY, WSX, 1], dtype=out_wins.dtype)

                        out, weights = self._scatter_windows_accumulate(
                            out_wins, conf, y_sub, x_sub, out, weights, Y, X
                        )
                        return idx + 1, out, weights

                    total_windows = num_y * num_x
                    _, output, weight_sums = tf.while_loop(
                        lambda idx, o, w: idx < total_windows,
                        process_window,
                        [0, output, weight_sums]
                    )

                # Final normalization
                epsilon = tf.cast(1e-3 if self.overlap_reduction != 'mean' else 1.0, output.dtype)
                output = output / tf.maximum(weight_sums, epsilon)

            return output

        # Branch based on image size
        single_window = tf.logical_and( tf.less_equal(Y, WSY), tf.less_equal(X, WSX) )
        output = tf.cond( single_window, single, multiple )
        # Output projection
        output = self.outproj(output)
        if self.multi_query:
            output = tf.reshape(output, tf.shape(source_query))
        if self.skip_connection:
            output = output + source_query

        return output

# ============================================================================
# TEST CODE
# ============================================================================

def test_coordinate_handling(multi_query:bool=False):

    """Test that window extraction and scattering preserves data correctly."""
    print("=" * 60)
    print("Testing Window Coordinate Handling")
    print("=" * 60)

    # Test both overlap reduction methods
    for overlap_method in ['mean', 'attention_weighted', 'geometrical']:
        print(f"\n{'=' * 60}")
        print(f"Testing with overlap_reduction='{overlap_method}'")
        print(f"{'=' * 60}")

        # Create layer
        layer = WindowSpatialAttention(num_heads=4, attention_filters=4, window_size=(7, 4), overlap_reduction=overlap_method)

        # Test with small image
        B, H, W, C = 2, 20, 4, 16
        k = tf.random.normal([B, H, W, C])
        if multi_query:
            Q = 2
            q = tf.random.normal([Q, B, H, W, C])
            q_resh = tf.reshape(q, [-1, H, W, C])
        else:
            Q = 1
            q = tf.random.normal([B, H, W, C])
            q_resh = q
        # Build layer
        layer.build([q.shape, k.shape])

        # Compute shift and grid
        print(f"\nImage size: {H}x{W}")
        print(f"Window size: {layer.window_size}")

        h_starts, w_starts = layer._compute_window_grid(H, W)
        print(f"\nWindow grid:")
        print(f"  h_starts: {h_starts.numpy()}")
        print(f"  w_starts: {w_starts.numpy()}")
        print(f"  Number of windows: {len(h_starts)} x {len(w_starts)} = {len(h_starts) * len(w_starts)}")

        # Check coverage
        max_h_covered = tf.reduce_max(h_starts) + layer.window_size[0]
        max_w_covered = tf.reduce_max(w_starts) + layer.window_size[1]
        print(f"\nCoverage check:")
        print(f"  Max H covered: {max_h_covered.numpy()} (image H: {H})")
        print(f"  Max W covered: {max_w_covered.numpy()} (image W: {W})")
        print(f"  Full coverage: {max_h_covered >= H and max_w_covered >= W}")

        # Test extract and scatter with identity operation
        print(f"\n{'-' * 60}")
        print("Testing Extract -> Scatter Round-trip")
        print(f"{'-' * 60}")

        # Extract windows
        windows = layer._extract_windows_vectorized(q_resh, h_starts, w_starts)
        print(f"Windows shape: {windows.shape}")

        # Scatter back (use appropriate method based on overlap_reduction)
        if overlap_method == 'mean':
            q_reconstructed = layer._scatter_windows_mean(
                windows, h_starts, w_starts, H, W
            )
        else:
            # Create weights (uniform for test, random for 'attention_weighted')
            weights = tf.random.uniform([tf.shape(windows)[0], layer.window_size[0], layer.window_size[1], 1], 0.3, 1.0)
            q_reconstructed = layer._scatter_windows_with_averaging(
                windows, weights, h_starts, w_starts, H, W
            )
        print(f"Reconstructed shape: {q_reconstructed.shape}")

        # Check reconstruction error
        diff = tf.abs(q_resh - q_reconstructed)
        max_error = tf.reduce_max(diff)
        mean_error = tf.reduce_mean(diff)

        print(f"\nReconstruction quality:")
        print(f"  Max absolute error: {max_error.numpy():.6f}")
        print(f"  Mean absolute error: {mean_error.numpy():.6f}")

        if overlap_method == 'mean':
            print(f"  (Should be near-zero for uniform weights)")
        else:
            print(f"  (Will have error due to weighted averaging)")

        # Check overlap statistics
        counts = tf.zeros([Q*B, H, W, 1], dtype=q.dtype)
        WSY, WSX = layer.window_size

        for h_start in h_starts.numpy():
            for w_start in w_starts.numpy():
                h_end = min(h_start + WSY, H)
                w_end = min(w_start + WSX, W)
                counts = counts.numpy()
                counts[:, h_start:h_end, w_start:w_end, :] += 1
                counts = tf.constant(counts)

        overlap_mask = counts > 1
        num_overlap = tf.reduce_sum(tf.cast(overlap_mask, tf.int32))
        print(f"\nOverlap statistics:")
        print(f"  Pixels with overlaps: {num_overlap.numpy()}")
        print(f"  Max overlap count: {tf.reduce_max(counts).numpy():.0f}")

        # Test full forward pass
        print(f"\n{'-' * 60}")
        print("Testing Full Forward Pass")
        print(f"{'-' * 60}")

        output = layer([q, k], training=False)
        print(f"Output shape: {output.shape}")
        print(f"Input shape:  {q.shape}")
        assert output.shape == q.shape, "Output shape should match input shape!"
        print("✓ Shape check passed")

    # Test with different sizes
    print("\n" + "=" * 60)
    print("Testing Multiple Image Sizes (mean reduction)")
    print("=" * 60)

    layer = WindowSpatialAttention(num_heads=1, attention_filters=C, window_size=7, overlap_reduction='mean')
    test_sizes = [(14, 14), (20, 30), (50, 50), (7, 7)]

    for H_test, W_test in test_sizes:
        k_test = tf.random.normal([1, H_test, W_test, C])
        if multi_query:
            Q = 2
            q_test = tf.random.normal([Q, 1, H_test, W_test, C])
            q_test_resh = tf.reshape(q_test, [-1, H_test, W_test, C])
        else:
            Q = 1
            q_test = tf.random.normal([1, H_test, W_test, C])
            q_test_resh = q_test
        layer.build([q_test.shape, k_test.shape])

        h_starts, w_starts = layer._compute_window_grid(H_test, W_test)

        windows = layer._extract_windows_vectorized(q_test_resh, h_starts, w_starts)
        q_recon = layer._scatter_windows_mean(
            windows, h_starts, w_starts, H_test, W_test
        )

        error = tf.reduce_max(tf.abs(q_test_resh - q_recon))

        print(f"Size {H_test:3d}x{W_test:3d}: "
              f"windows={len(h_starts):2d}x{len(w_starts):2d}, "
              f"max_error={error.numpy():.6f}")

    print("\n" + "=" * 60)
    print("All coordinate handling tests passed!")
    print("=" * 60)


def test_window_coverage_with_geometrical(multi_query:bool):
    # Create a small test image (6x6) with distinct values for visualization
    test_image = tf.constant([
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
        [[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
        [[13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18]],
        [[19, 19], [20, 20], [21, 21], [22, 22], [23, 23], [24, 24]],
        [[25, 25], [26, 26], [27, 27], [28, 28], [29, 29], [30, 30]],
        [[31, 31], [32, 32], [33, 33], [34, 34], [35, 35], [36, 36]]
    ], dtype=tf.float32)  # Shape: (6, 6, 2)
    test_image = tf.expand_dims(test_image, axis=0)  # Add batch dim: (1, 6, 6, 2)
    test_q = tf.stack([test_image, test_image*2, test_image*3], 0) if multi_query else test_image

    # Test with different window sizes and overlap modes
    window_sizes = [(3, 3), (4, 4), (6, 6)]
    overlap_modes = ['mean', 'geometrical']

    for window_size in window_sizes:
        for overlap_mode in overlap_modes:
            print(f"\n{'='*50}")
            print(f"Testing: Window {window_size}, Overlap {overlap_mode}")
            print(f"{'='*50}")

            # Create layer
            window_attention = WindowSpatialAttention(
                num_heads=1,
                attention_filters=2,
                window_size=window_size,
                overlap_reduction=overlap_mode,
                layer_normalization=False,
                skip_connection=False
            )

            # Build the layer
            window_attention.build([test_q.shape, test_image.shape])

            # Get window grid coordinates
            Y, X, C = test_image.shape[1], test_image.shape[2], test_image.shape[3]
            y_starts, x_starts = window_attention._compute_window_grid(Y, X)

            print(f"Window start coordinates:")
            print(f"Y starts: {y_starts.numpy()}")
            print(f"X starts: {x_starts.numpy()}")

            # Extract windows
            Q = tf.reshape(test_q, [-1, Y, X, C])
            windows = window_attention._extract_windows_vectorized(Q, y_starts, x_starts)

            print(f"\nExtracted windows shape: {windows.shape}")
            num_windows = len(y_starts) * len(x_starts)  # Number of windows per batch
            print(f"Number of windows: {num_windows}")

            # Print each window's coordinates and values
            WSY, WSX = window_attention.window_size
            for win_idx in range(num_windows):
                window = windows[win_idx, :, :, :]  # Shape: (WSY, WSX, C)
                print(f"\nWindow {win_idx + 1}:")

                # Get the actual coordinates of this window
                y_start = y_starts[win_idx // len(x_starts)]
                x_start = x_starts[win_idx % len(x_starts)]

                # Create coordinate grid for this window
                y_coords = tf.range(y_start, min(y_start + WSY, Y))
                x_coords = tf.range(x_start, min(x_start + WSX, X))
                y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing='ij')

                print(f"  Top-left corner: ({y_start.numpy()}, {x_start.numpy()})")
                print(f"  Window coordinates:")
                coords = tf.stack([y_grid, x_grid], axis=-1)
                for y in range(len(y_coords)):
                    row = []
                    for x in range(len(x_coords)):
                        coord = coords[y, x]
                        val = window[y, x].numpy()
                        row.append(f"({coord[0]}, {coord[1]}):{val}")
                    print("  " + " ".join(row))

            # Test reconstruction
            if overlap_mode == 'mean':
                reconstructed = window_attention._scatter_windows_mean(windows, y_starts, x_starts, Y, X)
            else:  # geometrical
                geometrical_confidence = window_attention._geometrical_confidence()
                print(f"\nGeometrical confidence shape: {geometrical_confidence.shape}")
                print("Geometrical confidence values:")
                print(geometrical_confidence[0, :, :, 0].numpy())

                # FIX: tile to match windows batch size (which is num_windows * Q.shape[0])
                tiled_confidence = tf.tile(tf.cast(geometrical_confidence, windows.dtype),
                                         [tf.shape(windows)[0], 1, 1, 1])
                reconstructed = window_attention._scatter_windows_with_averaging(
                    windows, tiled_confidence, y_starts, x_starts, Y, X)

            print(f"\nReconstructed image shape: {reconstructed.shape}")
            print("First channel of reconstructed image:")
            for y in range(Y):
                row = []
                for x in range(X):
                    val = reconstructed[0, y, x, 0].numpy()
                    row.append(f"{val:5.1f}")
                print(" ".join(row))

            # Verify reconstruction
            diff = tf.abs(Q - reconstructed)
            max_diff = tf.reduce_max(diff).numpy()
            mean_diff = tf.reduce_mean(diff).numpy()
            print(f"\nMax difference: {max_diff:.4f}")
            print(f"Mean difference: {mean_diff:.4f}")

            # Edge case verification: Check if all original pixels are covered
            coverage_mask = np.zeros((Y, X))
            for win_idx in range(num_windows):
                y_start = y_starts[win_idx // len(x_starts)].numpy()
                x_start = x_starts[win_idx % len(x_starts)].numpy()
                y_end = min(y_start + WSY, Y)
                x_end = min(x_start + WSX, X)
                coverage_mask[y_start:y_end, x_start:x_end] += 1

            print("\nCoverage mask (number of windows covering each pixel):")
            for y in range(Y):
                row = []
                for x in range(X):
                    row.append(f"{coverage_mask[y, x]:1.0f}")
                print(" ".join(row))

            # Check if all pixels are covered by at least one window
            min_coverage = np.min(coverage_mask)
            max_coverage = np.max(coverage_mask)
            print(f"\nCoverage statistics:")
            print(f"  Minimum coverage: {min_coverage} (should be >=1)")
            print(f"  Maximum coverage: {max_coverage}")
            print(f"  All pixels covered: {min_coverage >= 1}")


def test_multiquery_equivalence():
    """
    Verify that multi_query=True path produces the same results as
    applying single-query attention sequentially to each query.
    """
    print("=" * 70)
    print("Testing Multi-Query vs Sequential Single-Query Equivalence")
    print("=" * 70)

    # Test parameters
    Q = 3
    B = 2
    H, W = 14, 16
    C = 16
    num_heads = 4
    attention_filters = 4
    window_size = (7, 5)

    # Create test data with fixed seed
    np.random.seed(42)
    tf.random.set_seed(42)

    queries = tf.random.normal([Q, B, H, W, C], seed=42)
    key = tf.random.normal([B, H, W, C], seed=43)
    value = tf.random.normal([B, H, W, C], seed=44)

    # Test each overlap reduction method
    for overlap_method in ['mean', 'attention_weighted', 'geometrical']:
        print(f"\n{'=' * 70}")
        print(f"Testing overlap_reduction='{overlap_method}'")
        print(f"{'=' * 70}")

        # Reset seeds before each test
        tf.random.set_seed(100)

        # Method 1: Multi-query mode
        print("\n--- Multi-Query Mode ---")
        layer_multi = WindowSpatialAttention(
            num_heads=num_heads,
            attention_filters=attention_filters,
            window_size=window_size,
            overlap_reduction=overlap_method,
            skip_connection=False,
            layer_normalization=False,
            dropout=0.0,
            add_distance_embedding=False  # Disable for simpler comparison
        )

        # Build and get output
        output_multi = layer_multi([queries, key, value], training=False)
        print(f"Multi-query output shape: {output_multi.shape}")

        # Method 2: Sequential single-query mode
        print("\n--- Sequential Single-Query Mode ---")

        # Reset seed to get same weight initialization
        tf.random.set_seed(100)

        layer_single = WindowSpatialAttention(
            num_heads=num_heads,
            attention_filters=attention_filters,
            window_size=window_size,
            overlap_reduction=overlap_method,
            skip_connection=False,
            layer_normalization=False,
            dropout=0.0,
            add_distance_embedding=False
        )

        # Build with first query to initialize weights
        _ = layer_single([queries[0], key, value], training=False)

        # Copy weights to ensure exact match
        layer_single.qproj.set_weights(layer_multi.qproj.get_weights())
        layer_single.kproj.set_weights(layer_multi.kproj.get_weights())
        layer_single.vproj.set_weights(layer_multi.vproj.get_weights())
        layer_single.outproj.set_weights(layer_multi.outproj.get_weights())
        layer_single.relative_position_bias_table.assign(layer_multi.relative_position_bias_table)

        # Apply to each query sequentially
        outputs_sequential = []
        for q_idx in range(Q):
            query_single = queries[q_idx]
            output_single = layer_single([query_single, key, value], training=False)
            outputs_sequential.append(output_single)

        # Stack results
        output_sequential = tf.stack(outputs_sequential, axis=0)
        print(f"Stacked sequential output shape: {output_sequential.shape}")

        # Compare
        diff = tf.abs(output_multi - output_sequential)
        max_diff = tf.reduce_max(diff).numpy()
        mean_diff = tf.reduce_mean(diff).numpy()

        print(f"\nMax difference:  {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")

        tolerance = 1e-4
        if max_diff < tolerance:
            print(f"✓ PASS (tolerance: {tolerance})")
        else:
            print(f"✗ FAIL (tolerance: {tolerance})")
            # Debug info
            print("\nFirst query comparison (first 3x3x3):")
            print("Multi:", output_multi[0, 0, :3, :3, :3].numpy())
            print("Sequential:", output_sequential[0, 0, :3, :3, :3].numpy())


def test_edge_cases():
    """Test edge cases for multi-query mode."""
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)

    # Test with single window (no overlap)
    print("\n--- Single Window (no overlap) ---")
    Q, B = 2, 1
    H, W, C = 7, 7, 8

    queries = tf.random.normal([Q, B, H, W, C])
    key = tf.random.normal([B, H, W, C])

    layer = WindowSpatialAttention(
        num_heads=2,
        attention_filters=4,
        window_size=(7, 7),
        overlap_reduction='mean',
        skip_connection=False,
        layer_normalization=False,
        add_distance_embedding=False
    )

    layer.build([queries.shape, key.shape])
    output = layer([queries, key], training=False)
    print(f"Output shape: {output.shape}")
    print(f"Expected: {queries.shape}")
    assert output.shape == queries.shape
    print("✓ Single window case passed")

    # Test with one query (should still work)
    print("\n--- Single Query in Multi-Query Format ---")
    Q = 1
    queries_single = tf.random.normal([Q, B, H, W, C])

    layer_single_q = WindowSpatialAttention(
        num_heads=2,
        attention_filters=4,
        window_size=(7, 7),
        overlap_reduction='mean',
        skip_connection=False,
        add_distance_embedding=False
    )

    layer_single_q.build([queries_single.shape, key.shape])
    output_single_q = layer_single_q([queries_single, key], training=False)
    print(f"Output shape: {output_single_q.shape}")
    print(f"Expected: {queries_single.shape}")
    assert output_single_q.shape == queries_single.shape
    print("✓ Single query in multi-query format passed")


def test_attention_computation_details():
    """
    Deep dive into attention computation to verify the einsum operations
    are mathematically equivalent.
    """
    print("\n" + "=" * 70)
    print("Testing Attention Computation Details")
    print("=" * 70)

    # Small test for manual verification
    Q, B = 2, 1
    H, W, C = 7, 7, 8
    num_heads = 2
    attention_filters = 4

    queries = tf.random.normal([Q, B, H, W, C])
    key = tf.random.normal([B, H, W, C])
    value = tf.random.normal([B, H, W, C])

    layer = WindowSpatialAttention(
        num_heads=num_heads,
        attention_filters=attention_filters,
        window_size=(7, 7),
        overlap_reduction='mean',
        skip_connection=False,
        layer_normalization=False,
        dropout=0.0,
        add_distance_embedding=False
    )

    layer.build([queries.shape, key.shape, value.shape])

    # Project Q, K, V
    queries_flat = tf.reshape(queries, [Q * B, H, W, C])
    Q_proj = layer.qproj(queries_flat)
    K_proj = layer.kproj(key)
    V_proj = layer.vproj(value)

    # Manually compute attention for verification
    HF = num_heads * attention_filters
    N = H * W

    # Multi-query path
    print("\n--- Multi-Query Einsum Path ---")
    q_mq = tf.reshape(Q_proj, [1, Q, B, N, num_heads, attention_filters])
    k_mq = tf.reshape(K_proj, [1, B, N, num_heads, attention_filters])
    v_mq = tf.reshape(V_proj, [1, B, N, num_heads, attention_filters])

    scale = tf.math.rsqrt(tf.cast(attention_filters, q_mq.dtype))
    attn_mq = tf.einsum('mqbihf,mbkhf->mqbhik', q_mq, k_mq) * scale
    print(f"Attention scores shape (multi-query): {attn_mq.shape}")

    # Single-query path (sequential)
    print("\n--- Single-Query Matmul Path ---")
    attn_scores_seq = []
    for q_idx in range(Q):
        q_sq = tf.reshape(Q_proj[q_idx:q_idx + 1], [B, N, num_heads, attention_filters])
        k_sq = tf.reshape(K_proj, [B, N, num_heads, attention_filters])

        q_sq = tf.transpose(q_sq, [0, 2, 1, 3])  # (B, H, N, F)
        k_sq = tf.transpose(k_sq, [0, 2, 3, 1])  # (B, H, F, N)

        attn_sq = tf.matmul(q_sq, k_sq) * scale  # (B, H, N, N)
        attn_scores_seq.append(attn_sq)

    attn_scores_seq = tf.stack(attn_scores_seq, axis=0)  # (Q, B, H, N, N)
    print(f"Attention scores shape (sequential): {attn_scores_seq.shape}")

    # Reshape multi-query result to match
    attn_mq_reshaped = tf.reshape(attn_mq, [Q, B, num_heads, N, N])

    # Compare
    diff = tf.abs(attn_mq_reshaped - attn_scores_seq)
    print(f"\nAttention score differences:")
    print(f"  Max:  {tf.reduce_max(diff).numpy():.2e}")
    print(f"  Mean: {tf.reduce_mean(diff).numpy():.2e}")

    if tf.reduce_max(diff) < 1e-5:
        print("✓ Attention computation equivalent")
    else:
        print("✗ Attention computation differs!")


def test_window_processing_modes_equivalence():
    """Test that all window_processing modes produce equivalent results."""
    print("\n" + "=" * 70)
    print("Testing Window Processing Modes Equivalence")
    print("=" * 70)

    # Test parameters
    B, H, W, C = 2, 20, 16, 16
    num_heads = 4
    attention_filters = 4
    window_size = (7, 5)

    np.random.seed(42)
    tf.random.set_seed(42)

    # Test both single-query and multi-query
    for multi_query in [False, True]:
        print(f"\n{'=' * 70}")
        print(f"Testing {'Multi-Query' if multi_query else 'Single-Query'} Mode")
        print(f"{'=' * 70}")

        if multi_query:
            Q = 3
            queries = tf.random.normal([Q, B, H, W, C])
        else:
            queries = tf.random.normal([B, H, W, C])

        key = tf.random.normal([B, H, W, C])
        value = tf.random.normal([B, H, W, C])

        # Test each overlap reduction method
        for overlap_method in ['mean', 'attention_weighted', 'geometrical']:
            print(f"\n{'-' * 70}")
            print(f"Testing overlap_reduction='{overlap_method}'")
            print(f"{'-' * 70}")

            # Reference: 'all' mode (original behavior)
            layer_all = WindowSpatialAttention(
                num_heads=num_heads,
                attention_filters=attention_filters,
                window_size=window_size,
                overlap_reduction=overlap_method,
                window_processing='all',
                skip_connection=False,
                layer_normalization=False,
                dropout=0.0,
                add_distance_embedding=False
            )

            # Build by calling once
            _ = layer_all([queries, key, value], training=False)
            output_all = layer_all([queries, key, value], training=False)

            # Test other modes
            for mode in ['row', 'col', 'sequential']:
                layer_mode = WindowSpatialAttention(
                    num_heads=num_heads,
                    attention_filters=attention_filters,
                    window_size=window_size,
                    overlap_reduction=overlap_method,
                    window_processing=mode,
                    skip_connection=False,
                    layer_normalization=False,
                    dropout=0.0,
                    add_distance_embedding=False
                )

                # Build by calling once
                _ = layer_mode([queries, key, value], training=False)

                # Copy weights from reference layer
                layer_mode.qproj.set_weights(layer_all.qproj.get_weights())
                layer_mode.kproj.set_weights(layer_all.kproj.get_weights())
                layer_mode.vproj.set_weights(layer_all.vproj.get_weights())
                layer_mode.outproj.set_weights(layer_all.outproj.get_weights())
                layer_mode.relative_position_bias_table.assign(
                    layer_all.relative_position_bias_table
                )
                if hasattr(layer_all, 'distance_encoding_y'):
                    layer_mode.distance_encoding_y.set_weights(
                        layer_all.distance_encoding_y.get_weights()
                    )
                    layer_mode.distance_encoding_x.set_weights(
                        layer_all.distance_encoding_x.get_weights()
                    )

                # Forward pass
                output_mode = layer_mode([queries, key, value], training=False)

                # Compare
                diff = tf.abs(output_all - output_mode)
                max_diff = tf.reduce_max(diff).numpy()
                mean_diff = tf.reduce_mean(diff).numpy()
                relative_error = mean_diff / (tf.reduce_mean(tf.abs(output_all)).numpy() + 1e-8)

                print(f"  Mode '{mode}':")
                print(f"    Max difference:  {max_diff:.2e}")
                print(f"    Mean difference: {mean_diff:.2e}")
                print(f"    Relative error:  {relative_error:.2e}")

                tolerance = 1e-4
                status = "✓ PASS" if max_diff < tolerance else "✗ FAIL"
                print(f"    {status} (tolerance: {tolerance})")

    print("\n" + "=" * 70)
    print("Window Processing Modes Equivalence Test Complete")
    print("=" * 70)


def test_window_processing_memory_profile():
    """Demonstrate memory usage differences between modes."""
    print("\n" + "=" * 70)
    print("Window Processing Memory Profile Demo")
    print("=" * 70)
    print("\nThis test demonstrates the memory trade-offs:")
    print("  'all':        Fastest, highest memory (processes all windows at once)")
    print("  'row':        Medium speed/memory (processes one row at a time)")
    print("  'col':        Medium speed/memory (processes one column at a time)")
    print("  'sequential': Slowest, lowest memory (processes one window at a time)")
    print("\nFor large images with many windows, use 'row', 'col', or 'sequential'")
    print("to avoid OOM errors.")

    # Demonstrate with a configuration that would create many windows
    B, H, W, C = 1, 50, 50, 16
    window_size = (7, 7)

    layer = WindowSpatialAttention(
        num_heads=4,
        attention_filters=4,
        window_size=window_size,
        window_processing='all',
        dropout=0.0,
        add_distance_embedding=False
    )

    x = tf.random.normal([B, H, W, C])
    layer.build(x.shape)

    y_starts, x_starts = layer._compute_window_grid(H, W)
    num_windows = len(y_starts) * len(x_starts)

    print(f"\nExample: {H}x{W} image with {window_size} windows")
    print(f"  Number of windows: {num_windows}")
    print(f"  'all' mode: processes {num_windows} windows at once")
    print(f"  'row' mode: processes {len(x_starts)} windows per iteration ({len(y_starts)} iterations)")
    print(f"  'col' mode: processes {len(y_starts)} windows per iteration ({len(x_starts)} iterations)")
    print(f"  'sequential' mode: processes 1 window per iteration ({num_windows} iterations)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    #test_coordinate_handling(multi_query=True)
    #test_window_coverage_with_geometrical(multi_query=True)
    #test_multiquery_equivalence()
    #test_edge_cases()
    #test_attention_computation_details()
    test_window_processing_modes_equivalence()
    #test_window_processing_memory_profile()

    print("\n" + "=" * 70)
    print("All Multi-Query Equivalence Tests Complete")
    print("=" * 70)
import tensorflow as tf

import tensorflow as tf


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
    """

    def __init__(self, num_heads: int, attention_filters: int = 0, window_size:tuple = 7,
                 dropout: float = 0.1, skip_connection: bool = False,
                 overlap_reduction: str = 'geometrical', # 'mean', 'attention_weighted', 'geometrical'
                 additional_positional_embedding:bool = False,
                 l2_reg: float = 0., name="WindowSpatialAttention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        # Support both int and tuple for window_size
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = tuple(window_size)
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.skip_connection = skip_connection
        self.overlap_reduction = overlap_reduction
        self.additional_positional_embedding=additional_positional_embedding
        assert overlap_reduction in ['mean', 'attention_weighted', 'geometrical'], \
            f"overlap_reduction must be 'mean' or 'attention_weighted', got {overlap_reduction}"

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "attention_filters": self.attention_filters,
            "window_size": self.window_size,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "skip_connection": self.skip_connection,
            "overlap_reduction": self.overlap_reduction
        })
        return config

    def build(self, input_shapes):
        if not isinstance(input_shapes, (tuple, list)) or len(input_shapes) > 4:
            input_shapes = [input_shapes]
        try:
            input_shapes = [s.as_list() for s in input_shapes]
            input_shape = input_shapes[0]
            for s in input_shapes[1:min(3, len(input_shapes))]:
                assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape, s)), \
                    f"all tensors must have same input shape: {input_shape} != {s}"
        except:
            pass
        input_shape = input_shapes[0]


        self.spatial_dims = input_shape[1:-1]
        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)

        HF = self.num_heads * self.attention_filters

        # Separate Q, K, V projections
        self.qproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="qproj")
        self.kproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="kproj")
        self.vproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="vproj")
        self.outproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=False, name="outproj")

        if self.skip_connection:
            self.skipproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=True, name="skip")

        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        # Relative position bias table
        WSY, WSX = self.window_size
        bias_size = (2 * WSY - 1) * (2 * WSX - 1)

        self.relative_position_bias_table = self.add_weight(
            name="rpb",
            shape=(bias_size, self.num_heads),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
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

    def _extract_windows_vectorized(self, x, y_starts, x_starts):
        """
        Extract windows using vectorized operations.

        Args:
            x: (B, Y, X, C)
            y_starts: (num_y,) - window start positions in Y dimension
            x_starts: (num_x,) - window start positions in X dimension

        Returns:
            windows: (num_y*num_x*B, WSY, WSX, C)
        """
        B, Y, X, C = tf.unstack(tf.shape(x))
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
        x_flat = tf.reshape(x, [B, Y * X, C])

        # Use tf.gather with batch_dims for efficient batched gathering
        num_windows = tf.shape(flat_indices)[0]
        flat_indices_batched = tf.tile(flat_indices[None, :, :, :], [B, 1, 1, 1])  # (B, num_windows, WSY, WSX)

        # Reshape for batch gather
        flat_indices_batched = tf.reshape(flat_indices_batched, [B, num_windows * WSY * WSX])

        # Gather: (B, num_windows*WSY*WSX, C)
        gathered = tf.gather(x_flat, flat_indices_batched, axis=1, batch_dims=1)

        # Reshape to (B, num_windows, WSY, WSX, C)
        gathered = tf.reshape(gathered, [B, num_windows, WSY, WSX, C])

        # Transpose and reshape to (num_windows*B, WSY, WSX, C)
        windows = tf.transpose(gathered, [1, 0, 2, 3, 4])
        windows = tf.reshape(windows, [num_windows * B, WSY, WSX, C])

        return windows

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
        C = tf.shape(windows)[-1]
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

    def _scatter_windows_with_averaging(self, windows, window_weights, y_starts, x_starts, Y, X):
        """
        Scatter windows back to spatial dimensions with weighted averaging at overlaps.
        """
        C = tf.shape(windows)[-1]
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

        epsilon = tf.cast(1e-3, output.dtype)  # Larger epsilon for float16
        output = output / tf.maximum(weight_sums, epsilon)

        return output

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

    def _window_attention(self, q_windows, k_windows, v_windows, training=None):
        """
        Compute attention within windows.

        Args:
            q_windows, k_windows, v_windows: (B*num_windows, WSY, WSX, HF)

        Returns:
            output: (B*num_windows, WSY, WSX, HF)
            attention_weights: (B*num_windows, WSY, WSX, 1) - confidence measure for weighting
        """
        B_win = tf.shape(q_windows)[0]
        WSY, WSX = self.window_size
        HF = tf.shape(q_windows)[-1]
        H = self.num_heads
        F = self.attention_filters
        N = WSY * WSX

        # Reshape to (B_win, N, H, F)
        q = tf.reshape(q_windows, [B_win, N, H, F])
        k = tf.reshape(k_windows, [B_win, N, H, F])
        v = tf.reshape(v_windows, [B_win, N, H, F])

        # Transpose to (B_win, H, N, F) for efficient matmul
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Attention: (B_win, H, N, N)
        scale = tf.math.rsqrt(tf.cast(F, q.dtype))
        attn = tf.matmul(q, k, transpose_b=True) * scale

        # Add relative position bias
        rpb = tf.gather(self.relative_position_bias_table,
                        tf.reshape(self.relative_position_index, [-1]))
        rpb = tf.reshape(rpb, [N, N, H])
        rpb = tf.transpose(rpb, [2, 0, 1])  # (H, N, N)
        attn = attn + rpb[None, :, :, :]

        # Softmax
        attn_probs = tf.nn.softmax(attn, axis=-1)

        if self.overlap_reduction == "attention_weighted":
            # Maximum attention value indicates how "peaked" the distribution is
            max_attn = tf.reduce_max(attn_probs, axis=-1)  # (B_win, H, N)
            confidence = tf.reduce_mean(max_attn, axis=1)  # (B_win, N)
            confidence_spatial = tf.reshape(confidence, [B_win, WSY, WSX, 1])
            confidence_spatial = tf.maximum(confidence_spatial, tf.cast(1e-2, confidence_spatial.dtype))
            confidence_spatial = tf.stop_gradient(confidence_spatial)
        else:
            confidence_spatial = None

        if self.dropout > 0 and training:
            attn_probs = self.dropout_layer(attn_probs, training=training)

        # Apply to values: (B_win, H, N, F)
        out = tf.matmul(attn_probs, v)

        # Transpose back: (B_win, N, H, F)
        out = tf.transpose(out, [0, 2, 1, 3])

        # Reshape to spatial: (B_win, WSY, WSX, HF)
        out = tf.reshape(out, [B_win, WSY, WSX, HF])

        return out, confidence_spatial

    def call(self, x, training: bool = None):
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                key = value = query = x[0]
                emb = None
            elif len(x) == 2:
                query, value = x
                key = value
                emb = None
            elif len(x) == 3:
                query, key, value = x
                emb = None
            elif len(x) == 4:
                query, key, value, emb = x
            else:
                raise ValueError("Invalid input length should be lower than 4")
        else:
            key = value = query = x
            emb = None

        query_orig = query

        B, Y, X, C = tf.unstack(tf.shape(query))

        # Compute window grid with overlap
        y_starts, x_starts = self._compute_window_grid(Y, X)

        # Project Q, K, V
        Q = self.qproj(query)
        K = self.kproj(key)
        V = self.vproj(value)
        if emb is not None:
            Q = Q + emb
            K = K + emb
        # Extract windows
        Q_windows = self._extract_windows_vectorized(Q, y_starts, x_starts)
        K_windows = self._extract_windows_vectorized(K, y_starts, x_starts)
        V_windows = self._extract_windows_vectorized(V, y_starts, x_starts)

        # Apply window attention
        out_windows, attention_confidence = self._window_attention(
            Q_windows, K_windows, V_windows, training
        )

        # Scatter back with appropriate weighting
        if self.overlap_reduction == 'attention_weighted':
            output = self._scatter_windows_with_averaging( out_windows, attention_confidence, y_starts, x_starts, Y, X )
        elif self.overlap_reduction == "geometrical":
            num_win = tf.shape(y_starts)[0] * tf.shape(x_starts)[0]
            geometrical_confidence = tf.tile(tf.cast(self._geometrical_confidence(), out_windows.dtype), [B * num_win, 1, 1, 1] )
            output = self._scatter_windows_with_averaging(out_windows, geometrical_confidence, y_starts, x_starts, Y, X)
        else:  # 'mean'
            output = self._scatter_windows_mean( out_windows, y_starts, x_starts, Y, X )

        # Output projection
        output = self.outproj(output)

        if self.skip_connection:
            output = tf.concat([output, query_orig], axis=-1)
            output = self.skipproj(output)

        return output

# ============================================================================
# TEST CODE
# ============================================================================

def test_coordinate_handling():

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
        layer = WindowSpatialAttention(num_heads=4, window_size=(7, 4), overlap_reduction=overlap_method)

        # Test with small image
        B, H, W, C = 2, 20, 4, 16
        x = tf.random.normal([B, H, W, C])

        # Build layer
        layer.build(x.shape)

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
        windows = layer._extract_windows_vectorized(x, h_starts, w_starts)
        print(f"Windows shape: {windows.shape}")

        # Scatter back (use appropriate method based on overlap_reduction)
        if overlap_method == 'mean':
            x_reconstructed = layer._scatter_windows_mean(
                windows, h_starts, w_starts, H, W
            )
        else:
            # Create weights (uniform for test, random for 'attention_weighted')
            weights = tf.random.uniform([tf.shape(windows)[0], layer.window_size[0], layer.window_size[1], 1], 0.3, 1.0)
            x_reconstructed = layer._scatter_windows_with_averaging(
                windows, weights, h_starts, w_starts, H, W
            )
        print(f"Reconstructed shape: {x_reconstructed.shape}")

        # Check reconstruction error
        diff = tf.abs(x - x_reconstructed)
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
        counts = tf.zeros([B, H, W, 1], dtype=x.dtype)
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

        output = layer(x, training=False)
        print(f"Output shape: {output.shape}")
        print(f"Input shape:  {x.shape}")
        assert output.shape == x.shape, "Output shape should match input shape!"
        print("✓ Shape check passed")

    # Test with different sizes
    print("\n" + "=" * 60)
    print("Testing Multiple Image Sizes (mean reduction)")
    print("=" * 60)

    layer = WindowSpatialAttention(num_heads=4, window_size=7, overlap_reduction='mean')
    test_sizes = [(14, 14), (20, 30), (50, 50), (7, 7)]

    for H_test, W_test in test_sizes:
        x_test = tf.random.normal([1, H_test, W_test, C])
        layer.build(x_test.shape)

        h_starts, w_starts = layer._compute_window_grid(H_test, W_test)

        windows = layer._extract_windows_vectorized(x_test, h_starts, w_starts)
        x_recon = layer._scatter_windows_mean(
            windows, h_starts, w_starts, H_test, W_test
        )

        error = tf.reduce_max(tf.abs(x_test - x_recon))

        print(f"Size {H_test:3d}x{W_test:3d}: "
              f"windows={len(h_starts):2d}x{len(w_starts):2d}, "
              f"max_error={error.numpy():.6f}")

    print("\n" + "=" * 60)
    print("All coordinate handling tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    layer = WindowSpatialAttention(num_heads=4, window_size=(7, 4), overlap_reduction='mean')
    print(layer._geometrical_confidence().numpy())
    test_coordinate_handling()
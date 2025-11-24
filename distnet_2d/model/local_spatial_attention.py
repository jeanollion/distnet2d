import tensorflow as tf
import numpy as np


class LocalSpatialAttention(tf.keras.layers.Layer):
    """
    Optimized spatial attention - fast and memory efficient.
    Key optimization for speed: loop over spatial neighbors, rest of operation is vectorized
    Key optimization for memory: online softmax (value accumulation, no score stored)
    """

    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 dropout: float = 0.1, skip_connection: bool = False, l2_reg: float = 0.,
                 use_online_softmax: bool = False, name="LocalSpatialAttention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        self.spatial_radius = spatial_radius
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.skip_connection = skip_connection
        self.use_online_softmax = use_online_softmax

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "spatial_radius": self.spatial_radius,
            "skip_connection": self.skip_connection,
            "use_online_softmax": self.use_online_softmax
        })
        return config

    def build(self, input_shapes):
        if not isinstance(input_shapes, (tuple, list)) or len(input_shapes) > 3:
            input_shapes = [input_shapes]  # self-attention -> single input
        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass
        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape, s)), \
                f"all tensors must have same input shape: {input_shape} != {s}"

        self.spatial_dims = input_shape[1:-1]
        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)

        self.radius_y, self.radius_x = self.spatial_radius
        self.patch_height = 2 * self.radius_y + 1
        self.patch_width = 2 * self.radius_x + 1
        self.patch_size = self.patch_height * self.patch_width

        HF = self.num_heads * self.attention_filters

        # Projection layers
        self.qproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="qproj")
        self.kproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="kproj")
        self.vproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="vproj")
        self.outproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=False, name="outproj")

        if self.skip_connection:
            self.skipproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=True, name="skip")

        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        # Spatial positional embeddings
        self.spatial_pos_embedding = tf.keras.layers.Embedding(
            self.patch_size,
            HF,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="SpatialPosEnc"
        )

        # Pre-compute spatial offsets for graph mode compatibility
        spatial_offsets = []
        for dy in range(-self.radius_y, self.radius_y + 1):
            for dx in range(-self.radius_x, self.radius_x + 1):
                spatial_offsets.append([dy, dx])
        self.spatial_offsets = tf.constant(spatial_offsets, dtype=tf.int32)

        super().build(input_shape)

    def _get_query_spatial_indices(self, Y, X):
        """Get spatial positional indices for queries."""
        y_coords = tf.range(Y, dtype=tf.int32)
        x_coords = tf.range(X, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, Y - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, X - self.radius_x - 1)
        y_relative = y_coords - y_patch_center + self.radius_y
        x_relative = x_coords - x_patch_center + self.radius_x
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')
        spatial_indices = y_grid * self.patch_width + x_grid
        return spatial_indices

    def _compute_center_indices(self, Y, X):
        """Compute patch center coordinates."""
        y_coords = tf.range(Y, dtype=tf.int32)
        x_coords = tf.range(X, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, Y - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, X - self.radius_x - 1)
        y_grid, x_grid = tf.meshgrid(y_patch_center, x_patch_center, indexing='ij')
        return y_grid, x_grid

    def call(self, x, training: bool = None):
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                key = x[0]
                value = x[0]
                query = x[0]
            elif len(x) == 2:
                key = x[0]
                query = x[1]
                value = key
            else:
                key, query, value = x
        else:
            key = x
            value = x
            query = x

        C = self.filters
        shape = tf.shape(key)
        B, Y, X = shape[0], shape[1], shape[2]
        H = self.num_heads
        F = self.attention_filters
        HF = H * F
        S = self.patch_size
        dtype = key.dtype

        # Project
        K = self.kproj(key)
        Q = self.qproj(query)
        V = self.vproj(value)

        # Add query spatial positional encoding
        query_spatial_indices = self._get_query_spatial_indices(Y, X)  # (Y, X)
        q_s_emb = self.spatial_pos_embedding(query_spatial_indices)  # (Y, X, HF)
        Q = Q + q_s_emb  # (B, Y, X, HF)

        # Reshape Q for attention computation
        Q = tf.reshape(Q, (B, Y, X, H, F))

        # Get spatial embeddings for keys
        spatial_indices = tf.range(S, dtype=tf.int32)
        k_s_emb = self.spatial_pos_embedding(spatial_indices)  # (S, HF)

        # Compute patch centers
        cy_grid, cx_grid = self._compute_center_indices(Y, X)

        # Pre-compute batch indices (reused in both passes)
        b_idx = tf.tile(tf.range(B, dtype=tf.int32)[:, None, None], [1, Y, X])

        # Scaling factor
        scale = tf.math.rsqrt(tf.cast(F, dtype))

        if self.use_online_softmax:
            # Online softmax: accumulate exp-weighted values without storing scores
            min_val = tf.constant(-65504.0 if dtype == tf.float16 else -1e9, dtype=dtype)
            max_scores = tf.fill([B, Y, X, H, 1], min_val)
            sum_exp = tf.zeros([B, Y, X, H, 1], dtype=dtype)
            out_acc = tf.zeros([B, Y, X, H, F], dtype=dtype)

            # LOOP: Over spatial neighbors
            for s_idx in range(S):
                dy = self.spatial_offsets[s_idx, 0]
                dx = self.spatial_offsets[s_idx, 1]

                # Spatial embedding for this neighbor
                spatial_idx_scalar = (dy + self.radius_y) * self.patch_width + (dx + self.radius_x)
                neighbor_s_emb = tf.reshape(k_s_emb[spatial_idx_scalar], (1, 1, 1, HF))

                # Neighbor coordinates with edge handling
                ny = tf.clip_by_value(cy_grid + dy, 0, Y - 1)
                nx = tf.clip_by_value(cx_grid + dx, 0, X - 1)
                ny_b = tf.tile(ny[None, :, :], [B, 1, 1])
                nx_b = tf.tile(nx[None, :, :], [B, 1, 1])
                idx = tf.stack([b_idx, ny_b, nx_b], axis=-1)

                # Gather keys and values
                K_n = tf.gather_nd(K, idx) + neighbor_s_emb  # (B, Y, X, HF)
                V_n = tf.gather_nd(V, idx)  # (B, Y, X, HF)
                K_nh = tf.reshape(K_n, (B, Y, X, H, F))
                V_nh = tf.reshape(V_n, (B, Y, X, H, F))

                # Compute scores: (B, Y, X, H, 1)
                scores_s = tf.reduce_sum(Q * K_nh, axis=-1, keepdims=True) * scale

                # Update running max
                new_max = tf.maximum(max_scores, scores_s)

                # Rescale previous accumulations
                exp_diff = tf.exp(max_scores - new_max)
                sum_exp = sum_exp * exp_diff
                out_acc = out_acc * exp_diff

                # Add new contributions
                exp_scores = tf.exp(scores_s - new_max)
                sum_exp = sum_exp + exp_scores

                # Accumulate weighted values
                out_acc = out_acc + exp_scores * V_nh

                max_scores = new_max

            # Final normalization
            out_acc = out_acc / (sum_exp + 1e-8)

            # Apply dropout
            if self.dropout > 0 and training:
                out_acc = self.dropout_layer(out_acc, training=training)

        else:
            # TWO-PASS: Compute all scores first, then accumulate values
            # PASS 1: Compute all scores
            scores_array = tf.TensorArray(
                dtype=dtype,
                size=S,
                element_shape=None,
                clear_after_read=False
            )

            for s_idx in range(S):
                dy = self.spatial_offsets[s_idx, 0]
                dx = self.spatial_offsets[s_idx, 1]

                # Spatial embedding for this neighbor
                spatial_idx_scalar = (dy + self.radius_y) * self.patch_width + (dx + self.radius_x)
                neighbor_s_emb = tf.reshape(k_s_emb[spatial_idx_scalar], (1, 1, 1, HF))

                # Neighbor coordinates with edge handling
                ny = tf.clip_by_value(cy_grid + dy, 0, Y - 1)
                nx = tf.clip_by_value(cx_grid + dx, 0, X - 1)
                ny_b = tf.tile(ny[None, :, :], [B, 1, 1])
                nx_b = tf.tile(nx[None, :, :], [B, 1, 1])
                idx = tf.stack([b_idx, ny_b, nx_b], axis=-1)

                # Gather keys
                K_n = tf.gather_nd(K, idx) + neighbor_s_emb  # (B, Y, X, HF)
                K_nh = tf.reshape(K_n, (B, Y, X, H, F))

                # Compute scores
                scores = tf.reduce_sum(Q * K_nh, axis=-1) * scale  # (B, Y, X, H)
                scores_array = scores_array.write(s_idx, scores)

            # Stack all scores: (S, B, Y, X, H) -> (B, Y, X, H, S)
            scores = tf.transpose(scores_array.stack(), [1, 2, 3, 4, 0])

            # Softmax over all neighbors
            weights = tf.nn.softmax(scores, axis=-1)
            if self.dropout > 0 and training:
                weights = self.dropout_layer(weights, training=training)

            # PASS 2: Accumulate weighted values
            out_acc = tf.zeros([B, Y, X, H, F], dtype=dtype)

            for s_idx in range(S):
                dy = self.spatial_offsets[s_idx, 0]
                dx = self.spatial_offsets[s_idx, 1]

                ny = tf.clip_by_value(cy_grid + dy, 0, Y - 1)
                nx = tf.clip_by_value(cx_grid + dx, 0, X - 1)
                ny_b = tf.tile(ny[None, :, :], [B, 1, 1])
                nx_b = tf.tile(nx[None, :, :], [B, 1, 1])
                idx = tf.stack([b_idx, ny_b, nx_b], axis=-1)

                V_n = tf.gather_nd(V, idx)  # (B, Y, X, HF)
                V_nh = tf.reshape(V_n, (B, Y, X, H, F))

                # weights for this spatial neighbor: (B, Y, X, H, 1)
                w_s = tf.expand_dims(weights[..., s_idx], -1)
                out_acc = out_acc + w_s * V_nh

        # Output projection
        out_acc = tf.reshape(out_acc, (B, Y, X, HF))
        out = self.outproj(out_acc)

        if self.skip_connection:
            out = tf.concat([out, query], axis=-1)
            out = self.skipproj(out)

        return out


class LocalSpatialAttentionPatch(tf.keras.layers.Layer):
    """
    Regional spatial attention using extract_patches for parallelization.
    Faster but with higher memory footprint than the loop version.

    This version:
    - Extracts all spatial patches upfront using tf.image.extract_patches
    - Performs batched attention computation (no loops over spatial offsets)
    - Better GPU utilization and parallelism
    - Uses ~3x more memory than loop version
    - 2-4x faster for small/medium radius (r≤3)
    """

    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 skip_connection:bool=False,
                 dropout: float = 0.1, l2_reg: float = 0., name="RegionalSpatialAttentionHighMem"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        self.spatial_radius = spatial_radius
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.skip_connection=skip_connection

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "spatial_radius": self.spatial_radius,
            "skip_connection":self.skip_connection
        })
        return config

    def build(self, input_shapes):
        if not isinstance(input_shapes, (tuple, list)) or len(input_shapes) > 3:
            input_shapes = [input_shapes]
        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass
        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape, s)), \
                f"all tensors must have same input shape: {input_shape} != {s}"

        self.spatial_dims = input_shape[1:-1]
        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)

        self.radius_y, self.radius_x = self.spatial_radius
        self.patch_height = 2 * self.radius_y + 1
        self.patch_width = 2 * self.radius_x + 1
        self.neighborhood_size = self.patch_height * self.patch_width

        Ck = self.num_heads * self.attention_filters

        # Projections
        self.qproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="qproj")
        self.kproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="kproj")
        self.vproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="vproj")
        self.outproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=False, name="outproj")
        if self.skip_connection:
            self.skipproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=True, name="skip")
        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        # Spatial positional embeddings
        self.spatial_pos_embedding = tf.keras.layers.Embedding(
            self.patch_height * self.patch_width,
            Ck,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="SpatialPosEnc"
        )

        super().build(input_shape)

    def _get_query_spatial_indices(self, H, W):
        """
        For each query pixel, get the spatial index within the patch
        that corresponds to its position in the neighborhood.

        Returns: (H, W) array of spatial indices
        """
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)

        # Calculate which valid patch center this pixel would map to
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, H - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, W - self.radius_x - 1)

        # Calculate relative position within the patch
        y_relative = y_coords - y_patch_center + self.radius_y
        x_relative = x_coords - x_patch_center + self.radius_x

        # Create meshgrid
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')

        # Convert 2D position to flat index
        spatial_indices = y_grid * self.patch_width + x_grid

        return spatial_indices

    def _extract_patches_with_edge_handling(self, tensor):
        """
        Extract spatial patches with edge handling:
        - Interior pixels get their centered neighborhood
        - Edge pixels reuse patches from their nearest interior neighbor

        Args:
            tensor: (B, H, W, C) input tensor

        Returns:
            patches: (B, H, W, patch_h * patch_w, C)
        """
        B = tf.shape(tensor)[0]
        H = tf.shape(tensor)[1]
        W = tf.shape(tensor)[2]
        C = tf.shape(tensor)[3]

        # Extract patches with VALID padding (only for interior pixels)
        patches_valid = tf.image.extract_patches(
            images=tensor,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )  # (B, H_valid, W_valid, patch_h * patch_w * C)

        H_valid = H - 2 * self.radius_y
        W_valid = W - 2 * self.radius_x

        # Reshape for easier indexing
        patch_size = self.patch_height * self.patch_width
        patches_valid = tf.reshape(patches_valid, [B, H_valid, W_valid, patch_size, C])

        # Create index mapping for edge handling
        # Edge pixels use the patch from the nearest valid interior position

        # For y coordinate: clamp to [radius_y, H - radius_y - 1]
        y_indices = tf.range(H, dtype=tf.int32)
        y_indices = tf.clip_by_value(y_indices, self.radius_y, H - self.radius_y - 1)
        y_indices = y_indices - self.radius_y  # Convert to valid patch indices [0, H_valid)

        # For x coordinate: clamp to [radius_x, W - radius_x - 1]
        x_indices = tf.range(W, dtype=tf.int32)
        x_indices = tf.clip_by_value(x_indices, self.radius_x, W - self.radius_x - 1)
        x_indices = x_indices - self.radius_x  # Convert to valid patch indices [0, W_valid)

        # Create meshgrid of indices
        y_grid, x_grid = tf.meshgrid(y_indices, x_indices, indexing='ij')  # (H, W)

        # Create batch indices
        batch_indices = tf.range(B, dtype=tf.int32)
        batch_indices = tf.reshape(batch_indices, [B, 1, 1])
        batch_indices = tf.tile(batch_indices, [1, H, W])  # (B, H, W)

        # Stack indices for gather_nd
        y_grid_expanded = tf.tile(tf.expand_dims(y_grid, 0), [B, 1, 1])  # (B, H, W)
        x_grid_expanded = tf.tile(tf.expand_dims(x_grid, 0), [B, 1, 1])  # (B, H, W)

        indices = tf.stack([batch_indices, y_grid_expanded, x_grid_expanded], axis=-1)  # (B, H, W, 3)

        # Gather patches
        patches_full = tf.gather_nd(patches_valid, indices)  # (B, H, W, patch_size, C)

        return patches_full

    def call(self, x, training: bool = None):
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                key = x[0]
                value = x[0]
                query = x[0]
            elif len(x) == 2:
                key = x[0]
                query = x[1]
                value = key
            else:
                key, query, value = x
        else:
            key = x
            value = x
            query = x

        shape = tf.shape(key)
        B, H, W = shape[0], shape[1], shape[2]
        heads = self.num_heads
        d = self.attention_filters
        Ck = heads * d

        # Project Q, K, V
        Q = self.qproj(query)  # (B, H, W, Ck)
        K = self.kproj(key)  # (B, H, W, Ck)
        V = self.vproj(value)  # (B, H, W, Ck)

        # Add query spatial positional encoding
        query_spatial_indices = self._get_query_spatial_indices(H, W)  # (H, W)
        query_spatial_emb = self.spatial_pos_embedding(query_spatial_indices)  # (H, W, Ck)
        Q = Q + query_spatial_emb  # (B, H, W, Ck)

        # Reshape Q for multi-head attention
        Q = tf.reshape(Q, (B, H, W, heads, d))  # (B, H, W, heads, d)

        # Extract spatial patches for keys and values
        K_patches = self._extract_patches_with_edge_handling(K)  # (B, H, W, patch_size, Ck)
        V_patches = self._extract_patches_with_edge_handling(V)  # (B, H, W, patch_size, Ck)

        # Add spatial positional embeddings to key patches
        spatial_indices = tf.range(self.patch_height * self.patch_width, dtype=tf.int32)
        key_spatial_emb = self.spatial_pos_embedding(spatial_indices)  # (patch_size, Ck)
        key_spatial_emb = tf.reshape(key_spatial_emb, (1, 1, 1, self.neighborhood_size, Ck))

        K_patches = K_patches + key_spatial_emb  # (B, H, W, patch_size, Ck)

        # Reshape for multi-head attention
        K_patches = tf.reshape(K_patches, (B, H, W, self.neighborhood_size, heads, d))
        V_patches = tf.reshape(V_patches, (B, H, W, self.neighborhood_size, heads, d))

        # Compute attention scores
        # Q: (B, H, W, heads, d)
        # K_patches: (B, H, W, neighborhood_size, heads, d)
        # Want: (B, H, W, heads, neighborhood_size)

        scale = tf.math.rsqrt(tf.cast(d, tf.float32))

        # Use einsum for efficient computation
        scores = tf.einsum('bhwnd,bhwknd->bhwnk', Q, K_patches) * scale  # (B, H, W, heads, neighborhood_size)

        # Apply softmax
        weights = tf.nn.softmax(scores, axis=-1)  # (B, H, W, heads, neighborhood_size)

        if self.dropout > 0 and training:
            weights = self.dropout_layer(weights, training=training)

        # Apply attention weights to values
        # weights: (B, H, W, heads, neighborhood_size)
        # V_patches: (B, H, W, neighborhood_size, heads, d)
        # Want: (B, H, W, heads, d)

        out = tf.einsum('bhwnk,bhwknd->bhwnd', weights, V_patches)  # (B, H, W, heads, d)

        # Reshape back
        out = tf.reshape(out, (B, H, W, Ck))

        # Output projection
        out = self.outproj(out)
        if self.skip_connection:
            out = tf.concat([out, query], axis=-1)
            out = self.skipproj(out)
        return out

class LocalSpatialAttentionPatchKeras(tf.keras.layers.Layer):
    """
    Regional spatial attention using extract_patches and MultiHeadAttention.
    Faster but with higher memory footprint than the loop version.

    Key differences from loop version:
    - Uses tf.keras.layers.MultiHeadAttention for attention computation
    - Extracts all spatial patches upfront using tf.image.extract_patches
    - Performs batched attention computation (no loops over spatial offsets)
    - Better GPU utilization and parallelism
    - Uses ~3x more memory than loop version
    - 2-4x faster for small/medium radius (r≤3)
    - Spatial positional encoding added BEFORE projection (vs AFTER in loop version)
    """

    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 dropout: float = 0.1, l2_reg: float = 0., name="RegionalSpatialAttentionHighMem"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        self.spatial_radius = spatial_radius
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.attention_layer = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "spatial_radius": self.spatial_radius,
            "attention_filters": self.attention_filters,
        })
        return config

    def build(self, input_shapes):
        if not isinstance(input_shapes, (tuple, list)) or len(input_shapes) > 3:
            input_shapes = [input_shapes]
        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass
        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape, s)), \
                f"all tensors must have same input shape: {input_shape} != {s}"

        self.spatial_dims = input_shape[1:-1]
        self.filters = input_shape[-1]

        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)

        self.radius_y, self.radius_x = self.spatial_radius
        self.patch_height = 2 * self.radius_y + 1
        self.patch_width = 2 * self.radius_x + 1
        self.neighborhood_size = self.patch_height * self.patch_width

        # MultiHeadAttention layer
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.attention_filters,
            output_shape=self.filters,
            dropout=self.dropout,
            name="MultiHeadAttention"
        )

        # Spatial positional embeddings
        # Note: In this version, embeddings are added BEFORE MHA
        # In loop version, they're added AFTER projection
        self.spatial_pos_embedding = tf.keras.layers.Embedding(
            self.patch_height * self.patch_width,
            self.filters,  # Note: self.filters, not Ck
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="SpatialPosEnc"
        )

        super().build(input_shape)

    def _get_query_spatial_indices(self, H, W):
        """
        For each query pixel, get the spatial index within the patch
        that corresponds to its position in the neighborhood.

        Returns: (H, W) array of spatial indices
        """
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)

        # Calculate which valid patch center this pixel would map to
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, H - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, W - self.radius_x - 1)

        # Calculate relative position within the patch
        y_relative = y_coords - y_patch_center + self.radius_y
        x_relative = x_coords - x_patch_center + self.radius_x

        # Create meshgrid
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')

        # Convert 2D position to flat index
        spatial_indices = y_grid * self.patch_width + x_grid

        return spatial_indices

    def _extract_patches_with_edge_handling(self, tensor):
        """
        Extract spatial patches with edge handling:
        - Interior pixels get their centered neighborhood
        - Edge pixels reuse patches from their nearest interior neighbor

        Args:
            tensor: (B, H, W, C) input tensor

        Returns:
            patches: (B, H, W, patch_h * patch_w, C)
        """
        B = tf.shape(tensor)[0]
        H = tf.shape(tensor)[1]
        W = tf.shape(tensor)[2]
        C = tf.shape(tensor)[3]

        # Extract patches with VALID padding (only for interior pixels)
        patches_valid = tf.image.extract_patches(
            images=tensor,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )  # (B, H_valid, W_valid, patch_h * patch_w * C)

        H_valid = H - 2 * self.radius_y
        W_valid = W - 2 * self.radius_x

        # Reshape for easier indexing
        patch_size = self.patch_height * self.patch_width
        patches_valid = tf.reshape(patches_valid, [B, H_valid, W_valid, patch_size, C])

        # Create index mapping for edge handling
        # Edge pixels use the patch from the nearest valid interior position

        # For y coordinate: clamp to [radius_y, H - radius_y - 1]
        y_indices = tf.range(H, dtype=tf.int32)
        y_indices = tf.clip_by_value(y_indices, self.radius_y, H - self.radius_y - 1)
        y_indices = y_indices - self.radius_y  # Convert to valid patch indices [0, H_valid)

        # For x coordinate: clamp to [radius_x, W - radius_x - 1]
        x_indices = tf.range(W, dtype=tf.int32)
        x_indices = tf.clip_by_value(x_indices, self.radius_x, W - self.radius_x - 1)
        x_indices = x_indices - self.radius_x  # Convert to valid patch indices [0, W_valid)

        # Create meshgrid of indices
        y_grid, x_grid = tf.meshgrid(y_indices, x_indices, indexing='ij')  # (H, W)

        # Create batch indices
        batch_indices = tf.range(B, dtype=tf.int32)
        batch_indices = tf.reshape(batch_indices, [B, 1, 1])
        batch_indices = tf.tile(batch_indices, [1, H, W])  # (B, H, W)

        # Stack indices for gather_nd
        y_grid_expanded = tf.tile(tf.expand_dims(y_grid, 0), [B, 1, 1])  # (B, H, W)
        x_grid_expanded = tf.tile(tf.expand_dims(x_grid, 0), [B, 1, 1])  # (B, H, W)

        indices = tf.stack([batch_indices, y_grid_expanded, x_grid_expanded], axis=-1)  # (B, H, W, 3)

        # Gather patches
        patches_full = tf.gather_nd(patches_valid, indices)  # (B, H, W, patch_size, C)

        return patches_full

    def call(self, x, training: bool = None):
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                key = x[0]
                value = x[0]
                query = x[0]
            elif len(x) == 2:
                key = x[0]
                query = x[1]
                value = key
            else:
                key, query, value = x
        else:
            key = x
            value = x
            query = x

        shape = tf.shape(key)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        # Add query spatial positional encoding BEFORE MHA
        query_spatial_indices = self._get_query_spatial_indices(H, W)  # (H, W)
        query_spatial_emb = self.spatial_pos_embedding(query_spatial_indices)  # (H, W, C)
        query_spatial_emb = tf.expand_dims(query_spatial_emb, 0)  # (1, H, W, C)

        query_with_pos = query + query_spatial_emb  # (B, H, W, C)

        # Extract spatial patches for keys and values
        # Add spatial positional encoding to keys BEFORE MHA
        spatial_indices = tf.range(self.patch_height * self.patch_width, dtype=tf.int32)
        key_spatial_emb = self.spatial_pos_embedding(spatial_indices)  # (patch_size, C)
        key_spatial_emb = tf.reshape(key_spatial_emb, (1, 1, 1, self.neighborhood_size, C))

        key_with_pos = key + tf.zeros_like(key)  # Placeholder for consistency
        K_patches = self._extract_patches_with_edge_handling(key_with_pos)  # (B, H, W, patch_size, C)
        K_patches = K_patches + key_spatial_emb  # Add spatial encoding to patches

        V_patches = self._extract_patches_with_edge_handling(value)  # (B, H, W, patch_size, C)

        # Flatten spatial dimensions for MHA
        # Query: (B, H, W, C) -> (B*H*W, 1, C)
        query_flat = tf.reshape(query_with_pos, [B * H * W, 1, C])

        # Key/Value patches: (B, H, W, neighborhood_size, C) -> (B*H*W, neighborhood_size, C)
        key_flat = tf.reshape(K_patches, [B * H * W, self.neighborhood_size, C])
        value_flat = tf.reshape(V_patches, [B * H * W, self.neighborhood_size, C])

        # Apply MultiHeadAttention
        attention_output = self.attention_layer(
            query=query_flat,
            key=key_flat,
            value=value_flat,
            training=training
        )  # (B*H*W, 1, C)

        # Reshape back to spatial dimensions
        output = tf.reshape(attention_output, [B, H, W, C])

        return output

from math import ceil
import tensorflow as tf
import numpy as np
from distnet_2d.model.layers import InferenceLayer


class SpatioTemporalAttention(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 intra_mode: bool = True, inference_idx: int = None, return_list: bool = False,
                 dropout: float = 0.1, l2_reg: float = 0., name="SpatioTemporalAttention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        self.spatial_radius = spatial_radius
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.intra_mode = intra_mode
        self.temporal_dim = None
        self.inference_idx = inference_idx
        if self.intra_mode:
            assert inference_idx is not None and (
                min(inference_idx) >= 0 if isinstance(inference_idx, (list, tuple)) else inference_idx >= 0)
        self.return_list = return_list

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "return_list": self.return_list,
            "intra_mode": self.intra_mode,
            "inference_idx": self.inference_idx,
            "spatial_radius": self.spatial_radius,
        })
        return config

    def build(self, input_shape):
        if self.intra_mode:
            input_shapes = input_shape
        else:
            query_shapes, input_shapes = input_shape
        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass
        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape, s)), \
                f"all tensors must have same input shape: {input_shape} != {s}"
        if not self.intra_mode:
            query_shape = query_shapes[0]
            if len(query_shapes) > 1:
                for s in query_shapes[1:]:
                    assert len(s) == len(query_shape) and all(i == j for i, j in zip(query_shape, s)), \
                        f"all query tensors must have same input shape: {query_shape} != {s}"
            try:
                query_shape = query_shape[0].as_list()
            except:
                pass
        self.spatial_dims = input_shape[1:-1]
        self.temporal_dim = len(input_shapes)
        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)
        self.radius_y, self.radius_x = self.spatial_radius
        self.patch_height = 2 * self.radius_y + 1
        self.patch_width = 2 * self.radius_x + 1
        self.neighborhood_size = self.patch_height * self.patch_width * self.temporal_dim
        Ck = self.num_heads * self.attention_filters
        self.qproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="qproj")
        self.kproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="kproj")
        self.vproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="vproj")
        self.outproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=False, name="outproj")
        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.temp_embedding = tf.keras.layers.Embedding(
            self.temporal_dim,
            self.filters,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="TempEnc"
        )

        # Per-head spatial embeddings
        self.spatial_pos_embedding = tf.keras.layers.Embedding(
            self.patch_height * self.patch_width,
            Ck,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="SpatialPosEnc"
        )

        # Pre-compute spatial offsets for graph mode compatibility
        spatial_offsets = []
        for dy in range(-self.radius_y, self.radius_y + 1):
            for dx in range(-self.radius_x, self.radius_x + 1):
                spatial_offsets.append([dy, dx])
        self.spatial_offsets = tf.constant(spatial_offsets, dtype=tf.int32)
        self.num_spatial = len(spatial_offsets)

        super().build(input_shape)

    def _get_query_spatial_indices(self, H, W):
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, H - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, W - self.radius_x - 1)
        y_relative = y_coords - y_patch_center + self.radius_y
        x_relative = x_coords - x_patch_center + self.radius_x
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')
        spatial_indices = y_grid * self.patch_width + x_grid
        return spatial_indices

    def _compute_center_indices(self, H, W):
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, H - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, W - self.radius_x - 1)
        y_grid, x_grid = tf.meshgrid(y_patch_center, x_patch_center, indexing='ij')
        return y_grid, x_grid

    def call(self, x, training: bool = None):
        if self.intra_mode:
            all_values = x
        else:
            query_list, all_values = x
        C = self.filters
        T = self.temporal_dim
        shape = tf.shape(all_values[0]) if self.intra_mode else tf.shape(query_list[0])
        B, H, W = shape[0], shape[1], shape[2]
        heads = self.num_heads
        d = self.attention_filters
        Ck = heads * d

        # Temporal embeddings
        t_index = tf.range(T, dtype=tf.int32)
        t_emb = self.temp_embedding(t_index)  # (T, C)

        # Stack all values and add temporal embeddings for keys
        stacked_values = tf.stack(all_values, axis=1)  # (B, T, H, W, C)
        t_emb_broadcast = tf.reshape(t_emb, (1, T, 1, 1, C))
        stacked_keys_with_temp = stacked_values + t_emb_broadcast  # (B, T, H, W, C)

        # Project keys and values in one batch
        BT = B * T
        stacked_keys_resh = tf.reshape(stacked_keys_with_temp, (BT, H, W, C))
        stacked_values_resh = tf.reshape(stacked_values, (BT, H, W, C))

        K_all = tf.reshape(self.kproj(stacked_keys_resh), (B, T, H, W, Ck))
        V_all = tf.reshape(self.vproj(stacked_values_resh), (B, T, H, W, Ck))

        # Get spatial embeddings
        spatial_indices = tf.range(self.patch_height * self.patch_width, dtype=tf.int32)
        key_spatial_emb = self.spatial_pos_embedding(spatial_indices)  # (patch_h*patch_w, Ck)

        query_spatial_indices = self._get_query_spatial_indices(H, W)  # (H, W)
        query_spatial_emb = self.spatial_pos_embedding(query_spatial_indices)  # (H, W, Ck)
        query_spatial_emb = tf.reshape(query_spatial_emb, (1, 1, H, W, Ck))  # (1, 1, H, W, Ck)

        # Prepare queries
        if self.intra_mode:
            if self.inference_mode:
                idx_list = [self.inference_idx] if not isinstance(self.inference_idx,
                                                                  (list, tuple)) else self.inference_idx
            else:
                idx_list = list(range(T))
            # Gather queries from stacked values + temporal embeddings
            idx_tensor = tf.constant(idx_list, dtype=tf.int32)
            Qstacked = tf.gather(stacked_values, idx_tensor, axis=1)  # (B, Qcount, H, W, C)
            q_t_emb = tf.gather(t_emb, idx_tensor)  # (Qcount, C)
            Qstacked = Qstacked + tf.reshape(q_t_emb, (1, len(idx_list), 1, 1, C))
        else:
            Qstacked = tf.stack(query_list, axis=1)  # (B, Qcount, H, W, C)

        Qcount = Qstacked.shape[1] if Qstacked.shape[1] is not None else tf.shape(Qstacked)[1]
        Qresh = tf.reshape(Qstacked, (B * Qcount, H, W, C))
        Qproj_all = self.qproj(Qresh)  # (B*Qcount, H, W, Ck)
        Qproj_all = tf.reshape(Qproj_all, (B, Qcount, H, W, Ck))
        Qproj_all = Qproj_all + query_spatial_emb  # (B, Qcount, H, W, Ck)
        Qproj_all = tf.reshape(Qproj_all, (B, Qcount, H, W, heads, d))

        cy_grid, cx_grid = self._compute_center_indices(H, W)

        # Pre-compute batch and temporal indices (reused in both passes)
        b_idx = tf.tile(tf.range(B, dtype=tf.int32)[:, None, None, None], [1, T, H, W])
        t_idx = tf.tile(tf.range(T, dtype=tf.int32)[None, :, None, None], [B, 1, H, W])

        # Scaling factor
        scale = tf.math.rsqrt(tf.cast(d, tf.float32))

        # PASS 1: Compute scores using TensorArray
        scores_array = tf.TensorArray(dtype=Qproj_all.dtype, size=self.num_spatial, dynamic_size=False)

        for s_idx in tf.range(self.num_spatial):
            dy = self.spatial_offsets[s_idx, 0]
            dx = self.spatial_offsets[s_idx, 1]

            # Spatial embedding for this neighbor
            spatial_idx = (dy + self.radius_y) * self.patch_width + (dx + self.radius_x)
            neighbor_spatial_emb = tf.reshape(key_spatial_emb[spatial_idx], (1, 1, 1, 1, Ck))

            # Neighbor coordinates with edge handling
            ny = tf.clip_by_value(cy_grid + dy, 0, H - 1)
            nx = tf.clip_by_value(cx_grid + dx, 0, W - 1)
            ny_b = tf.tile(ny[None, None, :, :], [B, T, 1, 1])
            nx_b = tf.tile(nx[None, None, :, :], [B, T, 1, 1])
            idx = tf.stack([b_idx, t_idx, ny_b, nx_b], axis=-1)

            # Gather keys for all temporal frames
            K_n = tf.gather_nd(K_all, idx) + neighbor_spatial_emb  # (B, T, H, W, Ck)
            K_nh = tf.reshape(K_n, (B, T, H, W, heads, d))

            # Compute scores
            scores = tf.einsum('bqhwnd,bthwnd->bqthwn', Qproj_all, K_nh) * scale
            scores_array = scores_array.write(s_idx, scores)

        # Stack all scores and flatten for softmax
        scores_stack = scores_array.stack()  # (S, B, Qcount, T, H, W, heads)
        scores_stack = tf.transpose(scores_stack, [1, 2, 4, 5, 6, 0, 3])  # (B, Qcount, H, W, heads, S, T)
        scores_flat = tf.reshape(scores_stack, (B, Qcount, H, W, heads, -1))  # (B, Qcount, H, W, heads, S*T)

        # Softmax over all neighbors
        weights = tf.nn.softmax(scores_flat, axis=-1)
        if self.dropout > 0 and training:
            weights = self.dropout_layer(weights, training=training)

        # Reshape weights back
        weights = tf.reshape(weights, (B, Qcount, H, W, heads, self.num_spatial, T))

        # PASS 2: Accumulate weighted values
        out_acc = tf.zeros((B, Qcount, H, W, heads, d), dtype=Qproj_all.dtype)

        for s_idx in tf.range(self.num_spatial):
            dy = self.spatial_offsets[s_idx, 0]
            dx = self.spatial_offsets[s_idx, 1]

            ny = tf.clip_by_value(cy_grid + dy, 0, H - 1)
            nx = tf.clip_by_value(cx_grid + dx, 0, W - 1)
            ny_b = tf.tile(ny[None, None, :, :], [B, T, 1, 1])
            nx_b = tf.tile(nx[None, None, :, :], [B, T, 1, 1])
            idx = tf.stack([b_idx, t_idx, ny_b, nx_b], axis=-1)

            V_n = tf.gather_nd(V_all, idx)  # (B, T, H, W, Ck)
            V_nh = tf.reshape(V_n, (B, T, H, W, heads, d))

            # weights for this spatial neighbor: (B, Qcount, H, W, heads, T)
            w_s = weights[..., s_idx, :]

            # Weighted sum over T: einsum for clarity
            # w_s: (B, Qcount, H, W, heads, T), V_nh: (B, T, H, W, heads, d)
            contrib = tf.einsum('bqhwnt,bthwnd->bqhwnd', w_s, V_nh)
            out_acc = out_acc + contrib

        # Output projection
        out_acc = tf.reshape(out_acc, (B * Qcount, H, W, Ck))
        out = self.outproj(out_acc)
        out = tf.reshape(out, (B, Qcount, H, W, C))

        # Unstack outputs
        attention_output_list = tf.unstack(out, axis=1)

        if self.return_list:
            return attention_output_list
        if self.inference_mode and not isinstance(self.inference_idx, (list, tuple)):
            return attention_output_list[0]
        else:
            return tf.concat(attention_output_list, 0)


# faster but with high-memory footprint
class SpatioTemporalAttentionHighMem(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 intra_mode: bool = True, inference_idx: int = None, return_list: bool = False,
                 dropout: float = 0.1, l2_reg: float = 0., name="SpatioTemporalAttention"):
        '''
        Spatio-temporal attention with limited spatial neighborhood using extract_patches.
        Supports variable image dimensions (FCN compatible).

        Args:
            num_heads: Number of attention heads
            attention_filters: Number of filters for attention (auto if 0)
            spatial_radius: (radius_y, radius_x) defining neighborhood size
            intra_mode: if True, input is (idx, array), otherwise (query_tensor, array)
            inference_idx: index(es) for inference mode
            return_list: whether to return list or concatenated tensor
            dropout: dropout rate
            l2_reg: L2 regularization for embeddings
        '''
        super().__init__(name=name)
        self.attention_layer = None
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        self.spatial_radius = spatial_radius
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.intra_mode = intra_mode
        self.temporal_dim = None
        self.inference_idx = inference_idx
        if self.intra_mode:
            assert inference_idx is not None and (
                min(inference_idx) >= 0 if isinstance(inference_idx, (list, tuple)) else inference_idx >= 0)
        self.return_list = return_list

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "return_list": self.return_list,
            "intra_mode": self.intra_mode,
            "inference_idx": self.inference_idx,
            "spatial_radius": self.spatial_radius
        })
        return config

    def build(self, input_shape):
        if self.intra_mode:
            input_shapes = input_shape
        else:
            query_shapes, input_shapes = input_shape

        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass

        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape, s)), \
                f"all tensors must have same input shape: {input_shape} != {s}"

        if not self.intra_mode:
            query_shape = query_shapes[0]
            if len(query_shapes) > 1:
                for s in query_shapes[1:]:
                    assert len(s) == len(query_shape) and all(i == j for i, j in zip(query_shape, s)), \
                        f"all query tensors must have same input shape: {query_shape} != {s}"
            try:
                query_shape = query_shape[0].as_list()
            except:
                pass
            assert len(query_shape) == len(input_shape) and all(
                i == j for i, j in zip(input_shape[:-1], query_shape[:-1])), \
                f"query tensor must have same spatial shape: {input_shape} != {query_shapes}"

        self.spatial_dims = input_shape[1:-1]
        self.temporal_dim = len(input_shapes)
        self.filters = input_shape[-1]

        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(ceil(self.filters / self.num_heads))

        # Calculate neighborhood size
        self.radius_y, self.radius_x = self.spatial_radius
        self.patch_height = 2 * self.radius_y + 1
        self.patch_width = 2 * self.radius_x + 1
        self.neighborhood_size = self.patch_height * self.patch_width * self.temporal_dim

        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            self.num_heads,
            key_dim=self.attention_filters,
            output_shape=self.filters,
            dropout=self.dropout,
            name="MultiHeadAttention"
        )

        # Temporal embeddings
        self.temp_embedding = tf.keras.layers.Embedding(
            self.temporal_dim,
            self.filters,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="TempEnc"
        )

        # Spatial positional embeddings (relative positions within patch)
        # One embedding for each position in the spatial patch
        self.spatial_pos_embedding = tf.keras.layers.Embedding(
            self.patch_height * self.patch_width,
            self.filters,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="SpatialPosEnc"
        )

        super().build(input_shape)

    def _get_query_spatial_indices(self, H, W):
        """
        For each query pixel position, get the spatial index within the patch
        that corresponds to its position in the neighborhood.

        For interior pixels: center position (radius_y, radius_x)
        For edge pixels: adjusted position based on how close to the edge

        Returns: (H, W) array of spatial indices in range [0, patch_h*patch_w)
        """
        # For each pixel (y, x), calculate its offset from the patch center
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)

        # Calculate which valid patch center this pixel would map to
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, H - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, W - self.radius_x - 1)

        # Calculate relative position within the patch
        # For interior: y - patch_center = 0, so relative_y = radius_y (center)
        # For edge y=0, patch_center=radius_y: relative_y = 0 - radius_y + radius_y = 0
        y_relative = y_coords - y_patch_center + self.radius_y
        x_relative = x_coords - x_patch_center + self.radius_x

        # Create meshgrid
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')  # (H, W)

        # Convert 2D position to flat index
        spatial_indices = y_grid * self.patch_width + x_grid  # (H, W)

        return spatial_indices

    def _extract_patches_with_edge_handling(self, tensor):
        """
        Extract spatial patches with edge handling:
        - Interior pixels get their centered neighborhood
        - Edge pixels reuse patches from their nearest interior neighbor

        For example with radius (1,1):
        - Pixel (0,0) uses the same patch as pixel (radius_y, radius_x) = (1,1)
        - Pixel (0,1) uses the same patch as pixel (1,1)
        - etc.

        Returns patches of shape (B, H, W, patch_h * patch_w * C)
        """
        B = tf.shape(tensor)[0]
        H = tf.shape(tensor)[1]
        W = tf.shape(tensor)[2]
        C = tf.shape(tensor)[3]

        # Extract patches with VALID padding (only for interior pixels where full neighborhood exists)
        patches_valid = tf.image.extract_patches(
            images=tensor,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )  # (B, H_valid, W_valid, patch_h * patch_w * C)
        # where H_valid = H - 2*radius_y, W_valid = W - 2*radius_x

        H_valid = H - 2 * self.radius_y
        W_valid = W - 2 * self.radius_x

        # Reshape for easier indexing: (B, H_valid, W_valid, patch_size, C)
        patch_size = self.patch_height * self.patch_width
        patches_valid = tf.reshape(patches_valid, [B, H_valid, W_valid, patch_size, C])

        # Create index mapping: for each pixel (y, x) in [0, H) x [0, W),
        # determine which patch to use
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

        # Gather patches using the index mapping
        # We need to gather from patches_valid using y_grid and x_grid
        # patches_valid is (B, H_valid, W_valid, patch_size, C)

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

        # Reshape to expected output format
        patches_full = tf.reshape(patches_full, [B, H, W, patch_size * C])

        return patches_full

    def call(self, x, training: bool = None):
        '''
        x: [idx, all_values] or [query_list, all_values]
           each tensor with shape (batch_size, H, W, C)
           H and W can be None (variable dimensions)
        '''
        if self.intra_mode:
            all_values = x
        else:
            query_list, all_values = x

        C = self.filters
        T = self.temporal_dim
        shape = tf.shape(all_values[0]) if self.intra_mode else tf.shape(query_list[0])
        B, H, W = shape[0], shape[1], shape[2]

        # Temporal embeddings
        t_index = tf.range(T, dtype=tf.int32)
        t_emb = self.temp_embedding(t_index)  # (T, C)

        # Add temporal embeddings to keys (but not values)
        all_keys_with_temp = [all_values[i] + tf.reshape(t_emb[i], (1, 1, 1, C)) for i in range(T)]

        # Extract spatial patches for all temporal frames
        all_patches = []
        for t in range(T):
            patches_t = self._extract_patches_with_edge_handling(all_keys_with_temp[t])
            # Reshape to (B, H, W, patch_h, patch_w, C)
            patches_t = tf.reshape(patches_t, [B, H, W, self.patch_height, self.patch_width, C])
            # Flatten spatial patch dimensions (B, H, W, patch_h*patch_w, C)
            patches_t = tf.reshape(patches_t, [B, H, W, self.patch_height * self.patch_width, C])
            all_patches.append(patches_t)

        # Stack along temporal dimension: (B, H, W, T, patch_h*patch_w, C)
        all_patches_stacked = tf.stack(all_patches, axis=3)

        # Reshape to (B, H, W, T*patch_h*patch_w, C) for keys
        key_tensor = tf.reshape(all_patches_stacked, [B, H, W, -1, C])

        # Add spatial positional embeddings to keys
        # Spatial embeddings are the same for all temporal frames
        spatial_indices = tf.range(self.patch_height * self.patch_width, dtype=tf.int32)
        spatial_emb = self.spatial_pos_embedding(spatial_indices)  # (patch_h*patch_w, C)
        spatial_emb = tf.reshape(spatial_emb, (1, 1, 1, self.patch_height * self.patch_width, self.filters))
        # Tile for all temporal frames
        spatial_emb_tiled = tf.tile(spatial_emb, [1, 1, 1, T, 1])  # (1, 1, 1, T*patch_h*patch_w, self.filters)

        key_tensor = key_tensor + spatial_emb_tiled

        # Extract value patches without temporal embeddings
        all_value_patches = []
        for t in range(T):
            patches_t = self._extract_patches_with_edge_handling(all_values[t])
            patches_t = tf.reshape(patches_t, [B, H, W, self.patch_height, self.patch_width, C])
            patches_t = tf.reshape(patches_t, [B, H, W, self.patch_height * self.patch_width, C])
            all_value_patches.append(patches_t)

        all_value_patches_stacked = tf.stack(all_value_patches, axis=3)
        value_tensor = tf.reshape(all_value_patches_stacked, [B, H, W, -1, C])

        # Prepare queries
        if self.intra_mode:
            if self.inference_mode:
                idx_list = [self.inference_idx] if not isinstance(self.inference_idx,
                                                                  (list, tuple)) else self.inference_idx
            else:
                idx_list = range(T)
            query_list = [all_values[i] + tf.reshape(t_emb[i], (1, 1, 1, C)) for i in idx_list]

        # Get spatial positional indices for queries
        query_spatial_indices = self._get_query_spatial_indices(H, W)  # (H, W)
        query_spatial_emb = self.spatial_pos_embedding(query_spatial_indices)  # (H, W, C)
        query_spatial_emb = tf.expand_dims(query_spatial_emb, 0)  # (1, H, W, C)

        # Add spatial position to queries
        query_list = [q + query_spatial_emb for q in query_list]

        attention_output_list = []

        for query in query_list:
            # Flatten spatial dimensions for attention
            # Query: (B, H, W, C) -> (B*H*W, 1, C)
            query_flat = tf.reshape(query, [B * H * W, 1, C])

            # Key/Value: (B, H, W, N_neighbors, C) -> (B*H*W, N_neighbors, C)
            key_flat = tf.reshape(key_tensor, [B * H * W, self.neighborhood_size, C])
            value_flat = tf.reshape(value_tensor, [B * H * W, self.neighborhood_size, C])

            # Apply attention
            attention_output = self.attention_layer(
                query=query_flat,
                value=value_flat,
                key=key_flat,
                training=training
            )  # (B*H*W, 1, C)

            # Reshape back to spatial dimensions
            attention_output = tf.reshape(attention_output, [B, H, W, C])
            attention_output_list.append(attention_output)

        if self.return_list:
            return attention_output_list
        if self.inference_mode and not isinstance(self.inference_idx, (list, tuple)):
            return attention_output_list[0]
        else:
            return tf.concat(attention_output_list, 0)
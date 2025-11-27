from math import ceil
import tensorflow as tf
import numpy as np
from distnet_2d.model.layers import InferenceLayer, RelativeTemporalEmbedding

from math import ceil
import tensorflow as tf
import numpy as np
from distnet_2d.model.layers import InferenceLayer, RelativeTemporalEmbedding


class LocalSpatioTemporalAttention(InferenceLayer, tf.keras.layers.Layer):
    """
    Optimized spatio-temporal attention - fast and memory efficient.
    Key optimization: for speed : chunked loop over spatial rows/cols, rest of operation is vectorized
    Key optimization for memory: online softmax (value accumulation, no score stored)

    """

    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 skip_connection: bool = False, frame_aware: bool = False, frame_max_distance: int = 0,
                 training_query_idx: list = None, inference_query_idx: int = None,
                 dropout: float = 0, l2_reg: float = 0.,
                 spatial_chunk_mode: str = 'col',  # 'row' or 'col'
                 name="SpatioTemporalAttentionOptimized"):

        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        self.spatial_radius = spatial_radius
        self.skip_connection = skip_connection
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.temporal_dim = None
        self.training_query_idx = training_query_idx
        self.inference_query_idx = inference_query_idx
        self.frame_aware = frame_aware
        if self.frame_aware:
            assert frame_max_distance > 0, "in frame_aware mode frame max distance must be provided"
        self.frame_max_distance = frame_max_distance
        assert spatial_chunk_mode in ["row", "col"], f"invalid spatial_chunk_mode: {spatial_chunk_mode} must be in [row, col]"
        self.spatial_chunk_mode = spatial_chunk_mode

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "filters": self.filters,
            "l2_reg": self.l2_reg,
            "inference_query_idx": self.inference_query_idx,
            "training_query_idx": self.training_query_idx,
            "spatial_radius": self.spatial_radius,
            "skip_connection": self.skip_connection,
            "frame_aware": self.frame_aware,
            "frame_max_distance": self.frame_max_distance,
            "spatial_chunk_mode": self.spatial_chunk_mode
        })
        return config

    def build(self, input_shape):
        if self.frame_aware:
            input_shapes, t_index_shape = input_shape
        else:
            input_shapes = input_shape

        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass
        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s) == len(input_shape) and all(i == j for i, j in zip(input_shape, s)), \
                f"all tensors must have same input shape: {input_shape} != {s}"

        self.temporal_dim = len(input_shapes)
        if self.training_query_idx is None:
            self.training_query_idx = list(range(self.temporal_dim))
        elif not isinstance(self.training_query_idx, (list, tuple)):
            self.training_query_idx = [self.training_query_idx]
        if self.inference_query_idx is None:
            self.inference_query_idx = self.training_query_idx
        elif not isinstance(self.inference_query_idx, (list, tuple)):
            self.inference_query_idx = [self.inference_query_idx]

        self.spatial_dims = input_shape[1:-1]
        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters <= 0:
            self.attention_filters = int(self.filters / self.num_heads)
        self.radius_y, self.radius_x = self.spatial_radius
        self.patch_height = 2 * self.radius_y + 1
        self.patch_width = 2 * self.radius_x + 1
        self.patch_size = self.patch_height * self.patch_width
        self.neighborhood_size = self.patch_size * self.temporal_dim

        HF = self.num_heads * self.attention_filters

        # Projection layers
        self.qproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="qproj")
        self.kproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="kproj")
        self.vproj = tf.keras.layers.Conv2D(HF, 1, padding='same', use_bias=False, name="vproj")
        self.outproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=False, name="outproj")

        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        # Temporal embeddings
        self.temp_embedding = tf.keras.layers.Embedding(
            self.temporal_dim,
            HF,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="TempEnc"
        ) if not self.frame_aware else RelativeTemporalEmbedding(
            max(self.temporal_dim, self.frame_max_distance),
            HF,
            l2_reg=self.l2_reg,
            name="TempEnc"
        )

        # Spatial embeddings
        self.spatial_pos_embedding = tf.keras.layers.Embedding(
            self.patch_size,
            HF,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="SpatialPosEnc"
        )

        if self.skip_connection:
            self.skipproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=True, name="skip")

        # Pre-compute spatial offsets for gather_nd
        spatial_offsets = []
        for dy in range(-self.radius_y, self.radius_y + 1):
            for dx in range(-self.radius_x, self.radius_x + 1):
                spatial_offsets.append([dy, dx])
        self.spatial_offsets = tf.constant(spatial_offsets, dtype=tf.int32)

        # Pre-compute row/column structured offsets
        if self.spatial_chunk_mode == 'row':
            # Group by rows: [[row0_offsets], [row1_offsets], ...]
            self.spatial_offsets_structured = []
            for dy in range(-self.radius_y, self.radius_y + 1):
                row_offsets = []
                for dx in range(-self.radius_x, self.radius_x + 1):
                    row_offsets.append([dy, dx])
                self.spatial_offsets_structured.append(tf.constant(row_offsets, dtype=tf.int32))
        elif self.spatial_chunk_mode == 'col':
            # Group by columns: [[col0_offsets], [col1_offsets], ...]
            self.spatial_offsets_structured = []
            for dx in range(-self.radius_x, self.radius_x + 1):
                col_offsets = []
                for dy in range(-self.radius_y, self.radius_y + 1):
                    col_offsets.append([dy, dx])
                self.spatial_offsets_structured.append(tf.constant(col_offsets, dtype=tf.int32))
        else:
            self.spatial_offsets_structured = None

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

    def call(self, input, training: bool = None):
        if self.frame_aware:
            frame_list, t_index = input
        else:
            frame_list = input
            t_index = tf.range(self.temporal_dim, dtype=tf.int32)

        C = self.filters
        T = self.temporal_dim
        shape = tf.shape(frame_list[0])
        B, Y, X = shape[0], shape[1], shape[2]
        H = self.num_heads
        F = self.attention_filters
        HF = H * F
        S = self.patch_size
        dtype = frame_list[0].dtype

        # Get query indices
        idx_list = self.inference_query_idx if self.inference_mode else self.training_query_idx
        nQ = len(idx_list)

        frames = tf.stack(frame_list, axis=0)
        # Project all frames at once
        frames_batch = tf.reshape(frames, (T * B, Y, X, C))
        Q_all_input = tf.concat([frame_list[i] for i in idx_list], axis=0)  # nQ * B, Y, X, F

        # Fuse projections to reduce intermediate tensors
        K_all = self.kproj(frames_batch)
        V_all = self.vproj(frames_batch)
        Q_all = self.qproj(Q_all_input)

        # Reshape once after projection
        K_all = tf.reshape(K_all, (T, B, Y, X, H, F))
        V_all = tf.reshape(V_all, (T, B, Y, X, H, F))
        Q_all = tf.reshape(Q_all, (nQ, B, Y, X, H, F))

        t_emb = self.temp_embedding(t_index)

        if self.frame_aware:
            k_t_emb = tf.transpose(t_emb, [1, 0, 2])
            k_t_emb = tf.reshape(k_t_emb, (T, B, 1, 1, H, F))
        else:
            k_t_emb = tf.reshape(t_emb, (T, 1, 1, 1, H, F))

        if self.frame_aware:
            q_t_emb = tf.gather(t_emb, tf.constant(idx_list, dtype=tf.int32), axis=1)
            q_t_emb = tf.transpose(q_t_emb, [1, 0, 2])
            q_t_emb = tf.reshape(q_t_emb, (nQ, B, 1, 1, H, F))
        else:
            q_t_emb = tf.gather(t_emb, tf.constant(idx_list, dtype=tf.int32))
            q_t_emb = tf.reshape(q_t_emb, (nQ, 1, 1, 1, H, F))

        # Spatial embeddings
        spatial_indices = tf.range(S, dtype=tf.int32)
        k_s_emb = self.spatial_pos_embedding(spatial_indices)
        k_s_emb = tf.reshape(k_s_emb, (S, 1, 1, 1, 1, H, F))
        k_s_emb = tf.tile(k_s_emb, [1, T, 1, 1, 1, 1, 1])
        query_spatial_indices = self._get_query_spatial_indices(Y, X)
        q_s_emb = self.spatial_pos_embedding(query_spatial_indices)
        q_s_emb = tf.reshape(q_s_emb, (1, 1, Y, X, H, F))

        # Add embeddings to projected K & Q
        K_all = K_all + k_t_emb  # spatial embedding is added later for the neighborhood
        Q_all = Q_all + q_s_emb + q_t_emb

        # Scaling factor for softmax
        scale = tf.math.rsqrt(tf.cast(F, dtype))

        cy_grid, cx_grid = self._compute_center_indices(Y, X)

        # Pre-compute base index grids (shared across all spatial iterations)
        t_idx_base = tf.range(T, dtype=tf.int32)
        b_idx_base = tf.range(B, dtype=tf.int32)

        # Create base grids once
        t_idx_grid = tf.tile(t_idx_base[None, :, None, None, None], [1, 1, B, Y, X])
        b_idx_grid = tf.tile(b_idx_base[None, None, :, None, None], [1, T, 1, Y, X])

        min_val = tf.constant(-65504.0 if dtype == tf.float16 else -1e9, dtype=dtype)
        max_scores = tf.fill([nQ, B, Y, X, H, 1], min_val)
        sum_exp = tf.zeros([nQ, B, Y, X, H, 1], dtype=dtype)
        out_acc = tf.zeros([nQ, B, Y, X, H, F], dtype=dtype)

        num_chunks = len(self.spatial_offsets_structured)
        chunk_size = tf.shape(self.spatial_offsets_structured[0])[0]

        for chunk_idx in range(num_chunks):  # CHUNKED LOOP by row or column
            # Get all offsets for this row/column
            chunk_offsets = self.spatial_offsets_structured[chunk_idx]  # Shape: (chunk_size, 2)

            # Extract dy and dx as vectors
            dy_vec = chunk_offsets[:, 0]  # Shape: (chunk_size,)
            dx_vec = chunk_offsets[:, 1]  # Shape: (chunk_size,)

            # Compute spatial indices for embeddings
            spatial_idx_vec = (dy_vec + self.radius_y) * self.patch_width + (dx_vec + self.radius_x)

            # Vectorized neighbor coordinate computation
            # cy_grid, cx_grid shape: (Y, X)
            # Expand to (chunk_size, Y, X) and add offsets
            cy_expanded = cy_grid[None, :, :]  # (1, Y, X)
            cx_expanded = cx_grid[None, :, :]  # (1, Y, X)
            dy_expanded = dy_vec[:, None, None]  # (chunk_size, 1, 1)
            dx_expanded = dx_vec[:, None, None]  # (chunk_size, 1, 1)

            ny_chunk = tf.clip_by_value(cy_expanded + dy_expanded, 0, Y - 1)  # (chunk_size, Y, X)
            nx_chunk = tf.clip_by_value(cx_expanded + dx_expanded, 0, X - 1)  # (chunk_size, Y, X)

            # Expand to (chunk_size, T, B, Y, X)
            ny_grid_chunk = ny_chunk[:, None, None, :, :]  # (chunk_size, 1, 1, Y, X)
            nx_grid_chunk = nx_chunk[:, None, None, :, :]  # (chunk_size, 1, 1, Y, X)

            # Broadcast to full shape (chunk_size, T, B, Y, X)
            ny_grid_chunk = tf.broadcast_to(ny_grid_chunk, [chunk_size, T, B, Y, X])
            nx_grid_chunk = tf.broadcast_to(nx_grid_chunk, [chunk_size, T, B, Y, X])
            t_idx_chunk = tf.broadcast_to(t_idx_grid, [chunk_size, T, B, Y, X])
            b_idx_chunk = tf.broadcast_to(b_idx_grid, [chunk_size, T, B, Y, X])

            # Stack indices: (chunk_size, T, B, Y, X, 4)
            idx_chunk = tf.stack([t_idx_chunk, b_idx_chunk, ny_grid_chunk, nx_grid_chunk], axis=-1)

            # Reshape for single gather: (chunk_size * T * B * Y * X, 4)
            idx_flat = tf.reshape(idx_chunk, [-1, 4])

            # Single gather for K and V
            K_flat = tf.gather_nd(K_all, idx_flat)  # (chunk_size * T * B * Y * X, H, F)
            V_flat = tf.gather_nd(V_all, idx_flat)  # (chunk_size * T * B * Y * X, H, F)

            spatial_emb_chunk = tf.gather(k_s_emb, spatial_idx_vec, axis=0)  # (chunk_size, T, 1, 1, 1, H, F)

            # Reshape back: (chunk_size, T, B, Y, X, H, F)
            K_chunk = tf.reshape(K_flat, [chunk_size * T, B, Y, X, H, F])
            V_chunk = tf.reshape(V_flat, [chunk_size * T, B, Y, X, H, F])

            K_chunk = K_chunk + tf.reshape(spatial_emb_chunk, [chunk_size * T, 1, 1, 1, H, F])

            # Compute scores
            scores_chunk = tf.einsum('qbyxhf,sbyxhf->qbyxhs', Q_all, K_chunk) * scale

            # Update running max
            new_max = tf.maximum(max_scores, tf.reduce_max(scores_chunk, axis=-1, keepdims=True))

            # Rescale previous accumulations
            exp_diff = tf.exp(max_scores - new_max)
            sum_exp = sum_exp * exp_diff
            out_acc = out_acc * exp_diff

            # Add new contributions
            exp_scores_flat = tf.exp(scores_chunk - new_max)
            sum_exp = sum_exp + tf.reduce_sum(exp_scores_flat, axis=-1, keepdims=True)

            # Accumulate weighted values
            contrib = tf.einsum('qbyxhs,sbyxhf->qbyxhf', exp_scores_flat, V_chunk)
            out_acc = out_acc + contrib

            max_scores = new_max

        # Final normalization
        out_acc = out_acc / (sum_exp + 1e-8)

        if self.dropout > 0 and training:
            out_acc = self.dropout_layer(out_acc, training=training)

        # Output projection
        out_acc = tf.reshape(out_acc, (nQ * B, Y, X, HF))
        out = self.outproj(out_acc)
        if self.skip_connection:
            out = tf.concat([out, Q_all_input], axis=-1)
            out = self.skipproj(out)

        return tf.reshape(out, (nQ, B, Y, X, C))


class LocalSpatioTemporalAttentionPatch(InferenceLayer, tf.keras.layers.Layer):
    """
    Spatio-temporal attention using extract_patches for fast parallelization.
    Uses einsum operations instead of MultiHeadAttention for better performance.

    Key features:
    - Extracts all spatial patches upfront using tf.image.extract_patches
    - Uses einsum for efficient batched attention computation
    - Positional encoding added AFTER projection (in attention space)
    - 2-4x faster than loop version
    - ~3x more memory than loop version
    - Fully compatible with variable image dimensions (FCN)

    Args:
        num_heads: Number of attention heads
        attention_filters: Number of filters per head (auto if 0)
        spatial_radius: (radius_y, radius_x) defining neighborhood size
        intra_mode: if True, queries from same sequence; else separate query list
        inference_idx: index(es) for inference mode
        return_list: whether to return list or concatenated tensor
        dropout: dropout rate
        l2_reg: L2 regularization for embeddings
    """

    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 skip_connection: bool = False, frame_aware: bool = False, frame_max_distance: int = 0,
                 intra_mode: bool = True, inference_idx: int = None, return_list: bool = False,
                 dropout: float = 0.1, l2_reg: float = 0., name="SpatioTemporalAttentionHighMem"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.attention_filters = attention_filters
        self.spatial_radius = spatial_radius
        self.filters = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.intra_mode = intra_mode
        self.temporal_dim = None
        self.skip_connection = skip_connection
        self.inference_idx = inference_idx
        self.frame_aware = frame_aware
        self.frame_max_distance = frame_max_distance
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
            "attention_filters": self.attention_filters,
            "skip_connection": self.skip_connection,
            "frame_aware": self.frame_aware,
            "frame_max_distance": self.frame_max_distance
        })
        return config

    def build(self, input_shape):
        if self.intra_mode:
            if self.frame_aware:
                input_shapes, t_index_shape = input_shape
            else:
                input_shapes = input_shape
        else:
            if self.frame_aware:
                query_shapes, input_shapes, timepoints = input_shape
            else:
                query_shape, input_shapes = input_shape
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

        # Projection layers
        self.qproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="qproj")
        self.kproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="kproj")
        self.vproj = tf.keras.layers.Conv2D(Ck, 1, padding='same', use_bias=False, name="vproj")
        self.outproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=False, name="outproj")

        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        # Temporal embeddings
        self.temp_embedding = tf.keras.layers.Embedding(
            self.temporal_dim,
            Ck,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="TempEnc"
        )

        # Spatial positional embeddings (in Ck space, added after projection)
        self.spatial_pos_embedding = tf.keras.layers.Embedding(
            self.patch_height * self.patch_width,
            Ck,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="SpatialPosEnc"
        )
        if self.skip_connection:
            self.skipproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=True, name="skip")
        super().build(input_shape)

    def _get_query_spatial_indices(self, H, W):
        """Get spatial positional indices for queries."""
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, self.radius_y, H - self.radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, self.radius_x, W - self.radius_x - 1)
        y_relative = y_coords - y_patch_center + self.radius_y
        x_relative = x_coords - x_patch_center + self.radius_x
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')
        spatial_indices = y_grid * self.patch_width + x_grid
        return spatial_indices

    def _extract_patches_with_edge_handling(self, tensor):
        """
        Extract spatial patches with edge handling.

        Args:
            tensor: (B, H, W, C) or (B, T, H, W, C)

        Returns:
            patches: (B, H, W, patch_size, C) or (B, T, H, W, patch_size, C)
        """
        input_shape = tf.shape(tensor)
        ndims = len(tensor.shape)

        if ndims == 5:  # (B, T, H, W, C)
            B, T = input_shape[0], input_shape[1]
            H, W, C = input_shape[2], input_shape[3], input_shape[4]
            # Merge B and T for patch extraction
            tensor_merged = tf.reshape(tensor, [B * T, H, W, C])
        else:  # (B, H, W, C)
            B = input_shape[0]
            H, W, C = input_shape[1], input_shape[2], input_shape[3]
            tensor_merged = tensor

        # Extract patches with VALID padding
        patches_valid = tf.image.extract_patches(
            images=tensor_merged,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        H_valid = H - 2 * self.radius_y
        W_valid = W - 2 * self.radius_x
        patch_size = self.patch_height * self.patch_width

        if ndims == 5:
            patches_valid = tf.reshape(patches_valid, [B, T, H_valid, W_valid, patch_size, C])
        else:
            patches_valid = tf.reshape(patches_valid, [B, H_valid, W_valid, patch_size, C])

        # Create index mapping for edge handling
        y_indices = tf.range(H, dtype=tf.int32)
        y_indices = tf.clip_by_value(y_indices, self.radius_y, H - self.radius_y - 1)
        y_indices = y_indices - self.radius_y

        x_indices = tf.range(W, dtype=tf.int32)
        x_indices = tf.clip_by_value(x_indices, self.radius_x, W - self.radius_x - 1)
        x_indices = x_indices - self.radius_x

        y_grid, x_grid = tf.meshgrid(y_indices, x_indices, indexing='ij')

        if ndims == 5:
            # Create indices for (B, T, H, W)
            batch_indices = tf.range(B, dtype=tf.int32)
            batch_indices = tf.reshape(batch_indices, [B, 1, 1, 1])
            batch_indices = tf.tile(batch_indices, [1, T, H, W])

            time_indices = tf.range(T, dtype=tf.int32)
            time_indices = tf.reshape(time_indices, [1, T, 1, 1])
            time_indices = tf.tile(time_indices, [B, 1, H, W])

            y_grid_expanded = tf.tile(tf.reshape(y_grid, [1, 1, H, W]), [B, T, 1, 1])
            x_grid_expanded = tf.tile(tf.reshape(x_grid, [1, 1, H, W]), [B, T, 1, 1])

            indices = tf.stack([batch_indices, time_indices, y_grid_expanded, x_grid_expanded], axis=-1)
        else:
            # Create indices for (B, H, W)
            batch_indices = tf.range(B, dtype=tf.int32)
            batch_indices = tf.reshape(batch_indices, [B, 1, 1])
            batch_indices = tf.tile(batch_indices, [1, H, W])

            y_grid_expanded = tf.tile(tf.expand_dims(y_grid, 0), [B, 1, 1])
            x_grid_expanded = tf.tile(tf.expand_dims(x_grid, 0), [B, 1, 1])

            indices = tf.stack([batch_indices, y_grid_expanded, x_grid_expanded], axis=-1)

        patches_full = tf.gather_nd(patches_valid, indices)
        return patches_full

    def call(self, x, training: bool = None):
        if self.intra_mode:
            if self.frame_aware:
                all_values, t_index = x
            else:
                all_values = x
        else:
            if self.frame_aware:
                query_list, all_values, t_index = x
            else:
                query_list, all_values = x

        C = self.filters
        T = self.temporal_dim
        shape = tf.shape(all_values[0]) if self.intra_mode else tf.shape(query_list[0])
        B, H, W = shape[0], shape[1], shape[2]
        heads = self.num_heads
        d = self.attention_filters
        Ck = heads * d

        stacked_values = tf.stack(all_values, axis=1)  # (B, T, H, W, C)

        # Project keys and values
        BT = B * T
        stacked_values_resh = tf.reshape(stacked_values, (BT, H, W, C))

        K_all = tf.reshape(self.kproj(stacked_values_resh), (B, T, H, W, Ck))
        V_all = tf.reshape(self.vproj(stacked_values_resh), (B, T, H, W, Ck))

        # Temporal embeddings
        if not self.frame_aware:
            t_index = tf.range(T, dtype=tf.int32)
        t_emb = self.temp_embedding(t_index)  # (T, Ck)

        if self.frame_aware:
            K_all = K_all + tf.reshape(t_emb, (B, T, 1, 1, Ck))
        else:
            K_all = K_all + tf.reshape(t_emb, (1, T, 1, 1, Ck))

        # Extract spatial patches for K and V
        # K_all, V_all: (B, T, H, W, Ck)
        K_patches = self._extract_patches_with_edge_handling(K_all)  # (B, T, H, W, patch_size, Ck)
        V_patches = self._extract_patches_with_edge_handling(V_all)  # (B, T, H, W, patch_size, Ck)

        # Add spatial positional embeddings to key patches
        spatial_indices = tf.range(self.patch_height * self.patch_width, dtype=tf.int32)
        key_spatial_emb = self.spatial_pos_embedding(spatial_indices)  # (patch_size, Ck)
        key_spatial_emb = tf.reshape(key_spatial_emb, (1, 1, 1, 1, self.patch_height * self.patch_width, Ck))

        K_patches = K_patches + key_spatial_emb  # (B, T, H, W, patch_size, Ck)

        # Reshape patches: (B, T, H, W, patch_size, Ck) -> (B, H, W, T*patch_size, Ck)
        K_patches = tf.transpose(K_patches, [0, 2, 3, 1, 4, 5])  # (B, H, W, T, patch_size, Ck)
        V_patches = tf.transpose(V_patches, [0, 2, 3, 1, 4, 5])

        K_patches = tf.reshape(K_patches, [B, H, W, T * self.patch_height * self.patch_width, Ck])
        V_patches = tf.reshape(V_patches, [B, H, W, T * self.patch_height * self.patch_width, Ck])

        # Reshape for multi-head attention
        K_patches = tf.reshape(K_patches, [B, H, W, self.neighborhood_size, heads, d])
        V_patches = tf.reshape(V_patches, [B, H, W, self.neighborhood_size, heads, d])

        # Prepare queries
        if self.intra_mode:
            if self.inference_mode:
                idx_list = [self.inference_idx] if not isinstance(self.inference_idx,  (list, tuple)) else self.inference_idx
            else:
                idx_list = list(range(T))
            # Gather queries from stacked values + temporal embeddings
            idx_tensor = tf.constant(idx_list, dtype=tf.int32)
            Qstacked = tf.gather(stacked_values, idx_tensor, axis=1)  # (B, Qcount, H, W, C)
            if self.frame_aware:
                q_t_emb = tf.gather(t_emb, idx_tensor, axis=1)
                q_t_emb = tf.reshape(q_t_emb, (B, len(idx_list), 1, 1, Ck))
            else:
                q_t_emb = tf.gather(t_emb, idx_tensor)  # (Qcount, Ck)
                q_t_emb = tf.reshape(q_t_emb, (1, len(idx_list), 1, 1, Ck))
        else:
            Qstacked = tf.stack(query_list, axis=1)  # (B, Qcount, H, W, C)

        Qcount = Qstacked.shape[1] if Qstacked.shape[1] is not None else tf.shape(Qstacked)[1]

        # Project queries
        Qresh = tf.reshape(Qstacked, (B * Qcount, H, W, C))
        Qproj_all = self.qproj(Qresh)  # (B*Qcount, H, W, Ck)
        Qproj_all = tf.reshape(Qproj_all, (B, Qcount, H, W, Ck))
        if self.intra_mode:
            Qproj_all = Qproj_all + q_t_emb

        # Add query spatial positional embeddings
        query_spatial_indices = self._get_query_spatial_indices(H, W)  # (H, W)
        query_spatial_emb = self.spatial_pos_embedding(query_spatial_indices)  # (H, W, Ck)
        query_spatial_emb = tf.reshape(query_spatial_emb, (1, 1, H, W, Ck))

        Qproj_all = Qproj_all + query_spatial_emb  # (B, Qcount, H, W, Ck)
        Qproj_all = tf.reshape(Qproj_all, (B, Qcount, H, W, heads, d))

        # Compute attention scores using einsum
        # Q: (B, Qcount, H, W, heads, d)
        # K: (B, H, W, neighborhood_size, heads, d)
        # Output: (B, Qcount, H, W, heads, neighborhood_size)

        scale = tf.math.rsqrt(tf.cast(d, Qproj_all.dtype))
        scores = tf.einsum('bqhwnd,bhwknd->bqhwnk', Qproj_all, K_patches) * scale

        # Softmax
        weights = tf.nn.softmax(scores, axis=-1)  # (B, Qcount, H, W, heads, neighborhood_size)

        if self.dropout > 0 and training:
            weights = self.dropout_layer(weights, training=training)

        # Apply attention to values
        # weights: (B, Qcount, H, W, heads, neighborhood_size)
        # V: (B, H, W, neighborhood_size, heads, d)
        # Output: (B, Qcount, H, W, heads, d)

        out = tf.einsum('bqhwnk,bhwknd->bqhwnd', weights, V_patches)

        # Reshape and project output
        out = tf.reshape(out, (B * Qcount, H, W, Ck))
        out = self.outproj(out)
        if self.skip_connection:
            out = tf.concat([out, Qresh], axis=-1)
            out = self.skipproj(out)
        out = tf.reshape(out, (B, Qcount, H, W, C))
        return tf.transpose(out, [1, 0, 2, 3, 4])


# faster but with high-memory footprint
class LocalSpatioTemporalAttentionPatchKeras(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, num_heads: int, attention_filters: int = 0, spatial_radius: tuple = (2, 2),
                 skip_connection: bool = False, frame_aware: bool = False, frame_max_distance: int = 0,
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
        self.skip_connection = skip_connection
        self.inference_idx = inference_idx
        self.frame_aware = frame_aware
        self.frame_max_distance = frame_max_distance
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
            "skip_connection": self.skip_connection,
            "frame_aware": self.frame_aware,
            "frame_max_distance": self.frame_max_distance
        })
        return config

    def build(self, input_shape):
        if self.intra_mode:
            if self.frame_aware:
                input_shapes, t_index_shape = input_shape
            else:
                input_shapes = input_shape
        else:
            if self.frame_aware:
                query_shapes, input_shapes, timepoints = input_shape
            else:
                query_shape, input_shapes = input_shape

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
        if self.skip_connection:
            self.skipproj = tf.keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=True, name="skip")
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
            if self.frame_aware:
                all_values, t_index = x
            else:
                all_values = x
        else:
            if self.frame_aware:
                query_list, all_values, t_index = x
            else:
                query_list, all_values = x

        C = self.filters
        T = self.temporal_dim
        shape = tf.shape(all_values[0]) if self.intra_mode else tf.shape(query_list[0])
        B, H, W = shape[0], shape[1], shape[2]

        # Temporal embeddings
        if not self.frame_aware:
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
        if self.skip_connection:
            out = tf.concat(attention_output_list, axis=0) # Q*B, H, W, C
            queries = tf.concat(query_list, axis=0) # Q*B, H, W, C
            out = tf.concat([out, queries], axis=-1)
            out = self.skipproj(out)
            return tf.reshape(out, (-1, B, H, W, C))
        else:
            return tf.stack(attention_output_list)
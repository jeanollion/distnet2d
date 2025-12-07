import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from .layers import Combine, RelativeTemporalEmbedding, SplitBatch, Stack, InferenceLayer
from .window_spatial_attention import WindowSpatialAttention

class TemporalPyramid(Layer):
    """
    Hierarchical down and upsampling along time dimension.
    Time interval T = 2W + 1 (always centered and uneven).
    Operations are symmetrical around center (index W).

    For W=3: indices [0,1,2,3,4,5,6] represent [A-3,A-2,A-1,A0,A1,A2,A3]
    Center is always at index W (A0).

    Args:
        window_spatial_attention_kwargs: kwargs for WindowSpatialAttention (used in build)
        down_layer: Layer that takes [prev, curr, next] and outputs downsampled item (optional, for testing)
        up_layer: Layer that takes [prev_neighbor, center_from_down, next_neighbor]
                  and outputs upsampled item (optional, for testing)
        verbose: If True, print index information during build

    Input shape: (T, B, Y, X, C)
    Output shape: (T, B, Y, X, C)
    """

    def __init__(self, window_spatial_attention_kwargs, layer_normalization=True, filter_increase_factor:float=1, filter_increase_mode_log:bool=False, embedding_l2_reg=1e-5, verbose=False, **kwargs):
        super(TemporalPyramid, self).__init__(**kwargs)
        self.window_spatial_attention_kwargs = window_spatial_attention_kwargs
        self.window_spatial_attention_kwargs["layer_normalization"] = False
        self.layer_normalization=layer_normalization
        self.filter_increase_factor=filter_increase_factor
        self.filter_increase_mode_log=filter_increase_mode_log
        self.verbose = verbose
        self.embedding_l2_reg=embedding_l2_reg

    def _precompute_indices(self):
        """Pre-compute all indices for downsampling at each level using stride-based formula."""
        self.down_indices = []
        current_size = self.T
        current_center = self.W
        level = 0
        while current_size > 1:
            center_idx = self._get_indices(current_center, current_size)
            next_center = np.where(center_idx == current_center)[0][0]
            next_size = len(center_idx)
            self.down_indices.append({
                'center': center_idx,      # Indices kept at this level
                'prev':  center_idx - 1,   # Left neighbors for downsampling
                'next': center_idx + 1,    # Right neighbors for downsampling
            })
            if self.verbose:
                symbolic = [f'A{idx - current_center:+d}' if idx != current_center else 'A0'
                           for idx in center_idx]
                print(f"\nLevel {level + 1}: size {current_size} -> {next_size}, center {current_center} -> {next_center}")
                print(f"  Center indices: {center_idx.tolist()}")
                print(f"  Prev indices: {self.down_indices[-1]['prev'].tolist()}")
                print(f"  Next indices: {self.down_indices[-1]['next'].tolist()}")
                print(f"  Symbolic: {symbolic}")
            current_size = next_size
            current_center = next_center
            level += 1
        self.num_levels = len(self.down_indices)

    @staticmethod
    def _get_indices(current_center:int, current_size:int): # Keep elements at even offsets from center: [center-2k, ..., center, ..., center+2k]
        right_indices = list( range(current_center, current_size, 2))  # Right side (including center): [center, center+2, center+4, ...]
        left_indices = list( range(current_center - 2, -1, -2))  # Left side (excluding center): [center-2, center-4, ...]
        center_idx = left_indices[::-1] + right_indices  # Combine: left (reversed) + right
        center_idx = np.array(center_idx, dtype=np.int32)
        return np.clip(center_idx, 1, current_size - 2)  # center is never at edge

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape)==2:
            self.frame_aware = True
            tensor_shape, _ = input_shape
        else:
            if isinstance(input_shape, list) and len(input_shape)==1:
                tensor_shape=input_shape[0]
            else:
                tensor_shape = input_shape
            self.frame_aware = False
        self.T = tensor_shape[0]
        assert self.T % 2 == 1, "T must be odd (T = 2W + 1)"
        self.W = (self.T - 1) // 2
        self.C = tensor_shape[-1]
        Y, X = tensor_shape[2:4]
        # Pre-compute all indices for each level
        self._precompute_indices()
        self.temp_emb = []
        self.ln = []
        self.down_op = []
        for i in range(len(self.down_indices)):
            input_filters = self._compute_filters(i, self.C)
            output_filters = self._compute_filters(i+1, self.C)
            print(f"level: {i} filters: {input_filters} -> {output_filters}")
            att_layer = WindowSpatialAttention(**self.window_spatial_attention_kwargs, name=f"down_att{i}")
            conv_layer = Combine(filters=output_filters, name=f"down_comb{i}")
            input_layers = [tf.keras.layers.Input([Y, X, input_filters]), tf.keras.layers.Input([Y, X, input_filters]), tf.keras.layers.Input([Y, X, input_filters])]
            q = tf.keras.layers.Concatenate(axis=0, name=f"concat_q{i}")( [input_layers[1], input_layers[0], input_layers[1], input_layers[2]] )
            kv = tf.keras.layers.Concatenate(axis=0, name=f"concat_kv{i}")( [input_layers[0], input_layers[1], input_layers[2], input_layers[1]] )
            att = att_layer([q, kv])
            out = SplitBatch(n_splits=4, name=f"att_split{i}")(att)
            out = conv_layer(out)
            self.down_op.append(tf.keras.Model(input_layers, out))
            self.ln.append(tf.keras.layers.LayerNormalization())
            self.temp_emb.append(RelativeTemporalEmbedding(embedding_dim=input_filters, multiplicative=False, l2_reg=self.embedding_l2_reg))
        super(TemporalPyramid, self).build(input_shape)

    def call(self, inputs, training=None):
        if self.frame_aware:
            inputs, frame_index = inputs
        elif isinstance(inputs, list):
            inputs = inputs[0]
        input_shape = tf.shape(inputs)
        _, B, Y, X, _ = tf.unstack(input_shape)
        if not self.frame_aware:
            frame_index = tf.range(self.T) - tf.cast(self.W, tf.int32) # relative to center frame

        current_frame_index = frame_index # B, T if frame_aware, else T
        down_layers = [inputs]
        for level in range(self.num_levels):
            current = down_layers[-1]
            indices = self.down_indices[level]
            T, _, _, _, C = tf.unstack(tf.shape(current))

            # temporal embedding
            if self.frame_aware:
                t_emb = self.temp_emb[level](current_frame_index, training=training)  # B, T, C
                t_emb = tf.transpose(t_emb, [1, 0, 2])  # T, B, C
                t_emb = tf.reshape(t_emb, [T, B, 1, 1, C])
            else:
                t_emb = self.temp_emb[level](current_frame_index, training=training)  # T, C
                t_emb = tf.reshape(t_emb, [T, 1, 1, 1, C])

            current = current + t_emb
            if self.layer_normalization:
                current = self.ln[level](current, training=training)

            # Gather triplets
            prev_neighbors = tf.gather(current, indices['prev'])
            centers = tf.gather(current, indices['center'])
            next_neighbors = tf.gather(current, indices['next'])

            # Transpose + reshape for batch processing
            next_size = len(indices['center'])
            prev_neighbors = tf.reshape(prev_neighbors, [next_size * B, Y, X, C])
            centers = tf.reshape(centers, [next_size * B, Y, X, C])
            next_neighbors = tf.reshape(next_neighbors, [next_size * B, Y, X, C])

            # Apply downsampling
            downsampled = self.down_op[level]([prev_neighbors, centers, next_neighbors], training=training)
            if next_size > 1: # Reshape back
                downsampled = tf.reshape(downsampled, [next_size, B, Y, X, tf.shape(downsampled)[-1]])
            down_layers.append(downsampled)
            # update frame index for temporal encoding
            current_frame_index = tf.gather(current_frame_index, indices['center'], axis=1 if self.frame_aware else 0) # update frame index
        return down_layers[-1], down_layers[1]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        n_filters_global = self._compute_filters(self.num_levels, input_shape[-1])
        n_filters_l1 = self._compute_filters(1, input_shape[-1])
        return input_shape[1:-1] + (n_filters_global, ), input_shape[:-1] + (n_filters_l1, )

    def _compute_filters(self, level:int, base_filters:int):
        if level == 0:
            return base_filters
        if self.filter_increase_mode_log:
            compression_ratio = self.T / len(self.down_indices[level-1]['center'])
            return int(base_filters * compression_ratio * self.filter_increase_factor)
        else:
            return base_filters + int(base_filters * self.filter_increase_factor) * level

    def get_config(self):
        config = super(TemporalPyramid, self).get_config()
        config.update({
            'window_spatial_attention_kwargs': self.window_spatial_attention_kwargs,
            "filter_increase_factor":self.filter_increase_factor,
            "filter_increase_mode_log":self.filter_increase_mode_log,
            "layer_normalization":self.layer_normalization,
            'verbose': self.verbose
        })
        return config

# reconstruct features through independent convolutions that inputs both features, global context and level 1 features from pyramid
class TemporalFeatureReconstructor(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, output_filters, inference_idx:list, **kwargs):
        super().__init__(**kwargs)
        self.output_filters = output_filters
        self.inference_idx=inference_idx
        self.frame_convs = []

    def get_config(self):
        config = super().get_config()
        config.update({ 'output_filters': self.output_filters, 'inference_idx': self.inference_idx })
        return config

    def build(self, input_shape):
        features_level0, features_level1, global_features = input_shape
        self.T = features_level0[0]

        # Create T independent 1x1 convolutions
        for i in range(self.T):
            conv = tf.keras.layers.Conv2D(
                filters=self.output_filters,
                kernel_size=1,
                use_bias=True,
                name=f'frame_{i}_conv'
            )
            self.frame_convs.append(conv)
            self.feature_level1_indices = self._get_closest_indices(self.T)
        super().build(input_shape)

    @staticmethod
    def _get_closest_indices(T):
        next_indices = TemporalPyramid._get_indices((T - 1) // 2, T)
        closest_indices = []
        for i in np.arange(T):
            distances = np.abs(next_indices - i)
            min_distance_indices = np.where(distances == np.min(distances))[0]
            if len(min_distance_indices) > 1:  # multiple indices
                closest_indices.append(min_distance_indices.tolist())
            else:
                closest_indices.append(min_distance_indices.tolist())
        return closest_indices

    def call(self, inputs, training=None):
        features_level0, features_level1, global_features = inputs
        outputs = []
        idx_list = self.inference_idx if self.inference_mode and self.inference_idx is not None else list(range(self.T))
        for i in idx_list:
            idx = self.feature_level1_indices[i]
            input_list = [features_level0[i], global_features, features_level1[idx[0]]]
            if len(idx) == 2: # feature is in between two level 1 features -> also add the second one to inputs
                input_list.append(features_level1[idx[1]])
            combined_i = tf.concat(input_list, axis=-1)
            output_i = self.frame_convs[i](combined_i, training=training)
            outputs.append(output_i)
        return tf.stack(outputs, axis=0)
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from .layers import Combine, RelativeTemporalEmbedding, SplitBatch
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

    def __init__(self, window_spatial_attention_kwargs, filter_increase_factor:float=1, embedding_l2_reg=1e-5, verbose=False, **kwargs):
        super(TemporalPyramid, self).__init__(**kwargs)
        self.window_spatial_attention_kwargs = window_spatial_attention_kwargs
        self.layer_normalization = self.window_spatial_attention_kwargs.pop("layer_normalization", True)
        self.filter_increase_factor=filter_increase_factor
        self.verbose = verbose
        self.embedding_l2_reg=embedding_l2_reg

    def _precompute_indices(self):
        """Pre-compute all indices for down/upsampling at each level using stride-based formula."""

        self.down_indices = []
        self.level_sizes = []

        current_size = self.T
        current_center = self.W
        self.level_sizes.append(current_size)

        if self.verbose:
            print(f"\n=== Building Hierarchy for T={self.T}, W={self.W} ===")
            print(f"Center is at index {self.W}")
            print(f"Level 0 (input): size={current_size}, center={current_center}")

        # Build downsampling hierarchy
        level = 0
        while current_size > 1:
            # Keep elements at even offsets from center: [center-2k, ..., center, ..., center+2k]
            # Right side (including center): [center, center+2, center+4, ...]
            right_indices = list(range(current_center, current_size, 2))
            # Left side (excluding center): [center-2, center-4, ...]
            left_indices = list(range(current_center - 2, -1, -2))
            # Combine: left (reversed) + right
            kept_indices = left_indices[::-1] + right_indices

            next_size = len(kept_indices)
            next_center = kept_indices.index(current_center)

            # Store downsampling triplets
            kept_indices = np.array(kept_indices, dtype=np.int32)
            prev_idx = np.clip(kept_indices - 1, 0, current_size - 1)
            next_idx = np.clip(kept_indices + 1, 0, current_size - 1)

            self.down_indices.append({
                'center': kept_indices,      # Indices kept at this level
                'prev': prev_idx,          # Left neighbors for downsampling
                'next': next_idx           # Right neighbors for downsampling
            })

            if self.verbose:
                symbolic = [f'A{idx - current_center:+d}' if idx != current_center else 'A0'
                           for idx in kept_indices]
                print(f"\nLevel {level + 1}: size {current_size} -> {next_size}, center {current_center} -> {next_center}")
                print(f"  Kept indices: {kept_indices.tolist()}")
                print(f"  Symbolic: {symbolic}")
                print(f"  Downsampling triplets (prev, center, next):")
                for i, (p, c, n) in enumerate(zip(prev_idx, kept_indices, next_idx)):
                    print(f"    Out[{i}] = down(In[{p}], In[{c}], In[{n}])")

            self.level_sizes.append(next_size)
            current_size = next_size
            current_center = next_center
            level += 1
        self.num_levels = len(self.level_sizes) -1

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
        filter_increase = int(self.C * self.filter_increase_factor)
        # Pre-compute all indices for each level
        self._precompute_indices()

        self.down_op = []
        for i in range(len(self.down_indices)):
            input_filters = self.C + filter_increase * i
            output_filters = input_filters + filter_increase
            att_layer = WindowSpatialAttention(**self.window_spatial_attention_kwargs, layer_normalization=False, name=f"down_att{i}")
            conv_layer = Combine(filters=output_filters, name=f"down_comb{i}")
            input_layers = [tf.keras.layers.Input([Y, X, input_filters]), tf.keras.layers.Input([Y, X, input_filters]), tf.keras.layers.Input([Y, X, input_filters])]
            q = tf.keras.layers.Concatenate(axis=0, name=f"concat_q{i}")( [input_layers[1], input_layers[0], input_layers[1], input_layers[2]] )
            kv = tf.keras.layers.Concatenate(axis=0, name=f"concat_kv{i}")( [input_layers[0], input_layers[1], input_layers[2], input_layers[1]] )
            att = att_layer([q, kv])
            out = SplitBatch(n_splits=4, name=f"att_split{i}")(att)
            out = conv_layer(out)
            self.down_op.append(tf.keras.Model(input_layers, out))
        self.ln = tf.keras.layers.LayerNormalization() if self.layer_normalization else None
        self.tem_emb = RelativeTemporalEmbedding(embedding_dim=self.C, multiplicative=False, l2_reg=self.embedding_l2_reg) if self.frame_aware else tf.keras.layers.Embedding(
            input_dim = self.T, output_dim=self.C,
            embeddings_regularizer=tf.keras.regularizers.l2(self.embedding_l2_reg) if self.embedding_l2_reg > 0 else None
        )
        super(TemporalPyramid, self).build(input_shape)

    def call(self, inputs, training=None):
        if self.frame_aware:
            inputs, frame_index = inputs
        elif isinstance(inputs, list):
            inputs = inputs[0]
        input_shape = tf.shape(inputs)
        _, B, Y, X, _ = tf.unstack(input_shape)

        if self.frame_aware:
            t_emb = self.tem_emb(frame_index) # B, T, C
            t_emb = tf.transpose(t_emb, [1, 0, 2]) # T, B, C
            t_emb = tf.reshape(t_emb, [self.T, B, 1, 1, self.C])
        else:
            t_emb = self.tem_emb(tf.range(self.T)) # T, C
            t_emb = tf.reshape(t_emb, [self.T, 1, 1, 1, self.C])
        inputs_emb = inputs + t_emb
        if self.layer_normalization:
            inputs_emb = self.ln(inputs_emb)
        down_layers = [inputs_emb]

        for level in range(self.num_levels):
            current = down_layers[-1]
            indices = self.down_indices[level]
            C = tf.shape(current)[-1]

            # Gather triplets
            prev_neighbors = tf.gather(current, indices['prev'])
            centers = tf.gather(current, indices['center'])
            next_neighbors = tf.gather(current, indices['next'])

            # Batch processing
            next_size = len(indices['center'])
            prev_neighbors = tf.reshape(prev_neighbors, [next_size * B, Y, X, C])
            centers = tf.reshape(centers, [next_size * B, Y, X, C])
            next_neighbors = tf.reshape(next_neighbors, [next_size * B, Y, X, C])

            # Apply downsampling
            downsampled = self.down_op[level]([prev_neighbors, centers, next_neighbors], training=training)

            if next_size > 1: # Reshape back
                downsampled = tf.reshape(downsampled, [next_size, B, Y, X, tf.shape(downsampled)[-1]])
            down_layers.append(downsampled)

        return down_layers[-1]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        n_filters = input_shape[-1] + int(input_shape[-1] * self.filter_increase_factor) * self.num_levels
        return input_shape[1:-1] + (n_filters, )

    def get_config(self):
        config = super(TemporalPyramid, self).get_config()
        config.update({
            'window_spatial_attention_kwargs': self.window_spatial_attention_kwargs,
            "filter_increase_factor":self.filter_increase_factor,
            'verbose': self.verbose
        })
        return config

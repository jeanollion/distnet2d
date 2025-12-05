import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from .layers import Combine, RelativeTemporalEmbedding
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

    def __init__(self, window_spatial_attention_kwargs, embedding_l2_reg=1e-5, skip_connection:bool=True, down_layer=None, up_layer=None, verbose=False, **kwargs):
        super(TemporalPyramid, self).__init__(**kwargs)
        self.window_spatial_attention_kwargs = window_spatial_attention_kwargs
        self.skip_connection=skip_connection
        self.down_op = down_layer
        self.up_op = up_layer
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
        self.num_levels = len(self.level_sizes)

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape)==2:
            self.frame_aware = True
            input_shape, _ = input_shape
        else:
            if isinstance(input_shape, (list, tuple)) and len(input_shape)==1:
                input_shape=input_shape[0]
            self.frame_aware = False
        self.T = input_shape[0]
        assert self.T % 2 == 1, "T must be odd (T = 2W + 1)"
        self.W = (self.T - 1) // 2
        self.C = input_shape[-1]
        Y, X = input_shape[2:4]
        filter_increase = self.C // 2
        # Pre-compute all indices for each level
        self._precompute_indices()

        if self.down_op is None:
            self.down_op = []
            for i in range(len(self.down_indices)):
                input_filters = self.C + filter_increase * i
                output_filters = input_filters + filter_increase
                att_layer = WindowSpatialAttention(**self.window_spatial_attention_kwargs, skip_connection=False, name=f"down_att{i}")
                conv_layer = Combine(filters=output_filters, name=f"down_comb{i}")
                input_layers = [tf.keras.layers.Input([Y, X, input_filters]), tf.keras.layers.Input([Y, X, input_filters]), tf.keras.layers.Input([Y, X, input_filters])]
                q = tf.concat([input_layers[1], input_layers[0], input_layers[1], input_layers[2]], axis=0)
                kv = tf.concat([input_layers[0], input_layers[1], input_layers[2], input_layers[1]], axis=0)
                att = att_layer([q, kv])
                out = conv_layer(tf.split(att, 4))
                self.down_op.append(tf.keras.Model(input_layers, out))
        else:
            # Test mode: repeat the provided operation
            self.down_op = [self.down_op] * len(self.down_indices)

        if self.up_op is None:
            self.up_op = WindowSpatialAttention(**self.window_spatial_attention_kwargs, skip_connection=self.skip_connection, name=f"up_att")
        self.tem_emb = RelativeTemporalEmbedding(embedding_dim=self.C, multiplicative=False) if self.frame_aware else tf.keras.layers.Embedding(
            input_dim = self.T, output_dim=self.C,
            embeddings_regularizer=tf.keras.regularizers.l2(self.embedding_l2_reg) if self.embedding_l2_reg > 0 else None
        )
        super(TemporalPyramid, self).build(input_shape)

    def call(self, inputs, training=None):
        if self.frame_aware:
            inputs, frame_index = inputs
        elif isinstance(inputs, (tuple, list)):
            inputs = inputs[0]
        input_shape = tf.shape(inputs)
        _, B, Y, X, _ = tf.unstack(input_shape)

        down_layers = [inputs]

        # Downsample phase
        for level in range(self.num_levels - 1):
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

            # Reshape back
            downsampled = tf.reshape(downsampled, [next_size, B, Y, X, tf.shape(downsampled)[-1]])
            down_layers.append(downsampled)

        # Upsample phase
        central_feature = down_layers[-1][0]
        if self.frame_aware:
            t_emb = self.tem_emb(frame_index) # B, T, C
            t_emb = tf.transpose(t_emb, [1, 0, 2]) # T, B, C
            t_emb = tf.reshape(t_emb, [self.T, B, 1, 1, self.C])
        else:
            t_emb = self.tem_emb(tf.range(self.T)) # T, C
            t_emb = tf.reshape(t_emb, [self.T, 1, 1, 1, self.C])
        inputs_emb = inputs + t_emb
        return self.up_op([inputs_emb, central_feature]) # multi-query mode: all queries attend to the same KV

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(TemporalPyramid, self).get_config()
        config.update({
            'window_spatial_attention_kwargs': self.window_spatial_attention_kwargs,
            'skip_connection':self.skip_connection,
            'verbose': self.verbose
        })
        return config

# Test code
def test_left_neighbor_downsampling():
    """Test that downsampling correctly uses the LEFT neighbor (prev)."""
    print("\n" + "=" * 80)
    print("TEST 1: LEFT NEIGHBOR DOWNSAMPLING")
    print("=" * 80)

    W = 3
    T = 2 * W + 1  # T = 7
    B, Y, X, C = 1, 1, 1, 1

    print(f"\nSetup: W={W}, T={T}, B={B}, Y={Y}, X={X}, C={C}")
    print("Down layer: outputs prev (first input)")
    print("Up layer: outputs center (second input, identity)")

    # Down layer returns the PREV (left) neighbor
    down_layer = tf.keras.layers.Lambda(lambda x: x[0])
    # Up layer returns center (identity)
    up_layer = tf.keras.layers.Lambda(lambda x: x[1])

    hier_layer = TemporalPyramid(None, down_layer, up_layer, verbose=True)

    # Input: [10, 20, 30, 40, 50, 60, 70]
    test_input = tf.constant([10, 20, 30, 40, 50, 60, 70], dtype=tf.float32)
    test_input = tf.reshape(test_input, [T, B, Y, X, C])

    print(f"\nInput:  {test_input[:, 0, 0, 0, 0].numpy()}")

    output = hier_layer(test_input)
    output_vals = output[:, 0, 0, 0, 0].numpy()

    print(f"Output: {output_vals}")
    print("\n✓ Left neighbor downsampling test completed")


def test_right_neighbor_downsampling():
    """Test that downsampling correctly uses the RIGHT neighbor (next)."""
    print("\n" + "=" * 80)
    print("TEST 2: RIGHT NEIGHBOR DOWNSAMPLING")
    print("=" * 80)

    W = 3
    T = 2 * W + 1
    B, Y, X, C = 1, 1, 1, 1

    print(f"\nSetup: W={W}, T={T}, B={B}, Y={Y}, X={X}, C={C}")
    print("Down layer: outputs next (third input)")
    print("Up layer: outputs center (second input, identity)")

    # Down layer returns the NEXT (right) neighbor
    down_layer = tf.keras.layers.Lambda(lambda x: x[2])
    # Up layer returns center (identity)
    up_layer = tf.keras.layers.Lambda(lambda x: x[1])

    hier_layer = TemporalPyramid(None, down_layer, up_layer, verbose=True)

    test_input = tf.constant([10, 20, 30, 40, 50, 60, 70], dtype=tf.float32)
    test_input = tf.reshape(test_input, [T, B, Y, X, C])

    print(f"\nInput:  {test_input[:, 0, 0, 0, 0].numpy()}")

    output = hier_layer(test_input)
    output_vals = output[:, 0, 0, 0, 0].numpy()

    print(f"Output: {output_vals}")
    print("\n✓ Right neighbor downsampling test completed")


def test_left_neighbor_upsampling():
    """Test that upsampling correctly uses the LEFT neighbor."""
    print("\n" + "=" * 80)
    print("TEST 3: LEFT NEIGHBOR UPSAMPLING")
    print("=" * 80)

    W = 3
    T = 2 * W + 1
    B, Y, X, C = 1, 1, 1, 1

    print(f"\nSetup: W={W}, T={T}, B={B}, Y={Y}, X={X}, C={C}")
    print("Down layer: outputs center (identity)")
    print("Up layer: outputs prev (first input, left neighbor)")

    # Down layer returns center (identity)
    down_layer = tf.keras.layers.Lambda(lambda x: x[1])
    # Up layer returns the PREV (left) neighbor
    up_layer = tf.keras.layers.Lambda(lambda x: x[0])

    hier_layer = TemporalPyramid(None, down_layer, up_layer, verbose=True)

    test_input = tf.constant([10, 20, 30, 40, 50, 60, 70], dtype=tf.float32)
    test_input = tf.reshape(test_input, [T, B, Y, X, C])

    print(f"\nInput:  {test_input[:, 0, 0, 0, 0].numpy()}")
    print("Expected: [10, 20, 20, 40, 40, 40, 40]")

    output = hier_layer(test_input)
    output_vals = output[:, 0, 0, 0, 0].numpy()

    print(f"Output:   {output_vals}")

    expected = np.array([10, 20, 20, 40, 40, 40, 40], dtype=np.float32)
    if np.allclose(output_vals, expected):
        print("✓ Left neighbor upsampling test PASSED!")
    else:
        print("✗ Left neighbor upsampling test FAILED!")
        print(f"Difference: {output_vals - expected}")


def test_right_neighbor_upsampling():
    """Test that upsampling correctly uses the RIGHT neighbor."""
    print("\n" + "=" * 80)
    print("TEST 4: RIGHT NEIGHBOR UPSAMPLING")
    print("=" * 80)

    W = 3
    T = 2 * W + 1
    B, Y, X, C = 1, 1, 1, 1

    print(f"\nSetup: W={W}, T={T}, B={B}, Y={Y}, X={X}, C={C}")
    print("Down layer: outputs center (identity)")
    print("Up layer: outputs next (third input, right neighbor)")

    # Down layer returns center (identity)
    down_layer = tf.keras.layers.Lambda(lambda x: x[1])
    # Up layer returns the NEXT (right) neighbor
    up_layer = tf.keras.layers.Lambda(lambda x: x[2])

    hier_layer = TemporalPyramid(None, down_layer, up_layer, verbose=True)

    test_input = tf.constant([10, 20, 30, 40, 50, 60, 70], dtype=tf.float32)
    test_input = tf.reshape(test_input, [T, B, Y, X, C])

    print(f"\nInput:  {test_input[:, 0, 0, 0, 0].numpy()}")
    print("Expected: [40, 40, 40, 40, 60, 60, 70]")

    output = hier_layer(test_input)
    output_vals = output[:, 0, 0, 0, 0].numpy()

    print(f"Output:   {output_vals}")

    expected = np.array([40, 40, 40, 40, 60, 60, 70], dtype=np.float32)
    if np.allclose(output_vals, expected):
        print("✓ Right neighbor upsampling test PASSED!")
    else:
        print("✗ Right neighbor upsampling test FAILED!")
        print(f"Difference: {output_vals - expected}")


def test_hierarchical_layer():
    """Original comprehensive test."""
    print("=" * 80)
    print("COMPREHENSIVE IDENTITY TEST")
    print("=" * 80)

    W = 4
    T = 2 * W + 1
    B, Y, X, C = 2, 4, 4, 8

    # Down and up layers both return center (identity)
    down_layer = tf.keras.layers.Lambda(lambda x: x[1])
    up_layer = tf.keras.layers.Lambda(lambda x: x[1])

    hier_layer = TemporalPyramid(None, down_layer, up_layer, verbose=True)

    test_input = tf.constant(
        np.arange(T).reshape(T, 1, 1, 1, 1) * np.ones((T, B, Y, X, C)),
        dtype=tf.float32
    )

    output = hier_layer(test_input)

    assert output.shape == test_input.shape, f"Shape mismatch!"

    if np.allclose(output.numpy(), test_input.numpy()):
        print("✓ Identity test PASSED!")
    else:
        print("✗ Identity test FAILED!")
        print(f"Input:  {test_input[:, 0, 0, 0, 0].numpy()}")
        print(f"Output: {output[:, 0, 0, 0, 0].numpy()}")


if __name__ == "__main__":
    test_left_neighbor_upsampling()
    test_right_neighbor_upsampling()
    test_left_neighbor_downsampling()
    test_right_neighbor_downsampling()
    test_hierarchical_layer()
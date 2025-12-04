import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from .layers import Combine
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

    def __init__(self, window_spatial_attention_kwargs, down_layer=None, up_layer=None, verbose=False, **kwargs):
        super(TemporalPyramid, self).__init__(**kwargs)
        self.window_spatial_attention_kwargs = window_spatial_attention_kwargs
        self.down_op = down_layer
        self.up_op = up_layer
        self.verbose = verbose

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
                'kept': kept_indices,      # Indices kept at this level
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

        # Build upsampling indices
        self.up_indices = []

        if self.verbose:
            print(f"\n=== Upsampling Phase ===")
            print(f"Number of levels: {self.num_levels}")

        for level_idx in range(self.num_levels - 2, -1, -1):
            target_size = self.level_sizes[level_idx]
            kept_indices = self.down_indices[level_idx]['kept']

            # Build position mappings
            kept_set = set(kept_indices)
            keep_target_pos = []    # Target positions that are kept
            keep_from_prev = []     # Corresponding indices in prev_up
            up_positions = []       # Positions that need upsampling

            for target_pos in range(target_size):
                if target_pos in kept_set:
                    # This position comes from prev_up
                    prev_up_idx = np.where(kept_indices == target_pos)[0][0]
                    keep_target_pos.append(target_pos)
                    keep_from_prev.append(prev_up_idx)
                else:
                    # This position needs upsampling
                    up_positions.append(target_pos)

            # For each upsampled position, determine neighbors
            prev_neighbor_idx = []
            prev_is_center = []     # True if prev neighbor is center (edge case)
            next_neighbor_idx = []
            next_is_center = []     # True if next neighbor is center (edge case)

            for pos in up_positions:
                # Find left neighbor from kept indices
                left_kept = kept_indices[kept_indices < pos]
                if len(left_kept) > 0:
                    # Use the rightmost kept index to the left
                    left_target_pos = left_kept[-1]
                    left_prev_up_idx = np.where(kept_indices == left_target_pos)[0][0]
                    prev_neighbor_idx.append(left_prev_up_idx)
                    prev_is_center.append(False)
                else:
                    # No left neighbor - use center from down_layer
                    prev_neighbor_idx.append(pos)
                    prev_is_center.append(True)

                # Find right neighbor from kept indices
                right_kept = kept_indices[kept_indices > pos]
                if len(right_kept) > 0:
                    # Use the leftmost kept index to the right
                    right_target_pos = right_kept[0]
                    right_prev_up_idx = np.where(kept_indices == right_target_pos)[0][0]
                    next_neighbor_idx.append(right_prev_up_idx)
                    next_is_center.append(False)
                else:
                    # No right neighbor - use center from down_layer
                    next_neighbor_idx.append(pos)
                    next_is_center.append(True)

            self.up_indices.append({
                'keep_target_pos': np.array(keep_target_pos, dtype=np.int32),
                'keep_from_prev': np.array(keep_from_prev, dtype=np.int32),
                'up_positions': np.array(up_positions, dtype=np.int32),
                'prev_neighbor': np.array(prev_neighbor_idx, dtype=np.int32),
                'prev_is_center': np.array(prev_is_center, dtype=bool),
                'next_neighbor': np.array(next_neighbor_idx, dtype=np.int32),
                'next_is_center': np.array(next_is_center, dtype=bool)
            })

            if self.verbose:
                print(f"\nUp Level {level_idx}: size {self.level_sizes[level_idx + 1]} -> {target_size}")
                print(f"  Kept indices: {kept_indices.tolist()}")
                print(f"  Keep: target_pos {keep_target_pos} from prev_up {keep_from_prev}")
                print(f"  Upsample positions: {up_positions}")
                if len(up_positions) > 0:
                    print(f"  Upsampling operations:")
                    for i, pos in enumerate(up_positions):
                        pn = prev_neighbor_idx[i]
                        nn = next_neighbor_idx[i]
                        pn_src = f"DownLayer{level_idx}[{pos}]" if prev_is_center[i] else f"PrevUp[{pn}]"
                        nn_src = f"DownLayer{level_idx}[{pos}]" if next_is_center[i] else f"PrevUp[{nn}]"
                        print(f"    Out[{pos}] = up({pn_src}, DownLayer{level_idx}[{pos}], {nn_src})")

    def build(self, input_shape):
        self.T = input_shape[0]
        assert self.T % 2 == 1, "T must be odd (T = 2W + 1)"
        self.W = (self.T - 1) // 2
        self.C = input_shape[-1]
        Y, X = input_shape[2:4]

        # Pre-compute all indices for each level
        self._precompute_indices()

        if self.down_op is None:
            self.down_op = []
            for i in range(len(self.down_indices)):  # TODO add temporal position encoding ?
                att_layer = WindowSpatialAttention(**self.window_spatial_attention_kwargs, name=f"down_att{i}")
                conv_layer = Combine(filters=self.C, name=f"down_comb{i}")
                input_layers = [tf.keras.layers.Input([Y, X, self.C]), tf.keras.layers.Input([Y, X, self.C]), tf.keras.layers.Input([Y, X, self.C])]
                q = tf.concat([input_layers[1], input_layers[0], input_layers[1], input_layers[2]], axis=0)
                kv = tf.concat([input_layers[0], input_layers[1], input_layers[2], input_layers[1]], axis=0)
                att = att_layer([q, kv])
                out = conv_layer(tf.split(att, 4))
                self.down_op.append(tf.keras.Model(input_layers, out))
        else:
            # Test mode: repeat the provided operation
            self.down_op = [self.down_op] * len(self.down_indices)

        if self.up_op is None:
            self.up_op = []
            for i in range(len(self.up_indices)):
                self.up_op.append(Combine(filters=self.C, name=f"up_comb{i}"))
        else:
            # Test mode: repeat the provided operation
            self.up_op = [self.up_op] * len(self.up_indices)

        super(TemporalPyramid, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        B = input_shape[1]
        Y = input_shape[2]
        X = input_shape[3]

        down_layers = [inputs]

        # Downsample phase
        for level in range(self.num_levels - 1):
            current = down_layers[-1]
            indices = self.down_indices[level]
            C = tf.shape(current)[-1]

            # Gather triplets
            prev_neighbors = tf.gather(current, indices['prev'])
            centers = tf.gather(current, indices['kept'])
            next_neighbors = tf.gather(current, indices['next'])

            # Batch processing
            next_size = len(indices['kept'])
            prev_neighbors = tf.reshape(prev_neighbors, [next_size * B, Y, X, C])
            centers = tf.reshape(centers, [next_size * B, Y, X, C])
            next_neighbors = tf.reshape(next_neighbors, [next_size * B, Y, X, C])

            # Apply downsampling
            downsampled = self.down_op[level]([prev_neighbors, centers, next_neighbors], training=training)

            # Reshape back
            downsampled = tf.reshape(downsampled, [next_size, B, Y, X, tf.shape(downsampled)[-1]])
            down_layers.append(downsampled)

        # Upsample phase
        up_layers = [down_layers[-1]]

        for level_idx in range(len(self.up_indices)):
            prev_up = up_layers[-1]
            level = self.num_levels - 2 - level_idx
            down_layer_data = down_layers[level]
            indices = self.up_indices[level_idx]

            # Gather kept items
            kept_items = tf.gather(prev_up, indices['keep_from_prev'])

            # Perform upsampling
            if len(indices['up_positions']) > 0:
                num_ups = len(indices['up_positions'])

                # Centers come from down_layer at upsampled positions
                centers = tf.gather(down_layer_data, indices['up_positions'])

                # Build prev neighbors: gather from prev_up (non-edge) or down_layer (edge)
                prev_from_up = tf.gather(prev_up, indices['prev_neighbor'][~indices['prev_is_center']])
                prev_from_down = tf.gather(down_layer_data, indices['up_positions'][indices['prev_is_center']])
                # Determine insertion positions
                prev_up_positions = tf.reshape(tf.cast(tf.where(~indices['prev_is_center']), tf.int32), [-1])
                prev_down_positions = tf.reshape(tf.cast(tf.where(indices['prev_is_center']), tf.int32), [-1])
                prev_neighbors = tf.dynamic_stitch(
                    [prev_up_positions, prev_down_positions],
                    [prev_from_up, prev_from_down]
                )

                # Build next neighbors: gather from prev_up (non-edge) or down_layer (edge)
                next_from_up = tf.gather(prev_up, indices['next_neighbor'][~indices['next_is_center']])
                next_from_down = tf.gather(down_layer_data, indices['up_positions'][indices['next_is_center']])
                # Determine insertion positions
                next_up_positions = tf.reshape(tf.cast(tf.where(~indices['next_is_center']), tf.int32), [-1])
                next_down_positions = tf.reshape(tf.cast(tf.where(indices['next_is_center']), tf.int32), [-1])
                next_neighbors = tf.dynamic_stitch(
                    [next_up_positions, next_down_positions],
                    [next_from_up, next_from_down]
                )

                C = tf.shape(prev_neighbors)[-1]
                # Batch processing
                prev_neighbors = tf.reshape(prev_neighbors, [num_ups * B, Y, X, C])
                centers = tf.reshape(centers, [num_ups * B, Y, X, C])
                next_neighbors = tf.reshape(next_neighbors, [num_ups * B, Y, X, C])

                upsampled = self.up_op[level_idx]([prev_neighbors, centers, next_neighbors], training=training)

                # Reshape back
                upsampled = tf.reshape(upsampled, [num_ups, B, Y, X, tf.shape(upsampled)[-1]])

                # Stitch together kept and upsampled items
                all_indices = tf.concat([indices['keep_target_pos'], indices['up_positions']], axis=0)
                all_values = tf.concat([kept_items, upsampled], axis=0)
                result = tf.dynamic_stitch([all_indices], [all_values])
            else:
                # No upsampling needed
                result = tf.dynamic_stitch([indices['keep_target_pos']], [kept_items])

            up_layers.append(result)

        return up_layers[-1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(TemporalPyramid, self).get_config()
        config.update({
            'window_spatial_attention_kwargs': self.window_spatial_attention_kwargs,
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
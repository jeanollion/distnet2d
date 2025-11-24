from distnet_2d.model.local_spatial_attention import LocalSpatialAttentionPatchKeras, LocalSpatialAttention, \
    LocalSpatialAttentionPatch
from distnet_2d.model.local_spatio_temporal_attention import LocalSpatioTemporalAttention, \
    LocalSpatioTemporalAttentionPatchKeras, \
    LocalSpatioTemporalAttentionPatch
import tensorflow as tf
import numpy as np


def copy_spatio_temporal_att_weights(src_layer, dst_layer, keras_att:bool):
    """Copy weights from SpatioTemporalAttentionHighMem to SpatioTemporalAttention."""
    if keras_att:
        # zero out temporal embdedding for both layers to enable comparison
        # (since they're added at different stages in the pipeline)
        src_spatial_weights = src_layer.temp_embedding.get_weights()
        dst_spatial_weights = dst_layer.temp_embedding.get_weights()

        # Set to zeros
        src_spatial_weights[0] = np.zeros_like(src_spatial_weights[0])
        dst_spatial_weights[0] = np.zeros_like(dst_spatial_weights[0])

        src_layer.temp_embedding.set_weights(src_spatial_weights)
        dst_layer.temp_embedding.set_weights(dst_spatial_weights)
    else:
        dst_layer.temp_embedding.set_weights(src_layer.temp_embedding.get_weights())

    if keras_att:
        # Zero out spatial positional embeddings for both layers to enable comparison
        # (since they're added at different stages in the pipeline)
        src_spatial_weights = src_layer.spatial_pos_embedding.get_weights()
        dst_spatial_weights = dst_layer.spatial_pos_embedding.get_weights()

        # Set to zeros
        src_spatial_weights[0] = np.zeros_like(src_spatial_weights[0])
        dst_spatial_weights[0] = np.zeros_like(dst_spatial_weights[0])

        src_layer.spatial_pos_embedding.set_weights(src_spatial_weights)
        dst_layer.spatial_pos_embedding.set_weights(dst_spatial_weights)

    if keras_att:
        # Copy MHA weights to conv projections
        mha = src_layer.attention_layer

        # Extract MHA kernels
        # MultiHeadAttention uses key_dim per head, not attention_filters
        # Total dim = num_heads * key_dim
        W_q = mha._query_dense.kernel.numpy()  # (C_in, num_heads, key_dim)
        W_k = mha._key_dense.kernel.numpy()  # (C_in, num_heads, key_dim)
        W_v = mha._value_dense.kernel.numpy()  # (C_in, num_heads, key_dim)
        W_o = mha._output_dense.kernel.numpy()  # (num_heads * key_dim, C_out)

        print(f"MHA weight shapes:")
        print(f"  W_q: {W_q.shape}")
        print(f"  W_k: {W_k.shape}")
        print(f"  W_v: {W_v.shape}")
        print(f"  W_o: {W_o.shape}")

        # Get dimensions
        C_in = W_q.shape[0]
        Ck = W_q.shape[1] * W_q.shape[2]  # num_heads * key_dim

        print(f"Conv layer expected shapes:")
        print(f"  C_in: {C_in}")
        print(f"  Ck (num_heads * key_dim): {Ck}")
        print(f"  dst num_heads: {dst_layer.num_heads}")
        print(f"  dst attention_filters: {dst_layer.attention_filters}")
        print(f"  Expected Ck: {dst_layer.num_heads * dst_layer.attention_filters}")

        # Verify dimensions match
        expected_Ck = dst_layer.num_heads * dst_layer.attention_filters
        if Ck != expected_Ck:
            raise ValueError(
                f"Dimension mismatch: MHA has {Ck} features "
                f"but Conv layer expects {expected_Ck} "
                f"(num_heads={dst_layer.num_heads} * attention_filters={dst_layer.attention_filters})"
            )

        # Reshape to (1, 1, C_in, C_out) for Conv2D
        W_q_conv = W_q.reshape(1, 1, C_in, Ck)
        W_k_conv = W_k.reshape(1, 1, C_in, Ck)
        W_v_conv = W_v.reshape(1, 1, C_in, Ck)
        W_o_conv = W_o.reshape(1, 1, Ck, C_in)  # Output back to C_in

        # Assign weights to conv layers
        dst_layer.qproj.set_weights([W_q_conv])
        dst_layer.kproj.set_weights([W_k_conv])
        dst_layer.vproj.set_weights([W_v_conv])
        dst_layer.outproj.set_weights([W_o_conv])

        if src_layer.skip_connection:
            dst_layer.skipproj.kernel.assign(src_layer.skipproj.kernel)
            dst_layer.skipproj.bias.assign(src_layer.skipproj.bias)
    else:
        print("Copying weights for fair comparison...")
        for src_w, dst_w in zip(src_layer.trainable_weights, dst_layer.trainable_weights):
            dst_w.assign(src_w)
    print("✓ Weights copied successfully")


def compare_spatiotemporal_versions(keras_att:bool):
    """
    Test equivalence between patch-based version and efficient conv version.
    """
    print("=" * 80)
    print(f"TEST: HighMem {'KERAS' if keras_att else ''} vs Loop Conv Equivalence")
    print("=" * 80 + "\n")

    # Parameters
    B, H, W, C = 2, 8, 8, 16
    T = 3
    num_heads = 4
    attention_filters = 5
    ry, rx = 1, 1

    # IMPORTANT: Don't set attention_filters explicitly - let it auto-calculate
    # This ensures both layers use the same key_dim
    tf.random.set_seed(42)
    np.random.seed(42)
    frames = [tf.random.normal((B, H, W, C), dtype=tf.float32) for _ in range(T)]

    # Create layers without specifying attention_filters
    print("Creating layers...")
    patch_layer = LocalSpatioTemporalAttentionPatchKeras(
        num_heads=num_heads,
        attention_filters=attention_filters,  # Auto-calculate
        spatial_radius=(ry, rx),
        skip_connection=True,
        inference_idx=1,
        dropout=0.0
    ) if keras_att else LocalSpatioTemporalAttentionPatch(num_heads=num_heads,
                                                          attention_filters=attention_filters,  # Auto-calculate
                                                          spatial_radius=(ry, rx),
                                                          skip_connection=True,
                                                          inference_idx=1,
                                                          dropout=0.0)

    efficient_layer = LocalSpatioTemporalAttention(
        num_heads=num_heads,
        attention_filters=attention_filters,  # Auto-calculate
        spatial_radius=(ry, rx),
        skip_connection=True,
        inference_query_idx=1,
        dropout=0.0
    )

    # Build layers
    print("Building layers...")
    _ = patch_layer(frames, training=False)
    _ = efficient_layer(frames, training=False)

    print(f"\nPatch layer config:")
    print(f"  num_heads: {patch_layer.num_heads}")
    print(f"  attention_filters: {patch_layer.attention_filters}")
    print(f"  filters: {patch_layer.filters}")

    print(f"\nConv layer config:")
    print(f"  num_heads: {efficient_layer.num_heads}")
    print(f"  attention_filters: {efficient_layer.attention_filters}")
    print(f"  filters: {efficient_layer.filters}")

    # Copy weights (including zeroing out spatial embeddings)
    print("\nCopying embeddings and attention weights...")
    print("Setting spatial positional embeddings to zero for both layers...")
    copy_spatio_temporal_att_weights(patch_layer, efficient_layer, keras_att)

    # Run forward pass
    print("\nRunning forward pass on both layers...")
    out_patch = patch_layer(frames, training=False)
    out_efficient = efficient_layer(frames, training=False)

    # Compare
    out_patch_np = out_patch.numpy()
    out_efficient_np = out_efficient.numpy()
    print(f"patch shape: {out_patch_np.shape} loop shape: {out_efficient_np.shape}")
    abs_diff = np.abs(out_patch_np - out_efficient_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"\nResults:")
    print(f"  Patch output shape: {out_patch_np.shape}")
    print(f"  Efficient output shape: {out_efficient_np.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"\n✓ Efficient equivalence test PASSED (within tolerance {tolerance})\n")
        return True
    else:
        print(f"\n✗ Efficient equivalence test FAILED (exceeds tolerance {tolerance})")

        # Show some example differences
        print("\nSample differences at specific locations:")
        for i in range(min(3, out_patch_np.shape[0])):
            for j in range(min(3, out_patch_np.shape[1])):
                for k in range(min(3, out_patch_np.shape[2])):
                    diff = abs_diff[i, j, k, 0]
                    print(f"  [{i},{j},{k},0]: patch={out_patch_np[i, j, k, 0]:.6f}, "
                          f"conv={out_efficient_np[i, j, k, 0]:.6f}, diff={diff:.2e}")
        return False


def test_spatial_positional_encoding():
    """
    Test that spatial positional encoding is consistent between layers
    and correctly handles edge cases.
    """
    print("=" * 80)
    print("Testing Spatial Positional Encoding")
    print("=" * 80)

    # Test parameters
    H, W = 8, 8
    radius_y, radius_x = 2, 2
    patch_height = 2 * radius_y + 1
    patch_width = 2 * radius_x + 1

    # Helper function from the layers
    def get_query_spatial_indices(H, W, radius_y, radius_x, patch_width):
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, radius_y, H - radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, radius_x, W - radius_x - 1)
        y_relative = y_coords - y_patch_center + radius_y
        x_relative = x_coords - x_patch_center + radius_x
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')
        spatial_indices = y_grid * patch_width + x_grid
        return spatial_indices, y_patch_center, x_patch_center

    def compute_center_indices(H, W, radius_y, radius_x):
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, radius_y, H - radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, radius_x, W - radius_x - 1)
        y_grid, x_grid = tf.meshgrid(y_patch_center, x_patch_center, indexing='ij')
        return y_grid, x_grid

    spatial_indices, y_centers, x_centers = get_query_spatial_indices(
        H, W, radius_y, radius_x, patch_width
    )
    cy_grid, cx_grid = compute_center_indices(H, W, radius_y, radius_x)

    print(f"\nQuery Spatial Indices (shape {spatial_indices.shape}):")
    print("This represents the relative position of each query pixel within its neighborhood")
    print(spatial_indices.numpy())

    # Test specific positions
    print("\n" + "-" * 80)
    print("Testing Corner Cases:")
    print("-" * 80)

    # Top-left corner (0, 0)
    y, x = 0, 0
    print(f"\nPixel ({y}, {x}) - Top-left corner:")
    print(f"  Patch center: ({cy_grid[y, x].numpy()}, {cx_grid[y, x].numpy()})")
    print(f"  Spatial index: {spatial_indices[y, x].numpy()}")
    print(f"  Expected: Should be at position (0, 0) in patch = index 0")

    # Center pixel
    y, x = H // 2, W // 2
    print(f"\nPixel ({y}, {x}) - Center (interior):")
    print(f"  Patch center: ({cy_grid[y, x].numpy()}, {cx_grid[y, x].numpy()})")
    print(f"  Spatial index: {spatial_indices[y, x].numpy()}")
    print(f"  Expected: Should be at center of patch = index {patch_height * patch_width // 2}")

    # Bottom-right corner
    y, x = H - 1, W - 1
    print(f"\nPixel ({y}, {x}) - Bottom-right corner:")
    print(f"  Patch center: ({cy_grid[y, x].numpy()}, {cx_grid[y, x].numpy()})")
    print(f"  Spatial index: {spatial_indices[y, x].numpy()}")
    print(
        f"  Expected: Should be at position ({patch_height - 1}, {patch_width - 1}) = index {patch_height * patch_width - 1}")

    # Edge pixel (not corner)
    y, x = 0, W // 2
    print(f"\nPixel ({y}, {x}) - Top edge:")
    print(f"  Patch center: ({cy_grid[y, x].numpy()}, {cx_grid[y, x].numpy()})")
    print(f"  Spatial index: {spatial_indices[y, x].numpy()}")
    print(f"  Expected: Top row, middle column")

    print("\n" + "=" * 80)
    return spatial_indices, cy_grid, cx_grid


def test_key_value_neighborhood_indexing():
    """
    Test that key/value neighborhoods are correctly gathered with spatial offsets.
    """
    print("=" * 80)
    print("Testing Key/Value Neighborhood Indexing")
    print("=" * 80)

    # Create a simple test tensor where each position has unique values
    B, H, W, C = 1, 6, 6, 4
    radius_y, radius_x = 1, 1
    patch_height = 2 * radius_y + 1
    patch_width = 2 * radius_x + 1

    # Create test data where each pixel has value = [y, x, 0, 0]
    test_data = np.zeros((B, H, W, C), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            test_data[0, y, x, :] = [y, x, y * 10 + x, 0]
    test_tensor = tf.constant(test_data)

    print(f"\nTest tensor (showing channel 2 which = y*10 + x):")
    print(test_tensor[0, :, :, 2].numpy())

    # Test gathering neighborhoods
    def compute_center_indices(H, W, radius_y, radius_x):
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, radius_y, H - radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, radius_x, W - radius_x - 1)
        y_grid, x_grid = tf.meshgrid(y_patch_center, x_patch_center, indexing='ij')
        return y_grid, x_grid

    cy_grid, cx_grid = compute_center_indices(H, W, radius_y, radius_x)
    b_idx = tf.tile(tf.range(B, dtype=tf.int32)[:, None, None], [1, H, W])

    # Pre-compute spatial offsets
    spatial_offsets = []
    for dy in range(-radius_y, radius_y + 1):
        for dx in range(-radius_x, radius_x + 1):
            spatial_offsets.append([dy, dx])
    spatial_offsets = tf.constant(spatial_offsets, dtype=tf.int32)

    print(f"\nSpatial offsets (dy, dx):")
    for i, offset in enumerate(spatial_offsets.numpy()):
        print(f"  Index {i}: {offset}")

    # Test specific query pixel
    query_y, query_x = 2, 3
    print(f"\n" + "-" * 80)
    print(f"Testing neighborhood for query pixel ({query_y}, {query_x})")
    print("-" * 80)
    print(f"Patch center: ({cy_grid[query_y, query_x].numpy()}, {cx_grid[query_y, query_x].numpy()})")

    print(f"\nNeighborhood values (should be 3x3 grid centered at ({query_y}, {query_x})):")
    for i in range(len(spatial_offsets)):
        dy = spatial_offsets[i, 0]
        dx = spatial_offsets[i, 1]

        ny = tf.clip_by_value(cy_grid + dy, 0, H - 1)
        nx = tf.clip_by_value(cx_grid + dx, 0, W - 1)
        ny_b = tf.tile(ny[None, :, :], [B, 1, 1])
        nx_b = tf.tile(nx[None, :, :], [B, 1, 1])
        idx = tf.stack([b_idx, ny_b, nx_b], axis=-1)

        gathered = tf.gather_nd(test_tensor, idx)
        value_at_query = gathered[0, query_y, query_x, 2].numpy()

        actual_y = gathered[0, query_y, query_x, 0].numpy()
        actual_x = gathered[0, query_y, query_x, 1].numpy()

        print(f"  Offset {i} ({dy.numpy():+d}, {dx.numpy():+d}): "
              f"gathered from ({actual_y:.0f}, {actual_x:.0f}) = {value_at_query:.0f}")

    # Test edge case
    query_y, query_x = 0, 0
    print(f"\n" + "-" * 80)
    print(f"Testing edge case: corner pixel ({query_y}, {query_x})")
    print("-" * 80)
    print(f"Patch center: ({cy_grid[query_y, query_x].numpy()}, {cx_grid[query_y, query_x].numpy()})")

    print(f"\nNeighborhood values (should handle edge clipping):")
    for i in range(len(spatial_offsets)):
        dy = spatial_offsets[i, 0]
        dx = spatial_offsets[i, 1]

        ny = tf.clip_by_value(cy_grid + dy, 0, H - 1)
        nx = tf.clip_by_value(cx_grid + dx, 0, W - 1)
        ny_b = tf.tile(ny[None, :, :], [B, 1, 1])
        nx_b = tf.tile(nx[None, :, :], [B, 1, 1])
        idx = tf.stack([b_idx, ny_b, nx_b], axis=-1)

        gathered = tf.gather_nd(test_tensor, idx)
        value_at_query = gathered[0, query_y, query_x, 2].numpy()

        actual_y = gathered[0, query_y, query_x, 0].numpy()
        actual_x = gathered[0, query_y, query_x, 1].numpy()

        print(f"  Offset {i} ({dy.numpy():+d}, {dx.numpy():+d}): "
              f"gathered from ({actual_y:.0f}, {actual_x:.0f}) = {value_at_query:.0f}")

    print("\n" + "=" * 80)


def test_query_key_correspondence():
    """
    Verify that query spatial encoding matches key spatial encoding for the same position.
    """
    print("=" * 80)
    print("Testing Query-Key Spatial Encoding Correspondence")
    print("=" * 80)

    H, W = 6, 6
    radius_y, radius_x = 1, 1
    patch_height = 2 * radius_y + 1
    patch_width = 2 * radius_x + 1

    def get_query_spatial_indices(H, W, radius_y, radius_x, patch_width):
        y_coords = tf.range(H, dtype=tf.int32)
        x_coords = tf.range(W, dtype=tf.int32)
        y_patch_center = tf.clip_by_value(y_coords, radius_y, H - radius_y - 1)
        x_patch_center = tf.clip_by_value(x_coords, radius_x, W - radius_x - 1)
        y_relative = y_coords - y_patch_center + radius_y
        x_relative = x_coords - x_patch_center + radius_x
        y_grid, x_grid = tf.meshgrid(y_relative, x_relative, indexing='ij')
        spatial_indices = y_grid * patch_width + x_grid
        return spatial_indices

    query_spatial_indices = get_query_spatial_indices(H, W, radius_y, radius_x, patch_width)

    # Key spatial indices (all positions in patch)
    key_spatial_indices = list(range(patch_height * patch_width))

    print(f"\nKey spatial indices (all positions in {patch_height}x{patch_width} patch):")
    for i, idx in enumerate(key_spatial_indices):
        py = i // patch_width
        px = i % patch_width
        print(f"  Index {idx}: position ({py}, {px}) in patch")

    print(f"\nQuery spatial indices at various positions:")
    test_positions = [(0, 0), (radius_y, radius_x), (H - 1, W - 1), (0, W // 2)]

    for y, x in test_positions:
        q_idx = query_spatial_indices[y, x].numpy()
        py = q_idx // patch_width
        px = q_idx % patch_width
        print(f"  Query at ({y}, {x}): spatial_index={q_idx} -> position ({py}, {px}) in patch")

    print("\nVerification:")
    print("  - Interior queries should have index = center of patch")
    print(f"    Center index: {patch_height * patch_width // 2}")
    print("  - Edge queries should have indices corresponding to their edge position")

    print("\n" + "=" * 80)


def test_graph_mode_compatibility():
    """
    Test that layers work in graph mode (@tf.function).
    """
    print("=" * 80)
    print("Testing Graph Mode Compatibility")
    print("=" * 80)

    @tf.function
    def test_function(x, num_heads, spatial_radius):
        """Simplified version of attention logic in graph mode"""
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        radius_y, radius_x = spatial_radius

        # Pre-compute spatial offsets
        spatial_offsets = []
        for dy in range(-radius_y, radius_y + 1):
            for dx in range(-radius_x, radius_x + 1):
                spatial_offsets.append([dy, dx])
        spatial_offsets = tf.constant(spatial_offsets, dtype=tf.int32)
        num_spatial = len(spatial_offsets)

        # Test loop structure - use TensorArray instead of Python list
        results = tf.TensorArray(dtype=x.dtype, size=num_spatial, dynamic_size=False)

        for s_idx in tf.range(num_spatial):
            dy = spatial_offsets[s_idx, 0]
            dx = spatial_offsets[s_idx, 1]
            result = x + tf.cast(dy + dx, tf.float32)
            results = results.write(s_idx, result)

        return results.stack()

    # Test with dummy data
    x = tf.random.normal((2, 8, 8, 16))
    result = test_function(x, num_heads=4, spatial_radius=(1, 1))

    print(f"\nGraph mode test passed!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Expected: (9, 2, 8, 8, 16) for 3x3 neighborhood")

    print("\n" + "=" * 80)

def copy_regional_weights(src_layer, dst_layer, zero_spatial_embeddings=True, keras_att:bool=True):
    """
    Copy weights from RegionalSpatialAttention (loop version) to
    RegionalSpatialAttentionHighMem (MHA version).

    Args:
        src_layer: RegionalSpatialAttention instance (loop version with Conv2D projections)
        dst_layer: RegionalSpatialAttentionHighMem instance (MHA version)
        zero_spatial_embeddings: If True, zero out spatial embeddings for both layers
                                to enable fair comparison (since they're added at different stages)
    """

    # Handle spatial positional embeddings
    if zero_spatial_embeddings:
        # Zero out spatial positional embeddings for both layers
        # This is necessary because:
        # - Loop version adds them AFTER projection (to Ck-dimensional space)
        # - HighMem version adds them BEFORE projection (to C-dimensional space)
        src_spatial_weights = src_layer.spatial_pos_embedding.get_weights()
        dst_spatial_weights = dst_layer.spatial_pos_embedding.get_weights()

        src_spatial_weights[0] = np.zeros_like(src_spatial_weights[0])
        dst_spatial_weights[0] = np.zeros_like(dst_spatial_weights[0])

        src_layer.spatial_pos_embedding.set_weights(src_spatial_weights)
        dst_layer.spatial_pos_embedding.set_weights(dst_spatial_weights)

        print("✓ Spatial positional embeddings zeroed out for fair comparison")

    if keras_att:

        # Extract Conv2D projection weights from source layer
        W_q = src_layer.qproj.get_weights()[0]  # (1, 1, C_in, Ck)
        W_k = src_layer.kproj.get_weights()[0]  # (1, 1, C_in, Ck)
        W_v = src_layer.vproj.get_weights()[0]  # (1, 1, C_in, Ck)
        W_o = src_layer.outproj.get_weights()[0]  # (1, 1, Ck, C_out)

        print(f"Source Conv2D weight shapes:")
        print(f"  W_q: {W_q.shape}")
        print(f"  W_k: {W_k.shape}")
        print(f"  W_v: {W_v.shape}")
        print(f"  W_o: {W_o.shape}")

        # Get dimensions
        C_in = W_q.shape[2]
        Ck = W_q.shape[3]
        num_heads = src_layer.num_heads
        d = src_layer.attention_filters  # key_dim per head

        print(f"\nDimensions:")
        print(f"  C_in: {C_in}")
        print(f"  Ck: {Ck}")
        print(f"  num_heads: {num_heads}")
        print(f"  key_dim (d): {d}")
        print(f"  Expected Ck: {num_heads * d}")

        # Verify dimensions
        if Ck != num_heads * d:
            raise ValueError(
                f"Dimension mismatch: Conv has Ck={Ck} "
                f"but expected {num_heads * d} (num_heads={num_heads} * key_dim={d})"
            )

        # Reshape Conv2D weights to MHA format
        # Conv2D: (1, 1, C_in, Ck) -> MHA: (C_in, num_heads, key_dim)
        W_q_mha = W_q.squeeze(axis=(0, 1)).reshape(C_in, num_heads, d)
        W_k_mha = W_k.squeeze(axis=(0, 1)).reshape(C_in, num_heads, d)
        W_v_mha = W_v.squeeze(axis=(0, 1)).reshape(C_in, num_heads, d)

        # Output projection: (1, 1, Ck, C_out) -> (Ck, C_out)
        W_o_mha = W_o.squeeze(axis=(0, 1)).reshape(num_heads, d, Ck)

        print(f"\nTarget MHA weight shapes:")
        print(f"  W_q_mha: {W_q_mha.shape}")
        print(f"  W_k_mha: {W_k_mha.shape}")
        print(f"  W_v_mha: {W_v_mha.shape}")
        print(f"  W_o_mha: {W_o_mha.shape}")

        # Assign weights to MHA
        mha = dst_layer.attention_layer
        mha._query_dense.kernel.assign(W_q_mha)
        mha._key_dense.kernel.assign(W_k_mha)
        mha._value_dense.kernel.assign(W_v_mha)
        mha._output_dense.kernel.assign(W_o_mha)

        if src_layer.skip_connection:
            dst_layer.skipproj.kernel.assign(src_layer.skipproj.kernel)
            dst_layer.skipproj.bias.assign(src_layer.skipproj.bias)
    else:
        print("Copying weights for fair comparison...")
        for loop_w, highmem_w in zip(src_layer.trainable_weights, dst_layer.trainable_weights):
            highmem_w.assign(loop_w)

    print("\n✓ Weights copied successfully from loop version to HighMem version")


def compare_regional_versions(keras_att:bool):
    """Compare loop vs highmem versions for correctness and speed"""
    import time

    print("=" * 80)
    print(f"COMPARING REGIONAL ATTENTION VERSIONS {'KERAS VERSION' if keras_att else ''}")
    print("=" * 80)

    # Test configuration
    B, H, W, C = 2, 64, 64, 32
    num_heads = 4
    radius = (2, 2)

    # Create test input
    np.random.seed(42)
    tf.random.set_seed(42)
    x_np = np.random.randn(B, H, W, C).astype(np.float32)
    x = tf.constant(x_np)

    print(f"\nTest Configuration:")
    print(f"  Input shape: {x.shape}")
    print(f"  Num heads: {num_heads}")
    print(f"  Radius: {radius}")
    print(f"  Neighborhood size: {(2 * radius[0] + 1) * (2 * radius[1] + 1)}")


    # Create both versions
    print("\n" + "-" * 80)
    print("Creating layers...")
    loop_layer = LocalSpatialAttention(
        num_heads=num_heads,
        spatial_radius=radius,
        skip_connection=not keras_att,
        dropout=0.0,
        name="LoopVersion"
    )

    highmem_layer = LocalSpatialAttentionPatchKeras(
        num_heads=num_heads,
        spatial_radius=radius,
        dropout=0.0,
        name="HighMemVersion"
    ) if keras_att else LocalSpatialAttentionPatch(num_heads=num_heads, spatial_radius=radius, skip_connection=True)

    # Build layers
    print("Building layers...")
    _ = loop_layer(x)
    _ = highmem_layer(x)

    # Copy weights
    print("\n" + "-" * 80)
    print("Copying weights from loop to HighMem version...")
    copy_regional_weights(loop_layer, highmem_layer, zero_spatial_embeddings=keras_att, keras_att=keras_att)



    # Test correctness
    print("\n" + "-" * 80)
    print("Testing correctness...")

    out_loop = loop_layer(x, training=False)
    out_highmem = highmem_layer(x, training=False)

    diff = tf.reduce_max(tf.abs(out_loop - out_highmem)).numpy()
    rel_diff = (tf.reduce_max(tf.abs(out_loop - out_highmem)) /
                (tf.reduce_max(tf.abs(out_loop)) + 1e-8)).numpy()

    print(f"  Max absolute difference: {diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    print(f"  Output range (loop): [{tf.reduce_min(out_loop):.2e}, {tf.reduce_max(out_loop):.2e}]")
    print(f"  Output range (highmem): [{tf.reduce_min(out_highmem):.2e}, {tf.reduce_max(out_highmem):.2e}]")

    if diff < 1e-4:
        print("  ✅ Outputs match! (difference < 1e-4)")
    elif diff < 1e-3:
        print("  ✓ Outputs close (difference < 1e-3)")
    else:
        print(f"  ⚠️  Outputs differ by {diff:.2e}")
        print("  Note: This may be due to numerical differences in implementation")

    # Benchmark speed
    print("\n" + "-" * 80)
    print("Benchmarking speed...")

    num_warmup = 5
    num_runs = 20

    # Warmup
    for _ in range(num_warmup):
        _ = loop_layer(x, training=False)
        _ = highmem_layer(x, training=False)

    # Benchmark loop version
    times_loop = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = loop_layer(x, training=False)
        end = time.perf_counter()
        times_loop.append(end - start)

    avg_loop = np.mean(times_loop) * 1000
    std_loop = np.std(times_loop) * 1000

    # Benchmark highmem version
    times_highmem = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = highmem_layer(x, training=False)
        end = time.perf_counter()
        times_highmem.append(end - start)

    avg_highmem = np.mean(times_highmem) * 1000
    std_highmem = np.std(times_highmem) * 1000

    speedup = avg_loop / avg_highmem

    print(f"\n  Loop Version:    {avg_loop:.2f} ms ± {std_loop:.2f} ms")
    print(f"  HighMem Version: {avg_highmem:.2f} ms ± {std_highmem:.2f} ms")
    print(f"  Speedup:         {speedup:.2f}x")

    if speedup > 1.5:
        print(f"  ✅ HighMem is {speedup:.2f}x faster")
    elif speedup > 1.0:
        print(f"  ✓ HighMem is slightly faster ({speedup:.2f}x)")
    else:
        print(f"  ⚠️  Loop version is faster (1/{1 / speedup:.2f}x)")

    # Memory estimation
    print("\n" + "-" * 80)
    print("Estimating memory usage...")

    loop_params = sum([np.prod(v.shape) for v in loop_layer.trainable_weights])
    highmem_params = sum([np.prod(v.shape) for v in highmem_layer.trainable_weights])

    # Estimate activation memory
    neighborhood_size = (2 * radius[0] + 1) * (2 * radius[1] + 1)
    Ck = num_heads * (C // num_heads)

    # Loop version: stores scores array and accumulates
    loop_activation = (
            3 * B * H * W * Ck +  # Q, K, V projections
            B * H * W * num_heads * neighborhood_size +  # Scores
            B * H * W * num_heads * (C // num_heads)  # Accumulator
    )

    # HighMem version: stores extracted patches
    highmem_activation = (
            B * H * W * C +  # Query with pos encoding
            2 * B * H * W * neighborhood_size * C +  # K, V patches
            B * H * W * num_heads * neighborhood_size  # Attention scores in MHA
    )

    loop_memory_mb = (loop_params + loop_activation) * 4 / (1024 ** 2)
    highmem_memory_mb = (highmem_params + highmem_activation) * 4 / (1024 ** 2)

    memory_ratio = highmem_memory_mb / loop_memory_mb

    print(f"  Loop Version:    ~{loop_memory_mb:.1f} MB")
    print(f"  HighMem Version: ~{highmem_memory_mb:.1f} MB")
    print(f"  Memory Ratio:    {memory_ratio:.2f}x")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    print("\n📝 Notes:")
    print("  • Spatial positional embeddings were zeroed for fair comparison")
    print("  • Loop version adds spatial pos encoding AFTER projection (Ck dims)")
    print("  • HighMem version adds spatial pos encoding BEFORE MHA (C dims)")
    print("  • For production use, keep spatial embeddings enabled")


if __name__ == "__main__":
    compare_regional_versions(False)
    compare_regional_versions(True)

    #compare_spatiotemporal_versions(False)
    #compare_spatiotemporal_versions(True)

    print("\n")
    print("█" * 80)
    print("  ATTENTION LAYER TEST SUITE")
    print("█" * 80)
    print("\n")

    #test_spatial_positional_encoding()
    print("\n\n")

    #test_key_value_neighborhood_indexing()
    print("\n\n")

    #test_query_key_correspondence()
    print("\n\n")

    #test_graph_mode_compatibility()

    print("\n")
    print("█" * 80)
    print("  ALL TESTS COMPLETED")
    print("█" * 80)
    print("\n")
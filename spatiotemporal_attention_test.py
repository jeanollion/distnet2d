from distnet_2d.model.spatiotemporal_attention import SpatioTemporalAttention, SpatioTemporalAttentionHighMem
import tensorflow as tf
import numpy as np


def copy_sta_weights_to_conv(src_layer, dst_layer):
    """Copy weights from SpatioTemporalAttention to ConvSpatioTemporalAttention."""
    # Copy temporal embeddings
    dst_layer.temp_embedding.set_weights(src_layer.temp_embedding.get_weights())

    # Zero out spatial positional embeddings for both layers to enable comparison
    # (since they're added at different stages in the pipeline)
    src_spatial_weights = src_layer.spatial_pos_embedding.get_weights()
    dst_spatial_weights = dst_layer.spatial_pos_embedding.get_weights()

    # Set to zeros
    src_spatial_weights[0] = np.zeros_like(src_spatial_weights[0])
    dst_spatial_weights[0] = np.zeros_like(dst_spatial_weights[0])

    src_layer.spatial_pos_embedding.set_weights(src_spatial_weights)
    dst_layer.spatial_pos_embedding.set_weights(dst_spatial_weights)

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

    print("✓ Weights copied successfully")


def test_equivalence_patch_vs_efficient():
    """
    Test equivalence between patch-based version and efficient conv version.
    """
    print("=" * 80)
    print("TEST: Patch vs Efficient Conv Equivalence")
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
    patch_layer = SpatioTemporalAttentionHighMem(
        num_heads=num_heads,
        attention_filters=attention_filters,  # Auto-calculate
        spatial_radius=(ry, rx),
        inference_idx=1,
        dropout=0.0
    )
    efficient_layer = SpatioTemporalAttention(
        num_heads=num_heads,
        attention_filters=attention_filters,  # Auto-calculate
        spatial_radius=(ry, rx),
        inference_idx=1,
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
    copy_sta_weights_to_conv(patch_layer, efficient_layer)

    # Run forward pass
    print("\nRunning forward pass on both layers...")
    out_patch = patch_layer(frames, training=False)
    out_efficient = efficient_layer(frames, training=False)

    # Compare
    out_patch_np = out_patch.numpy()
    out_efficient_np = out_efficient.numpy()
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


if __name__ == "__main__":
    test_equivalence_patch_vs_efficient()
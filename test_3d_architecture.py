import unittest
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from distnet_2d.model.layers import (
    NConvToBatch, ResConv, ConvBNDrop, ConvTransposeBNDrop, UpSamplingWithDtype
)
from distnet_2d.model.architectures import (
    get_kernels_and_dilation, get_downsampling_factor, spatial_contraction_product
)
from distnet_2d.model.spatial_attention import SpatialAttention
from distnet_2d.model.temporal_pyramid import TemporalPyramid


B, Z, Y, X, C = 2, 4, 8, 8, 16


class TestNConvToBatch(unittest.TestCase):
    def _run(self, input_shape):
        layer = NConvToBatch(n_conv=3, inference_idx=0, filters=8, l2_reg=0)
        x = tf.random.normal(input_shape)
        out = layer(x)
        expected = list(input_shape)
        expected[0] *= 3
        expected[-1] = 8
        self.assertEqual(out.shape.as_list(), expected)

    def test_2d(self):
        self._run([B, Y, X, C])

    def test_3d(self):
        self._run([B, Z, Y, X, C])

    def test_3d_uses_conv3d(self):
        layer = NConvToBatch(n_conv=2, inference_idx=0, filters=4, l2_reg=0)
        layer(tf.random.normal([B, Z, Y, X, C]))
        self.assertIsInstance(layer.convs[0], tf.keras.layers.Conv3D)

    def test_2d_uses_conv2d(self):
        layer = NConvToBatch(n_conv=2, inference_idx=0, filters=4, l2_reg=0)
        layer(tf.random.normal([B, Y, X, C]))
        self.assertIsInstance(layer.convs[0], tf.keras.layers.Conv2D)


class TestResConv(unittest.TestCase):
    def _run(self, input_shape):
        layer = ResConv(kernel_size=3)
        x = tf.random.normal(input_shape)
        out = layer(x)
        self.assertEqual(out.shape.as_list(), list(input_shape))

    def test_2d(self):
        self._run([B, Y, X, C])

    def test_3d(self):
        self._run([B, Z, Y, X, C])

    def test_3d_uses_conv3d(self):
        layer = ResConv(kernel_size=3)
        layer(tf.random.normal([B, Z, Y, X, C]))
        self.assertIsInstance(layer.conv1, tf.keras.layers.Conv3D)

    def test_2d_uses_conv2d(self):
        layer = ResConv(kernel_size=3)
        layer(tf.random.normal([B, Y, X, C]))
        self.assertIsInstance(layer.conv1, tf.keras.layers.Conv2D)

    def test_with_batchnorm(self):
        layer = ResConv(kernel_size=3, batch_norm=True)
        x = tf.random.normal([B, Z, Y, X, C])
        out = layer(x, training=True)
        self.assertEqual(out.shape.as_list(), [B, Z, Y, X, C])

    def test_with_dilation(self):
        layer = ResConv(kernel_size=3, dilation=2)
        x = tf.random.normal([B, Z, Y, X, C])
        out = layer(x)
        self.assertEqual(out.shape.as_list(), [B, Z, Y, X, C])


class TestConvBNDrop(unittest.TestCase):
    def _run(self, input_shape, filters=8, strides=1):
        layer = ConvBNDrop(filters=filters, kernel_size=3, strides=strides)
        x = tf.random.normal(input_shape)
        out = layer(x)
        expected = list(input_shape)
        expected[-1] = filters
        for i in range(1, len(input_shape) - 1):
            expected[i] = (expected[i] + strides - 1) // strides
        self.assertEqual(out.shape.as_list(), expected)

    def test_2d(self):
        self._run([B, Y, X, C])

    def test_3d(self):
        self._run([B, Z, Y, X, C])

    def test_3d_stride2(self):
        self._run([B, Z, Y, X, C], strides=2)

    def test_3d_uses_conv3d(self):
        layer = ConvBNDrop(filters=8, kernel_size=3)
        layer(tf.random.normal([B, Z, Y, X, C]))
        self.assertIsInstance(layer.conv, tf.keras.layers.Conv3D)

    def test_2d_uses_conv2d(self):
        layer = ConvBNDrop(filters=8, kernel_size=3)
        layer(tf.random.normal([B, Y, X, C]))
        self.assertIsInstance(layer.conv, tf.keras.layers.Conv2D)


class TestConvTransposeBNDrop(unittest.TestCase):
    def _run(self, input_shape, filters=8, strides=2):
        layer = ConvTransposeBNDrop(filters=filters, kernel_size=4, strides=strides)
        x = tf.random.normal(input_shape)
        out = layer(x)
        expected = list(input_shape)
        expected[-1] = filters
        for i in range(1, len(input_shape) - 1):
            expected[i] *= strides
        self.assertEqual(out.shape.as_list(), expected)

    def test_2d(self):
        self._run([B, Y, X, C])

    def test_3d(self):
        self._run([B, Z, Y, X, C])

    def test_3d_uses_conv3d_transpose(self):
        layer = ConvTransposeBNDrop(filters=8, kernel_size=4, strides=2)
        layer(tf.random.normal([B, Z, Y, X, C]))
        self.assertIsInstance(layer.conv, tf.keras.layers.Conv3DTranspose)

    def test_2d_uses_conv2d_transpose(self):
        layer = ConvTransposeBNDrop(filters=8, kernel_size=4, strides=2)
        layer(tf.random.normal([B, Y, X, C]))
        self.assertIsInstance(layer.conv, tf.keras.layers.Conv2DTranspose)


class TestUpSamplingWithDtype(unittest.TestCase):
    def test_2d(self):
        layer = UpSamplingWithDtype(size=2, interpolation="nearest")
        x = tf.random.normal([B, Y, X, C])
        out = layer(x)
        self.assertEqual(out.shape.as_list(), [B, Y*2, X*2, C])

    def test_3d(self):
        layer = UpSamplingWithDtype(size=2, interpolation="nearest")
        x = tf.random.normal([B, Z, Y, X, C])
        out = layer(x)
        self.assertEqual(out.shape.as_list(), [B, Z*2, Y*2, X*2, C])

    def test_3d_uses_upsampling3d(self):
        layer = UpSamplingWithDtype(size=2, interpolation="nearest")
        layer(tf.random.normal([B, Z, Y, X, C]))
        self.assertIsInstance(layer.up_op, tf.keras.layers.UpSampling3D)

    def test_2d_uses_upsampling2d(self):
        layer = UpSamplingWithDtype(size=2, interpolation="nearest")
        layer(tf.random.normal([B, Y, X, C]))
        self.assertIsInstance(layer.up_op, tf.keras.layers.UpSampling2D)


class TestGetKernelsAndDilation(unittest.TestCase):
    def test_2d_no_cap(self):
        ker, dil = get_kernels_and_dilation(3, 1, [64, 64], 1, tridimensional_mode=False)
        self.assertEqual(ker, 3)
        self.assertEqual(dil, 1)

    def test_2d_large_kernel(self):
        ker, dil = get_kernels_and_dilation(5, 2, [64, 64], 1, tridimensional_mode=False)
        self.assertEqual(ker, 5)
        self.assertEqual(dil, 2)

    def test_2d_none_dims(self):
        ker, dil = get_kernels_and_dilation(5, 2, None, 1, tridimensional_mode=False)
        self.assertEqual(ker, 5)
        self.assertEqual(dil, 2)

    def test_3d_z_kernel_capped(self):
        ker, dil = get_kernels_and_dilation(5, 2, [8, 64, 64], 1, tridimensional_mode=True)
        self.assertIsInstance(ker, list)
        self.assertEqual(len(ker), 3)
        self.assertLessEqual(ker[0], 3)  # Z kernel capped at 3
        self.assertEqual(dil[0], 1)      # Z dilation always 1

    def test_3d_yx_match_target(self):
        ker, dil = get_kernels_and_dilation(5, 1, [8, 64, 64], 1, tridimensional_mode=True)
        self.assertEqual(ker[1], 5)
        self.assertEqual(ker[2], 5)

    def test_none_dimensions(self):
        ker, dil = get_kernels_and_dilation(5, 2, None, 1, tridimensional_mode=True)
        self.assertEqual(ker, 5)
        self.assertEqual(dil, 2)


class TestGetDownsamplingFactor(unittest.TestCase):
    def test_2d(self):
        ds = get_downsampling_factor(2, [64, 64], 1, tridimensional_mode=False)
        self.assertEqual(ds, 2)

    def test_3d(self):
        ds = get_downsampling_factor(2, [8, 64, 64], 1, tridimensional_mode=True)
        self.assertIsInstance(ds, list)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[1], 2)
        self.assertEqual(ds[2], 2)

    def test_3d_small_z(self):
        ds = get_downsampling_factor(2, [2, 64, 64], 1, tridimensional_mode=True)
        self.assertLessEqual(ds[0], 2)  # Z should be reduced if too small

    def test_none_dimensions(self):
        ds = get_downsampling_factor(2, None, 1, tridimensional_mode=True)
        self.assertEqual(ds, 2)


class TestSpatialContractionProduct(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(spatial_contraction_product(2, 2), 4)

    def test_list(self):
        result = spatial_contraction_product([1, 2, 2], [1, 2, 2])
        self.assertEqual(result, [1, 4, 4])

    def test_mixed(self):
        result = spatial_contraction_product(2, [1, 2, 2])
        self.assertEqual(result, [2, 4, 4])


class TestSpatialAttention(unittest.TestCase):
    def test_2d(self):
        shape = [Y, X, C]
        layer = SpatialAttention(num_heads=2, positional_encoding="1d", attention_filters=8)
        inp1 = tf.keras.layers.Input(shape)
        inp2 = tf.keras.layers.Input(shape)
        out = layer([inp1, inp2])
        self.assertEqual(out.shape.as_list()[1:], shape)

    def test_3d(self):
        shape = [Z, Y, X, C]
        layer = SpatialAttention(num_heads=2, positional_encoding="1d", attention_filters=8)
        inp1 = tf.keras.layers.Input(shape)
        inp2 = tf.keras.layers.Input(shape)
        out = layer([inp1, inp2])
        self.assertEqual(out.shape.as_list()[1:], shape)

    def test_3d_attention_axes(self):
        shape = [Z, Y, X, C]
        layer = SpatialAttention(num_heads=2, positional_encoding="1d", attention_filters=8)
        inp1 = tf.keras.layers.Input(shape)
        inp2 = tf.keras.layers.Input(shape)
        layer([inp1, inp2])
        self.assertEqual(layer.attention_layer._attention_axes, (1, 2, 3))

    def test_2d_attention_axes(self):
        shape = [Y, X, C]
        layer = SpatialAttention(num_heads=2, positional_encoding="1d", attention_filters=8)
        inp1 = tf.keras.layers.Input(shape)
        inp2 = tf.keras.layers.Input(shape)
        layer([inp1, inp2])
        self.assertEqual(layer.attention_layer._attention_axes, (1, 2))


class TestTemporalPyramid(unittest.TestCase):
    def _make_wsa_kwargs(self, tridim=False):
        return dict(
            window_size=[2, 4, 4] if tridim else [4, 4],
            num_heads=2,
            attention_filters=8,
        )

    def test_2d(self):
        T = 3
        wsa_kwargs = self._make_wsa_kwargs()
        layer = TemporalPyramid(window_spatial_attention_kwargs=wsa_kwargs)
        x = tf.random.normal([T, B, Y, X, C])
        out_global, out_l1 = layer(x)
        # output is (B, Y, X, C') — T=1 squeezed and merged with B
        self.assertEqual(out_global.shape[0], B)
        self.assertEqual(out_global.shape[1], Y)
        self.assertEqual(out_global.shape[2], X)

    def test_3d(self):
        T = 3
        wsa_kwargs = self._make_wsa_kwargs(tridim=True)
        layer = TemporalPyramid(window_spatial_attention_kwargs=wsa_kwargs)
        x = tf.random.normal([T, B, Z, Y, X, C])
        out_global, out_l1 = layer(x)
        self.assertTrue(layer.tridim_mode)
        # output is (B, Z, Y, X, C')
        self.assertEqual(out_global.shape[0], B)
        self.assertEqual(out_global.shape[1], Z)
        self.assertEqual(out_global.shape[2], Y)
        self.assertEqual(out_global.shape[3], X)

    def test_2d_not_tridim(self):
        T = 3
        wsa_kwargs = self._make_wsa_kwargs()
        layer = TemporalPyramid(window_spatial_attention_kwargs=wsa_kwargs)
        layer(tf.random.normal([T, B, Y, X, C]))
        self.assertFalse(layer.tridim_mode)


class TestFullModelBlend(unittest.TestCase):
    """End-to-end test: build a full DiSTNet2D model with Blend architecture and run prediction."""

    def _build_and_predict(self, spa_dims, tridim, n_downsampling=2):
        from distnet_2d.model import get_distnet_2d
        from distnet_2d.model.architectures import get_architecture
        fw = 2
        arch = get_architecture("blend",
            filters=32,
            n_inputs=1,
            spatial_dimensions=spa_dims,
            frame_window=fw,
            frame_max_distance=fw+1,
            segmentation=True,
            tracking=False,
            frame_aware=False,
            attention=0,
            self_attention=0,
            n_downsampling=n_downsampling,
            early_downsampling=False,
            dropout=0,
        )
        model = get_distnet_2d(arch)
        model.set_inference(True)
        model.compile()
        n_frames = fw * 2 + 1
        input_shape = [1] + list(spa_dims) + [n_frames]
        out = model.predict(tf.zeros(input_shape), verbose=0)
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0)
        for o in out:
            self.assertEqual(o.shape[0], 1)
            if tridim:
                self.assertEqual(len(o.shape), 5)  # B, Z, Y, X, C
            else:
                self.assertEqual(len(o.shape), 4)  # B, Y, X, C
        return out

    def test_2d(self):
        self._build_and_predict(spa_dims=(32, 32), tridim=False)

    def test_3d(self):
        self._build_and_predict(spa_dims=(4, 32, 32), tridim=True)


class TestFullModelTemPy(unittest.TestCase):
    """End-to-end test: build a full DiSTNet2D model with TemPy architecture and run prediction."""

    def _build_and_predict(self, spa_dims, tridim, n_downsampling=2):
        from distnet_2d.model import get_distnet_2d
        from distnet_2d.model.architectures import get_architecture
        fw = 2
        arch = get_architecture("tempy",
            filters=32,
            n_inputs=1,
            spatial_dimensions=spa_dims,
            frame_window=fw,
            segmentation=True,
            tracking=False,
            window_attention=4,
            attention_spatial_radius=8,
            n_downsampling=n_downsampling,
            early_downsampling=True,
            dropout=0,
        )
        model = get_distnet_2d(arch)
        model.set_inference(True)
        model.compile()
        n_frames = fw * 2 + 1
        input_shape = [1] + list(spa_dims) + [n_frames]
        image_input = tf.zeros(input_shape)
        fi_shape = [1] + [1]*len(spa_dims) + [n_frames]
        frame_index = tf.reshape(tf.range(n_frames), fi_shape)
        out = model.predict([image_input, frame_index], verbose=0)
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0)
        for o in out:
            self.assertEqual(o.shape[0], 1)
            if tridim:
                self.assertEqual(len(o.shape), 5)
            else:
                self.assertEqual(len(o.shape), 4)
        return out

    def test_2d(self):
        self._build_and_predict(spa_dims=(32, 32), tridim=False)

    def test_3d(self):
        self._build_and_predict(spa_dims=(4, 32, 32), tridim=True)


class TestTrainingStep(unittest.TestCase):
    """Test that train_step works in graph mode for both 2D and 3D, with and without tracking."""

    @staticmethod
    def _build_model(spa_dims, tridim, tracking=True, predict_edm_derivatives=False, return_weight_map=False):
        from distnet_2d.model import get_distnet_2d
        from distnet_2d.model.architectures import get_architecture
        fw = 2
        arch = get_architecture("blend",
            filters=32,
            n_inputs=1,
            spatial_dimensions=spa_dims,
            frame_window=fw,
            frame_max_distance=fw + 1,
            segmentation=True,
            tracking=tracking,
            frame_aware=False,
            attention=0,
            self_attention=0,
            n_downsampling=2,
            early_downsampling=False,
            dropout=0,
            predict_edm_derivatives=predict_edm_derivatives,
        )
        model = get_distnet_2d(arch, return_weight_map=return_weight_map)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
        return model, arch

    @staticmethod
    def _make_data(spa_dims, tridim, tracking=True, predict_edm_derivatives=False, return_weight_map=False):
        fw = 2
        n_frames = fw * 2 + 1
        n_displacement = 3 if tridim else 2
        batch_size = 2
        input_shape = [batch_size] + list(spa_dims) + [n_frames]
        x = [tf.random.normal(input_shape)]

        # EDM target: value (+ derivatives if predict_edm_derivatives)
        n_edm_channels = (n_displacement + 1) * n_frames if predict_edm_derivatives else n_frames
        edm_shape = [batch_size] + list(spa_dims) + [n_edm_channels]
        edm = tf.abs(tf.random.normal(edm_shape))  # EDM should be >= 0

        # CDM target
        cdm_shape = [batch_size] + list(spa_dims) + [n_frames]
        cdm = tf.random.normal(cdm_shape)

        # number of frame pairs
        n_frame_pairs = n_frames - 1 + (fw - 1) * 2  # long_term with future_frames
        # with predict_fw: n_fp_mul = 2
        n_pair_channels = n_frame_pairs * 2  # predict_fw doubles it

        y = [edm, cdm]
        if tracking:
            for _ in range(n_displacement):
                disp_shape = [batch_size] + list(spa_dims) + [n_pair_channels]
                y.append(tf.random.normal(disp_shape))
            # link multiplicity: n_pair_channels channels, values in {1,2,3}
            lm_shape = [batch_size] + list(spa_dims) + [n_pair_channels]
            y.append(tf.ones(lm_shape))
        if return_weight_map:
            wm_shape = [batch_size] + list(spa_dims) + [n_frames]
            y.append(tf.ones(wm_shape))
        return x, y

    def _run_train_step(self, spa_dims, tridim, tracking=True, predict_edm_derivatives=False, return_weight_map=False):
        with tf.device('/CPU:0'):
            model, arch = self._build_model(spa_dims, tridim, tracking=tracking, predict_edm_derivatives=predict_edm_derivatives, return_weight_map=return_weight_map)
            x, y = self._make_data(spa_dims, tridim, tracking=tracking, predict_edm_derivatives=predict_edm_derivatives, return_weight_map=return_weight_map)

            # Run in graph mode via tf.function
            @tf.function
            def train_fn(data):
                return model.train_step(data)

            metrics = train_fn((x, y))
            self.assertIn("loss", metrics)
            if tracking:
                self.assertIn("dY", metrics)
                self.assertIn("dX", metrics)
                if tridim:
                    self.assertIn("dZ", metrics)
            return metrics

    def test_2d_seg_only(self):
        self._run_train_step(spa_dims=(32, 32), tridim=False, tracking=False)

    def test_2d_with_tracking(self):
        self._run_train_step(spa_dims=(32, 32), tridim=False, tracking=True)

    def test_3d_seg_only(self):
        self._run_train_step(spa_dims=(4, 32, 32), tridim=True, tracking=False)

    def test_3d_with_tracking(self):
        self._run_train_step(spa_dims=(4, 32, 32), tridim=True, tracking=True)

    def test_2d_edm_derivatives(self):
        self._run_train_step(spa_dims=(32, 32), tridim=False, tracking=False, predict_edm_derivatives=True)

    def test_2d_weight_map(self):
        self._run_train_step(spa_dims=(32, 32), tridim=False, tracking=True, return_weight_map=True)

    def test_3d_weight_map(self):
        self._run_train_step(spa_dims=(4, 32, 32), tridim=True, tracking=False, return_weight_map=True)

    def test_3d_edm_derivatives(self):
        self._run_train_step(spa_dims=(4, 32, 32), tridim=True, tracking=False, predict_edm_derivatives=True)


class TestIteratorModelCompat(unittest.TestCase):
    """Verify that iterator output shape/order matches model train_step expectations."""

    def _check_compat(self, tridim, tracking, predict_edm_derivatives=False):
        """Build model and mock iterator-like data, verify train_step runs."""
        spa_dims = (4, 32, 32) if tridim else (32, 32)
        model, arch = TestTrainingStep._build_model(spa_dims, tridim, tracking=tracking,
                                                     predict_edm_derivatives=predict_edm_derivatives)
        fw = 2
        n_frames = fw * 2 + 1
        n_displacement = 3 if tridim else 2
        batch_size = 1
        spa = list(spa_dims)

        # Simulate iterator output order: [edm, cdm, (dz), dy, dx, link_mult]
        n_motion = n_frames - 1 + (fw - 1) * 2  # long_term + future_frames
        n_pair_channels = n_motion * 2  # predict_fw

        # EDM: n_frames channels, or (n_der+1)*n_frames if predict_edm_derivatives
        n_edm_chan = (n_displacement + 1) * n_frames if predict_edm_derivatives else n_frames
        edm = np.abs(np.random.randn(batch_size, *spa, n_edm_chan).astype("float32"))
        cdm = np.random.randn(batch_size, *spa, n_frames).astype("float32")

        y = [edm, cdm]
        if tracking:
            if tridim:
                y.append(np.random.randn(batch_size, *spa, n_pair_channels).astype("float32"))  # dZ
            y.append(np.random.randn(batch_size, *spa, n_pair_channels).astype("float32"))  # dY
            y.append(np.random.randn(batch_size, *spa, n_pair_channels).astype("float32"))  # dX
            y.append(np.ones((batch_size, *spa, n_pair_channels), dtype="float32"))  # link mult

        x = [np.random.randn(batch_size, *spa, n_frames).astype("float32")]

        with tf.device('/CPU:0'):
            @tf.function
            def step_fn(data):
                return model.train_step(data)
            metrics = step_fn((x, y))

        self.assertIn("loss", metrics)
        loss_val = metrics["loss"].numpy()
        self.assertFalse(np.isnan(loss_val), "loss is NaN")

    def test_2d_seg_tracking(self):
        self._check_compat(tridim=False, tracking=True)

    def test_3d_seg_tracking(self):
        self._check_compat(tridim=True, tracking=True)

    def test_2d_seg_only(self):
        self._check_compat(tridim=False, tracking=False)

    def test_3d_seg_only(self):
        self._check_compat(tridim=True, tracking=False)

    def test_3d_edm_derivatives(self):
        self._check_compat(tridim=True, tracking=False, predict_edm_derivatives=True)


class TestWeightMap(unittest.TestCase):
    """Test weight map generation and category frequency balancing in the iterator."""

    def test_category_frequency_balancing(self):
        """Verify that category_frequencies produces correct keep probabilities and binary exclusion."""
        from distnet_2d.data.distnet_iterator import DistnetIterator
        from scipy.ndimage import find_objects

        # category_frequencies: cat 0 is 4x more frequent than cat 1
        cat_freq = [0.8, 0.2]
        # expected keep_prob: min_freq / freq = [0.2/0.8, 0.2/0.2] = [0.25, 1.0]
        expected_keep_prob = np.array([0.25, 1.0])

        it = object.__new__(DistnetIterator)
        cat_freq_arr = np.array(cat_freq, dtype=np.float64)
        min_freq = np.min(cat_freq_arr[cat_freq_arr > 0])
        it.category_keep_prob = min_freq / np.maximum(cat_freq_arr, 1e-10)
        np.testing.assert_allclose(it.category_keep_prob, expected_keep_prob, rtol=1e-6)

        # Simulate weight map construction over many trials to verify probabilities
        label = np.zeros((16, 16), dtype=np.int32)
        label[2:6, 2:6] = 1    # object 1: cat 0 (common)
        label[2:6, 10:14] = 2  # object 2: cat 1 (rare)
        label[10:14, 2:6] = 3  # object 3: cat 0 (common)
        object_slices = find_objects(label)
        cat_array = np.array([0, 1, 0])  # categories for objects 1, 2, 3

        n_trials = 2000
        kept_counts = np.zeros(3)  # count how many times each object is kept
        np.random.seed(42)
        for _ in range(n_trials):
            weight_map = np.ones_like(label, dtype=np.float32)
            for obj_idx, sl in enumerate(object_slices):
                if sl is not None:
                    cat = int(cat_array[obj_idx])
                    if np.random.random() >= it.category_keep_prob[cat]:
                        mask = label[sl] == obj_idx + 1
                        weight_map[sl][mask] = 0
            # Check object kept/excluded
            for obj_idx in range(3):
                # Sample a pixel known to be in the object
                if obj_idx == 0:
                    kept_counts[obj_idx] += weight_map[3, 3]
                elif obj_idx == 1:
                    kept_counts[obj_idx] += weight_map[3, 12]
                else:
                    kept_counts[obj_idx] += weight_map[12, 3]

        keep_rates = kept_counts / n_trials
        # Object 2 (cat 1, rare) should be kept ~100% of the time
        self.assertGreater(keep_rates[1], 0.98, f"Rare category keep rate too low: {keep_rates[1]}")
        # Objects 1 and 3 (cat 0, common) should be kept ~25% of the time
        for i in [0, 2]:
            self.assertAlmostEqual(keep_rates[i], 0.25, delta=0.05,
                                   msg=f"Object {i+1} (common cat) keep rate {keep_rates[i]} not close to 0.25")
        # Weight map should always be binary
        # (already ensured by construction: values are 0 or 1)

    def test_weight_map_training_step_2d(self):
        """Verify that training step works in graph mode with weight map enabled."""
        self._run_weight_map_train_step(spa_dims=(32, 32), tridim=False, tracking=True)

    def test_weight_map_training_step_3d(self):
        """Verify that training step works in graph mode with weight map enabled (3D, seg only)."""
        self._run_weight_map_train_step(spa_dims=(4, 32, 32), tridim=True, tracking=False)

    def _run_weight_map_train_step(self, spa_dims, tridim, tracking):
        with tf.device('/CPU:0'):
            model, arch = TestTrainingStep._build_model(spa_dims, tridim, tracking=tracking, return_weight_map=True)
            x, y = TestTrainingStep._make_data(spa_dims, tridim, tracking=tracking, return_weight_map=True)

            @tf.function
            def train_fn(data):
                return model.train_step(data)

            metrics = train_fn((x, y))
            self.assertIn("loss", metrics)
            loss_val = metrics["loss"].numpy()
            self.assertFalse(np.isnan(loss_val), "loss is NaN with weight map")


    def test_cell_line_exclusion_tracking(self):
        """Verify that excluding an object in the central frame excludes the whole cell line across all frames."""
        # Setup: 3 frames, 8x8 spatial, batch=1
        # Frame 0: object 1 (label=1, cat=0)
        # Frame 1 (central): object 2 (label=1, cat=0) — descended from obj 1 in frame 0
        # Frame 2: object 3 (label=1, cat=0) — descended from obj 2 in frame 1
        # Also: object 4 (label=2) present in all frames, cat=1 (rare, never excluded)
        #
        # labels_map_prev[bidx][c] maps labels in frame c+1 → labels in frame c
        #   labels_map_prev[0][0]: frame 1 labels → frame 0 labels: {1: {1}, 2: {2}}
        #   labels_map_prev[0][1]: frame 2 labels → frame 1 labels: {1: {1}, 2: {2}}
        from scipy.ndimage import find_objects

        n_frames = 3
        frame_window = 1  # central frame index = 1
        H, W = 8, 8

        labelIms = np.zeros((1, H, W, n_frames), dtype=np.int32)
        # Object 1 (label=1) in all frames, different positions to make it interesting
        labelIms[0, 1:4, 1:4, 0] = 1  # frame 0
        labelIms[0, 1:4, 1:4, 1] = 1  # frame 1 (central)
        labelIms[0, 2:5, 2:5, 2] = 1  # frame 2 (moved)
        # Object 2 (label=2) in all frames
        labelIms[0, 5:7, 5:7, 0] = 2
        labelIms[0, 5:7, 5:7, 1] = 2
        labelIms[0, 5:7, 5:7, 2] = 2

        # Category array: cat_array[bidx, obj_idx, frame]
        # obj_idx is 0-based (object with label k+1 is at index k)
        cat_array = np.zeros((1, 2, n_frames), dtype=np.int32)
        cat_array[0, 0, :] = 0  # object 1: category 0 (common)
        cat_array[0, 1, :] = 1  # object 2: category 1 (rare)

        # labels_map_prev: list of length batch_size, each is a list of dicts
        # labels_map_prev[b][c] maps label in frame c+1 → set of labels in frame c
        labels_map_prev = [
            [
                {1: {1}, 2: {2}},  # frame 1 → frame 0
                {1: {1}, 2: {2}},  # frame 2 → frame 1
            ]
        ]

        # Object slices per (batch, frame)
        object_slices = {}
        for c in range(n_frames):
            object_slices[(0, c)] = find_objects(labelIms[0, ..., c])

        # category_keep_prob: cat 0 always excluded (keep_prob=0), cat 1 always kept (keep_prob=1)
        category_keep_prob = np.array([0.0, 1.0])

        # Run the cell line tracing logic (extracted from _get_output_batch)
        weight_map = np.ones(labelIms.shape, dtype=np.float32)
        for b in range(labelIms.shape[0]):
            excluded_labels = {c: set() for c in range(n_frames)}
            central_c = frame_window  # = 1
            cur_cat = cat_array[b, :, central_c]
            for obj_idx, sl in enumerate(object_slices[(b, central_c)]):
                if sl is not None:
                    cat = int(cur_cat[obj_idx])
                    if 0 <= cat < len(category_keep_prob):
                        if np.random.random() >= category_keep_prob[cat]:
                            excluded_labels[central_c].add(obj_idx + 1)
            # Trace backward
            for c in range(central_c - 1, -1, -1):
                lmp = labels_map_prev[b][c]
                for label in excluded_labels[c + 1]:
                    for prev_label in lmp.get(label, []):
                        excluded_labels[c].add(prev_label)
            # Trace forward
            for c in range(central_c, n_frames - 1):
                lmp = labels_map_prev[b][c]
                fwd = {}
                for next_label, prev_labels in lmp.items():
                    for pl in prev_labels:
                        fwd.setdefault(pl, []).append(next_label)
                for label in excluded_labels[c]:
                    for next_label in fwd.get(label, []):
                        excluded_labels[c + 1].add(next_label)
            # Apply exclusions
            for c in range(n_frames):
                for label in excluded_labels[c]:
                    mask = labelIms[b, ..., c] == label
                    weight_map[b, ..., c][mask] = 0

        # Verify: object 1 (cat 0, keep_prob=0) should be excluded in ALL frames
        # Frame 0: label=1 pixels at [1:4, 1:4]
        self.assertEqual(weight_map[0, 2, 2, 0], 0, "Object 1 should be excluded in frame 0 (backward trace)")
        # Frame 1 (central): label=1 pixels at [1:4, 1:4]
        self.assertEqual(weight_map[0, 2, 2, 1], 0, "Object 1 should be excluded in frame 1 (central)")
        # Frame 2: label=1 pixels at [2:5, 2:5]
        self.assertEqual(weight_map[0, 3, 3, 2], 0, "Object 1 should be excluded in frame 2 (forward trace)")

        # Verify: object 2 (cat 1, keep_prob=1) should be kept in ALL frames
        self.assertEqual(weight_map[0, 5, 5, 0], 1, "Object 2 should be kept in frame 0")
        self.assertEqual(weight_map[0, 5, 5, 1], 1, "Object 2 should be kept in frame 1")
        self.assertEqual(weight_map[0, 5, 5, 2], 1, "Object 2 should be kept in frame 2")

        # Verify: background pixels are always 1
        self.assertEqual(weight_map[0, 0, 0, 0], 1, "Background should always be 1")
        self.assertEqual(weight_map[0, 7, 7, 1], 1, "Background should always be 1")

    def test_cell_line_exclusion_with_division(self):
        """Verify cell line tracing works with cell division (one cell becomes two)."""
        # Frame 0: object A (label=1, cat=0)
        # Frame 1 (central): object B (label=1, cat=0) — same cell
        # Frame 2: objects C (label=1) and D (label=2) — B divided into C and D
        # If B is excluded, both C and D must also be excluded, and A must be excluded.
        from scipy.ndimage import find_objects

        n_frames = 3
        frame_window = 1
        H, W = 16, 16

        labelIms = np.zeros((1, H, W, n_frames), dtype=np.int32)
        labelIms[0, 2:6, 2:6, 0] = 1    # frame 0: cell A
        labelIms[0, 2:6, 2:6, 1] = 1    # frame 1: cell B
        labelIms[0, 2:6, 2:4, 2] = 1    # frame 2: cell C (left half)
        labelIms[0, 2:6, 4:6, 2] = 2    # frame 2: cell D (right half)

        # Also a non-excluded object (label=3, cat=1) present in all frames
        labelIms[0, 10:13, 10:13, :] = 3

        cat_array = np.zeros((1, 3, n_frames), dtype=np.int32)
        cat_array[0, 0, :] = 0  # label 1: cat 0 (common, to be excluded)
        cat_array[0, 1, :] = 0  # label 2: cat 0
        cat_array[0, 2, :] = 1  # label 3: cat 1 (rare, always kept)

        # labels_map_prev[b][c]: label in frame c+1 → labels in frame c
        labels_map_prev = [
            [
                {1: {1}, 3: {3}},           # frame 1 → frame 0
                {1: {1}, 2: {1}, 3: {3}},   # frame 2 → frame 1 (both C=1 and D=2 come from B=1)
            ]
        ]

        object_slices = {}
        for c in range(n_frames):
            object_slices[(0, c)] = find_objects(labelIms[0, ..., c])

        category_keep_prob = np.array([0.0, 1.0])  # cat 0 always excluded

        weight_map = np.ones(labelIms.shape, dtype=np.float32)
        for b in range(1):
            excluded_labels = {c: set() for c in range(n_frames)}
            central_c = frame_window
            cur_cat = cat_array[b, :, central_c]
            for obj_idx, sl in enumerate(object_slices[(b, central_c)]):
                if sl is not None:
                    cat = int(cur_cat[obj_idx])
                    if 0 <= cat < len(category_keep_prob):
                        if np.random.random() >= category_keep_prob[cat]:
                            excluded_labels[central_c].add(obj_idx + 1)
            for c in range(central_c - 1, -1, -1):
                lmp = labels_map_prev[b][c]
                for label in excluded_labels[c + 1]:
                    for prev_label in lmp.get(label, []):
                        excluded_labels[c].add(prev_label)
            for c in range(central_c, n_frames - 1):
                lmp = labels_map_prev[b][c]
                fwd = {}
                for next_label, prev_labels in lmp.items():
                    for pl in prev_labels:
                        fwd.setdefault(pl, []).append(next_label)
                for label in excluded_labels[c]:
                    for next_label in fwd.get(label, []):
                        excluded_labels[c + 1].add(next_label)
            for c in range(n_frames):
                for label in excluded_labels[c]:
                    mask = labelIms[b, ..., c] == label
                    weight_map[b, ..., c][mask] = 0

        # Cell B (label=1) excluded at central frame → whole line excluded
        # Frame 0: A (label=1) excluded via backward trace
        self.assertEqual(weight_map[0, 3, 3, 0], 0, "Cell A should be excluded (backward trace from B)")
        # Frame 1: B (label=1) excluded directly
        self.assertEqual(weight_map[0, 3, 3, 1], 0, "Cell B should be excluded (central frame)")
        # Frame 2: C (label=1) excluded via forward trace
        self.assertEqual(weight_map[0, 3, 2, 2], 0, "Cell C should be excluded (forward trace from B)")
        # Frame 2: D (label=2) also excluded via forward trace (division daughter)
        self.assertEqual(weight_map[0, 3, 5, 2], 0, "Cell D should be excluded (division daughter, forward trace)")

        # Non-excluded object (label=3) kept in all frames
        self.assertEqual(weight_map[0, 11, 11, 0], 1, "Label 3 kept in frame 0")
        self.assertEqual(weight_map[0, 11, 11, 1], 1, "Label 3 kept in frame 1")
        self.assertEqual(weight_map[0, 11, 11, 2], 1, "Label 3 kept in frame 2")


class TestObjectwiseHelpers3D(unittest.TestCase):
    """Tests for objectwise_computation_tf helpers in 3D mode."""

    def test_argmax_by_object_2d(self):
        from distnet_2d.utils.objectwise_computation_tf import get_argmax_2d_by_object_fun
        fun = get_argmax_2d_by_object_fun(tridimensional_mode=False)
        data = tf.constant([[0., 1., 0.], [0., 0., 5.], [0., 0., 0.]], dtype=tf.float32)  # max at (1, 2)
        mask = tf.ones_like(data)
        out = fun(data, mask, 9).numpy()
        np.testing.assert_array_equal(out, [1.0, 2.0])

    def test_argmax_by_object_3d(self):
        from distnet_2d.utils.objectwise_computation_tf import get_argmax_2d_by_object_fun
        fun = get_argmax_2d_by_object_fun(tridimensional_mode=True)
        data = np.zeros((4, 8, 8), dtype=np.float32)
        data[2, 3, 5] = 10.0  # max at (z=2, y=3, x=5)
        mask = np.ones_like(data)
        out = fun(tf.constant(data), tf.constant(mask), 4 * 8 * 8).numpy()
        np.testing.assert_array_equal(out, [2.0, 3.0, 5.0])

    def test_argmax_by_object_3d_nan_return(self):
        from distnet_2d.utils.objectwise_computation_tf import get_argmax_2d_by_object_fun
        fun = get_argmax_2d_by_object_fun(tridimensional_mode=True)
        data = tf.zeros((4, 8, 8), dtype=tf.float32)
        mask = tf.zeros_like(data)
        out = fun(data, mask, 0).numpy()
        self.assertEqual(out.shape, (3,))  # 3D should return 3 nan values
        self.assertTrue(np.all(np.isnan(out)))

    def test_mean_by_object_2d(self):
        from distnet_2d.utils.objectwise_computation_tf import get_mean_by_object_fun
        fun = get_mean_by_object_fun(channel_axis=False, tridimensional_mode=False)
        data = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=tf.float32)
        mask = tf.cast(tf.constant([[1, 1, 0], [1, 1, 0], [0, 0, 0]]), tf.float32)
        out = fun(data, mask, 4).numpy()
        np.testing.assert_allclose(out, (1 + 2 + 4 + 5) / 4)

    def test_mean_by_object_3d(self):
        from distnet_2d.utils.objectwise_computation_tf import get_mean_by_object_fun
        fun = get_mean_by_object_fun(channel_axis=False, tridimensional_mode=True)
        data = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
        mask = np.zeros_like(data)
        mask[0, 0, 0] = 1
        mask[1, 2, 2] = 1
        out = fun(tf.constant(data), tf.constant(mask), 2).numpy()
        expected = (data[0, 0, 0] + data[1, 2, 2]) / 2
        np.testing.assert_allclose(out, expected)

    def test_mean_by_object_3d_with_channel(self):
        from distnet_2d.utils.objectwise_computation_tf import get_mean_by_object_fun
        fun = get_mean_by_object_fun(channel_axis=True, tridimensional_mode=True)
        # data: (Z, Y, X, C) = (2, 2, 2, 3)
        data = np.arange(2 * 2 * 2 * 3, dtype=np.float32).reshape(2, 2, 2, 3)
        mask = np.ones((2, 2, 2), dtype=np.float32)
        out = fun(tf.constant(data), tf.constant(mask), 8).numpy()
        self.assertEqual(out.shape, (3,))  # one mean per channel
        expected = data.mean(axis=(0, 1, 2))
        np.testing.assert_allclose(out, expected)

    def test_max_by_object_3d(self):
        from distnet_2d.utils.objectwise_computation_tf import get_max_by_object_fun
        fun = get_max_by_object_fun(channel_axis=False, tridimensional_mode=True)
        data = np.zeros((3, 4, 5), dtype=np.float32)
        data[1, 2, 3] = 42.0
        mask = np.ones_like(data)
        out = fun(tf.constant(data), tf.constant(mask), 60).numpy()
        np.testing.assert_allclose(out, 42.0)

    def test_objectwise_compute_3d(self):
        from distnet_2d.utils.objectwise_computation_tf import (
            get_label_size, get_mean_by_object_fun, objectwise_compute
        )
        labels = np.zeros((1, 2, 3, 3), dtype=np.int32)  # (1, Z, Y, X)
        labels[0, 0, 0:2, 0:2] = 1  # object 1
        labels[0, 1, 1:3, 1:3] = 2  # object 2
        data = np.ones((2, 3, 3), dtype=np.float32) * 5.0  # (Z, Y, X)
        ids, sizes, N = get_label_size(tf.constant(labels), max_objects_number=2)
        mean_fun = get_mean_by_object_fun(channel_axis=False, tridimensional_mode=True)
        out = objectwise_compute(tf.constant(data), mean_fun, tf.constant(labels[0]), ids[0], sizes[0]).numpy()
        np.testing.assert_allclose(out, [5.0, 5.0])

    def test_spherical_kernel(self):
        from distnet_2d.utils.objectwise_computation_tf import spherical_kernel
        ker, rad = spherical_kernel(2.0)
        ker = ker.numpy()
        self.assertEqual(ker.shape, (5, 5, 5))
        # Center should be 1
        self.assertEqual(ker[2, 2, 2], 1)
        # Corners should be 0 (distance > 2)
        self.assertEqual(ker[0, 0, 0], 0)
        # The kernel should be symmetric
        np.testing.assert_array_equal(ker, ker[::-1, ::-1, ::-1])

    def test_dilate_3d_slicewise(self):
        from distnet_2d.utils.objectwise_computation_tf import _dilate_mask, CONV_2D
        mask = np.zeros((1, 8, 16, 16), dtype=bool)
        mask[0, 2:6, 4:12, 4:12] = True  # 4x8x8 = 256 voxels
        n_before = mask.sum()
        # CONV_2D on a 3D input escalates to CONV_2D_SLICEWISE via _auto_mode
        @tf.function
        def fn(m):
            return _dilate_mask(m, radius=1.5, tolerance=0.25, mode=CONV_2D)
        dilated = fn(tf.constant(mask)).numpy()
        self.assertGreater(dilated.sum(), n_before)

    def test_dilate_3d_slicewise_does_not_spread_along_z(self):
        """CONV_2D on rank-4 input escalates to CONV_2D_SLICEWISE: empty Z-slices stay empty."""
        from distnet_2d.utils.objectwise_computation_tf import _dilate_mask, CONV_2D
        mask = np.zeros((1, 8, 16, 16), dtype=bool)
        mask[0, 3, 4:12, 4:12] = True
        @tf.function
        def fn(m):
            return _dilate_mask(m, radius=2.0, tolerance=0.25, mode=CONV_2D)
        dilated = fn(tf.constant(mask)).numpy()
        self.assertGreater(dilated[0, 3].sum(), mask[0, 3].sum())
        for z in range(8):
            if z != 3:
                self.assertEqual(dilated[0, z].sum(), 0,
                                 f"Dilation leaked into Z-slice {z} (expected empty)")

    def test_dilate_3d_true_3d_spreads_along_z(self):
        """CONV_3D dilation: a single-slice mask should spread into adjacent Z-slices."""
        from distnet_2d.utils.objectwise_computation_tf import _dilate_mask, CONV_3D
        mask = np.zeros((1, 8, 16, 16), dtype=bool)
        mask[0, 3, 4:12, 4:12] = True
        @tf.function
        def fn(m):
            return _dilate_mask(m, radius=2.0, tolerance=0.25, mode=CONV_3D)
        dilated = fn(tf.constant(mask)).numpy()
        # Neighbouring slices must now contain dilated voxels
        self.assertGreater(dilated[0, 2].sum(), 0, "CONV_3D dilation should spread to Z=2")
        self.assertGreater(dilated[0, 4].sum(), 0, "CONV_3D dilation should spread to Z=4")

    def test_contours_3d(self):
        """auto-mode on rank-4 input → CONV_3D contour detection."""
        from distnet_2d.utils.objectwise_computation_tf import _compute_contours
        mask = np.zeros((1, 6, 16, 16), dtype=bool)
        mask[0, 1:5, 4:12, 4:12] = True  # 4x8x8 solid block
        @tf.function
        def fn(m):
            return _compute_contours(m)  # auto: rank 4 → CONV_3D
        contours = fn(tf.constant(mask)).numpy()
        # Interior voxel (2,7,7) is surrounded by mask on all 6 faces — not a contour
        self.assertFalse(contours[0, 2, 7, 7], "interior voxel should not be a contour")
        # A face voxel should be a contour
        self.assertTrue(contours[0, 1, 7, 7], "Z-face voxel must be a contour with CONV_3D")
        self.assertTrue(contours[0, 2, 4, 7], "Y-face voxel must be a contour with CONV_3D")

    def test_auto_mode_detection(self):
        """mode=None should auto-detect: rank 3 → CONV_2D, rank 4 → CONV_3D."""
        from distnet_2d.utils.objectwise_computation_tf import _auto_mode, CONV_2D, CONV_3D
        img2d = tf.zeros((1, 8, 8), dtype=tf.float32)
        img3d = tf.zeros((1, 4, 8, 8), dtype=tf.float32)
        self.assertEqual(_auto_mode(img2d, None), CONV_2D)
        self.assertEqual(_auto_mode(img3d, None), CONV_3D)
        # explicit mode is honored
        from distnet_2d.utils.objectwise_computation_tf import CONV_2D_SLICEWISE
        self.assertEqual(_auto_mode(img3d, CONV_2D_SLICEWISE), CONV_2D_SLICEWISE)

    def test_iou_fp_auto_mode(self):
        """IoU/FP without explicit mode should auto-detect from input rank — in graph mode."""
        from distnet_2d.utils.objectwise_computation_tf import IoU, FP

        @tf.function
        def iou_fn(t, p):
            return IoU(t, p)

        @tf.function
        def fp_fn(t, p):
            return FP(t, p)

        # 2D inputs → CONV_2D path
        m2d = np.zeros((1, 8, 8), dtype=bool)
        m2d[0, 2:6, 2:6] = True
        self.assertAlmostEqual(iou_fn(tf.constant(m2d), tf.constant(m2d)).numpy(), 1.0)
        # 3D inputs → CONV_3D path (auto-detect)
        m3d = np.zeros((1, 4, 8, 8), dtype=bool)
        m3d[0, 1:3, 2:6, 2:6] = True
        self.assertAlmostEqual(iou_fn(tf.constant(m3d), tf.constant(m3d)).numpy(), 1.0)
        self.assertEqual(fp_fn(tf.constant(m3d), tf.constant(m3d)).numpy(), 0.0)

    def test_convolve_modes_graph_compatible(self):
        """All three conv modes must run inside @tf.function (no eager-only ops)."""
        from distnet_2d.utils.objectwise_computation_tf import _convolve, circular_kernel, spherical_kernel, CONV_2D, CONV_3D, CONV_2D_SLICEWISE

        @tf.function
        def conv2d(img, ker, rad):
            return _convolve(img, ker, rad, symmetric_padding=True, mode=CONV_2D)

        @tf.function
        def conv3d(img, ker, rad):
            return _convolve(img, ker, rad, symmetric_padding=True, mode=CONV_3D)

        @tf.function
        def conv_slicewise(img, ker, rad):
            return _convolve(img, ker, rad, symmetric_padding=True, mode=CONV_2D_SLICEWISE)

        ker2, rad = circular_kernel(1.5)
        ker3, _ = spherical_kernel(1.5)
        img2d = tf.cast(tf.random.uniform((1, 8, 8), maxval=2, dtype=tf.int32), tf.int32)
        img3d = tf.cast(tf.random.uniform((1, 4, 8, 8), maxval=2, dtype=tf.int32), tf.int32)
        # Should not raise
        out2 = conv2d(img2d, ker2, rad)
        out3 = conv3d(img3d, ker3, rad)
        out_s = conv_slicewise(img3d, ker2, rad)
        self.assertEqual(out2.shape.as_list(), [1, 8, 8])
        self.assertEqual(out3.shape.as_list(), [1, 4, 8, 8])
        self.assertEqual(out_s.shape.as_list(), [1, 4, 8, 8])

    def test_contours_2d_slicewise_ignores_z_boundaries(self):
        """CONV_2D on rank-4 input escalates to CONV_2D_SLICEWISE: Z-faces are NOT contours."""
        from distnet_2d.utils.objectwise_computation_tf import _compute_contours, CONV_2D
        mask = np.zeros((1, 6, 16, 16), dtype=bool)
        mask[0, 1:5, 4:12, 4:12] = True
        @tf.function
        def fn_slicewise(m):
            return _compute_contours(m, mode=CONV_2D)  # rank 4: CONV_2D → CONV_2D_SLICEWISE
        @tf.function
        def fn_3d(m):
            return _compute_contours(m)
        c_slicewise = fn_slicewise(tf.constant(mask)).numpy()
        c_3d = fn_3d(tf.constant(mask)).numpy()
        # CONV_3D: Z-face voxel marked as contour; CONV_2D_SLICEWISE: interior of Y/X plane → not contour
        self.assertTrue(c_3d[0, 1, 7, 7])
        self.assertFalse(c_slicewise[0, 1, 7, 7],
                         "slicewise must not flag Z-face voxel that is interior in its Y/X slice")

    def test_FP_3d_slicewise(self):
        from distnet_2d.utils.objectwise_computation_tf import FP, CONV_2D
        true_fg = np.zeros((1, 8, 16, 16), dtype=bool)
        true_fg[0, 3:5, 7:9, 7:9] = True
        @tf.function
        def fp_fn(tf_fg, pf_fg):
            return FP(tf_fg, pf_fg, rate=False, tolerance_radius=0, mode=CONV_2D)
        fp_perfect = fp_fn(tf.constant(true_fg), tf.constant(true_fg)).numpy()
        self.assertEqual(fp_perfect, 0.0)
        pred_fg = np.copy(true_fg)
        pred_fg[0, 0, 0:6, 0:6] = True
        fp = fp_fn(tf.constant(true_fg), tf.constant(pred_fg)).numpy()
        self.assertGreater(fp, 0)

    def test_IoU_3d_slicewise(self):
        from distnet_2d.utils.objectwise_computation_tf import IoU, CONV_2D
        true_fg = np.zeros((1, 4, 8, 8), dtype=bool)
        true_fg[0, 1:3, 3:5, 3:5] = True
        @tf.function
        def iou_fn(t, p):
            return IoU(t, p, tolerance_radius=0, mode=CONV_2D)
        iou = iou_fn(tf.constant(true_fg), tf.constant(true_fg)).numpy()
        np.testing.assert_allclose(iou, 1.0)
        pred_fg2 = np.zeros_like(true_fg)
        pred_fg2[0, 1:3, 3:5, 4:6] = True
        iou2 = iou_fn(tf.constant(true_fg), tf.constant(pred_fg2)).numpy()
        self.assertGreater(iou2, 0)
        self.assertLess(iou2, 1.0)


from dataset_iterator.datasetIO import DictDatasetIO


def _build_synth_dataset(tridim, Z=4):
    """Build an in-memory synthetic dataset with merge, division and moving rectangles.

    Layout (5 frames, 32x32 spatial, optional Z dim):
      - Cell 1: 5x5 rect moving diagonally — persists all 5 frames
      - Cell 2: rect that DIVIDES at frame 2 into labels 2 and 3
      - Cell 4 + cell 5: two rects that MERGE at frame 3 (both → label 4)
    """
    N_FRAMES, H, W = 5, 32, 32
    spatial = (Z, H, W) if tridim else (H, W)

    # raw — just random
    raw = np.random.rand(N_FRAMES, *spatial).astype(np.float32)
    labels = np.zeros((N_FRAMES,) + spatial, dtype=np.int32)

    # In 3D, put cells on a 2-Z-slice slab — Z slice indices
    def zslab(idx0):
        return slice(idx0, idx0 + 2)

    for t in range(N_FRAMES):
        # Cell 1: diagonal motion
        if tridim:
            labels[t, zslab(0), 2+t:7+t, 2+t:7+t] = 1
        else:
            labels[t, 2+t:7+t, 2+t:7+t] = 1

    # Cell 2 → divides at t=2
    for t in [0, 1]:
        if tridim:
            labels[t, zslab(1), 2:7, 20-t:25-t] = 2
        else:
            labels[t, 2:7, 20-t:25-t] = 2
    for t in [2, 3, 4]:
        offset = t - 2
        if tridim:
            labels[t, zslab(1), 2:5, 18-offset:23-offset] = 2
            labels[t, zslab(1), 5:8, 18-offset:23-offset] = 3
        else:
            labels[t, 2:5, 18-offset:23-offset] = 2
            labels[t, 5:8, 18-offset:23-offset] = 3

    # Cell 4 + 5 → merge at t=3
    for t in [0, 1, 2]:
        if tridim:
            labels[t, zslab(2), 20:25, 5+2*t:10+2*t] = 4
            labels[t, zslab(2), 20:25, 17-2*t:22-2*t] = 5
        else:
            labels[t, 20:25, 5+2*t:10+2*t] = 4
            labels[t, 20:25, 17-2*t:22-2*t] = 5
    for t in [3, 4]:
        if tridim:
            labels[t, zslab(2), 20:25, 9+t-3:18+t-3] = 4
        else:
            labels[t, 20:25, 9+t-3:18+t-3] = 4

    # linksPrev: (N_entries, 2, N_FRAMES) — [current_label, prev_label]
    max_entries = 5
    links = np.zeros((max_entries, 2, N_FRAMES), dtype=np.int32)
    # t=1 from t=0
    links[0, :, 1] = [1, 1]; links[1, :, 1] = [2, 2]; links[2, :, 1] = [4, 4]; links[3, :, 1] = [5, 5]
    # t=2 from t=1 — division of cell 2
    links[0, :, 2] = [1, 1]; links[1, :, 2] = [2, 2]; links[2, :, 2] = [3, 2]
    links[3, :, 2] = [4, 4]; links[4, :, 2] = [5, 5]
    # t=3 from t=2 — merge of 4+5 → 4
    links[0, :, 3] = [1, 1]; links[1, :, 3] = [2, 2]; links[2, :, 3] = [3, 3]
    links[3, :, 3] = [4, 4]; links[4, :, 3] = [4, 5]
    # t=4 from t=3 — stable
    links[0, :, 4] = [1, 1]; links[1, :, 4] = [2, 2]; links[2, :, 4] = [3, 3]; links[3, :, 4] = [4, 4]

    return DictDatasetIO({
        "/posA/raw": raw,
        "/posA/regionLabels": labels,
        "/posA/linksPrev": links,
    })


def _make_iterator(ds_io, tridim, tracking, Z=4, metrics:bool=False):
    """Build a DistnetIterator from an in-memory DictDatasetIO.

    When `metrics=True` the iterator is configured to emit the extra outputs
    used by `get_metrics_fun` (label rank, prev-label array, center array) and
    only the central frame for masks (`output_central_only=True`,
    `return_label_rank=True`, `incomplete_last_batch_mode=0`).
    Otherwise it returns full-frame outputs for model training.
    """
    from distnet_2d.data import DistnetIterator
    from dataset_iterator.image_data_generator import get_image_data_generator
    from dataset_iterator import extract_tile_random_zoom_function

    data_gen = get_image_data_generator()
    mask_gen = get_image_data_generator()
    H, W = 32, 32
    tile_shape = (Z, H, W) if tridim else (H, W)
    it = DistnetIterator(
        dataset=ds_io,
        extract_tile_function=extract_tile_random_zoom_function(tile_shape=tile_shape, n_tiles=1, perform_augmentation=False),
        frame_window=2,
        aug_frame_subsampling=None,
        erase_edge_cell_size=0,
        return_label_rank=metrics,
        return_link_multiplicity=True,
        segmentation=True,
        tracking=tracking,
        image_data_generators=[data_gen, mask_gen],
        batch_size=1,
        step_number=0,
        tridimensional_mode=tridim,
        verbose=False,
        shuffle=False,
    )
    it.disable_random_transforms(True, True)
    if metrics:
        it.output_central_only = True
        it.return_label_rank = True
        it.incomplete_last_batch_mode = 0
    else:
        it.output_central_only = False
        it.return_label_rank = False
    return it


class TestIteratorTrainingStep(unittest.TestCase):
    """Run a model train_step on real DistnetIterator output (full-frame mode)."""

    @staticmethod
    def _iter_train_inputs(tridim, tracking):
        """Build iterator in full-frame mode and return (x, y) ready for train_step."""
        ds_io = _build_synth_dataset(tridim=tridim, Z=4)
        it = _make_iterator(ds_io, tridim=tridim, tracking=tracking, metrics=False)
        x, y = it[0]
        y_train = [tf.constant(yy, dtype=tf.float32) for yy in y]
        # x: iterator returns numpy without explicit batch dim — wrap into [batch=1] tensor
        x_train = [tf.constant(xx, dtype=tf.float32)[tf.newaxis] if xx.ndim == 3 + int(tridim)
                   else tf.constant(xx, dtype=tf.float32) for xx in x]
        return x_train, y_train

    def _run(self, tridim, tracking):
        with tf.device('/CPU:0'):
            x, y = self._iter_train_inputs(tridim=tridim, tracking=tracking)
            spa_dims = tuple(int(d) for d in y[0].shape[1:-1])
            model, _ = TestTrainingStep._build_model(spa_dims, tridim=tridim, tracking=tracking)

            @tf.function
            def step_fn(data):
                return model.train_step(data)
            metrics = step_fn((x, y))
        self.assertIn("loss", metrics)
        loss_val = metrics["loss"].numpy()
        self.assertFalse(np.isnan(loss_val), f"loss is NaN on iterator data (tridim={tridim}, tracking={tracking})")

    def test_iterator_train_2d_seg_only(self):
        self._run(tridim=False, tracking=False)

    def test_iterator_train_2d_tracking(self):
        self._run(tridim=False, tracking=True)

    def test_iterator_train_3d_seg_only(self):
        self._run(tridim=True, tracking=False)

    def test_iterator_train_3d_tracking(self):
        self._run(tridim=True, tracking=True)


class TestMetricsFunWithIterator(unittest.TestCase):
    """End-to-end metric tests: synthetic in-memory dataset → DistnetIterator → metrics_fun."""

    def _run_metrics(self, tridim, tracking, n_label_max_override=None):
        from distnet_2d.utils.metrics_tf import get_metrics_fun
        ds_io = _build_synth_dataset(tridim=tridim, Z=4)
        it = _make_iterator(ds_io, tridim=tridim, tracking=tracking, metrics=True)
        if n_label_max_override is not None:
            it.n_label_max = n_label_max_override

        x, y = it[0]
        # y order with seg + tracking: EDM, CDM, dY, dX, LM, labels, prev_labels, centerArr
        # In 3D iterator there should also be dZ — check by counting elements
        # 2D tracking expects 8 outputs; 3D tracking expects 9 (extra dZ)
        # Seg-only (tracking=False): EDM, CDM, labels, centerArr → 4 outputs
        if tracking:
            if tridim:
                self.assertEqual(len(y), 9, f"3D+tracking should yield 9 outputs, got {len(y)}")
                true_edm, true_cdm, true_dZ, true_dY, true_dX, true_lm, labels, prev_labels, centerArr = y
            else:
                self.assertEqual(len(y), 8, f"2D+tracking should yield 8 outputs, got {len(y)}")
                true_edm, true_cdm, true_dY, true_dX, true_lm, labels, prev_labels, centerArr = y
                true_dZ = tf.zeros_like(true_dY)  # unused in 2D
        else:
            self.assertEqual(len(y), 4, f"seg-only should yield 4 outputs, got {len(y)}")
            true_edm, true_cdm, labels, centerArr = y

        # Build "perfect-ish" predictions from ground truth where possible
        edm_pred = tf.constant(true_edm, dtype=tf.float32)
        gdcm_pred = tf.constant(true_cdm, dtype=tf.float32)
        # cat: not used (category=False) but signature requires it
        n_frames = int(true_edm.shape[-1])
        cat_shape = list(true_edm.shape[:-1]) + [3]  # 3 classes
        cat_pred = tf.zeros(cat_shape, dtype=tf.float32)
        true_cat = tf.zeros(list(true_edm.shape), dtype=tf.float32)

        metric_fn = get_metrics_fun(
            scale=4.0,
            max_objects_number=it.n_label_max,
            category=False,
            segmentation=True,
            tracking=tracking,
            tridimensional_mode=tridim,
        )
        if tracking:
            # LM prediction: convert categorical ground truth (n_pair_chans channels with values in {1,2,3})
            # to one-hot (n_pair_chans * 3 channels) for "perfect" prediction
            n_pair = int(true_lm.shape[-1])
            lm_categorical = tf.cast(tf.clip_by_value(true_lm - 1, 0, 2), tf.int32)  # values 0..2
            lm_onehot = tf.one_hot(lm_categorical, depth=3, dtype=tf.float32)  # (..., n_pair, 3)
            # Flatten last two dims into (..., n_pair * 3)
            lm_shape = list(true_lm.shape) + [3]
            lm_pred = tf.reshape(lm_onehot, list(true_lm.shape[:-1]) + [n_pair * 3])

            dY_pred = tf.constant(true_dY, dtype=tf.float32)
            dX_pred = tf.constant(true_dX, dtype=tf.float32)
            dZ_pred = tf.constant(true_dZ, dtype=tf.float32)
            out = metric_fn(edm_pred, gdcm_pred, cat_pred,
                            dZ_pred, dY_pred, dX_pred, lm_pred,
                            tf.cast(true_edm, tf.float32), true_cat,
                            tf.cast(true_dZ, tf.float32), tf.cast(true_dY, tf.float32), tf.cast(true_dX, tf.float32),
                            tf.cast(true_lm, tf.float32),
                            tf.cast(labels, tf.int32), tf.cast(prev_labels, tf.int32),
                            tf.cast(centerArr, tf.float32))
        else:
            out = metric_fn(edm_pred, gdcm_pred, cat_pred,
                            tf.cast(true_edm, tf.float32), true_cat,
                            tf.cast(labels, tf.int32),
                            tf.cast(centerArr, tf.float32))
        result = out.numpy()
        self.assertTrue(np.all(np.isfinite(result)), f"metrics produced non-finite values: {result}")
        return result

    def test_iterator_metrics_2d_seg_only(self):
        self._run_metrics(tridim=False, tracking=False)

    def test_iterator_metrics_2d_tracking(self):
        self._run_metrics(tridim=False, tracking=True)

    def test_iterator_metrics_3d_seg_only(self):
        self._run_metrics(tridim=True, tracking=False)

    def test_iterator_metrics_3d_tracking(self):
        self._run_metrics(tridim=True, tracking=True)


if __name__ == '__main__':
    unittest.main()

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
    def _build_model(spa_dims, tridim, tracking=True, predict_edm_derivatives=False):
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
        model = get_distnet_2d(arch)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
        return model, arch

    @staticmethod
    def _make_data(spa_dims, tridim, tracking=True, predict_edm_derivatives=False):
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
        return x, y

    def _run_train_step(self, spa_dims, tridim, tracking=True, predict_edm_derivatives=False):
        with tf.device('/CPU:0'):
            model, arch = self._build_model(spa_dims, tridim, tracking=tracking, predict_edm_derivatives=predict_edm_derivatives)
            x, y = self._make_data(spa_dims, tridim, tracking=tracking, predict_edm_derivatives=predict_edm_derivatives)

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


if __name__ == '__main__':
    unittest.main()

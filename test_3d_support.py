import unittest
import numpy as np
from scipy.ndimage import find_objects
from distnet_2d.utils import image_derivatives_np as der
from distnet_2d.data.medoid import get_medoid
from distnet_2d.data.center_edm import compute_edm
from distnet_2d.data.distnet_iterator import (
    edt_smooth, _get_small_objects_at_edges_to_erase, _get_labels_and_centers,
    _draw_centers, derivatives_labelwise, _compute_outputs
)


def _make_label_2d(shape=(32, 32)):
    """Create a 2D label image with 2 non-touching rectangular objects."""
    lab = np.zeros(shape, dtype=np.int32)
    lab[4:12, 4:12] = 1
    lab[16:28, 16:28] = 2
    return lab


def _make_label_3d(shape=(6, 32, 32)):
    """Create a 3D label image with 2 non-touching rectangular objects."""
    lab = np.zeros(shape, dtype=np.int32)
    lab[1:5, 4:12, 4:12] = 1
    lab[1:5, 16:28, 16:28] = 2
    return lab


class TestDerivatives(unittest.TestCase):
    def test_der_2d_shape(self):
        img = np.random.rand(16, 16).astype(np.float32)
        for ax in range(2):
            d = der.der(img, ax)
            self.assertEqual(d.shape, img.shape)

    def test_der_3d_shape(self):
        img = np.random.rand(4, 16, 16).astype(np.float32)
        for ax in range(3):
            d = der.der(img, ax)
            self.assertEqual(d.shape, img.shape)

    def test_der_2d_linear_ramp(self):
        # Linear ramp along Y: derivative along Y should be ~constant
        img = np.tile(np.arange(16, dtype=np.float32).reshape(16, 1), (1, 16))
        dy = der.der(img, 0)
        # Interior pixels should have derivative = 0.5 * (f(y+1) - f(y-1)) = 1.0
        np.testing.assert_allclose(dy[1:-1, :], 1.0, atol=1e-6)
        dx = der.der(img, 1)
        np.testing.assert_allclose(dx, 0.0, atol=1e-6)

    def test_der_3d_linear_ramp(self):
        # Linear ramp along Z
        img = np.tile(np.arange(8, dtype=np.float32).reshape(8, 1, 1), (1, 4, 4))
        dz = der.der(img, 0)
        np.testing.assert_allclose(dz[1:-1, :, :], 1.0, atol=1e-6)
        dy = der.der(img, 1)
        np.testing.assert_allclose(dy, 0.0, atol=1e-6)

    def test_laplacian_2d(self):
        img = np.random.rand(16, 16).astype(np.float32)
        lap = der.laplacian(img)
        self.assertEqual(lap.shape, img.shape)

    def test_laplacian_3d(self):
        img = np.random.rand(4, 16, 16).astype(np.float32)
        lap = der.laplacian(img)
        self.assertEqual(lap.shape, img.shape)

    def test_backward_compat(self):
        img = np.random.rand(16, 16).astype(np.float32)
        d = der.der_2d(img, 0)
        self.assertEqual(d.shape, img.shape)
        lap = der.laplacian_2d(img)
        self.assertEqual(lap.shape, img.shape)
        gm = der.gradient_magnitude_2d(img)
        self.assertEqual(gm.shape, img.shape)


class TestMedoid(unittest.TestCase):
    def test_2d(self):
        Y = np.array([0, 0, 10, 10])
        X = np.array([0, 10, 0, 10])
        med = get_medoid(Y, X)
        self.assertEqual(len(med), 2)
        # Result must be one of the input points
        self.assertIn(med, [(0, 0), (0, 10), (10, 0), (10, 10)])

    def test_3d(self):
        Z = np.array([0, 0, 0, 5])
        Y = np.array([0, 10, 0, 5])
        X = np.array([0, 0, 10, 5])
        med = get_medoid(Z, Y, X)
        self.assertEqual(len(med), 3)
        # Result must be one of the input points
        points = set(zip(Z, Y, X))
        self.assertIn(med, points)

    def test_single_point_2d(self):
        med = get_medoid(np.array([5]), np.array([7]))
        self.assertEqual(med, (5, 7))

    def test_single_point_3d(self):
        med = get_medoid(np.array([1]), np.array([5]), np.array([7]))
        self.assertEqual(med, (1, 5, 7))


class TestComputeEdm(unittest.TestCase):
    def test_2d(self):
        output = np.zeros((16, 16), dtype=np.float32)
        centers = [[8, 8]]
        compute_edm(centers, output)
        self.assertEqual(output.shape, (16, 16))
        # Distance at center should be 0
        self.assertAlmostEqual(output[8, 8], 0.0, places=5)
        # Distance at (8, 10) should be 2.0
        self.assertAlmostEqual(output[8, 10], 2.0, places=5)
        # All values >= 0
        self.assertTrue(np.all(output >= 0))

    def test_3d(self):
        output = np.zeros((6, 16, 16), dtype=np.float32)
        centers = [[3, 8, 8]]
        compute_edm(centers, output)
        self.assertEqual(output.shape, (6, 16, 16))
        self.assertAlmostEqual(output[3, 8, 8], 0.0, places=5)
        # Distance at (3, 8, 10) should be 2.0
        self.assertAlmostEqual(output[3, 8, 10], 2.0, places=5)
        self.assertTrue(np.all(output >= 0))


class TestEdtSmooth(unittest.TestCase):
    def test_2d(self):
        lab = _make_label_2d()
        obj_slices = find_objects(lab)
        edm = edt_smooth(lab, obj_slices)
        self.assertEqual(edm.shape, lab.shape)
        # Inside object 1: edm > 0
        self.assertTrue(np.all(edm[5:11, 5:11] > 0))
        # Outside both objects: edm == 0
        self.assertEqual(edm[0, 0], 0.0)

    def test_3d(self):
        lab = _make_label_3d()
        obj_slices = find_objects(lab)
        edm = edt_smooth(lab, obj_slices, z_radius=1.0)
        self.assertEqual(edm.shape, lab.shape)
        self.assertTrue(np.all(edm[2:4, 6:10, 6:10] > 0))
        self.assertEqual(edm[0, 0, 0], 0.0)

    def test_3d_anisotropic(self):
        lab = _make_label_3d()
        obj_slices = find_objects(lab)
        edm = edt_smooth(lab, obj_slices, z_radius=0.5)
        self.assertEqual(edm.shape, lab.shape)
        self.assertTrue(np.all(edm[2:4, 6:10, 6:10] > 0))


class TestGetSmallObjectsAtEdgesToErase(unittest.TestCase):
    def test_2d_edge_object(self):
        lab = np.zeros((32, 32), dtype=np.int32)
        lab[0:2, 0:2] = 1  # small object at Y=0 edge
        lab[10:20, 10:20] = 2  # large interior object
        result = _get_small_objects_at_edges_to_erase(lab, min_size=10)
        self.assertIn(1, result)
        self.assertNotIn(2, result)

    def test_2d_interior_object(self):
        lab = np.zeros((32, 32), dtype=np.int32)
        lab[10:14, 10:14] = 1  # interior object
        result = _get_small_objects_at_edges_to_erase(lab, min_size=100)
        self.assertEqual(len(result), 0)

    def test_3d_z_edge_not_detected(self):
        """Object touching only Z edge should NOT be detected."""
        lab = np.zeros((8, 32, 32), dtype=np.int32)
        lab[0:1, 10:13, 10:13] = 1  # small object at Z=0 edge only
        result = _get_small_objects_at_edges_to_erase(lab, min_size=100)
        self.assertNotIn(1, result)

    def test_3d_x_edge_detected(self):
        """Object touching X edge should be detected."""
        lab = np.zeros((8, 32, 32), dtype=np.int32)
        lab[2:4, 10:13, 30:32] = 1  # small object at X=-1 edge
        result = _get_small_objects_at_edges_to_erase(lab, min_size=100)
        self.assertIn(1, result)


class TestGetLabelsAndCenters(unittest.TestCase):
    def test_2d_geometrical(self):
        lab = _make_label_2d()
        edm = edt_smooth(lab, find_objects(lab))
        lc = _get_labels_and_centers(lab, edm, "GEOMETRICAL")
        self.assertIn(1, lc)
        self.assertIn(2, lc)
        # Centers should have 2 coords
        self.assertEqual(len(lc[1]), 2)
        # Center should be inside the object
        c = lc[1]
        self.assertTrue(lab[int(round(c[0])), int(round(c[1]))] == 1)

    def test_2d_medoid(self):
        lab = _make_label_2d()
        edm = edt_smooth(lab, find_objects(lab))
        lc = _get_labels_and_centers(lab, edm, "MEDOID")
        self.assertIn(1, lc)
        c = lc[1]
        self.assertEqual(len(c), 2)
        self.assertTrue(lab[int(round(c[0])), int(round(c[1]))] == 1)

    def test_3d_geometrical(self):
        lab = _make_label_3d()
        edm = edt_smooth(lab, find_objects(lab))
        lc = _get_labels_and_centers(lab, edm, "GEOMETRICAL")
        self.assertIn(1, lc)
        self.assertIn(2, lc)
        c = lc[1]
        self.assertEqual(len(c), 3)
        self.assertTrue(lab[int(round(c[0])), int(round(c[1])), int(round(c[2]))] == 1)

    def test_3d_medoid(self):
        lab = _make_label_3d()
        edm = edt_smooth(lab, find_objects(lab))
        lc = _get_labels_and_centers(lab, edm, "MEDOID")
        self.assertIn(1, lc)
        c = lc[1]
        self.assertEqual(len(c), 3)
        self.assertTrue(lab[int(round(c[0])), int(round(c[1])), int(round(c[2]))] == 1)

    def test_3d_edm_max(self):
        lab = _make_label_3d()
        edm = edt_smooth(lab, find_objects(lab))
        lc = _get_labels_and_centers(lab, edm, "EDM_MAX")
        self.assertIn(1, lc)
        c = lc[1]
        self.assertEqual(len(c), 3)
        self.assertTrue(lab[int(round(c[0])), int(round(c[1])), int(round(c[2]))] == 1)


class TestDrawCenters(unittest.TestCase):
    def _run_draw_centers(self, ndim, mode, z_radius=None):
        if ndim == 2:
            lab = _make_label_2d()
        else:
            lab = _make_label_3d()
        edm = edt_smooth(lab, find_objects(lab), z_radius=z_radius)
        lc = _get_labels_and_centers(lab, edm, "GEOMETRICAL")
        obj_slices = find_objects(lab)
        center_im = np.zeros_like(lab, dtype=np.float32)
        _draw_centers(center_im, lc, lab, obj_slices, center_distance_mode=mode, z_radius=z_radius)
        return center_im, lab

    def test_geodesic_2d(self):
        cim, lab = self._run_draw_centers(2, "GEODESIC")
        # Inside objects: should have distance values > 0 (except at center pixel which could be ~0)
        self.assertTrue(np.any(cim[lab == 1] > 0))
        # Far from objects: should be 0 (near borders may have values due to label dilation)
        self.assertTrue(np.all(cim[0, :] == 0))

    def test_geodesic_3d(self):
        cim, lab = self._run_draw_centers(3, "GEODESIC", z_radius=1.0)
        self.assertTrue(np.any(cim[lab == 1] > 0))
        self.assertTrue(np.all(cim[0, 0, :] == 0))

    def test_geodesic_3d_anisotropic(self):
        cim, lab = self._run_draw_centers(3, "GEODESIC", z_radius=0.5)
        self.assertTrue(np.any(cim[lab == 1] > 0))

    def test_euclidean_2d(self):
        cim, lab = self._run_draw_centers(2, "EUCLIDEAN")
        self.assertEqual(cim.shape, lab.shape)
        # At least some positive values
        self.assertTrue(np.any(cim > 0))

    def test_euclidean_3d(self):
        cim, lab = self._run_draw_centers(3, "EUCLIDEAN")
        self.assertEqual(cim.shape, lab.shape)
        self.assertTrue(np.any(cim > 0))

    def test_cell_only_2d(self):
        cim, lab = self._run_draw_centers(2, "CELL_ONLY")
        self.assertTrue(np.any(cim[lab == 1] > 0))
        self.assertTrue(np.all(cim[lab == 0] == 0))

    def test_cell_only_3d(self):
        cim, lab = self._run_draw_centers(3, "CELL_ONLY", z_radius=0.5)
        self.assertTrue(np.any(cim[lab == 1] > 0))
        self.assertTrue(np.all(cim[lab == 0] == 0))


class TestDerivativesLabelwise(unittest.TestCase):
    def test_2d(self):
        lab = np.zeros((16, 16), dtype=np.int32)
        lab[4:12, 4:12] = 1
        obj_slices = find_objects(lab)
        # Create a linear ramp inside the label along Y
        image = np.zeros((16, 16), dtype=np.float32)
        image[4:12, 4:12] = np.tile(np.arange(4, 12, dtype=np.float32).reshape(8, 1), (1, 8))
        der_y = np.zeros_like(image)
        der_x = np.zeros_like(image)
        derivatives_labelwise(image, 0, None, der_y, der_x, lab, obj_slices)
        # Inside: der_y should be ~1.0 (away from borders), der_x should be ~0
        interior = lab == 1
        self.assertTrue(np.any(np.abs(der_y[interior]) > 0.1))
        np.testing.assert_allclose(der_x[6:10, 6:10], 0.0, atol=0.1)

    def test_3d(self):
        lab = np.zeros((6, 16, 16), dtype=np.int32)
        lab[1:5, 4:12, 4:12] = 1
        obj_slices = find_objects(lab)
        # Linear ramp along Z inside label
        image = np.zeros((6, 16, 16), dtype=np.float32)
        for z in range(1, 5):
            image[z, 4:12, 4:12] = float(z)
        der_z = np.zeros_like(image)
        der_y = np.zeros_like(image)
        der_x = np.zeros_like(image)
        derivatives_labelwise(image, 0, der_z, der_y, der_x, lab, obj_slices)
        interior = lab == 1
        # der_z should be nonzero inside
        self.assertTrue(np.any(np.abs(der_z[interior]) > 0.1))
        # der_x, der_y should be ~0 (constant along Y and X)
        np.testing.assert_allclose(der_x[2:4, 6:10, 6:10], 0.0, atol=0.1)
        np.testing.assert_allclose(der_y[2:4, 6:10, 6:10], 0.0, atol=0.1)


class TestComputeOutputs(unittest.TestCase):
    def _setup(self, ndim):
        """Create a 2-frame label setup with one object per frame, linked."""
        if ndim == 2:
            lab = np.zeros((16, 16, 2), dtype=np.int32)
            lab[4:12, 4:12, 0] = 1   # frame 0: object at y~8, x~8
            lab[6:14, 6:14, 1] = 1   # frame 1: object shifted by +2,+2
        else:
            lab = np.zeros((6, 16, 16, 2), dtype=np.int32)
            lab[1:5, 4:12, 4:12, 0] = 1
            lab[2:5, 6:14, 6:14, 1] = 1  # shifted in z by ~+0.5, y by +2, x by +2
        spatial_shape = lab.shape[:-1]
        # Compute EDM and centers for each frame
        obj_slices = []
        for c in range(2):
            obj_slices.append(find_objects(lab[..., c]))
        edm = np.zeros_like(lab, dtype=np.float32)
        for c in range(2):
            edm[..., c] = edt_smooth(lab[..., c], obj_slices[c])
        labels_and_centers = []
        for c in range(2):
            labels_and_centers.append(
                _get_labels_and_centers(lab[..., c], edm[..., c], "GEOMETRICAL")
            )
        # labels_map_prev: label 1 in frame 1 comes from label 1 in frame 0
        labels_map_prev = {1: {1}}
        return lab, labels_and_centers, labels_map_prev, obj_slices, spatial_shape, ndim

    def test_2d_displacement(self):
        lab, lc, lmp, obj_slices, sp_shape, ndim = self._setup(2)
        dyIm = np.zeros(sp_shape, dtype=np.float32)
        dxIm = np.zeros(sp_shape, dtype=np.float32)
        centerArr = np.full((1, 2), np.nan, dtype=np.float32)
        _compute_outputs(
            lc, lab, lmp, obj_slices,
            dyIm=dyIm, dxIm=dxIm,
            centerArr=centerArr,
            center_distance_mode="GEODESIC"
        )
        # Object in frame 1 is shifted +2 in both y and x from frame 0
        mask = lab[..., 1] == 1
        self.assertTrue(np.all(dyIm[mask] != 0))
        self.assertTrue(np.all(dxIm[mask] != 0))
        # centerArr should have 2 valid coords
        self.assertFalse(np.any(np.isnan(centerArr[0])))

    def test_3d_displacement(self):
        lab, lc, lmp, obj_slices, sp_shape, ndim = self._setup(3)
        dzIm = np.zeros(sp_shape, dtype=np.float32)
        dyIm = np.zeros(sp_shape, dtype=np.float32)
        dxIm = np.zeros(sp_shape, dtype=np.float32)
        centerArr = np.full((1, 3), np.nan, dtype=np.float32)
        _compute_outputs(
            lc, lab, lmp, obj_slices,
            dzIm=dzIm, dyIm=dyIm, dxIm=dxIm,
            centerArr=centerArr,
            center_distance_mode="GEODESIC"
        )
        mask = lab[..., 1] == 1
        self.assertTrue(np.all(dyIm[mask] != 0))
        self.assertTrue(np.all(dxIm[mask] != 0))
        # dz should also be nonzero (object shifted in Z)
        self.assertTrue(np.any(dzIm[mask] != 0))
        # centerArr should have 3 valid coords
        self.assertFalse(np.any(np.isnan(centerArr[0])))


if __name__ == '__main__':
    unittest.main()

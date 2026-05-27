"""Unit tests for arraylink.array_geometry."""
import numpy as np
import pytest
from arraylink.array_geometry import (
    upa_positions,
    center_dense_positions,
    uniform_grid_positions,
    place_subarrays,
    build_ground_station,
)


class TestUPAPositions:
    def test_shape(self):
        p = upa_positions(4, 8, 0.5)
        assert p.shape == (32, 3)

    def test_centred_at_origin(self):
        p = upa_positions(4, 4, 1.0)
        centroid = p.mean(axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-10)

    def test_spacing(self):
        spacing = 0.5
        p = upa_positions(3, 1, spacing)
        # x coordinates should be -spacing, 0, +spacing
        xs = np.sort(p[:, 0])
        np.testing.assert_allclose(np.diff(xs), spacing, atol=1e-10)

    def test_z_is_zero(self):
        p = upa_positions(4, 4, 0.5)
        assert np.all(p[:, 2] == 0.0)

    def test_center_offset(self):
        center = [1.0, 2.0, 3.0]
        p = upa_positions(2, 2, 1.0, center=center)
        centroid = p.mean(axis=0)
        np.testing.assert_allclose(centroid, center, atol=1e-10)


class TestCenterDensePositions:
    def test_shape(self):
        b = center_dense_positions(4, 4, 1.0, 1.0)
        assert b.shape == (16, 3)

    def test_within_aperture(self):
        Gx, Gy = 1.414, 1.0
        b = center_dense_positions(4, 4, Gx, Gy)
        assert np.all(b[:, 0] >= -Gx / 2 - 1e-9)
        assert np.all(b[:, 0] <= Gx / 2 + 1e-9)
        assert np.all(b[:, 1] >= -Gy / 2 - 1e-9)
        assert np.all(b[:, 1] <= Gy / 2 + 1e-9)

    def test_more_concentrated_than_uniform(self):
        """Center-dense should have more panels near origin than uniform grid."""
        Gx, Gy = 1.0, 1.0
        cd = center_dense_positions(4, 4, Gx, Gy, gamma=3.0)
        ug = uniform_grid_positions(4, 4, Gx, Gy)
        half_r = 0.25  # within central 25% radius
        cd_inner = np.sum(np.linalg.norm(cd[:, :2], axis=1) < half_r)
        ug_inner = np.sum(np.linalg.norm(ug[:, :2], axis=1) < half_r)
        assert cd_inner >= ug_inner, "Center-dense should concentrate more near origin"

    def test_reproducible_with_seed(self):
        b1 = center_dense_positions(4, 4, 1.0, 1.0, jitter_frac=0.1, seed=42)
        b2 = center_dense_positions(4, 4, 1.0, 1.0, jitter_frac=0.1, seed=42)
        np.testing.assert_array_equal(b1, b2)

    def test_different_seeds_differ(self):
        b1 = center_dense_positions(4, 4, 1.0, 1.0, jitter_frac=0.2, seed=1)
        b2 = center_dense_positions(4, 4, 1.0, 1.0, jitter_frac=0.2, seed=2)
        assert not np.allclose(b1, b2)


class TestUniformGridPositions:
    def test_shape(self):
        g = uniform_grid_positions(4, 4, 1.0, 1.0)
        assert g.shape == (16, 3)

    def test_centred_at_origin(self):
        g = uniform_grid_positions(4, 4, 1.0, 1.0)
        centroid = g.mean(axis=0)
        np.testing.assert_allclose(centroid[:2], [0, 0], atol=1e-10)


class TestPlaceSubarrays:
    def test_total_count(self):
        bases = np.zeros((4, 3))
        result = place_subarrays(bases, subarray_shape=(8, 8), element_spacing=0.005)
        assert result.shape == (4 * 64, 3)

    def test_single_base_centred(self):
        base = np.array([[0.0, 0.0, 0.0]])
        result = place_subarrays(base, subarray_shape=(4, 4), element_spacing=1.0)
        centroid = result.mean(axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-10)


class TestBuildGroundStation:
    def test_arraylink_mode(self):
        ants, bases = build_ground_station(
            mode='arraylink',
            subarray_shape=(32, 32),
            element_spacing=0.005,
            Nx=4, Ny=4, Lx=1.414, Ly=1.0,
        )
        assert bases.shape == (16, 3)
        assert ants.shape == (16 * 1024, 3)

    def test_uniform_mode(self):
        ants, bases = build_ground_station(
            mode='uniform',
            subarray_shape=(4, 4),
            element_spacing=0.005,
            Nx=2, Ny=2, Lx=0.5, Ly=0.5,
        )
        assert bases.shape == (4, 3)
        assert ants.shape == (64, 3)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            build_ground_station('invalid', (4, 4), 0.005, 2, 2, 0.5, 0.5)

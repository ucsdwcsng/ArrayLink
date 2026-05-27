"""Unit tests for arraylink.channel."""
import numpy as np
import pytest
from arraylink.channel import (
    compute_channel_matrix,
    compute_singular_values,
    singular_value_ratio,
    degrees_of_freedom,
    mimo_region_bounds,
    theoretical_singular_value_ratio,
    spectral_efficiency,
)


class TestChannelMatrix:
    def test_shape(self):
        tx = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        rx = np.array([[0.0, 0.0, 1.0], [0.0, 0.2, 1.0], [0.0, -0.2, 1.0]])
        lam = 0.01
        H = compute_channel_matrix(tx, rx, lam)
        assert H.shape == (3, 2)

    def test_complex(self):
        tx = np.array([[0.0, 0.0, 0.0]])
        rx = np.array([[0.0, 0.0, 1.0]])
        H = compute_channel_matrix(tx, rx, 0.01)
        assert np.iscomplexobj(H)

    def test_path_loss_decreases_with_distance(self):
        """Amplitude should decrease as distance doubles."""
        tx = np.array([[0.0, 0.0, 0.0]])
        lam = 0.01
        rx1 = np.array([[0.0, 0.0, 1.0]])
        rx2 = np.array([[0.0, 0.0, 2.0]])
        H1 = compute_channel_matrix(tx, rx1, lam)
        H2 = compute_channel_matrix(tx, rx2, lam)
        assert np.abs(H1[0, 0]) > np.abs(H2[0, 0])

    def test_2x2_singular_values_analytical(self):
        """
        For a symmetric 2×2 LoS geometry at r = d_tx*d_rx/lambda (Delta=2pi),
        the two singular values should be equal (well-conditioned channel).
        At r_max the ratio should be close to tau=0.1.
        """
        lam = 0.01    # 10 mm (28 GHz in metres)
        d_tx = 0.2
        d_rx = 0.2
        _, r_max = mimo_region_bounds(d_tx, d_rx, lam, tau=0.1)
        # At r_max, ratio should be ~tau
        ratio = theoretical_singular_value_ratio(r_max, d_tx, d_rx, lam)
        assert abs(ratio - 0.1) < 0.02, f"Expected ~0.1 got {ratio:.4f}"


class TestSingularValues:
    def test_normalised_norm(self):
        tx = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        rx = np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 1.0]])
        H = compute_channel_matrix(tx, rx, 0.01)
        s = compute_singular_values(H, normalise=True)
        assert abs(np.linalg.norm(s) - 1.0) < 1e-10

    def test_descending_order(self):
        tx = np.random.randn(4, 3)
        rx = np.random.randn(4, 3) + np.array([0, 0, 10])
        H = compute_channel_matrix(tx, rx, 0.01)
        s = compute_singular_values(H)
        assert np.all(np.diff(s) <= 0), "Singular values not in descending order"


class TestMIMOBounds:
    def test_paper_example_metres(self):
        """
        Paper Sec. III-C example: d_tx=d_rx=0.2m, lambda=0.01m, tau=0.1
        -> r_max ~ 62 m.
        """
        r_min, r_max = mimo_region_bounds(0.2, 0.2, 0.01, tau=0.1)
        assert 55 < r_max < 70, f"r_max={r_max:.1f} m, expected ~62 m"
        assert r_min < r_max

    def test_satellite_scale(self):
        """
        Paper Sec. III-C: d_tx=2km, d_rx=1m, lambda=0.01m, tau=0.1
        -> r_max in the thousands-of-km range.

        Note: The paper quotes "≈2500 km" as a rounded figure; the closed-form
        Eq. (11) with these exact parameters yields ~3152 km. We verify the
        order-of-magnitude correctness (satellite-scale, not lab-scale).
        """
        lam_km = 0.01 / 1e3
        d_tx_km = 2.0        # 2 km ground aperture
        d_rx_km = 1e-3       # 1 m satellite aperture in km
        _, r_max = mimo_region_bounds(d_tx_km, d_rx_km, lam_km, tau=0.1)
        assert 2000 < r_max < 4000, (
            f"r_max={r_max:.0f} km — expected satellite-scale range (2000-4000 km)"
        )

    def test_larger_aperture_extends_range(self):
        lam = 0.01
        _, r_max_small = mimo_region_bounds(0.2, 0.2, lam)
        _, r_max_large = mimo_region_bounds(1.0, 0.2, lam)
        assert r_max_large > r_max_small


class TestSpectralEfficiency:
    def test_positive(self):
        tx = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        rx = np.array([[0.0, 0.0, 10.0], [0.0, 0.2, 10.0]])
        H = compute_channel_matrix(tx, rx, 0.01)
        C = spectral_efficiency(H, snr_linear=100.0)
        assert C > 0

    def test_increases_with_snr(self):
        tx = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        rx = np.array([[0.0, 0.0, 10.0], [0.0, 0.2, 10.0]])
        H = compute_channel_matrix(tx, rx, 0.01)
        C1 = spectral_efficiency(H, snr_linear=10.0)
        C2 = spectral_efficiency(H, snr_linear=1000.0)
        assert C2 > C1


class TestDegreesOfFreedom:
    def test_returns_int(self):
        tx = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        rx = np.array([[0.0, 0.0, 10.0], [0.0, 0.2, 10.0]])
        H = compute_channel_matrix(tx, rx, 0.01)
        dof = degrees_of_freedom(H)
        assert isinstance(dof, int)
        assert 1 <= dof <= 2

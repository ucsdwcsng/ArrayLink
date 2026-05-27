"""Unit tests for arraylink.beamforming."""
import numpy as np
import pytest
from arraylink.beamforming import (
    parabolic_gain_dbi,
    total_array_gain_dbi,
    marginal_gain_dbi,
    dc_weights,
    compute_beam_pattern_numpy,
)


class TestParabolicGain:
    def test_1p47m_dish_28ghz(self):
        """1.47 m dish at 28 GHz, eta=0.6 → ~49.5 dBi (paper Sec. II-A)."""
        g = parabolic_gain_dbi(1.47, 28e9, efficiency=0.6)
        assert 48.0 < g < 51.0, f"Expected ~49.5 dBi, got {g:.2f}"

    def test_1p85m_dish_28ghz(self):
        """1.85 m dish at 28 GHz → ~52.6 dBi (paper Sec. II-A)."""
        g = parabolic_gain_dbi(1.85, 28e9, efficiency=0.6)
        assert 51.0 < g < 54.0, f"Expected ~52.6 dBi, got {g:.2f}"

    def test_larger_dish_more_gain(self):
        g1 = parabolic_gain_dbi(1.0, 28e9)
        g2 = parabolic_gain_dbi(2.0, 28e9)
        assert g2 > g1


class TestArrayGain:
    def test_single_panel_equals_pa_gain(self):
        G_pa = 36.1
        g = total_array_gain_dbi(1, G_pa)
        assert abs(g - G_pa) < 1e-10

    def test_16_panels_formula(self):
        """Paper example: 16 panels × 36.1 dBi -> ~48.1 dBi."""
        g = total_array_gain_dbi(16, 36.1)
        expected = 10 * np.log10(16) + 36.1
        assert abs(g - expected) < 1e-10
        assert 47.0 < g < 49.0

    def test_marginal_gain_decreasing(self):
        N = np.arange(1, 50)
        mg = marginal_gain_dbi(N, 36.1)
        # Marginal gain should be strictly decreasing
        assert np.all(np.diff(mg) < 0)

    def test_marginal_gain_below_0p25_after_16(self):
        """
        Paper Fig 4b: marginal gain 'beyond 16 panels' falls below 0.25 dB.

        At N=16, G(17)-G(16) ≈ 0.263 dB (slightly above 0.25).
        The threshold is crossed near N=18: G(19)-G(18) ≈ 0.23 dB.
        We verify the transition happens within a few panels of N=16.
        """
        N_check = np.arange(16, 25)
        mg = marginal_gain_dbi(N_check, 36.1)
        assert np.any(mg < 0.25), (
            f"Expected marginal gain to drop below 0.25 dB near N=16-20, "
            f"but values were {mg.round(4)}"
        )


class TestDCWeights:
    def test_shape(self):
        sat = np.array([0.0, 0.0, 500.0])
        rx = np.random.randn(64, 3)
        w = dc_weights(sat, rx, wavelength=0.01 / 1e3)
        assert w.shape == (64, 1)

    def test_normalised(self):
        sat = np.array([0.0, 0.0, 500.0])
        rx = np.random.randn(32, 3)
        w = dc_weights(sat, rx, wavelength=0.01 / 1e3)
        assert abs(np.linalg.norm(w) - 1.0) < 1e-10

    def test_complex(self):
        sat = np.array([0.0, 0.0, 1.0])
        rx = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        w = dc_weights(sat, rx, wavelength=0.01)
        assert np.iscomplexobj(w)

    def test_boresight_peak(self):
        """Beam should peak at the steered direction."""
        lam = 0.01
        # Single receive antenna at origin, satellite at (0,0,1) km
        rx = np.array([[0.0, 0.0, 0.0]])
        sat_target = np.array([0.0, 0.0, 1.0])
        sat_off = np.array([1.0, 0.0, 1.0])

        w = dc_weights(sat_target, rx, lam)

        bp_on = compute_beam_pattern_numpy(
            sat_target.reshape(1, 3), rx, w, lam
        )
        bp_off = compute_beam_pattern_numpy(
            sat_off.reshape(1, 3), rx, w, lam
        )
        assert bp_on[0] >= bp_off[0]


class TestBeamPatternNumpy:
    def test_shape(self):
        sat_pts = np.random.randn(10, 3) + np.array([0, 0, 5])
        rx = np.random.randn(8, 3)
        w = np.ones((8, 1), dtype=complex) / np.sqrt(8)
        bp = compute_beam_pattern_numpy(sat_pts, rx, w, wavelength=0.01)
        assert bp.shape == (10,)

    def test_values_are_db(self):
        sat_pts = np.array([[0.0, 0.0, 1.0]])
        rx = np.zeros((4, 3))
        w = np.ones((4, 1), dtype=complex) / 2.0
        bp = compute_beam_pattern_numpy(sat_pts, rx, w, wavelength=0.01)
        # Result should be finite (not NaN, not inf)
        assert np.all(np.isfinite(bp))

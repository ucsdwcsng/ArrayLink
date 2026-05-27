# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Free-space LoS channel matrix and near-field MIMO analysis.

All positions in the same unit (km recommended for satellite scale;
metre scale is fine for hardware experiments — just keep lambda consistent).

Mathematical reference: ArrayLink paper (INFOCOM 2026), Sec. III-B/C and Appendix.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Channel matrix
# ---------------------------------------------------------------------------

def compute_channel_matrix(tx_pos, rx_pos, wavelength):
    """
    Compute the Nr x Nt free-space spherical-wave channel matrix.

    h_ij = (lambda / (4*pi*d_ij)) * exp(-j*2*pi*d_ij / lambda)

    Assumption: amplitude ~1/d (near-field model); for far-field validation
    only phase terms matter and amplitude differences are negligible.

    Parameters
    ----------
    tx_pos     : (Nt, 3) transmit antenna positions
    rx_pos     : (Nr, 3) receive antenna positions
    wavelength : carrier wavelength (same units as positions)

    Returns
    -------
    H : (Nr, Nt) complex channel matrix
    """
    tx = np.asarray(tx_pos, dtype=float)  # (Nt, 3)
    rx = np.asarray(rx_pos, dtype=float)  # (Nr, 3)

    # pairwise distances  (Nr, Nt)
    diff = rx[:, None, :] - tx[None, :, :]   # (Nr, Nt, 3)
    d = np.linalg.norm(diff, axis=-1)         # (Nr, Nt)

    H = (wavelength / (4 * np.pi * d)) * np.exp(-1j * 2 * np.pi * d / wavelength)
    return H


def compute_singular_values(H, normalise=True):
    """
    SVD of H; optionally normalise so ||s||_2 = 1.

    Returns singular values in descending order.
    """
    s = np.linalg.svd(H, compute_uv=False)   # descending
    if normalise and s[0] > 0:
        s = s / np.linalg.norm(s)
    return s


def singular_value_ratio(H):
    """
    Return sigma_min / sigma_max from the normalised SVD of H.
    For a 2×2 channel this equals sigma_2 / sigma_1.
    Useful for checking MIMO feasibility (threshold tau ~ 0.1).
    """
    s = compute_singular_values(H, normalise=True)
    if s[0] == 0:
        return 0.0
    return float(s[-1] / s[0])


def degrees_of_freedom(H, threshold=0.1):
    """
    Number of spatial streams satisfying the paper's MIMO feasibility criterion.

    Counts k where σ_k / σ_1 ≥ threshold  (paper Eq. 7, default threshold=0.1).

    Note: uses the ratio σ_k/σ_1, NOT the L2-normalised absolute value.
    The two are equivalent only for rank-1 channels; for multi-stream channels
    the L2 norm suppresses all values and would under-count DoF.
    """
    s = compute_singular_values(H, normalise=False)
    if s[0] == 0:
        return 0
    return int(np.sum(s / s[0] >= threshold))


def spectral_efficiency(H, snr_linear):
    """
    MIMO Shannon capacity under uniform power allocation (bits/s/Hz).

    C = sum_k log2(1 + SNR/Nt * sigma_k^2)

    Parameters
    ----------
    H          : (Nr, Nt) channel matrix
    snr_linear : total receive SNR (linear)
    """
    Nt = H.shape[1]
    s = np.linalg.svd(H, compute_uv=False)
    return float(np.sum(np.log2(1.0 + (snr_linear / Nt) * s**2)))


# ---------------------------------------------------------------------------
# MIMO region boundaries (2×2 closed-form, paper Sec. III-C)
# ---------------------------------------------------------------------------

def mimo_region_bounds(d_tx, d_rx, wavelength, tau=0.1):
    """
    Analytical MIMO feasibility boundaries for a 2×2 LoS system.

    Equations (9) and (11) from the paper.

    Parameters
    ----------
    d_tx, d_rx : transmit and receive antenna spacings (any consistent unit)
    wavelength : carrier wavelength (same unit)
    tau        : singular-value ratio threshold (default 0.1)

    Returns
    -------
    r_min, r_max : distance boundaries (same unit as d_tx/d_rx/wavelength)

    Notes
    -----
    At r < r_min the channel is unstable (oscillating singular-value ratio).
    At r_min <= r <= r_max  MIMO is well-conditioned.
    At r > r_max the channel degrades toward rank-1.
    """
    dtdrl = d_tx * d_rx / wavelength
    r_min = (np.pi / (2 * np.arctan(1.0 / tau))) * dtdrl
    r_max = (np.pi / (2 * np.arctan(tau))) * dtdrl
    return float(r_min), float(r_max)


def theoretical_singular_value_ratio(r, d_tx, d_rx, wavelength):
    """
    Closed-form sigma_2/sigma_1 vs distance for a 2×2 perpendicular LoS channel.

    From Eq. (4)-(5): Delta = 2*pi*d_tx*d_rx / (lambda * r)
    sigma_1,2 = sqrt(2 +/- 2*|cos(Delta/2)|)
    ratio = sigma_2 / sigma_1

    Parameters
    ----------
    r          : distance (scalar or array, same unit as d_tx/d_rx/wavelength)
    d_tx, d_rx : antenna spacings
    wavelength : carrier wavelength

    Returns
    -------
    ratio : sigma_2 / sigma_1  (in [0, 1])
    """
    r = np.asarray(r, dtype=float)
    delta = 2 * np.pi * d_tx * d_rx / (wavelength * r)
    s1 = np.sqrt(2 + 2 * np.abs(np.cos(delta / 2)))
    s2 = np.sqrt(2 - 2 * np.abs(np.cos(delta / 2)))
    norm = np.sqrt(s1**2 + s2**2)
    s1n, s2n = s1 / norm, s2 / norm
    return s2n / s1n


# ---------------------------------------------------------------------------
# Multi-stream DoF sweep (satellite scale)
# ---------------------------------------------------------------------------

def dof_vs_distance(dist_array, gnd_coords, sat_params, wavelength, threshold=0.1):
    """
    Compute singular values and DoF for a 4-element satellite array vs. distance.

    Parameters
    ----------
    dist_array   : 1-D array of distances (km)
    gnd_coords   : (M, 3) ground antenna positions (km)
    sat_params   : dict with keys:
                     center_theta, center_phi (radians)
                     subarray_shape [nx, ny]
                     xdim_subarrays, ydim_subarrays
                     Dx_km, Dy_km, seed
    wavelength   : carrier wavelength (km)
    threshold    : singular-value ratio threshold for DoF count

    Returns
    -------
    dof_list          : list of int, DoF at each distance
    sing_vals_list    : list of normalised singular-value arrays
    """
    from .array_geometry import place_subarrays, center_dense_positions

    dof_list = []
    sing_vals_list = []

    for d_km in dist_array:
        # satellite centre in Cartesian
        ct = sat_params['center_theta']
        cp = sat_params['center_phi']
        sat_center = np.array([
            d_km * np.sin(ct) * np.cos(cp),
            d_km * np.sin(ct) * np.sin(cp),
            d_km * np.cos(ct),
        ])

        # satellite antenna positions (small 2-D array centred at sat_center)
        nx, ny = sat_params['subarray_shape']
        spacing = wavelength / 2.0
        sat_coords = place_subarrays(
            np.array([sat_center]),
            subarray_shape=(nx * sat_params['xdim_subarrays'],
                            ny * sat_params['ydim_subarrays']),
            element_spacing=spacing,
        )

        H = compute_channel_matrix(sat_coords, gnd_coords, wavelength)
        s = compute_singular_values(H, normalise=True)
        dof_list.append(int(np.sum(s >= threshold)))
        sing_vals_list.append(s)

    return dof_list, sing_vals_list

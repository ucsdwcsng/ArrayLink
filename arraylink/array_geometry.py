# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Antenna array placement functions.

Supports:
  - Uniform Planar Arrays (UPA / monolithic)
  - ArrayLink center-dense distributed placement
  - Uniform grid distributed placement
  - Subarray expansion (place an M×N panel at each base location)

All positions returned as (N, 3) NumPy arrays in km (or the unit passed in).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Monolithic UPA
# ---------------------------------------------------------------------------

def upa_positions(M, N, element_spacing, center=None):
    """
    Generate a M×N Uniform Planar Array centred at the origin (z=0 plane).

    Parameters
    ----------
    M, N            : element counts along x and y
    element_spacing : element pitch (same unit as returned positions)
    center          : optional (3,) offset; defaults to [0,0,0]

    Returns
    -------
    coords : (M*N, 3) antenna positions
    """
    xs = (np.arange(M) - (M - 1) / 2.0) * element_spacing
    ys = (np.arange(N) - (N - 1) / 2.0) * element_spacing
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), np.zeros(M * N)], axis=1)
    if center is not None:
        coords = coords + np.asarray(center)
    return coords


# ---------------------------------------------------------------------------
# ArrayLink: center-dense panel base placement
# ---------------------------------------------------------------------------

def center_dense_positions(Mx, My, Gx, Gy, gamma=3.0, seed=None, jitter_frac=0.0):
    """
    Generate 2-D center-dense panel base positions using a power-law transform.

    Starting from a uniform Mx×My grid on [-Gx/2, Gx/2] × [-Gy/2, Gy/2],
    apply s -> sign(s)*|s|^gamma to concentrate panels toward the centre,
    then optionally add random jitter.

    Paper reference: Sec. III-D, Fig. 7b.

    Parameters
    ----------
    Mx, My      : number of panels along x and y
    Gx, Gy      : aperture dimensions (km)
    gamma       : power-law exponent; higher = more centre-concentrated (default 3.0)
    seed        : random seed for jitter reproducibility
    jitter_frac : jitter as fraction of minimum inter-panel spacing (default 0 = no jitter)

    Returns
    -------
    bases : (Mx*My, 3) panel base positions (z=0)
    """
    tx = np.linspace(0, 1, Mx)
    ty = np.linspace(0, 1, My)
    sx = 2 * tx - 1                          # -> [-1, 1]
    sy = 2 * ty - 1
    sx = np.sign(sx) * np.abs(sx) ** gamma   # power-law transform
    sy = np.sign(sy) * np.abs(sy) ** gamma
    x_pos = (Gx / 2.0) * sx
    y_pos = (Gy / 2.0) * sy

    X, Y = np.meshgrid(x_pos, y_pos, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), np.zeros(Mx * My)], axis=1)

    if jitter_frac > 0.0 and (Mx > 1 or My > 1):
        rng = np.random.default_rng(seed)
        min_dx = np.min(np.diff(x_pos)) if Mx > 1 else Gx
        min_dy = np.min(np.diff(y_pos)) if My > 1 else Gy
        jitter_scale = jitter_frac * min(min_dx, min_dy)
        coords[:, :2] += rng.uniform(-jitter_scale, jitter_scale, (Mx * My, 2))

    return coords


def uniform_grid_positions(Nx, Ny, Lx, Ly, center=None):
    """
    Generate a uniform Nx×Ny grid spanning Lx×Ly, centred at origin.

    Parameters
    ----------
    Nx, Ny : number of panels along x and y
    Lx, Ly : physical extent (km)
    center : optional (3,) offset

    Returns
    -------
    bases : (Nx*Ny, 3) positions (z=0)
    """
    xs = np.linspace(-Lx / 2.0, Lx / 2.0, Nx)
    ys = np.linspace(-Ly / 2.0, Ly / 2.0, Ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), np.zeros(Nx * Ny)], axis=1)
    if center is not None:
        coords = coords + np.asarray(center)
    return coords


# ---------------------------------------------------------------------------
# Subarray placement
# ---------------------------------------------------------------------------

def place_subarrays(base_positions, subarray_shape, element_spacing):
    """
    Expand each base position into a subarray of shape (M, N).

    Parameters
    ----------
    base_positions  : (K, 3) panel base positions
    subarray_shape  : (M, N) elements per panel
    element_spacing : element pitch within a panel

    Returns
    -------
    all_positions : (K*M*N, 3) full antenna positions
    """
    M, N = subarray_shape
    offsets = []
    for i in range(M):
        for j in range(N):
            dx = (i - (M - 1) / 2.0) * element_spacing
            dy = (j - (N - 1) / 2.0) * element_spacing
            offsets.append([dx, dy, 0.0])
    offsets = np.array(offsets)   # (M*N, 3)

    all_pos = []
    for base in base_positions:
        all_pos.append(base + offsets)
    return np.vstack(all_pos)


# ---------------------------------------------------------------------------
# Full ground-station factory
# ---------------------------------------------------------------------------

def build_ground_station(mode, subarray_shape, element_spacing,
                         Nx, Ny, Lx, Ly,
                         gamma=3.0, seed=None, jitter_frac=0.0,
                         center=None):
    """
    Build ground-station antenna positions.

    Parameters
    ----------
    mode           : 'arraylink' (center-dense) or 'uniform'
    subarray_shape : (M, N) elements per panel
    element_spacing: element pitch within a panel (km)
    Nx, Ny         : number of panels along x and y
    Lx, Ly         : aperture dimensions (km)
    gamma          : power-law exponent for 'arraylink' mode
    seed           : RNG seed for jitter
    jitter_frac    : jitter as fraction of min spacing
    center         : optional (3,) ground-station centre offset (km)

    Returns
    -------
    ant_positions  : (Nx*Ny*M*N, 3) all antenna positions
    base_positions : (Nx*Ny, 3)   panel base positions
    """
    if mode == 'arraylink':
        bases = center_dense_positions(Nx, Ny, Lx, Ly, gamma=gamma,
                                       seed=seed, jitter_frac=jitter_frac)
    elif mode == 'uniform':
        bases = uniform_grid_positions(Nx, Ny, Lx, Ly)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'arraylink' or 'uniform'.")

    if center is not None:
        bases = bases + np.asarray(center)

    ants = place_subarrays(bases, subarray_shape, element_spacing)
    return ants, bases

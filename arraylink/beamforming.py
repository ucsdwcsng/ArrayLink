"""
Beamforming weight generation and beam pattern computation.

Heavy computation (satellite-scale) uses:
  - numexpr for fast exp(-j*psi*d)
  - numpy.memmap for out-of-core distance matrices
  - h5py for compressed beam-pattern storage

For small/quick runs (unit tests, smoke tests), pure-numpy fallbacks are used
when numexpr is not available.
"""
import os
import gc
import tempfile
import numpy as np

try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


# ---------------------------------------------------------------------------
# Parabolic dish reference
# ---------------------------------------------------------------------------

def parabolic_gain_dbi(D, frequency, efficiency=0.6):
    """
    Directional gain of a parabolic dish antenna (dBi).

    G = eta * (pi*D/lambda)^2

    Paper reference: Sec. II-A, Eq. before (2).

    Parameters
    ----------
    D         : dish diameter (metres)
    frequency : carrier frequency (Hz)
    efficiency: aperture efficiency eta (default 0.6, typical 0.5-0.7)

    Returns
    -------
    gain_dbi : float
    """
    c = 3e8
    lam = c / frequency
    gain_linear = efficiency * (np.pi * D / lam) ** 2
    return 10.0 * np.log10(gain_linear)


# ---------------------------------------------------------------------------
# Gain aggregation model
# ---------------------------------------------------------------------------

def total_array_gain_dbi(N, G_pa_dbi):
    """
    Total gain of N coherently combined phased-array panels.

    G_total = 10*log10(N) + G_PA   (paper Eq. (2), ignoring sync loss delta)

    Parameters
    ----------
    N       : number of panels (int or array)
    G_pa_dbi: gain of a single panel (dBi)

    Returns
    -------
    G_total_dbi : float or array
    """
    return 10.0 * np.log10(np.asarray(N, dtype=float)) + G_pa_dbi


def marginal_gain_dbi(N_array, G_pa_dbi):
    """
    Marginal gain from adding one more panel: G(N+1) - G(N).

    Parameters
    ----------
    N_array  : 1-D array of panel counts
    G_pa_dbi : gain of a single panel (dBi)

    Returns
    -------
    marginal : array of same length as N_array
    """
    N = np.asarray(N_array, dtype=float)
    G = total_array_gain_dbi(N + 1, G_pa_dbi) - total_array_gain_dbi(N, G_pa_dbi)
    return G


# ---------------------------------------------------------------------------
# Delay-compensation (DC) weights
# ---------------------------------------------------------------------------

def dc_weights(satellite_loc, rx_coords, wavelength):
    """
    Compute delay-compensation beamforming weights.

    w_k = exp(-j * 2*pi/lambda * d_k) / ||w||

    where d_k is the distance from the satellite to the k-th receive antenna.

    Convention: beam_pattern_numpy computes  af @ conj(w)  where
    af_k = exp(-j*psi*d_k_eval).  With w_k = exp(-j*psi*d_k_focal),
    conj(w_k) = exp(+j*psi*d_k_focal), so the product is
    exp(j*psi*(d_k_focal - d_k_eval)) which sums coherently (= N) at
    boresight.  Using +j here (phase conjugate of h) doubles the phase
    instead of cancelling it and gives completely wrong results for
    distributed arrays with aperture >> lambda.

    Parameters
    ----------
    satellite_loc : (3,) or (1,3) satellite position (km)
    rx_coords     : (M, 3) receive antenna positions (km)
    wavelength    : carrier wavelength (km)

    Returns
    -------
    w : (M, 1) complex normalised weight vector
    """
    sat = np.asarray(satellite_loc, dtype=float).reshape(1, 3)
    rx = np.asarray(rx_coords, dtype=float)
    d = np.linalg.norm(rx - sat, axis=1)          # (M,)
    w = np.exp(-1j * 2 * np.pi / wavelength * d)
    w = w / np.linalg.norm(w)
    return w.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Distance computation (batched, mmap-backed)
# ---------------------------------------------------------------------------

def compute_distances_batched(tx_points, rx_antennas, batch_size=1000,
                              tmp_file=None):
    """
    Compute Euclidean distances from every tx point to every rx antenna.

    Uses a memory-mapped file to avoid loading the full (Ntot×M) matrix
    into RAM at once — essential for satellite-scale grids.

    Parameters
    ----------
    tx_points   : (Ntot, 3) transmit/satellite grid points
    rx_antennas : (M, 3) receive antenna positions
    batch_size  : number of tx points processed per batch
    tmp_file    : optional path for the mmap file; uses tempfile if None

    Returns
    -------
    distances : np.memmap of shape (Ntot, M), float64
                Caller is responsible for deleting tmp_file when done.
    tmp_file  : str, path to the mmap file
    """
    tx = np.asarray(tx_points, dtype=np.float64)
    rx = np.asarray(rx_antennas, dtype=np.float64)
    Ntot, M = tx.shape[0], rx.shape[0]

    if tmp_file is None:
        fd, tmp_file = tempfile.mkstemp(suffix='.dat', prefix='arraylink_dist_')
        os.close(fd)

    distances = np.memmap(tmp_file, dtype=np.float64, mode='w+', shape=(Ntot, M))

    for start in range(0, Ntot, batch_size):
        end = min(start + batch_size, Ntot)
        dx = tx[start:end, 0:1] - rx[:, 0]   # (batch, M)
        dy = tx[start:end, 1:2] - rx[:, 1]
        dz = tx[start:end, 2:3] - rx[:, 2]
        if _HAS_NUMEXPR:
            distances[start:end] = ne.evaluate("sqrt(dx**2 + dy**2 + dz**2)")
        else:
            distances[start:end] = np.sqrt(dx**2 + dy**2 + dz**2)

    distances.flush()
    return distances, tmp_file


# ---------------------------------------------------------------------------
# Beam pattern (h5py-backed, batched over the first axis of the grid)
# ---------------------------------------------------------------------------

def compute_beam_pattern(
    grid_shape, satellite_points, rx_coords,
    weights, wavelength,
    gain_variations=True,
    outer_batch_size=10,
    min_limit_db=-50,
    output_file=None,
    dist_batch_size=1000,
):
    """
    Compute the beam pattern over a 3-D satellite grid.

    For each point (r, theta, phi) or (x, y, z) on the grid, evaluates

        P(point) = |A(point) * w*|^2   in dB

    where A is the array factor vector exp(-j*2*pi/lambda * d_k).

    When gain_variations=True the amplitude is scaled by the free-space
    path loss (1/d), matching the near-field model.

    Results are stored in an HDF5 file to keep memory bounded.

    Parameters
    ----------
    grid_shape       : (D, T, P) tuple matching satellite_points first 3 dims
    satellite_points : (D, T, P, 3) Cartesian satellite grid (km)
    rx_coords        : (M, 3) receive antenna positions (km)
    weights          : (M, 1) complex beamforming weights
    wavelength       : carrier wavelength (km)
    gain_variations  : include 1/d amplitude weighting (default True)
    outer_batch_size : rows of dim-0 processed per iteration
    min_limit_db     : floor clamp for the dB output
    output_file      : path for HDF5 output; uses tempfile if None
    dist_batch_size  : inner batch size for distance computation

    Returns
    -------
    output_file : str, path to the HDF5 file containing dataset 'beam_pattern_dB'
    """
    if not _HAS_H5PY:
        raise ImportError("h5py is required for compute_beam_pattern. "
                          "Install with: pip install h5py")

    D, T, P = grid_shape
    psi = 2 * np.pi / wavelength

    if output_file is None:
        fd, output_file = tempfile.mkstemp(suffix='.h5', prefix='arraylink_bp_')
        os.close(fd)

    with h5py.File(output_file, 'w') as f:
        dset = f.create_dataset(
            'beam_pattern_dB', shape=(D, T, P),
            dtype=np.float64, chunks=True, compression='gzip'
        )

        for start in range(0, D, outer_batch_size):
            end = min(start + outer_batch_size, D)
            batch_pts = satellite_points[start:end]          # (batch, T, P, 3)
            flat_pts = batch_pts.reshape(-1, 3)              # (batch*T*P, 3)

            dist_flat, tmp_dist = compute_distances_batched(
                flat_pts, rx_coords, batch_size=dist_batch_size
            )

            # Array factor:  exp(-j*psi*d)   shape (batch*T*P, M)
            if _HAS_NUMEXPR:
                af = ne.evaluate("exp(-1j * psi * dist_flat)")
            else:
                af = np.exp(-1j * psi * dist_flat)

            if gain_variations:
                # Scale by absolute satellite distance to account for path loss
                abs_dist = np.linalg.norm(flat_pts, axis=1, keepdims=True)  # (N,1)
                abs_dist = np.where(abs_dist == 0, np.finfo(float).eps, abs_dist)
                af = af / (dist_flat / abs_dist)   # relative amplitude weighting

            # Beam pattern: (batch*T*P, 1) -> dB
            bp_linear = af @ np.conj(weights)        # (N, 1)
            bp_db = 20.0 * np.log10(np.abs(bp_linear) + np.finfo(float).eps)
            bp_db = np.clip(bp_db, min_limit_db, None)
            dset[start:end] = bp_db.reshape(end - start, T, P)

            del dist_flat, af, bp_linear, bp_db
            os.remove(tmp_dist)
            gc.collect()

    return output_file


# ---------------------------------------------------------------------------
# Quick (pure-numpy, no h5py) beam pattern for small grids / tests
# ---------------------------------------------------------------------------

def compute_beam_pattern_numpy(satellite_points, rx_coords, weights, wavelength,
                               min_limit_db=-50):
    """
    Pure-numpy beam pattern computation for small grids (tests / quick mode).

    Parameters
    ----------
    satellite_points : (Ntot, 3) Cartesian satellite grid (km)
    rx_coords        : (M, 3) receive antenna positions (km)
    weights          : (M, 1) complex weights
    wavelength       : carrier wavelength (km)

    Returns
    -------
    bp_db : (Ntot,) beam pattern in dB
    """
    d = np.linalg.norm(
        satellite_points[:, None, :] - rx_coords[None, :, :], axis=-1
    )  # (Ntot, M)
    psi = 2 * np.pi / wavelength
    af = np.exp(-1j * psi * d)                      # (Ntot, M)
    bp = af @ np.conj(weights)                       # (Ntot, 1)
    bp_db = 20.0 * np.log10(np.abs(bp).ravel() + np.finfo(float).eps)
    return np.clip(bp_db, min_limit_db, None)

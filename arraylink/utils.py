# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Coordinate conversion, decibel helpers, and config loading.

All coordinates in kilometres unless otherwise noted in the function signature.
"""
import os
import numpy as np


def load_config(path):
    """
    Load a YAML configuration file.

    The path may be absolute or relative to the repository root.
    The repository root is inferred as the parent directory of the
    ``arraylink/`` package (i.e. two levels up from this file).

    Parameters
    ----------
    path : str
        Path to the YAML config file, e.g. ``'configs/arraylink_1km.yaml'``.

    Returns
    -------
    cfg : dict
        Parsed YAML contents.

    Examples
    --------
    >>> from arraylink.utils import load_config
    >>> cfg = load_config('configs/arraylink_1km.yaml')
    >>> cfg['frequency_hz']
    28000000000.0  # stored as 28.0e+9 in YAML → parsed as float by PyYAML
    """
    import yaml
    if not os.path.isabs(path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, path)
    with open(path) as f:
        return yaml.safe_load(f)


def spherical2cartesian(r, theta, phi, deg=False):
    """
    r     : radius (km)
    theta : inclination from +z axis  (0 <= theta <= pi)
    phi   : azimuth from +x axis      (-pi < phi <= pi)
    """
    if deg:
        theta = np.radians(theta)
        phi = np.radians(phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian2spherical(x, y, z, deg=False):
    """Returns (r, theta, phi). theta in [0, pi], phi in (-pi, pi]."""
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r == 0, np.finfo(float).eps, r)
    theta = np.arccos(z / r_safe)
    phi = np.arctan2(y, x)
    if deg:
        theta = np.degrees(theta)
        phi = np.degrees(phi)
    return r, theta, phi


def linear2db(x):
    """Convert linear amplitude to dB (20*log10)."""
    return 20.0 * np.log10(np.abs(x))


def db2linear(x_db):
    """Convert dB to linear amplitude."""
    return 10.0 ** (x_db / 20.0)


def generate_grid_points(x, y, z, spherical=False):
    """
    Build a 3-D grid of Cartesian points.

    Parameters
    ----------
    x, y, z : 1-D arrays
        When spherical=False  -> Cartesian axes (km).
        When spherical=True   -> r (km), theta (deg), phi (deg).

    Returns
    -------
    grid : (Nx, Ny, Nz, 3) array of Cartesian coordinates
    X, Y, Z : (Nx, Ny, Nz) component arrays
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if spherical:
        r = x[:, None, None]
        theta = np.deg2rad(y)[None, :, None]
        phi = np.deg2rad(z)[None, None, :]
        R, Theta, Phi = np.broadcast_arrays(r, theta, phi)
        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)
    else:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid = np.stack((X, Y, Z), axis=-1)
    return grid, X, Y, Z

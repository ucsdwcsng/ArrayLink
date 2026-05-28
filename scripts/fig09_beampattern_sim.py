# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Fig 9 — Simulation setup and beam pattern results.

(a) 2D UPA 128×128 element positions (λ/2 spacing)
(b) ArrayLink: 16 × 32×32 panel center positions in a 2 km × 2 km grid
(c) Beam pattern vs elevation angle θ ∈ [−90°, +90°] at r = 500 km
(d) Beam pattern vs distance r ∈ [0, 2000] km at θ = 0° (boresight)

Two ArrayLink layouts are shown (seed_0, seed_1 = two different random placements).
--center-dense adds a third ArrayLink curve using a 4×4 power-law grid.

Panel placement (random layout):
  - 4 corner panels fixed at (±1 km, ±1 km) to span the full aperture
  - Remaining 12 panels drawn uniformly within the 2 km × 2 km grid
  - Minimum separation between any two panel centres enforced

Usage
-----
  python scripts/fig09_beampattern_sim.py [--quick] [--center-dense]
                                          [--save-dir paper_figures]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

from arraylink.array_geometry import upa_positions, place_subarrays
from arraylink.beamforming import dc_weights, compute_beam_pattern_numpy
from arraylink.utils import load_config

# --------------------------------------------------------------------------
# Fig-09-specific constants (not in YAML — visualization layout choices)
# --------------------------------------------------------------------------
GRID_KM = 2.0    # panel centres drawn within ±1 km × ±1 km (display aperture)
MIN_GAP = 0.1    # minimum separation between panel centres [km]
N_UPA   = 128    # UPA dimension (128×128) used only in this figure


# --------------------------------------------------------------------------
# Panel placement helpers
# --------------------------------------------------------------------------
def _place_random_2d(n_panels, grid_km, min_gap_km, seed):
    """Random panel centres; 4 corners fixed, remainder uniformly drawn."""
    rng = np.random.default_rng(seed)
    half = grid_km / 2.0
    panels = [(-half, -half), (-half, half), (half, -half), (half, half)]
    attempts = 0
    while len(panels) < n_panels and attempts < 500_000:
        p = rng.uniform(-half, half, size=2)
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_gap_km for q in panels):
            panels.append(tuple(p))
        attempts += 1
    if len(panels) < n_panels:
        raise RuntimeError(
            f"Could only place {len(panels)}/{n_panels} panels "
            f"with min_gap={min_gap_km} km — reduce MIN_GAP or increase GRID_KM."
        )
    return np.array(panels)  # (n_panels, 2)


def _place_center_dense_2d(nx, ny, grid_km, gamma=3.0):
    """4×4 power-law grid — denser at centre (ArrayLink placement)."""
    half = grid_km / 2.0
    def warp(n):
        t = np.linspace(0, 1, n)
        s = 2 * t - 1
        return (half * np.sign(s) * np.abs(s) ** gamma)
    xs = warp(nx)
    ys = warp(ny)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])  # (nx*ny, 2)


def _build_arraylink(panel_xy_km, sub_shape, elem_spacing_km):
    """All element positions for a distributed array with given panel centres."""
    centres_3d = np.hstack([panel_xy_km, np.zeros((len(panel_xy_km), 1))])
    return place_subarrays(centres_3d, sub_shape, elem_spacing_km)  # (N, 3) km


# --------------------------------------------------------------------------
# Beam pattern helpers
# --------------------------------------------------------------------------
def _gain_vs_theta(ants_km, w, r_km, theta_deg, lam_km):
    thetas = np.deg2rad(theta_deg)
    sat_pts = np.column_stack([
        r_km * np.sin(thetas),
        np.zeros_like(thetas),
        r_km * np.cos(thetas),
    ])
    return compute_beam_pattern_numpy(sat_pts, ants_km, w, lam_km)  # dB


def _gain_vs_dist(ants_km, w, r_km_array, lam_km):
    sat_pts = np.column_stack([
        np.zeros(len(r_km_array)),
        np.zeros(len(r_km_array)),
        r_km_array,
    ])
    return compute_beam_pattern_numpy(sat_pts, ants_km, w, lam_km)  # dB


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Fig 9: simulation setup and beam pattern")
    ap.add_argument("--config", default="configs/arraylink_1km.yaml",
                    help="Path to YAML config (default: configs/arraylink_1km.yaml)")
    ap.add_argument("--quick", action="store_true",
                    help="Coarse grids for fast smoke test")
    ap.add_argument("--center-dense", action="store_true",
                    help="Add a third ArrayLink curve with center-dense 4×4 layout")
    ap.add_argument("--save-dir", default="paper_figures")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load config — students can change these in configs/arraylink_1km.yaml
    # and see the effect on all figures that use this config.
    # ------------------------------------------------------------------ #
    cfg       = load_config(args.config)
    F_HZ      = cfg['frequency_hz']
    LAM_KM    = 3e8 / F_HZ / 1e3
    R_FOCAL   = cfg['target_satellite']['r_km']
    SUB_SHAPE = tuple(cfg['ground_station']['subarray_shape'])
    N_PANELS  = cfg['ground_station']['Nx'] * cfg['ground_station']['Ny']
    ELEM_GAIN = cfg['computation']['element_gain_dbi']

    if args.quick:
        theta_deg  = np.linspace(-90, 90, 19)
        r_axis_km  = np.linspace(50, 2000, 11)
    else:
        theta_deg  = np.linspace(-90, 90, 1801)
        r_axis_km  = np.linspace(10, 2000, 500)

    # ------------------------------------------------------------------ #
    # Build arrays
    # ------------------------------------------------------------------ #
    # UPA baseline
    upa_ants = upa_positions(N_UPA, N_UPA, LAM_KM / 2)       # (16384, 3) km

    # ArrayLink: two random seeds
    al_xy_s0 = _place_random_2d(N_PANELS, GRID_KM, MIN_GAP, seed=0)
    al_ants_s0 = _build_arraylink(al_xy_s0, SUB_SHAPE, LAM_KM / 2)

    al_xy_s1 = _place_random_2d(N_PANELS, GRID_KM, MIN_GAP, seed=1)
    al_ants_s1 = _build_arraylink(al_xy_s1, SUB_SHAPE, LAM_KM / 2)

    # Optional center-dense
    if args.center_dense:
        al_xy_cd = _place_center_dense_2d(4, 4, GRID_KM, gamma=3.0)
        al_ants_cd = _build_arraylink(al_xy_cd, SUB_SHAPE, LAM_KM / 2)

    sat_boresight = np.array([0.0, 0.0, R_FOCAL])

    # DC weights (focused at 500 km boresight)
    w_upa = dc_weights(sat_boresight, upa_ants, LAM_KM)
    w_s0  = dc_weights(sat_boresight, al_ants_s0, LAM_KM)
    w_s1  = dc_weights(sat_boresight, al_ants_s1, LAM_KM)
    if args.center_dense:
        w_cd = dc_weights(sat_boresight, al_ants_cd, LAM_KM)

    print(f"UPA elements   : {len(upa_ants)}")
    print(f"ArrayLink elems: {len(al_ants_s0)}")

    # ------------------------------------------------------------------ #
    # Fig 9a — UPA element positions
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(upa_ants[:, 0] * 1e3, upa_ants[:, 1] * 1e3,
               s=0.3, color='royalblue', alpha=0.6)
    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_title(f"UPA {N_UPA}×{N_UPA} (λ/2 spacing)", fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Inset: zoom on top-right 32×32 subarray block
    corner_idx = [r * N_UPA + c
                  for r in range(N_UPA - 32, N_UPA)
                  for c in range(N_UPA - 32, N_UPA)]
    corner_elems = upa_ants[corner_idx]
    x_lo = corner_elems[:, 0].min() * 1e3
    x_hi = corner_elems[:, 0].max() * 1e3
    y_lo = corner_elems[:, 1].min() * 1e3
    y_hi = corner_elems[:, 1].max() * 1e3
    pad = max(x_hi - x_lo, y_hi - y_lo) * 0.2
    axins = ax.inset_axes([0.02, 0.58, 0.40, 0.40])
    axins.scatter(corner_elems[:, 0] * 1e3, corner_elems[:, 1] * 1e3,
                  s=3, color='royalblue', alpha=0.8)
    axins.set_xlim(x_lo - pad, x_hi + pad)
    axins.set_ylim(y_lo - pad, y_hi + pad)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.grid(True, alpha=0.3)
    ax.add_patch(Rectangle(
        (x_lo - pad, y_lo - pad), (x_hi - x_lo + 2 * pad), (y_hi - y_lo + 2 * pad),
        linewidth=1.5, edgecolor='darkorange', facecolor='none'
    ))

    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig09a_upa_positions.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Fig 9b — ArrayLink panel positions (seed 0)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(4, 4))
    # All elements (light)
    ax.scatter(al_ants_s0[:, 0], al_ants_s0[:, 1],
               s=0.3, color='royalblue', alpha=0.3, label="Subarray Elements")
    # Panel centres (prominent)
    ax.scatter(al_xy_s0[:, 0], al_xy_s0[:, 1],
               s=60, color='crimson', marker='x', linewidths=1.5,
               label="Subarray Centers")
    ax.set_xlabel("X [km]", fontsize=11)
    ax.set_ylabel("Y [km]", fontsize=11)
    ax.set_title(f"ArrayLink: {N_PANELS} × {SUB_SHAPE[0]}×{SUB_SHAPE[1]} UPAs", fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Inset: zoom on one panel (closest to grid centre)
    dists_from_origin = np.linalg.norm(al_xy_s0, axis=1)
    panel_idx = np.argmin(dists_from_origin)
    cx, cy = al_xy_s0[panel_idx]
    start = panel_idx * SUB_SHAPE[0] * SUB_SHAPE[1]
    end   = start + SUB_SHAPE[0] * SUB_SHAPE[1]
    sub_e = al_ants_s0[start:end]
    x_lo = sub_e[:, 0].min()
    x_hi = sub_e[:, 0].max()
    y_lo = sub_e[:, 1].min()
    y_hi = sub_e[:, 1].max()
    pad = max(x_hi - x_lo, y_hi - y_lo) * 0.25
    axins = ax.inset_axes([0.02, 0.58, 0.40, 0.40])
    axins.scatter(sub_e[:, 0], sub_e[:, 1], s=2, color='royalblue', alpha=0.8)
    axins.scatter([cx], [cy], s=40, color='crimson', marker='x', linewidths=1.5)
    axins.set_xlim(x_lo - pad, x_hi + pad)
    axins.set_ylim(y_lo - pad, y_hi + pad)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.grid(True, alpha=0.3)

    legend_elems = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
                   markersize=5, label='Subarray Elements'),
        plt.Line2D([0], [0], marker='x', color='crimson', markersize=7,
                   linewidth=0, label='Subarray Centers'),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc='lower right')
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig09b_arraylink_positions.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    if args.quick:
        print("Quick mode: running beam pattern on coarse grid.")

    # ------------------------------------------------------------------ #
    # Compute beam patterns
    # ------------------------------------------------------------------ #
    bp_upa_theta  = _gain_vs_theta(upa_ants,  w_upa, R_FOCAL, theta_deg, LAM_KM) + ELEM_GAIN
    bp_s0_theta   = _gain_vs_theta(al_ants_s0, w_s0, R_FOCAL, theta_deg, LAM_KM) + ELEM_GAIN
    bp_s1_theta   = _gain_vs_theta(al_ants_s1, w_s1, R_FOCAL, theta_deg, LAM_KM) + ELEM_GAIN

    bp_upa_dist   = _gain_vs_dist(upa_ants,  w_upa, r_axis_km, LAM_KM) + ELEM_GAIN
    bp_s0_dist    = _gain_vs_dist(al_ants_s0, w_s0, r_axis_km, LAM_KM) + ELEM_GAIN
    bp_s1_dist    = _gain_vs_dist(al_ants_s1, w_s1, r_axis_km, LAM_KM) + ELEM_GAIN

    if args.center_dense:
        bp_cd_theta = _gain_vs_theta(al_ants_cd, w_cd, R_FOCAL, theta_deg, LAM_KM) + ELEM_GAIN
        bp_cd_dist  = _gain_vs_dist(al_ants_cd, w_cd, r_axis_km, LAM_KM) + ELEM_GAIN

    boresight_upa = bp_upa_dist[np.argmin(np.abs(r_axis_km - R_FOCAL))]
    boresight_s0  = bp_s0_dist[np.argmin(np.abs(r_axis_km - R_FOCAL))]
    print(f"Boresight gain UPA      : {boresight_upa:.1f} dBi")
    print(f"Boresight gain ArrayLink: {boresight_s0:.1f} dBi")

    fig_kw = dict(fontsize=14, tick_labelsize=12)

    # ------------------------------------------------------------------ #
    # Fig 9c — Gain vs angle
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(theta_deg, bp_upa_theta, color='steelblue', linewidth=1.5, label="UPA")
    ax.plot(theta_deg, bp_s0_theta,  color='darkorange', linewidth=1.2,
            linestyle='-.', label="ArrayLink (rand., seed 0)")
    ax.plot(theta_deg, bp_s1_theta,  color='seagreen', linewidth=1.2,
            linestyle='--', label="ArrayLink (rand., seed 1)")
    if args.center_dense:
        ax.plot(theta_deg, bp_cd_theta, color='purple', linewidth=1.2,
                linestyle=':', label="ArrayLink (center-dense)")
    ax.set_xlabel(r"$\theta$ (in deg)", fontsize=fig_kw['fontsize'])
    ax.set_ylabel("Gain (dB)", fontsize=fig_kw['fontsize'])
    ax.set_xlim(-90, 90)
    ax.tick_params(labelsize=fig_kw['tick_labelsize'])
    ax.legend(fontsize=11, framealpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig09c_gain_vs_angle.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Fig 9d — Gain vs distance
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(r_axis_km, bp_upa_dist, color='steelblue', linewidth=1.5, label="UPA")
    ax.plot(r_axis_km, bp_s0_dist,  color='darkorange', linewidth=1.2,
            linestyle='-.', label="ArrayLink (rand., seed 0)")
    ax.plot(r_axis_km, bp_s1_dist,  color='seagreen', linewidth=1.2,
            linestyle='--', label="ArrayLink (rand., seed 1)")
    if args.center_dense:
        ax.plot(r_axis_km, bp_cd_dist, color='purple', linewidth=1.2,
                linestyle=':', label="ArrayLink (center-dense)")
    ax.set_xlabel("Distance (km)", fontsize=fig_kw['fontsize'])
    ax.set_ylabel("Gain (dB)", fontsize=fig_kw['fontsize'])
    ax.set_xlim(0, 2000)
    ax.axvline(R_FOCAL, color='gray', linestyle=':', linewidth=1,
               label=f"Focal point ({R_FOCAL:.0f} km)")
    ax.tick_params(labelsize=fig_kw['tick_labelsize'])
    ax.legend(fontsize=11, framealpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig09d_gain_vs_distance.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

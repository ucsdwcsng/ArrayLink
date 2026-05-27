# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Fig 12 — LoS MIMO degrees of freedom vs distance for ArrayLink.

Reproduces Fig. 12 from the ArrayLink paper (INFOCOM 2026).

Computes the 4x4 MIMO channel between a satellite 4-element array
(2x2 panels, 1.414m x 1m grid) and the ArrayLink ground station
(16 x 32x32 panels over 1.414km x 1km), then plots:
  (a) Normalised singular-value ratios sigma_k/sigma_1 vs distance
  (b) Effective number of spatial streams (DoF) vs distance

Usage
-----
  python scripts/fig12_mimo_dof.py [--quick] [--save-dir paper_figures]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

from arraylink.channel import compute_channel_matrix, compute_singular_values
from arraylink.array_geometry import build_ground_station, upa_positions
from arraylink.utils import spherical2cartesian, load_config


def parse_args():
    ap = argparse.ArgumentParser(description="Fig 12: MIMO DoF vs distance")
    ap.add_argument("--config", default="configs/arraylink_1km.yaml",
                    help="Path to YAML config (default: configs/arraylink_1km.yaml)")
    ap.add_argument("--quick", action="store_true",
                    help="Run on a coarse distance grid (fast, for smoke test)")
    ap.add_argument("--save-dir", default="paper_figures")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load config — students can change these in configs/arraylink_1km.yaml ---
    cfg = load_config(args.config)
    F_HZ   = cfg['frequency_hz']
    LAM_KM = 3e8 / F_HZ / 1e3   # wavelength in km (~1.07e-5 km)
    TAU    = 0.1                  # feasibility threshold

    # Ground station geometry from YAML
    GND_NX      = cfg['ground_station']['Nx']
    GND_NY      = cfg['ground_station']['Ny']
    GND_LX      = cfg['ground_station']['aperture_x_km']   # sqrt(2) km
    GND_LY      = cfg['ground_station']['aperture_y_km']   # 1.0 km
    PANEL_SHAPE = tuple(cfg['ground_station']['subarray_shape'])

    # Satellite: 2x2 element array (aperture from YAML, in metres → convert to km)
    SAT_DX_KM = cfg['satellite']['aperture_x_m'] / 1e3
    SAT_DY_KM = cfg['satellite']['aperture_y_m'] / 1e3

    if args.quick:
        dist_array_km = np.linspace(100, 3000, 5)
    else:
        dist_array_km = np.arange(100, 3001, 50, dtype=float)

    # Build ground station once (positions don't change with distance)
    gnd_ants, gnd_bases = build_ground_station(
        mode='arraylink',
        subarray_shape=PANEL_SHAPE,
        element_spacing=LAM_KM / 2.0,
        Nx=GND_NX, Ny=GND_NY,
        Lx=GND_LX, Ly=GND_LY,
        gamma=3.0, seed=0,
    )

    # 2x2 satellite array template (centred at origin, will be translated)
    sat_template = upa_positions(2, 2, element_spacing=SAT_DX_KM / 2.0)
    # The above gives spacing SAT_DX_KM/2 between elements; adjust so the
    # max separation is SAT_DX_KM x SAT_DY_KM
    span_x = np.ptp(sat_template[:, 0])
    span_y = np.ptp(sat_template[:, 1])
    sat_template[:, 0] *= (SAT_DX_KM / span_x) if span_x > 0 else 1
    sat_template[:, 1] *= (SAT_DY_KM / span_y) if span_y > 0 else 1

    # Collect singular-value ratios vs distance
    # sigma_k / sigma_1 for k=1,2,3 (paper Fig 12a uses 3 ratios beyond sigma_1)
    sing_ratios = {1: [], 2: [], 3: []}
    dof_list = []

    for d_km in dist_array_km:
        # Place satellite directly above (theta=0, phi=0) at distance d_km
        sat_center = np.array([0.0, 0.0, d_km])
        sat_coords = sat_template + sat_center

        H = compute_channel_matrix(sat_coords, gnd_ants, LAM_KM)
        s = compute_singular_values(H, normalise=True)

        # sigma_k / sigma_0 for k=1,2,3
        for k in [1, 2, 3]:
            ratio = float(s[k] / s[0]) if len(s) > k and s[0] > 0 else 0.0
            sing_ratios[k].append(ratio)

        # Paper criterion: σ_k/σ_1 ≥ τ  (Eq. 7).  s[k]/s[0] = σ_k/σ_1
        # (L2-normalisation factors cancel in the ratio).
        dof = int(np.sum(s / s[0] >= TAU)) if s[0] > 0 else 0
        dof_list.append(dof)

    if args.quick:
        print("DoF at each distance:", list(zip(dist_array_km.astype(int), dof_list)))
        return

    fig_params = dict(fontsize=16, tick_labelsize=14)
    colors = ['steelblue', 'darkorange', 'green']
    labels = [r'$\sigma_2/\sigma_1$', r'$\sigma_3/\sigma_1$', r'$\sigma_4/\sigma_1$']

    # --- Fig 12a: singular-value ratios ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, k in enumerate([1, 2, 3]):
        ax.plot(dist_array_km, sing_ratios[k], linewidth=2,
                color=colors[idx], label=labels[idx])
    ax.axhline(TAU, linestyle='--', color='firebrick', linewidth=1.5,
               label=f'Threshold (τ={TAU})')
    ax.set_xlabel("Distance (km)", fontsize=fig_params['fontsize'])
    ax.set_ylabel(r"Singular-value ratio $\sigma_k/\sigma_1$",
                  fontsize=fig_params['fontsize'])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 3000)
    ax.tick_params(labelsize=fig_params['tick_labelsize'])
    ax.legend(fontsize=13, framealpha=0.5)
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig12a_singular_value_ratios.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # --- Fig 12b: DoF vs distance ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(dist_array_km, dof_list, linewidth=2, color='steelblue', label='DoFs')
    ax.axhline(TAU, linestyle='--', color='firebrick', linewidth=1.5,
               label=f'Threshold (τ={TAU})', alpha=0)  # hidden, just for legend alignment
    ax.set_xlabel("Distance (km)", fontsize=fig_params['fontsize'])
    ax.set_ylabel(r"DoF (prominent $\sigma$ values)", fontsize=fig_params['fontsize'])
    ax.set_ylim(0.5, 4.5)
    ax.set_xlim(0, 3000)
    ax.set_yticks([1, 2, 3, 4])
    ax.tick_params(labelsize=fig_params['tick_labelsize'])
    ax.legend(fontsize=13, framealpha=0.5)
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig12b_dof_vs_distance.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

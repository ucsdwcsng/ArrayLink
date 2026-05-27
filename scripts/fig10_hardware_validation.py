# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Fig 10 — Hardware validation: singular-value ratio vs distance (4 aperture cases).

Reproduces Fig. 10 from the ArrayLink paper (INFOCOM 2026).

Overlays three curves for each aperture case (d_tx, d_rx):
  - Hardware measurements (from hardware_metrics.pkl)
  - Simulation (computed here from the channel model, Eq. 3)
  - Theory (closed-form, Eqs. 4-5)

Hardware data
-------------
Place `hardware_metrics.pkl` in the `data/` directory (default path).
The pkl file must be a dict keyed by case name with entries:
  {
    'case1': {'distances': [...], 'mean_sing_ratio': [...], 'std_sing_ratio': [...]},
    ...
  }

If the file is not found, the script prints instructions and exits.

Usage
-----
  python scripts/fig10_hardware_validation.py [--quick]
      [--data-path data/hardware_metrics.pkl]
      [--save-dir paper_figures]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

from arraylink.channel import (
    compute_channel_matrix, compute_singular_values,
    theoretical_singular_value_ratio,
)
from arraylink.array_geometry import upa_positions


# Hardware experiment aperture configurations (paper Sec. IV-B, Table I)
CASES = {
    'case1': {'d_tx_cm': 20, 'd_rx_cm': 20},
    'case2': {'d_tx_cm': 50, 'd_rx_cm': 20},
    'case3': {'d_tx_cm': 20, 'd_rx_cm': 50},
    'case4': {'d_tx_cm': 50, 'd_rx_cm': 50},
}

# The hardware_metrics.pkl (from satellite-ground-station) uses a different
# case numbering convention.  This map translates pkl keys → ArrayLink keys.
#
#   pkl key  Drx   Dtx     ArrayLink key
#   case1    50cm  50cm  → case4
#   case2    20cm  50cm  → case2  (same apertures, same label — coincidence)
#   case3    20cm  20cm  → case1
#   case4    50cm  20cm  → case3
#   case0    40cm  40cm  → (not used in ArrayLink)
#
PKL_TO_ARRAYLINK = {
    'case1': 'case4',
    'case2': 'case2',
    'case3': 'case1',
    'case4': 'case3',
}


def parse_args():
    ap = argparse.ArgumentParser(description="Fig 10: hardware validation")
    ap.add_argument("--quick", action="store_true",
                    help="Skip hardware file check; plot theory + sim only")
    ap.add_argument("--data-path", default="data/hardware_metrics.pkl",
                    help="Path to hardware_metrics.pkl (default: data/hardware_metrics.pkl)")
    ap.add_argument("--save-dir", default="paper_figures")
    return ap.parse_args()


def simulate_ratio(d_tx_m, d_rx_m, wavelength_m, dist_m_array):
    """Compute sigma_2/sigma_1 from the exact channel model (Eq. 3)."""
    ratios = []
    for r_m in dist_m_array:
        # 2-element TX and RX arrays, perpendicular to LoS
        tx_pos = np.array([[-d_tx_m / 2, 0, 0], [d_tx_m / 2, 0, 0]])
        rx_pos = np.array([[0, -d_rx_m / 2, r_m], [0, d_rx_m / 2, r_m]])
        H = compute_channel_matrix(tx_pos, rx_pos, wavelength_m)
        s = compute_singular_values(H, normalise=True)
        ratios.append(float(s[1] / s[0]) if s[0] > 0 else 0.0)
    return np.array(ratios)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    F_HZ = 27e9          # hardware experiment carrier (27 GHz)
    LAM_M = 3e8 / F_HZ  # ~0.0111 m
    TAU = 0.1

    # Try to load hardware data
    hw_data = None
    data_path = args.data_path
    if not os.path.isabs(data_path):
        # resolve relative to repo root (parent of scripts/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(script_dir), data_path)

    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        # Remap pkl case keys to ArrayLink's case convention (apertures differ)
        hw_data = {PKL_TO_ARRAYLINK[k]: v for k, v in raw.items()
                   if k in PKL_TO_ARRAYLINK}
        print(f"Loaded hardware data from {data_path} "
              f"({len(hw_data)} cases remapped)")
    else:
        if args.quick:
            print(f"[quick mode] Hardware data not found at {data_path}. "
                  "Plotting theory + simulation only.")
        else:
            print(f"\nHardware data not found at: {data_path}")
            print("To include hardware measurements in Fig 10:")
            print("  1. Obtain hardware_metrics.pkl from the data release")
            print("  2. Place it in data/hardware_metrics.pkl (or pass --data-path)")
            print("\nProceeding with theory + simulation curves only.\n")

    # Distance range for simulation / theory
    if args.quick:
        dist_theory_m = np.linspace(1, 100, 50)
        dist_sim_m = np.array([2.5, 5.0, 10.0, 20.0, 50.0, 100.0])
    else:
        dist_theory_m = np.linspace(0.5, 100, 2000)
        dist_sim_m = np.array([2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0,
                                50.0, 60.0, 70.0, 80.0, 90.0, 100.0])

    fig_params = dict(fontsize=16, tick_labelsize=14)

    for case_name, cfg in CASES.items():
        d_tx = cfg['d_tx_cm'] / 100.0   # metres
        d_rx = cfg['d_rx_cm'] / 100.0

        theory_ratio = theoretical_singular_value_ratio(dist_theory_m, d_tx, d_rx, LAM_M)
        sim_ratio = simulate_ratio(d_tx, d_rx, LAM_M, dist_sim_m)

        fig, ax = plt.subplots(figsize=(6, 4))

        # Theory
        ax.plot(dist_theory_m, theory_ratio, linestyle=':', linewidth=1.5,
                color='black', label='Theory (Eq. 4-5)')
        # Simulation
        ax.plot(dist_sim_m, sim_ratio, linestyle='--', linewidth=2,
                color='steelblue', label='Simulation (Eq. 3)')
        # Hardware (if available)
        if hw_data is not None and case_name in hw_data:
            hw = hw_data[case_name]
            ax.errorbar(
                hw['distances'], hw['mean_sing_ratio'],
                yerr=hw['std_sing_ratio'],
                fmt='-o', linewidth=2, markersize=5, capsize=3,
                color='darkorange', label='Hardware',
            )

        # MIMO feasible region
        from arraylink.channel import mimo_region_bounds
        r_min, r_max = mimo_region_bounds(d_tx, d_rx, LAM_M, tau=TAU)
        ax.axhline(TAU, linestyle='--', color='firebrick', linewidth=1.5,
                   label=f'Threshold (τ={TAU})')
        r_min_vis = max(r_min, dist_theory_m[0])
        r_max_vis = min(r_max, dist_theory_m[-1])
        if r_max_vis > r_min_vis:
            ax.axvspan(r_min_vis, r_max_vis, alpha=0.12, color='green',
                       label='MIMO feasible')

        ax.set_title(
            rf"$d_{{tx}}={cfg['d_tx_cm']}\,\mathrm{{cm}},\;"
            rf"d_{{rx}}={cfg['d_rx_cm']}\,\mathrm{{cm}}$",
            fontsize=14,
        )
        ax.set_xlabel("Distance (m)", fontsize=fig_params['fontsize'])
        ax.set_ylabel(r"Singular-value ratio $\sigma_2/\sigma_1$",
                      fontsize=fig_params['fontsize'])
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=fig_params['tick_labelsize'])
        ax.legend(fontsize=12, framealpha=0.5, loc='upper right')
        ax.grid(True)
        fig.tight_layout()
        out = os.path.join(args.save_dir, f"fig10_{case_name}_validation.pdf")
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out}")

        if args.quick:
            break   # one plot is enough for smoke test


if __name__ == "__main__":
    main()

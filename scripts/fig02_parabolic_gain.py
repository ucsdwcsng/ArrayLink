# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Fig 2 — Parabolic dish beam pattern vs scan angle.

Reproduces Fig. 2 from the ArrayLink paper (INFOCOM 2026).

Shows the far-field gain (dBi) of two circular parabolic dishes as a function
of scan angle θ ∈ [−90°, +90°] at 28 GHz:
  - D = 1.85 m  (η = 0.577)  →  ~52.3 dBi peak  [baseline dish]
  - D = 1.47 m  (η = 0.50)   →  ~49.7 dBi peak  [smaller reference]

The pattern uses the Airy-disk model for a uniformly illuminated circular
aperture:  G(θ) = G0 · [2 J1(x)/x]²,  x = π·D·sin(θ)/λ

Two output files
----------------
  fig02_parabolic_gain.pdf         — full ±90° scan range
  fig02_parabolic_gain_zoomed.pdf  — zoomed ±1°, 40–52.5 dB (main-lobe detail)

Usage
-----
  python scripts/fig02_parabolic_gain.py [--quick] [--save-dir paper_figures]
                                         [--config configs/arraylink_1km.yaml]
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

from arraylink.beamforming import parabolic_beam_pattern_dbi
from arraylink.utils import load_config


# ---------------------------------------------------------------------------
# Fig-2-specific dish configurations (paper Sec. II-A / Table I)
# η=0.6 (60% aperture efficiency) used consistently across all figures.
# ---------------------------------------------------------------------------
DISHES = [
    {"label": "Diameter=1.85 m", "D_m": 1.85, "eta": 0.6,
     "color": "steelblue",  "ls": "-"},
    {"label": "Diameter=1.47 m", "D_m": 1.47, "eta": 0.6,
     "color": "darkorange", "ls": "--"},
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fig 2: parabolic dish beam pattern vs scan angle"
    )
    ap.add_argument("--config", default="configs/arraylink_1km.yaml",
                    help="Path to YAML config (default: configs/arraylink_1km.yaml)")
    ap.add_argument("--quick", action="store_true",
                    help="Print peak gains and exit, skip plot save")
    ap.add_argument("--save-dir", default="paper_figures")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    cfg      = load_config(args.config)
    F_HZ     = cfg['frequency_hz']

    # Angular grid — high resolution to resolve narrow main lobe at 28 GHz
    if args.quick:
        theta_deg = np.linspace(-90, 90, 181)
    else:
        theta_deg = np.linspace(-90, 90, 18001)

    # Compute patterns
    patterns = []
    for dish in DISHES:
        gain = parabolic_beam_pattern_dbi(
            dish["D_m"], F_HZ, theta_deg, efficiency=dish["eta"]
        )
        patterns.append(gain)
        peak = gain.max()
        print(f"  {dish['label']:20s}  peak = {peak:.2f} dBi  (η={dish['eta']})")

    if args.quick:
        print("Quick mode: computations done, skipping plot save.")
        return

    fig_kw = dict(fontsize=18, tick_labelsize=15)

    # ------------------------------------------------------------------ #
    # Fig 2  — Full beam pattern (±90°)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 6))
    for dish, gain in zip(DISHES, patterns):
        ax.plot(theta_deg, gain,
                label=dish["label"],
                color=dish["color"],
                linestyle=dish["ls"],
                linewidth=1.5)

    ax.set_xlabel("Scan Angle θ (degrees)", fontsize=fig_kw['fontsize'])
    ax.set_ylabel("Gain (dB)", fontsize=fig_kw['fontsize'])
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 60)
    ax.tick_params(labelsize=fig_kw['tick_labelsize'])
    ax.legend(fontsize=16, loc="upper right", framealpha=0.5)
    ax.grid(True, which='major', alpha=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.3, linestyle=':')
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig02_parabolic_gain.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Fig 2 (zoomed) — Main-lobe detail (±1°, 40–52.5 dB)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 6))
    for dish, gain in zip(DISHES, patterns):
        ax.plot(theta_deg, gain,
                label=dish["label"],
                color=dish["color"],
                linestyle=dish["ls"],
                linewidth=3)

    ax.set_xlim(-1, 1)
    ax.set_ylim(40, 52.5)
    ax.tick_params(labelsize=fig_kw['tick_labelsize'])
    ax.grid(True, which='major', alpha=0.6)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig02_parabolic_gain_zoomed.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

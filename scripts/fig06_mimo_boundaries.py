"""
Fig 6 — Singular-value ratio vs distance showing MIMO feasibility boundaries.

Reproduces Fig. 6 from the ArrayLink paper (INFOCOM 2026).

Plots sigma_2/sigma_1 vs link distance for small-aperture (hardware) and
satellite-scale aperture configurations, with r_min and r_max annotated.

Usage
-----
  python scripts/fig06_mimo_boundaries.py [--quick] [--save-dir paper_figures]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

from arraylink.channel import mimo_region_bounds, theoretical_singular_value_ratio


def parse_args():
    ap = argparse.ArgumentParser(description="Fig 6: MIMO feasibility boundaries")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save-dir", default="paper_figures")
    return ap.parse_args()


def _plot_boundary(ax, r_km, ratio, r_min, r_max, tau, label_prefix, color, lam):
    ax.plot(r_km, ratio, linewidth=2, color=color, label=label_prefix)
    ax.axhline(tau, linestyle='--', color='firebrick', linewidth=1.5, label=f'Threshold (τ={tau})')
    ax.axvline(r_min, linestyle=':', color='green', linewidth=1.5,
               label=rf'$r_{{min}}$ = {r_min:.2f}')
    ax.axvline(r_max, linestyle=':', color='navy', linewidth=1.5,
               label=rf'$r_{{max}}$ = {r_max:.2f}')
    # shade MIMO-feasible region
    ylim = ax.get_ylim()
    ax.axvspan(r_min, r_max, alpha=0.12, color='green', label='MIMO feasible region')


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    TAU = 0.1      # feasibility threshold (paper Eq. 7)
    F_HZ = 28e9    # 28 GHz
    C = 3e8
    LAM_M = C / F_HZ            # wavelength in metres
    LAM_KM = LAM_M / 1e3        # wavelength in km

    fig_params = dict(fontsize=16, tick_labelsize=14)

    # ---- (a) Small aperture: hardware scale (d_tx = d_rx = 0.2 m) ----
    if args.quick:
        for d_tx, d_rx, label in [
            (0.2, 0.2, "d_tx=d_rx=0.2m"),  # 20 cm x 20 cm
            (np.sqrt(2e3), np.sqrt(2), "satellite scale"), # sqrt(2) km x sqrt(2) m
        ]:
            r_min, r_max = mimo_region_bounds(d_tx, d_rx, LAM_M, tau=TAU)
            print(f"{label}: r_min={r_min:.2f}, r_max={r_max:.2f}")
        return

    configs = [
        # (d_tx_m, d_rx_m, r_range_m, label, filename_suffix)
        (0.20, 0.20, np.linspace(0.1, 100,  5000),
         r"$d_{tx}=d_{rx}=20\,\mathrm{cm}$", "small"),
        (0.20, 0.50, np.linspace(0.1, 200,  5000),
         r"$d_{tx}=20\,\mathrm{cm},\;d_{rx}=50\,\mathrm{cm}$", "medium"),
    ]

    for d_tx, d_rx, r_array, label, suffix in configs:
        ratio = theoretical_singular_value_ratio(r_array, d_tx, d_rx, LAM_M)
        r_min, r_max = mimo_region_bounds(d_tx, d_rx, LAM_M, tau=TAU)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(r_array, ratio, linewidth=2, color='steelblue', label='Theory')
        ax.axhline(TAU, linestyle='--', color='firebrick', linewidth=1.5,
                   label=f'Threshold (τ={TAU})')
        ax.axvspan(r_min, r_max, alpha=0.15, color='green', label='MIMO feasible region')
        ax.axvline(r_min, linestyle=':', color='green', linewidth=1.5,
                   label=rf'$r_{{min}}={r_min:.1f}\,\mathrm{{m}}$')
        ax.axvline(r_max, linestyle=':', color='navy', linewidth=1.5,
                   label=rf'$r_{{max}}={r_max:.1f}\,\mathrm{{m}}$')
        ax.set_xlabel("Distance (m)", fontsize=fig_params['fontsize'])
        ax.set_ylabel(r"Singular-value ratio $\sigma_2/\sigma_1$",
                      fontsize=fig_params['fontsize'])
        ax.set_title(label, fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=fig_params['tick_labelsize'])
        ax.legend(fontsize=12, framealpha=0.5, loc='upper right')
        ax.grid(True)
        fig.tight_layout()
        out = os.path.join(args.save_dir, f"fig06_mimo_boundaries_{suffix}.pdf")
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out}   r_min={r_min:.2f} m, r_max={r_max:.2f} m")

    # ---- Satellite scale (km units) ----
    D_TX_KM = np.sqrt(2)        # sqrt(2) km ground aperture diagonal
    D_RX_KM = np.sqrt(2) / 1e3        # sqrt(2) m satellite aperture in km
    r_km = np.linspace(1, 3200, 10000)
    ratio_sat = theoretical_singular_value_ratio(r_km, D_TX_KM, D_RX_KM, LAM_KM)
    r_min_km, r_max_km = mimo_region_bounds(D_TX_KM, D_RX_KM, LAM_KM, tau=TAU)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_km, ratio_sat, linewidth=2, color='steelblue', label='Theory')
    ax.axhline(TAU, linestyle='--', color='firebrick', linewidth=1.5,
               label=f'Threshold (τ={TAU})')
    ax.axvspan(r_min_km, r_max_km, alpha=0.15, color='green', label='MIMO feasible region')
    ax.axvline(r_min_km, linestyle=':', color='green', linewidth=1.5,
               label=rf'$r_{{min}}={r_min_km:.0f}\,\mathrm{{km}}$')
    ax.axvline(r_max_km, linestyle=':', color='navy', linewidth=1.5,
               label=rf'$r_{{max}}={r_max_km:.0f}\,\mathrm{{km}}$')
    ax.set_xlabel("Distance (km)", fontsize=fig_params['fontsize'])
    ax.set_ylabel(r"Singular-value ratio $\sigma_2/\sigma_1$",
                  fontsize=fig_params['fontsize'])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 3200)
    ax.tick_params(labelsize=fig_params['tick_labelsize'])
    ax.legend(fontsize=12, framealpha=0.5, loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig06_mimo_boundaries_satellite.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}   r_min={r_min_km:.0f} km, r_max={r_max_km:.0f} km")


if __name__ == "__main__":
    main()

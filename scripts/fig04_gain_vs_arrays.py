"""
Fig 4a/4b — Total and marginal gain vs number of phased-array panels.

Reproduces Fig. 4 from the ArrayLink paper (INFOCOM 2026).

Usage
-----
  python scripts/fig04_gain_vs_arrays.py [--quick] [--save-dir paper_figures]
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

from arraylink.beamforming import total_array_gain_dbi, marginal_gain_dbi, parabolic_gain_dbi


def parse_args():
    ap = argparse.ArgumentParser(description="Fig 4: gain vs number of panels")
    ap.add_argument("--quick", action="store_true", help="Skip plot, just print values")
    ap.add_argument("--save-dir", default="paper_figures", help="Output directory for PDFs")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Parameters (paper Sec. III-A) ---
    G_PA_DBI = 36.1          # single 32x32 panel gain (6 dBi/element × 1024 elements)
    N_MAX = 50
    N = np.arange(1, N_MAX + 1)

    # Reference dish gains (paper Fig. 2)
    G_DISH_185 = parabolic_gain_dbi(1.85, 28e9, efficiency=0.6)   # ~52.6 dBi
    G_DISH_147 = parabolic_gain_dbi(1.47, 28e9, efficiency=0.6)   # ~49.5 dBi
    ARRAYLINK_N = 16   # number of panels chosen in the paper
    THRESHOLD_DB = 0.25  # marginal gain threshold beyond which returns diminish

    G_total = total_array_gain_dbi(N, G_PA_DBI)
    G_marginal = marginal_gain_dbi(N, G_PA_DBI)

    if args.quick:
        print(f"G(1)   = {G_total[0]:.2f} dBi")
        print(f"G(16)  = {G_total[15]:.2f} dBi")
        print(f"G(50)  = {G_total[-1]:.2f} dBi")
        print(f"Marginal gain at N=16: {G_marginal[15]:.4f} dB")
        print(f"1.47m dish: {G_DISH_147:.2f} dBi, 1.85m dish: {G_DISH_185:.2f} dBi")
        return

    fig_params = dict(fontsize=16, tick_labelsize=14)

    # --- Fig 4a: total gain vs N ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(N, G_total, linewidth=2, color='steelblue', label='ArrayLink')
    ax.axhline(G_DISH_185, linestyle='--', color='firebrick', linewidth=1.5,
               label=f'1.85 m dish ({G_DISH_185:.1f} dBi)')
    ax.axhline(G_DISH_147, linestyle=':', color='darkorange', linewidth=1.5,
               label=f'1.47 m dish ({G_DISH_147:.1f} dBi)')
    ax.axvline(ARRAYLINK_N, linestyle='--', color='gray', linewidth=1,
               label=f'ArrayLink (N={ARRAYLINK_N})')
    ax.set_xlabel("Number of phased-array panels", fontsize=fig_params['fontsize'])
    ax.set_ylabel("Total gain (dBi)", fontsize=fig_params['fontsize'])
    ax.tick_params(labelsize=fig_params['tick_labelsize'])
    ax.legend(fontsize=13, framealpha=0.5)
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig04a_total_gain_vs_arrays.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    # --- Fig 4b: marginal gain vs N ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(N, G_marginal, linewidth=2, color='steelblue', label='Marginal gain')
    ax.axhline(THRESHOLD_DB, linestyle='--', color='firebrick', linewidth=1.5,
               label=f'Threshold ({THRESHOLD_DB} dB)')
    ax.axvline(ARRAYLINK_N, linestyle='--', color='gray', linewidth=1,
               label=f'N={ARRAYLINK_N}')
    ax.set_xlabel("Number of phased-array panels", fontsize=fig_params['fontsize'])
    ax.set_ylabel("Marginal gain per added panel (dB)", fontsize=fig_params['fontsize'])
    ax.tick_params(labelsize=fig_params['tick_labelsize'])
    ax.legend(fontsize=13, framealpha=0.5)
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig04b_marginal_gain_vs_arrays.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved fig04a and fig04b to {args.save_dir}/")
    print(f"  ArrayLink (N=16): {total_array_gain_dbi(16, G_PA_DBI):.2f} dBi "
          f"vs 1.85m dish {G_DISH_185:.2f} dBi "
          f"(gap = {G_DISH_185 - total_array_gain_dbi(16, G_PA_DBI):.1f} dB)")


if __name__ == "__main__":
    main()

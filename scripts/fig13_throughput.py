# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Fig 13 — Per-link and aggregate throughput comparison.

Reproduces Fig. 13 from the ArrayLink paper (INFOCOM 2026).

Compares:
  - 1 stream @ ~52.5 dBi (1.85 m dish baseline, η=0.6)
  - 2 streams @ 48.14 dBi (ArrayLink)
  - 4 streams @ 48.14 dBi (ArrayLink)

Two models:
  (a) Per-link throughput: each configuration uses total bandwidth B
  (b) Aggregate throughput: each stream adds B_single Hz of bandwidth

Usage
-----
  python scripts/fig13_throughput.py [--quick] [--save-dir paper_figures]
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

from arraylink.beamforming import parabolic_gain_dbi, total_array_gain_dbi


def parse_args():
    ap = argparse.ArgumentParser(description="Fig 13: throughput comparison")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save-dir", default="paper_figures")
    return ap.parse_args()


def spectral_eff(snr_db, gap_db=0.0):
    """Shannon spectral efficiency (bps/Hz) with implementation gap."""
    return np.log2(1.0 + 10 ** ((snr_db - gap_db) / 10.0))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Parameters ---
    G_DISH_185 = parabolic_gain_dbi(1.85, 28e9, efficiency=0.6)    # ~52.6 dBi (baseline)
    G_ARRAYLINK = total_array_gain_dbi(16, 36.1)                    # ~48.14 dBi
    B_SINGLE_HZ = 250e6    # 250 MHz per link
    GAP_DB = 2.0           # implementation gap to Shannon

    # SNR baseline: defined for the 1-stream dish case
    if args.quick:
        SNR0_SWEEP = np.array([10.0, 15.0, 20.0])
    else:
        SNR0_SWEEP = np.arange(5, 26, 2.5)

    configs = [
        {"name": f"1 stream @ {G_DISH_185:.1f} dBi (dish)", "N": 1, "gain_dbi": G_DISH_185},
        {"name": f"2 streams @ {G_ARRAYLINK:.2f} dBi (AL)", "N": 2, "gain_dbi": G_ARRAYLINK},
        {"name": f"4 streams @ {G_ARRAYLINK:.2f} dBi (AL)", "N": 4, "gain_dbi": G_ARRAYLINK},
    ]
    markers = ['o', 's', '^']
    lines   = ['--', '-', '-.']
    colors  = ['steelblue', 'darkorange', 'seagreen']

    # Pre-compute throughput curves
    perlink_gbps = {c["name"]: [] for c in configs}
    agg_gbps     = {c["name"]: [] for c in configs}

    for snr0 in SNR0_SWEEP:
        for cfg in configs:
            dg = cfg["gain_dbi"] - G_DISH_185         # gain delta vs dish
            snr_db = snr0 + dg                         # per-link SNR
            eta = spectral_eff(snr_db, gap_db=GAP_DB) # bps/Hz
            # per-link: total BW / N streams shared
            perlink_gbps[cfg["name"]].append(eta * B_SINGLE_HZ / 1e9)
            # aggregate: each stream gets B_single
            agg_gbps[cfg["name"]].append(cfg["N"] * eta * B_SINGLE_HZ / 1e9)

    if args.quick:
        for cfg in configs:
            print(f"{cfg['name']}: per-link={perlink_gbps[cfg['name']]}, agg={agg_gbps[cfg['name']]}")
        print("Quick mode: computations done, skipping plot save.")
        return

    fig_params = dict(fontsize=20, tick_labelsize=18)

    # --- Fig 13a: per-link throughput ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cfg in enumerate(configs):
        ax.plot(SNR0_SWEEP, perlink_gbps[cfg["name"]], marker=markers[i],
                linestyle=lines[i], color=colors[i], linewidth=2.5, markersize=6,
                label=cfg["name"])
    ax.set_xlabel(f"Baseline SNR₀ for {G_DISH_185:.1f} dBi (dB)",
                  fontsize=fig_params['fontsize'])
    ax.set_ylabel("Per-link throughput (Gbps)", fontsize=fig_params['fontsize'])
    ax.tick_params(labelsize=fig_params['tick_labelsize'])
    ax.legend(fontsize=14, framealpha=1, loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig13a_perlink_throughput.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # --- Fig 13b: aggregate throughput ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cfg in enumerate(configs):
        ax.plot(SNR0_SWEEP, agg_gbps[cfg["name"]], marker=markers[i],
                linestyle=lines[i], color=colors[i], linewidth=2.5, markersize=6,
                label=cfg["name"])
    ax.set_xlabel(f"Baseline SNR₀ for {G_DISH_185:.1f} dBi (dB)",
                  fontsize=fig_params['fontsize'])
    ax.set_ylabel("Aggregate throughput (Gbps)", fontsize=fig_params['fontsize'])
    ax.tick_params(labelsize=fig_params['tick_labelsize'])
    ax.legend(fontsize=14, framealpha=1, loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig13b_aggregate_throughput.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

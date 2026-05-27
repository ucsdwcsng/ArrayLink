"""
Fig 11 — 2D beam patterns (transverse × range heat maps).

Reproduces Fig. 11 from the ArrayLink paper (INFOCOM 2026).

The satellite is placed on a Cartesian (X, Y) grid:
  X : transverse offset from boresight (km)
  Y : range / boresight distance (km)

  satellite position = (X, 0, Y)

DC weights are focused at (0, 0, 500) km (zenith, 500 km range).

Shows:
  (a) Monolithic UPA 128×128  — far-field gain ridge, constant in range
  (b) ArrayLink 16×(32×32)    — near-field focal spot at Y=500 km

ArrayLink panel placement  (--placement)
----------------------------------------
  random       [default]  16 panels placed uniformly at random within the
                           aperture (Lx × Ly).  Different each run unless
                           --seed is fixed.
  center-dense             Power-law concentrated toward boresight (gamma=3),
                           as used in the paper.
  uniform                  Regular 4×4 grid spanning the full aperture.

Resolution  (default: fast preview, ~30 s)
------------------------------------------
  Default      20 km X steps × 25 km Y steps  (301 × 119 ≈ 36 K pts)
  --full       5 km X steps × 20 km Y steps   (1201 × 148 ≈ 178 K pts, ~5 min)

Usage
-----
  # Fast preview with random panel placement (default):
  python scripts/fig11_2d_beampattern.py

  # Full-resolution, paper-exact center-dense layout:
  python scripts/fig11_2d_beampattern.py --placement center-dense --full

  # Reproducible random layout, full resolution:
  python scripts/fig11_2d_beampattern.py --placement random --seed 42 --full

  # Smoke test (tiny grid, no plot saved):
  python scripts/fig11_2d_beampattern.py --quick

  # Interactive Plotly figure (requires plotly):
  python scripts/fig11_2d_beampattern.py --interactive
"""
import argparse
import gc
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

from arraylink.array_geometry import build_ground_station, upa_positions, place_subarrays
from arraylink.beamforming import (
    dc_weights, compute_beam_pattern_numpy,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fig 11: 2D beam pattern heat map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--quick", action="store_true",
                    help="Tiny grid for smoke test (skips plot save)")
    ap.add_argument("--full", action="store_true",
                    help="Full resolution: 5 km × 20 km grid (~178 K pts, ~5 min). "
                         "Default is fast preview: 20 km × 25 km grid (~36 K pts, ~30 s).")
    ap.add_argument("--placement", default="random",
                    choices=["random", "center-dense", "uniform"],
                    help="ArrayLink panel placement strategy (default: random). "
                         "'random' places 16 panels uniformly at random within the aperture. "
                         "'center-dense' uses the paper's power-law layout (gamma=3). "
                         "'uniform' uses a regular 4×4 grid.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for random placement (default: 0)")
    ap.add_argument("--interactive", action="store_true",
                    help="Open Plotly interactive figure (requires plotly)")
    ap.add_argument("--save-dir", default="paper_figures")
    return ap.parse_args()


ELEMENT_GAIN_DBI = 6.0
TARGET_R_KM      = 500.0    # DC focus range (km) — matches paper Fig. 11

# ArrayLink aperture dimensions (km)
AL_LX = np.sqrt(2)          # 1.414 km along X
AL_LY = 1.0                 # 1.000 km along Y
AL_NX = 4                   # panels along X
AL_NY = 4                   # panels along Y


def build_arraylink(placement, subarray_shape, element_spacing, seed):
    """
    Build the ArrayLink antenna array according to the chosen placement strategy.

    Parameters
    ----------
    placement      : 'random' | 'center-dense' | 'uniform'
    subarray_shape : (M, N) elements per panel
    element_spacing: element pitch within a panel (km)
    seed           : RNG seed (used for 'random' placement)

    Returns
    -------
    ants  : (Nx*Ny*M*N, 3) all antenna positions
    bases : (Nx*Ny, 3)     panel base positions
    """
    n_panels = AL_NX * AL_NY

    if placement == "random":
        rng = np.random.default_rng(seed)
        bx = rng.uniform(-AL_LX / 2.0, AL_LX / 2.0, n_panels)
        by = rng.uniform(-AL_LY / 2.0, AL_LY / 2.0, n_panels)
        bases = np.stack([bx, by, np.zeros(n_panels)], axis=1)   # (16, 3)

    elif placement == "center-dense":
        _, bases = build_ground_station(
            mode='arraylink', subarray_shape=subarray_shape,
            element_spacing=element_spacing,
            Nx=AL_NX, Ny=AL_NY, Lx=AL_LX, Ly=AL_LY,
            gamma=3.0, seed=seed,
        )

    elif placement == "uniform":
        _, bases = build_ground_station(
            mode='uniform', subarray_shape=subarray_shape,
            element_spacing=element_spacing,
            Nx=AL_NX, Ny=AL_NY, Lx=AL_LX, Ly=AL_LY,
        )

    else:
        raise ValueError(f"Unknown placement '{placement}'")

    ants = place_subarrays(bases, subarray_shape, element_spacing)
    return ants, bases


def compute_2d_pattern(gnd_ants, weights, lam_km, x_axis, y_axis, quick,
                       batch_size=2000):
    """
    Compute beam pattern on a Cartesian (X, Y) grid.

    Satellite is placed at (X, 0, Y) km.  Returns (Ny, Nx) array of gain in dBi.
    Uses batched numpy evaluation (~400 MB RAM per batch).
    """
    Nx, Ny = len(x_axis), len(y_axis)
    XX, YY = np.meshgrid(x_axis, y_axis, indexing='ij')   # (Nx, Ny)
    flat_pts = np.stack([XX.ravel(), np.zeros(Nx * Ny), YY.ravel()], axis=-1)

    if quick:
        bp = compute_beam_pattern_numpy(flat_pts, gnd_ants, weights, lam_km)
        bp += ELEMENT_GAIN_DBI
        return bp.reshape(Nx, Ny).T   # (Ny, Nx)

    # Batched numpy: process batch_size points at a time to keep RAM bounded
    N = len(flat_pts)
    results = np.empty(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        results[start:end] = compute_beam_pattern_numpy(
            flat_pts[start:end], gnd_ants, weights, lam_km
        )
        if start % (batch_size * 20) == 0 and start > 0:
            pct = 100 * start / N
            print(f"    {pct:.0f}% ({start:,}/{N:,})", flush=True)
    gc.collect()
    return (results + ELEMENT_GAIN_DBI).reshape(Nx, Ny).T   # (Ny, Nx)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    F_HZ   = 28e9
    LAM_KM = 3e8 / F_HZ / 1e3

    if args.quick:
        # Minimal grid for smoke test — just verify code paths run
        x_axis = np.linspace(-3000, 3000, 13)
        y_axis = np.linspace(50, 3000, 7)
    elif args.full:
        # Full resolution — resolves UPA sidelobes and ArrayLink focal depth
        # 1201 × 148 ≈ 178 K pts  →  ~5 min on a single CPU core
        x_axis = np.arange(-3000, 3001,  5, dtype=float)   # 1201 pts, 5 km steps
        y_axis = np.arange(50,   3001, 20, dtype=float)    #  148 pts, 20 km steps
    else:
        # Fast preview — good enough to see main beam structure and focal spot
        # 301 × 119 ≈ 36 K pts  →  ~30 s on a single CPU core
        x_axis = np.arange(-3000, 3001, 20, dtype=float)   #  301 pts, 20 km steps
        y_axis = np.arange(50,   3001, 25, dtype=float)    #  119 pts, 25 km steps

    sat_loc = np.array([0.0, 0.0, TARGET_R_KM])

    # ---------- Build antenna arrays ----------
    # UPA 128×128 — compact monolithic array, λ/2 spacing
    upa_ants = upa_positions(128, 128, element_spacing=LAM_KM / 2.0)

    # ArrayLink 4×4 panels × 32×32 elements — placement set by --placement flag
    al_ants, al_bases = build_arraylink(
        placement=args.placement,
        subarray_shape=(32, 32),
        element_spacing=LAM_KM / 2.0,
        seed=args.seed,
    )

    w_upa = dc_weights(sat_loc, upa_ants, LAM_KM)
    w_al  = dc_weights(sat_loc, al_ants,  LAM_KM)

    res_tag = "full" if args.full else ("quick" if args.quick else "fast")
    print(f"UPA: {len(upa_ants)} elements  |  ArrayLink: {len(al_ants)} elements  "
          f"[placement={args.placement}, seed={args.seed}]")
    print(f"Grid: {len(x_axis)} × {len(y_axis)} = {len(x_axis)*len(y_axis):,} points  "
          f"[resolution={res_tag}]")

    print("Computing UPA beam pattern …", flush=True)
    bp_upa = compute_2d_pattern(upa_ants, w_upa, LAM_KM, x_axis, y_axis, args.quick)

    print("Computing ArrayLink beam pattern …", flush=True)
    bp_al  = compute_2d_pattern(al_ants,  w_al,  LAM_KM, x_axis, y_axis, args.quick)

    if args.quick:
        print("Quick mode: patterns computed, skipping plot save.")
        print(f"  UPA max gain    : {bp_upa.max():.1f} dBi")
        print(f"  ArrayLink max   : {bp_al.max():.1f} dBi")
        return

    peak = max(bp_upa.max(), bp_al.max())
    vmin = ELEMENT_GAIN_DBI   # floor = single-element gain (matches paper colorbar)
    vmax = peak
    print(f"Peak = {peak:.2f} dBi  |  colorbar [{vmin:.1f}, {vmax:.1f}] dBi")

    # ---------- Static matplotlib figure ----------
    al_title = {
        "random":       f"(b) ArrayLink — random (seed={args.seed})",
        "center-dense": "(b) ArrayLink — center-dense",
        "uniform":      "(b) ArrayLink — uniform",
    }[args.placement]

    X, Y = np.meshgrid(x_axis, y_axis)   # both (Ny, Nx)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, bp, title in [
        (axes[0], bp_upa, "(a) Monolithic UPA"),
        (axes[1], bp_al,  al_title),
    ]:
        c = ax.pcolormesh(X, Y, bp, shading='auto',
                          cmap='inferno', vmin=vmin, vmax=vmax)
        ax.axhline(TARGET_R_KM, color='white', lw=0.6, ls='--', alpha=0.45)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X (km)", fontsize=12)
        ax.set_ylabel("Y (km)", fontsize=12)
        fig.colorbar(c, ax=ax, label="Gain (dB)", shrink=0.88)
    fig.tight_layout()
    out = os.path.join(args.save_dir, "fig11_2d_beampattern.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ---------- Optional Plotly interactive ----------
    if args.interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fig_p = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Monolithic UPA", al_title])
            for col, (bp, lbl) in enumerate([(bp_upa, "UPA"), (bp_al, "AL")], start=1):
                fig_p.add_trace(
                    go.Heatmap(
                        x=x_axis, y=y_axis, z=bp,
                        colorscale='Hot', zmin=vmin, zmax=vmax,
                        colorbar=dict(title="dB", x=col * 0.5),
                        showscale=(col == 2),
                    ),
                    row=1, col=col,
                )
                fig_p.update_xaxes(title_text="X (km)", row=1, col=col)
                fig_p.update_yaxes(title_text="Y (km)", row=1, col=col)
            fig_p.update_layout(title="2D Beam Patterns: UPA vs ArrayLink")
            fig_p.show()
        except ImportError:
            print("Plotly not installed. Install with: mamba install -c conda-forge plotly")


if __name__ == "__main__":
    main()

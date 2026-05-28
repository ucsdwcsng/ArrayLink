# ArrayLink Simulator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[**Webpage**](https://wcsng.ucsd.edu/arraylink/) | [**Paper**](https://wcsng.ucsd.edu/files/arraylink_paper.pdf) | [**Slides**](https://wcsng.ucsd.edu/files/arraylink_slides.pdf) | [**Poster**](https://wcsng.ucsd.edu/files/arraylink_poster.pdf) | [**Demo**](https://www.youtube.com/watch?v=HJBeDMcrmWs)

Conventional satellite ground stations rely on a single large parabolic dish. ArrayLink replaces it with 16 distributed phased-array panels spanning a km-scale aperture — placing LEO/MEO satellites in the radiative near-field where the line-of-sight channel supports spatial multiplexing across independent streams, delivering throughput that no single dish can match.

This repository contains the simulator and hardware experiment results for the INFOCOM 2026 paper.

## Installation

Requires [mamba](https://mamba.readthedocs.io/) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

```bash
git clone https://github.com/ucsdwcsng/ArrayLink.git
cd ArrayLink
mamba env create -f environment.yml
mamba activate arraylink
```

## Quick Start

```python
from arraylink.channel import compute_channel_matrix, mimo_region_bounds
from arraylink.array_geometry import build_ground_station

# MIMO feasibility range for the ArrayLink aperture at 28 GHz
lam = 3e8 / 28e9
r_min, r_max = mimo_region_bounds(d_tx=2000, d_rx=1.0, wavelength=lam)
print(f"MIMO region: {r_min/1e3:.0f} – {r_max/1e3:.0f} km")

# Build the ArrayLink ground station (16 panels (each 32x32), √2 km × 1 km aperture)
gnd, _ = build_ground_station(mode="arraylink", subarray_shape=(32, 32),
                               element_spacing=lam/2/1e3, Nx=4, Ny=4,
                               Lx=1.4142, Ly=1.0)
print(f"Total antenna elements: {len(gnd)}")   # → 16384
```

## Reproducing Paper Figures

Run any figure script from the repo root:

| Script | Figure | Description |
|--------|--------|-------------|
| `scripts/fig02_parabolic_gain.py` | Fig. 2 | Parabolic dish beam pattern vs scan angle |
| `scripts/fig04_gain_vs_arrays.py` | Fig. 4 | Array gain vs number of panels |
| `scripts/fig06_mimo_boundaries.py` | Fig. 6 | Theoretical MIMO region boundaries |
| `scripts/fig09_beampattern_sim.py` | Fig. 9 | UPA/ArrayLink positions and beam pattern |
| `scripts/fig10_hardware_validation.py` | Fig. 10 | Hardware experiment validation |
| `scripts/fig11_2d_beampattern.py` | Fig. 11 | 2D beam pattern heatmap |
| `scripts/fig12_mimo_dof.py` | Fig. 12 | MIMO degrees of freedom vs distance |
| `scripts/fig13_throughput.py` | Fig. 13 | Throughput comparison |

```bash
# Generate all figures (outputs to paper_figures/; ~2–3 min total)
for s in scripts/fig*.py; do python $s; done
```

> **Fig. 11 interactive mode** — install `plotly` first, then run with `--interactive`:
> ```bash
> mamba install -c conda-forge plotly
> python scripts/fig11_2d_beampattern.py --interactive
> ```

> **Fig. 10 hardware curves** — place `hardware_metrics.pkl` at `data/hardware_metrics.pkl`
> (theory and simulation curves render without it; see [data/README.md](data/README.md)).

## Running Tests

```bash
pytest tests/ -v
```

## Package Structure

```
arraylink/
  channel.py        # Channel matrix, singular values, MIMO bounds, spectral efficiency
  array_geometry.py # UPA, center-dense, and uniform array placement
  beamforming.py    # DC weights, beam pattern, array/dish gain formulas
  utils.py          # Coordinate transforms, decibel helpers, config loading
scripts/            # One script per paper figure
configs/            # YAML parameter files (arraylink_1km.yaml, upa_baseline.yaml)
tests/              # Unit tests (42) + smoke tests (8)
data/               # Hardware experiment data (see data/README.md)
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Carrier frequency | 28 GHz |
| Panels | 16 (4×4, center-dense over √2 km × 1 km) |
| Elements per panel | 32×32 = 1024 (λ/2 spacing) |
| Dish aperture efficiency | 60% |
| MIMO threshold τ | 0.1 (σ₂/σ₁ > τ for spatial multiplexing) |

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia  
UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)

## Citation

If you use this simulator, please cite:

```bibtex
@article{vennam2025satellites,
  title   = {Satellites are closer than you think: A near field MIMO approach for Ground stations},
  author  = {Vennam, Rohith Reddy and Wilson, Luke and Jain, Ish Kumar and Bharadia, Dinesh},
  journal = {arXiv preprint arXiv:2508.09374},
  year    = {2025},
}
```
For more details refer webpage: [wcsng.ucsd.edu/arraylink/](https://wcsng.ucsd.edu/arraylink/)

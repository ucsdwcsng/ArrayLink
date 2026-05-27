# ArrayLink Simulator

Near-field LoS MIMO simulator for the ArrayLink distributed ground station (INFOCOM 2026).

ArrayLink uses 16 phased-array panels spread across a km-scale aperture to form a
distributed ground station that exploits spatial multiplexing in the radiative near-field
of LEO/MEO satellites.

## Installation

Requires [mamba](https://mamba.readthedocs.io/) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

```bash
git clone https://github.com/ucsdwcsng/ArrayLink.git
cd ArrayLink
mamba env create -f environment.yml
mamba activate arraylink
```

For the interactive 2D beam pattern (Fig. 11 `--interactive` flag):
```bash
mamba install -c conda-forge plotly
```

## Quick Start

```python
from arraylink.channel import compute_channel_matrix, mimo_region_bounds
from arraylink.array_geometry import build_ground_station
from arraylink.beamforming import total_array_gain_dbi, parabolic_gain_dbi

# MIMO feasibility range for the ArrayLink aperture at 28 GHz
lam = 3e8 / 28e9
r_min, r_max = mimo_region_bounds(d_tx=2000, d_rx=1.0, wavelength=lam)
print(f"MIMO region: {r_min/1e3:.0f} km – {r_max/1e3:.0f} km")

# Build the ArrayLink ground station (16 panels, center-dense layout)
# Lx/Ly are in km; element_spacing is also in km
gnd, _ = build_ground_station(
    mode="arraylink",
    subarray_shape=(32, 32),
    element_spacing=lam / 2 / 1e3,   # λ/2 converted to km
    Nx=4, Ny=4,
    Lx=1.4142, Ly=1.0,               # 1.414 km × 1.0 km aperture
)
print(f"Total antenna elements: {len(gnd)}")   # → 16384
```

## Reproducing Paper Figures

Run any figure script from the repo root:

| Script | Figure | Description |
|--------|--------|-------------|
| `scripts/fig04_gain_vs_arrays.py` | Fig. 4 | Array gain vs number of panels |
| `scripts/fig06_mimo_boundaries.py` | Fig. 6 | Theoretical MIMO region boundaries |
| `scripts/fig09_beampattern_sim.py` | Fig. 9 | Simulation setup: UPA/ArrayLink positions and gain vs θ / distance |
| `scripts/fig10_hardware_validation.py` | Fig. 10 | Hardware experiment validation |
| `scripts/fig11_2d_beampattern.py` | Fig. 11 | 2D beam pattern heatmap |
| `scripts/fig12_mimo_dof.py` | Fig. 12 | MIMO degrees of freedom vs distance |
| `scripts/fig13_throughput.py` | Fig. 13 | Throughput comparison |

```bash
# Generate all figures (outputs to paper_figures/)
for s in scripts/fig*.py; do python $s; done

# Interactive 2D beam pattern
python scripts/fig11_2d_beampattern.py --interactive
```

### Hardware data (Fig. 10)

Fig. 10 shows theory and simulation curves even without hardware data. To add the
measured curves, place `hardware_metrics.pkl` at `data/hardware_metrics.pkl`. See
[data/README.md](data/README.md) for the expected format.

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
  utils.py          # Coordinate transforms and helpers
scripts/            # One script per paper figure
configs/            # YAML parameter files for ArrayLink and UPA baseline
tests/              # Unit tests (42) + smoke tests (8)
data/               # Hardware experiment data (not included; see data/README.md)
environment.yml     # Mamba/conda environment
```

## Key Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Carrier frequency | 28 GHz | Ka-band (λ ≈ 10.7 mm) |
| Hardware experiment frequency | 27 GHz | Matches lab setup |
| Panel element gain | 6 dBi | Isotropic radiator baseline |
| Dish efficiency | 0.6 (60%) | Standard aperture efficiency |
| MIMO threshold τ | 0.1 | σ₂/σ₁ > τ for spatial multiplexing |
| Panels | 16 (4×4 layout) | Center-dense over √2 km × 1 km |
| Elements per panel | 32×32 = 1024 | Half-wavelength spacing |
| Center-dense exponent γ | 3.0 | Power-law placement transform |

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

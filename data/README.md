# Hardware Data

Hardware experiment data for Fig. 10 (singular-value ratio vs. distance).

## Included file

`hardware_metrics.pkl` is included in this directory.  
It contains pre-processed aggregate statistics extracted from the raw hardware
measurements and is all that is needed to reproduce Fig. 10.

## What the file contains

```python
import pickle
with open("data/hardware_metrics.pkl", "rb") as f:
    data = pickle.load(f)

# data is a dict keyed by case label (satellite-ground-station convention):
# {
#   'case1': {                         # Drx=50 cm, Dtx=50 cm
#     'distances':       np.array([2.5, 5.0, 10.0, ...]),  # metres
#     'mean_sing_ratio': np.array([...]),   # mean σ₂/σ₁ across packets
#     'std_sing_ratio':  np.array([...]),   # std  σ₂/σ₁ across packets
#     ... (additional keys not used by fig10)
#   },
#   'case2': { ... },   # Drx=20 cm, Dtx=50 cm
#   'case3': { ... },   # Drx=20 cm, Dtx=20 cm
#   'case4': { ... },   # Drx=50 cm, Dtx=20 cm
# }
```

`fig10_hardware_validation.py` remaps these keys to ArrayLink's case convention
automatically on load — no manual step needed.

## Running Fig. 10

```bash
python scripts/fig10_hardware_validation.py --save-dir paper_figures
```

## How `hardware_metrics.pkl` was generated

`hardware_metrics.pkl` was produced from the raw `.mat` channel files using:

```
python_simulator/targeted_scripts/hardware_experiments_analysis.py
```

At a high level, the script:
1. Reads a logbook CSV that maps each `.mat` file to its distance, case, and aperture config
2. Loads each `.mat` file (complex MIMO channel matrices, shape `[64 freq bins, 2×2, N packets]`)
3. At a fixed frequency bin, computes σ₂/σ₁ (singular-value ratio) for every captured packet
4. Aggregates per distance point → `mean_sing_ratio`, `std_sing_ratio`
5. Saves the result as `hardware_metrics.pkl`

> ⚠️ This script is part of the raw-data repository (link below) and requires
> the full `.mat` dataset to run. It is **not** needed to reproduce Fig. 10 —
> `hardware_metrics.pkl` is already included here.

## Raw channel data

> 📌 **Dataset link coming soon** — the full raw dataset will be released on Zenodo
> before camera-ready. This section will be updated with the DOI and download instructions.

Each raw `.mat` file contains:
- Complex MIMO channel matrices `H[64 freq bins, 2×2, N packets]`
- Measured at 27 GHz with a 2×2 MIMO testbed
- Cases correspond to four (d_tx, d_rx) aperture configurations from Table I

| Case | d_tx | d_rx |
|------|------|------|
| case1 | 50 cm | 50 cm |
| case2 | 50 cm | 20 cm |
| case3 | 20 cm | 20 cm |
| case4 | 20 cm | 50 cm |

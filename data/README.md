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

## Raw channel data

The full raw dataset (~31 GB of `.mat` channel matrix files) will be released
on Zenodo. A download link and DOI will be added here before camera-ready.

Each raw file contains:
- Complex MIMO channel matrices `H[64 freq bins, 2×2, N packets]`
- Measured at 27 GHz with a 2×2 MIMO testbed
- Cases correspond to four (d_tx, d_rx) aperture configurations from Table I

| Case | d_tx | d_rx |
|------|------|------|
| case1 | 50 cm | 50 cm |
| case2 | 50 cm | 20 cm |
| case3 | 20 cm | 20 cm |
| case4 | 20 cm | 50 cm |

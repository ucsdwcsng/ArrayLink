# Hardware Data

The hardware experiment data for Fig. 10 will be released as part of the
ArrayLink dataset.

## Expected file

Place `hardware_metrics.pkl` in this directory (i.e. `data/hardware_metrics.pkl`).

## Format

The pickle file should contain a Python `dict` with the following structure:

```python
{
    'case1': {
        'distances':       [2.5, 5.0, 10.0, ...],   # metres
        'mean_sing_ratio': [0.82, 0.75, 0.61, ...],  # mean sigma_2/sigma_1
        'std_sing_ratio':  [0.03, 0.04, 0.05, ...],  # std dev across packets
    },
    'case2': { ... },
    'case3': { ... },
    'case4': { ... },
}
```

where case1–case4 correspond to the four transmit/receive aperture configurations
in Table I of the paper:

| Case | d_tx (cm) | d_rx (cm) |
|------|-----------|-----------|
| 1    | 20        | 20        |
| 2    | 50        | 20        |
| 3    | 20        | 50        |
| 4    | 50        | 50        |

## Running Fig 10 without hardware data

`fig10_hardware_validation.py` will run with theory and simulation curves only
if the pkl file is absent. It prints a clear message and continues:

```
Hardware data not found at: data/hardware_metrics.pkl
Proceeding with theory + simulation curves only.
```

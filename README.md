# Behavioral responsiveness in epidemic resource allocation

Can county-level behavioral information improve the allocation of a fixed epidemic-intervention budget?

**Main experiment:** [Case-calibrated proxy analysis](results/case_calibrated_proxy/README.md) compares three planners over the same twelve weeks: epidemic only, a county predictor, and exact county scores. The replay uses NYT cases, observed mobility, and paired stochastic runs.

The project studies how knowledge of county responsiveness changes allocation in a mobility-coupled SIR model. The current code reproduces the U.S. county replay.

**Interpretation:** synthetic mechanism results demonstrate potential. Assigning survey fractions as responsiveness is a modeling assumption; the empirical-input simulations do not identify causal effects of spending.

## Main result at the smaller budget

In the 12-week U.S. replay, the reference with no added intervention produced **7.03 million modeled infections**. We chose a smaller weekly dose to bring epidemic-only planning near **3.5 million**, then held that dose fixed for all three planners across 16 paired simulation runs.

| Planner | Mean modeled infections | Fewer than epidemic-only planning |
|---|---:|---:|
| Epidemic only | 3,500,157 | Baseline |
| County predictor | 3,455,778 | **1.27%** |
| Exact county scores (oracle) | 3,326,142 | **4.97%** |

The case data calibrate transmission. The budget target is illustrative, and the intervention effect uses county masking ranks as a proxy for responsiveness. [Smaller-budget methods and results](results/case_calibrated_proxy/README.md#smaller-intervention-budget)

## Reproduce the result

Run from the repository root in Python 3.11 or newer. The data command downloads the required public sources and reuses cached copies.

```bash
python -m pip install -e .
python -m epidemic.data
python -m epidemic.us_weekly_case_calibration
python -m epidemic.us_case_calibrated_proxy --target-baseline 3500000
python -m unittest discover -s tests -v
```

| Module | Purpose |
|---|---|
| `data.py` | Download and prepare the county data and mobility matrices |
| `us_weekly_case_calibration.py` | Fit the case-based epidemic and write the replay inputs |
| `responsiveness_mechanisms.py` | Simulate expected infections and optimize the weekly allocation |
| `us_case_calibrated_proxy.py` | Build county predictions and compare the three plans in matched stochastic runs |

[Replay methods and results](results/case_calibrated_proxy/README.md) · [Data sources](DATA_SOURCES.md) · [Midterm archive with earlier code, results, and maps](https://github.com/yoshspider/epidemic-responsiveness-optimization/tree/02695ee)

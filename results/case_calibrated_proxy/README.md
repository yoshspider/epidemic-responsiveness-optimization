# One U.S. replay, three planners

The experiment asks how much exact county responsiveness would improve allocation, and how much of that benefit a county predictor can recover. All three planners operate on the same 3,133 counties, start from NYT reported cases on September 7, 2020, and use the same twelve observed GeoDS mobility weeks and weekly dose. The simulation runs continuously through November 29.

## What each planner sees

| Planner | County responsiveness used to plan |
|---|---|
| Epidemic only | Uses a fixed modeling value of 0.5 for every county. It can allocate unevenly based on the epidemic and mobility. |
| County predictor | Estimates each county score from pre-pandemic Census and CDC SVI features. Ridge regression trains on other states in five state-grouped folds. |
| Exact county scores | Receives the exact responsiveness scores used by the simulator. This estimates the potential benefit of better information for this model and optimizer. |

Responsiveness in this run is the percentile rank of each county's July NYT masking score. The same response map determines what happens under every policy. Percentiles specify a wider spread in assumed responsiveness than the raw masking fractions. The earlier run using raw fractions gave exact information a 2.58% advantage. This difference matters because masking behavior is a proxy for intervention response.

## Model and evaluation

The case fit estimates local and mobility transmission from eight weeks of reported county cases and checks the following four weeks. The fitted coefficients are 0.1047 and 0.0346 per day, respectively. The allocation replay overlaps the fit window. Earlier calibration checks also show that a constant-rate model misses much of the November surge.

Each planner selects twelve weekly county doses to minimize the number of new infections in an expected-count SIR rollout, with a population-weighted weekly dose of 0.8. The dose multiplies the daily infection hazard by $e^{-p_i d_{iw}}$. We then evaluate all three preplanned schedules in 16 matched stochastic runs, drawing daily infections and recoveries from binomial distributions. The paired intervals describe simulator variation under the fixed assumptions.

## Results

| Planner | Modeled infections | Fewer than epidemic only | Share of exact-score gain |
|---|---:|---:|---:|
| Epidemic only | 438,904 | — | — |
| County predictor | 425,932 | **2.96%** | **38.0%** |
| Exact county scores | 404,801 | **7.77%** | 100% |

Exact county scores avoid about **34,103** modeled infections. The county predictor avoids about **12,973** under the same budget. It recovers about two fifths of the benefit available when the planner sees the exact scores. See the [summary](summary.csv), [paired trials](trials.csv), and [model settings](manifest.json).

## Reproduce

Prepare the [case, mobility, survey, and county data](../us_weekly_information/REPRODUCE.md#2-download-and-prepare-the-inputs). This comparison uses the national bootstrap data, including the CDC SVI file.

```bash
python -m epidemic.us_weekly_case_calibration
python -m epidemic.us_case_calibrated_proxy
```

The second command writes the summary, per-seed trials, and model settings to outputs/case_calibrated_main. The [previous experimental record](https://github.com/yoshspider/epidemic-responsiveness-optimization/tree/e3f7b5b/results/case_calibrated_proxy) preserves the budget, longer-horizon, and tweet feature diagnostics.

# One U.S. replay, three planners

The experiment asks how much exact county responsiveness would improve allocation, and how much of that benefit a county predictor can recover. All three planners operate on the same 3,133 counties, start from NYT reported cases on September 7, 2020, and use the same twelve observed GeoDS mobility weeks and weekly dose. The simulation runs continuously through November 29.

## What each planner sees

| Planner | County responsiveness used to plan |
|---|---|
| Epidemic only | Uses a fixed modeling value of 0.5 for every county. It can allocate unevenly based on the epidemic and mobility. |
| County predictor | Estimates each county score from pre-pandemic Census and CDC SVI features. Ridge regression trains on other states in five state-grouped folds. |
| Exact county scores | Receives the exact responsiveness scores used by the simulator. This estimates the potential benefit of better information for this model and optimizer. |

Responsiveness in this run is the percentile rank of each county's July NYT masking score. The same response map determines what happens under every policy. This uses a wider spread of assumed responsiveness than the raw masking fractions. Masking ranks are a proxy for intervention response.

## Model and evaluation

The case fit estimates local and mobility transmission from eight weeks of reported county cases and checks the following four weeks. The fitted coefficients are 0.1047 and 0.0346 per day, respectively. The allocation replay overlaps the fit window. Earlier calibration checks also show that a constant-rate model misses much of the November surge.

Each planner selects twelve weekly county doses to minimize the number of new infections in an expected-count SIR rollout, with a population-weighted weekly dose of 0.8. The dose multiplies the daily infection hazard by $e^{-p_i d_{iw}}$. We then evaluate all three preplanned schedules in 16 matched stochastic runs, drawing daily infections and recoveries from binomial distributions.

## Results

| Planner | Fewer modeled infections vs. epidemic only |
|---|---:|
| Epidemic only | 0% (baseline) |
| County predictor | **2.96%** |
| Exact county scores | **7.77%** |

The percentage is the difference in mean modeled infections divided by the epidemic-only mean. All three planners use the same weekly budget. [Summary](summary.csv) · [Paired trials](trials.csv) · [Model settings](manifest.json)

## Reproduce

Prepare the [case, mobility, survey, and county data](../us_weekly_information/REPRODUCE.md#2-download-and-prepare-the-inputs). This comparison uses the national bootstrap data, including the CDC SVI file.

```bash
python -m epidemic.us_weekly_case_calibration
python -m epidemic.us_case_calibrated_proxy
```

The second command writes the summary, per-seed trials, and model settings to outputs/case_calibrated_main. The [previous experimental record](https://github.com/yoshspider/epidemic-responsiveness-optimization/tree/e3f7b5b/results/case_calibrated_proxy) preserves the budget, longer-horizon, and tweet feature diagnostics.

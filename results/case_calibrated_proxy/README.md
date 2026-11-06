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

The original constant-rate replay missed the November surge. This run carries the case-initialized SIR state forward continuously and fits one effective transmission rate per week to match that week's national case count on the same reporting county-days. The fitted rates range from 0.127 to 0.181 per day. The mobility share is fixed at 0.431 from the continuous fit on the first eight weeks. Every planner uses the same rates and observed mobility.

The national weekly match is by construction and uses all twelve weeks of outcomes. Treat this as a retrospective counterfactual with observed epidemic timing. It does not validate county-level trajectories or estimate what a planner could forecast in real time. [Weekly case fit](weekly_tracking_fit.csv)

Each planner selects twelve weekly county doses to minimize the number of new infections in an expected-count SIR rollout, with a population-weighted weekly dose of 0.8. The dose multiplies the daily infection hazard by $e^{-p_i d_{iw}}$. We then evaluate all three preplanned schedules in 16 matched stochastic runs, drawing daily infections and recoveries from binomial distributions.

## Results

| Planner | Fewer modeled infections vs. epidemic only |
|---|---:|
| Epidemic only | 0% (baseline) |
| County predictor | **4.67%** |
| Exact county scores | **11.38%** |

The percentage is the difference in mean modeled infections divided by the epidemic-only mean. All three planners use the same weekly budget. [Summary](summary.csv) · [Paired trials](trials.csv) · [Model settings](manifest.json)

## Smaller intervention budget

The fitted reference with no added intervention produces 7.03 million modeled infections. We chose a weekly dose of 0.10713 to put the epidemic-only plan near 3.5 million in an expected-count rollout. Budget selection used only the epidemic-only outcome. We then fixed the dose for all three planners and evaluated 16 paired stochastic runs.

| Planner | Mean modeled infections | Fewer than epidemic only |
|---|---:|---:|
| Epidemic only | 3,500,157 | 0% |
| County predictor | 3,455,778 | 1.27% |
| Exact county scores | 3,326,142 | 4.97% |

The smaller budget reduces the assumed overall intervention impact to about 50% of the reference. This is a budget sensitivity with an illustrative target. The historical case curve calibrates transmission, while the model assumes the intervention effect. [Summary](lower_budget_summary.csv) · [Paired trials](lower_budget_trials.csv) · [Model settings](lower_budget_manifest.json)

## Reproduce

Install the package with python -m pip install -e . and run these commands from the repository root. The data command prepares the cases, mobility, surveys, and CDC SVI county features.

```bash
python -m epidemic.data
python -m epidemic.us_weekly_case_calibration
python -m epidemic.us_case_calibrated_proxy
python -m epidemic.us_case_calibrated_proxy --target-baseline 3500000
```

The default replay writes results to outputs/case_calibrated_main. The smaller-budget run writes to outputs/case_calibrated_main_target_3.5e+06. The [previous experimental record](https://github.com/yoshspider/epidemic-responsiveness-optimization/tree/e3f7b5b/results/case_calibrated_proxy) preserves the budget, longer-horizon, and tweet feature diagnostics.

# Historical-input proxy analysis

**Research question:** Does county-level behavioral information improve weekly intervention allocation beyond what an epidemic-only planner can do? We compare several sources of information and separately ask whether tweet sentiment adds value beyond population and tweet volume.

## What the uninformed planner sees

The epidemic-only planner sees population, initial SIR state, mobility, fitted transmission rates and the weekly budget. It gets **no county responsiveness measurements, no fitted average or histogram, and no simulated response distribution**. For optimization, it uses the same prespecified response coefficient of 0.5 everywhere. This is a convention for translating dose into an effect when the planner has no behavioral data. Setting that coefficient to zero would make every feasible allocation have exactly the same predicted outcome. We also report **uniform dose per resident**, which uses no assumed response coefficient even to choose the allocation. The older U.S. benchmark uses a distribution-aware comparator and remains a separate experiment.

## 1. Fit the disease model

Use Census populations and the NYT county case series. For a week starting on date $t$, let $A_i(t)$ be cumulative reported cases through the previous day and $A_i(t-7)$ the count one week earlier. Initialize $I_i=A_i(t)-A_i(t-7)$, $R_i=A_i(t-7)$, and $S_i=N_i-A_i(t)$. Use reported cases directly. Missing intermediate reports are carried forward for state initialization. If a reporting revision produces an impossible initial state, bound it to a valid state for network simulation and exclude that county-week from fitting and scoring. Negative daily revisions and non-daily gaps are also excluded from the loss.

Let $W_{ij,w}$ be the recorded GeoDS flow from origin $j$ into destination $i$, normalized by total incoming flow in week $w$ after removing self-flows. Rows without recorded inflow remain zero. Define the daily hazard

$$
h_{it}^0 = eta_L I_{it}/N_i+eta_M sum_j W_{ij,w} I_{jt}/N_j.
$$

Fit nonnegative $eta_L,eta_M$ by minimizing the sum of squared county-day errors between $sqrt{widehat X_{it}+0.375}$ and $sqrt{X_{it}^{NYT}+0.375}$. For each training week, initialize from cases available before its start, then run a seven-day expected-count SIR forecast without intervention. Use weeks starting September 7 through October 26, 2020 for fitting. Freeze the rates and compare one-week forecasts starting November 2, 9, 16 and 23 with two controls: a 15% mobility split and local transmission alone. This checks forecasting at a one-week horizon given observed mobility and preweek case counts.

## 2. Define the two behavioral scenarios

The historical allocation replay starts **once** on November 2 from the NYT-based SIR state. It runs four weeks without resetting to observed cases. The scenarios share that starting state, rates, network, budget and random draws. The only difference in execution is the responsiveness surface used to turn dose into effect.

| Scenario | County response used by the simulator | What the proxy planners see |
|---|---|---|
| July NYT masking | Raw score $p_i=	ext{always}_i+	ext{frequently}_i+0.5,	ext{sometimes}_i$ | A prediction from log population. The oracle sees the survey score itself. |
| October CTIS masking | Raw October mean masking fraction for counties with at least 20 observed days. Elsewhere use the fixed 0.5 convention, including in the oracle. | Three nested predictors: log population, population plus log tweet count, and those features plus September TBCOV sentiment. |

The simulator treats masking levels as proxies for responsiveness. The October analysis is a retrospective comparison with a partly observed response surface. It does not establish an as-of-November survey release or an observed effect of spending. For CTIS, proxy-based plans are given the same label-coverage mask and use 0.5 for counties outside it. That means their comparison with the oracle applies to the covered portion of the response surface. They share the rest of the same nationwide network.

For prediction, split states into five groups. For each held-out group, fit to labeled counties in the other four groups. Inside each training fold, median-impute missing features, standardize, and fit ridge regression with penalty 10. Clip predictions to [0,1]. September tweet count enters as log(1 + count), with zero recorded posts for missing counties. Sentiment features are mean sentiment, positive share and negative share. Compare the sentiment model directly with the otherwise matched population-plus-activity model. This isolates sentiment from tweet volume.

## 3. Plan allocations with the same budget

For each intervention policy and each week, choose dose per resident $d_{iw}$ with

$$
0leq d_{iw}leq 8,qquad sum_i N_i d_{iw}=0.8sum_i N_i.
$$

The intervention changes the hazard to $h_{it}=exp(-p_i d_{iw})h_{it}^0$. The dose is normalized resources divided by county population, so this is the proposal's exponential attenuation. The solver plugs in 0.5, a held-out prediction, or the exact surrogate label according to the planner's information. It minimizes the total expected new infections in a deterministic SIR rollout over 28 days. For the deterministic planning rollout, replace each binomial draw below with its conditional mean and minimize the sum of daily new infections across counties. Each dose is 8 times the logistic function of a free county score plus a common shift for the week. Bisection sets that shift to meet the budget. L-BFGS-B uses derivatives propagated through the daily SIR updates and stops after at most 60 iterations. The oracle can start from any other feasible intervention plan. All allocations are preplanned using the observed mobility sequence. This is a numerical lookahead benchmark for the proposal's information comparison. Adaptive weekly PPO is a separate implementation.

## 4. Evaluate with the proposal's stochastic SIR

For each county-day, draw new infections $X_{it}$ from $mathrm{Binomial}(S_{it},1-e^{-h_{it}})$ and recoveries $Y_{it}$ from $mathrm{Binomial}(I_{it},1-e^{-1/7})$. Update $(S,I,R)$ by $(-X,X-Y,Y)$. Evaluate every policy on the same underlying scenario response surface, regardless of what it used for planning. Use 16 seeds, 1049200 through 1049215, with common county-day uniform random numbers passed through each policy's binomial inverse CDF. The paired intervals summarize epidemic draw variability conditional on fitted rates, coverage and surrogate assumptions.

Compare modeled infections against the uniform and epidemic-only policies. Use the oracle as a reference for information value. For the social-media question, compare the two CTIS predictors that differ only in sentiment features. The [earlier synthetic perfect-information and HHS survey experiments](../us_weekly_information/METHODS.md) have their own pipeline and use a distribution-aware reference. They share the exponential dose equation, but use synthetic initial states, fixed transmission settings, and different response maps.

## Results

The historical fit gives local and mobility coefficients of 0.1047 and 0.0346 per day, equivalent to total transmission 0.1392 and mobility weight 0.248. Holdout error on the square-root scale was 2.418, versus 2.421 when the weight was fixed at 15%. The difference is small. See the [calibration comparisons](calibration.csv) and [fit manifest](calibration_manifest.json).

| Policy | July NYT masking | October CTIS masking |
|---|---:|---:|
| No intervention | 2,162,029 | 2,162,029 |
| Uniform | 517,123 | 610,788 |
| Optimized epidemic only | 442,262 | 518,516 |
| Population predictor | 432,048 | 509,762 |
| Population and tweet volume | — | 509,789 |
| Population, tweet volume and sentiment | — | 509,719 |
| Survey oracle | 430,810 | 509,126 |

Values are average simulated infections, rounded to the nearest infection. The July oracle reduces infections by 2.59% versus epidemic-only allocation. The October sentiment predictor reduces infections by 1.70% versus epidemic-only allocation. The comparison that isolates sentiment is much smaller: about **70 fewer infections (0.014%)** than the otherwise matched population-plus-volume policy. The paired interval is 48–91 infections under the fixed model. Held-out CTIS prediction RMSE falls from 0.05823 to 0.05569 with sentiment. October labels cover 451 of 3,133 counties, and 436 of the 451 have tweet features. Survey, model, calibration, and geographic uncertainty are outside the paired interval.

[Policy summary](summary.csv) · [Paired trials](trials.csv) · [Prediction errors](prediction.csv) · [Scenario manifest](manifest.json)

## Reproduce

From the repository root, prepare the historical inputs and surveys using the [data setup guide](../us_weekly_information/REPRODUCE.md). Prepare the September tweet aggregates with the existing TBCOV pipeline. It requires the September 2020 archive and may take time.

```bash
python -m epidemic.tbcov_pipeline --archive data/raw/tbcov_september2020.zip
python -m epidemic.us_weekly_case_calibration
python -m epidemic.us_case_calibrated_proxy
```

The first command builds county tweet features from the archive. The next command writes calibration and initialization files to outputs/weekly_case_calibration. The final command writes policy schedules, prediction errors, and paired simulation results to outputs/case_calibrated_proxy.

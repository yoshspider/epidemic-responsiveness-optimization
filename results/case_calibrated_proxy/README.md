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

## Full 12-week replay and bridge

The full replay uses the NYT-initialized September 7 state and all twelve observed GeoDS weeks through November 29. It uses the same fitted transmission coefficients, weekly budget, planning algorithm, and paired binomial draws as the four-week run. The fitting period overlaps the first eight weeks, so this is a retrospective scenario. July masking precedes the start date. The October CTIS and September tweet comparison remains in the November window to preserve the ordering of the inputs.

The full replay tests both raw July masking fractions and their county percentile ranks. The rank sensitivity preserves the ordering of counties while changing the mean and spread of the response assumed by the simulator. Treat the two results as different intervention assumptions.

The bridge starts every run with the same November 2 county state. It crosses the fitted transmission rate or the earlier 0.36 setting with four or twelve weeks and raw or ranked July masking. For the twelve-week bridge, the November mobility sequence repeats. Each run replans for its own assumptions. The weekly budget stays fixed, so the longer horizon also gives three times as many weeks of intervention. Bridge totals use expected-count rollouts. A paired check on the original November raw scenario additionally compares a planner that knows only the national mean response with a planner that knows individual county scores.

### What changed over twelve weeks

| September–November replay | No intervention | Epidemic only | Population predictor | Survey oracle |
|---|---:|---:|---:|---:|
| Raw July masking | 3,655,533 | 263,982 | 258,207 | 257,163 |
| Ranked July masking sensitivity | 3,655,533 | 438,904 | 428,682 | 404,805 |

The epidemic-only policy avoids **3.39 million** modeled infections in the raw scenario. Exact county scores avoid a further **6,818 (2.58%)**, compared with **34,099 (7.77%)** when the response model uses ranks. These are paired stochastic results. See the [12-week summary](replay_12week_summary.csv), [paired trials](replay_12week_trials.csv) and [manifest](replay_12week_manifest.json).

| Bridge setup | Oracle advantage with raw masking | Oracle advantage with ranked masking |
|---|---:|---:|
| Fitted transmission, 4 weeks | 11,404 | 20,419 |
| Fitted transmission, 12 weeks | 12,167 | 72,206 |
| High transmission, 4 weeks | 150,411 | 1,519,881 |
| High transmission, 12 weeks | 90,541,077 | 65,122,827 |

Each bridge cell compares the oracle against the planner using a constant 0.5. The big raw-score difference in the high-transmission 12-week cell includes a correction to the assumed overall response level. A planner given only the national mean response reduced the modeled total from 131.65M to 44.68M. The oracle then reduced it to 41.11M, an additional **3.57M (8.0%)** from knowing the county scores in this high-transmission setting. The high transmission coefficient is an assumed sensitivity setting. See the [full bridge](bridge_summary.csv) and [paired November diagnostic](bridge_november_paired.csv).

## Longer horizons and intermediate growth

With fitted transmission, $\beta=0.139$ per day is close to the daily removal probability of 0.133. In the simple homogeneous, fully susceptible approximation, their ratio is about 1.05 before intervention. For comparison, the earlier high-transmission benchmark at $\beta=0.36$ has a ratio of about 2.70. At the lower rate, allocating the weekly dose usually suppresses spread early. Extra weeks can add substantial infections to the **no-intervention** run while adding few to the intervention runs, leaving little additional room for differences between those policies.

The long-horizon sensitivity repeats the November mobility sequence for 24 and 48 weeks, with the same starting state and weekly budget rule as the bridge. These are simulations beyond the observed replay window. The intermediate growth sensitivity instead keeps the horizon at 12 weeks, uses raw July masking, and varies the total transmission coefficient across 0.22, 0.26 and 0.30 per day. It compares the county predictor and oracle against a planner that knows only the national average score, to separate county-specific information from adjusting the overall assumed effect.

| Horizon at fitted rate | No intervention | Raw masking oracle advantage | Ranked masking oracle advantage |
|---|---:|---:|---:|
| 4 weeks | 2.16M | 11,404 (2.58%) | 20,419 (3.17%) |
| 12 weeks | 6.54M | 12,167 (2.35%) | 72,206 (7.96%) |
| 24 weeks | 12.48M | 12,863 (2.47%) | 114,986 (11.86%) |
| 48 weeks | 20.05M | 12,815 (2.46%) | 125,478 (12.78%) |

The two right-hand columns compare against epidemic-only planning separately within each response scale. The ranked masking predictor itself avoids about 33,600 infections at 48 weeks (3.42%). The rank experiment changes the assumed dose response, so its oracle percentage is distinct from the earlier HHS predictor's 7.82%. The 4- and 12-week rows come from the bridge with November initialization. See the [24- and 48-week results](long_horizon_summary.csv).

| Assumed total rate, 12 weeks | Additional infections avoided by the population proxy | Additional infections avoided by the oracle |
|---|---:|---:|
| 0.22/day | 28,362 | 65,480 |
| 0.26/day | 135,029 | 298,740 |
| 0.30/day | 536,228 | 1,172,957 |
| 0.36/day | 1,502,398 | 3,568,432 |

Each column is relative to the national-average planner under the same epidemic and raw masking assumption. The last row comes from the earlier bridge. The other rows are in the [intermediate growth results](growth_grid_summary.csv).

## Checking the historical fit and intervention dose

The faster-spread benchmark is a sensitivity experiment. Its approximately 300 million modeled infections without intervention are far above the reported-case scale of the September–November 2020 replay. In the replay, case data initialize the epidemic and fit its transmission parameters.

The first fit started from reported county states each week during calibration. The allocation simulation starts only once. To align those procedures, a second fit starts on September 7 and carries its states forward continuously. It minimizes the same square-root-scale county-day loss on the first eight weeks, then uses the frozen rates for the final four weeks. On this continuous holdout, county-day RMSE fell from 3.80 to 3.57. The continuous fit modeled 4.11 million cases across all twelve weeks, compared with 3.66 million under the original coefficients. On scored county-days in the four-week holdout, it predicted 1.55 million cases versus 4.08 million reported. The constant-rate replay still misses the rise in November. See the [continuous calibration](continuous_calibration.csv) and [matched weekly totals](continuous_weekly_totals.csv).

The budget sensitivity keeps the same twelve observed mobility weeks, September initialization and raw July masking scores. It uses mean-count rollouts at weekly doses 0.05, 0.15, 0.35 and 0.8. Each comparison holds the dose fixed across policies. The national-average planner receives a single nationwide response level. This makes it possible to distinguish county-specific information from simply correcting the epidemic-only planner's common 0.5 assumption.

| Weekly dose | Epidemic only | National average | Population proxy | County-score oracle | Proxy gain vs. national average |
|---|---:|---:|---:|---:|---:|
| 0.05 | 2,722,221 | 2,694,259 | 2,693,426 | 2,686,818 | 0.03% |
| 0.15 | 1,636,824 | 1,605,270 | 1,603,656 | 1,598,277 | 0.10% |
| 0.35 | 794,375 | 770,344 | 769,120 | 766,530 | 0.16% |
| 0.8 | 279,527 | 273,411 | 272,808 | 271,460 | 0.22% |

All doses had the same modeled total of 4.11 million without intervention. The population proxy improved on epidemic-only planning, but most of that improvement also came from giving the planner the national average. The proxy's incremental gain ranged from 0.03% to 0.22% over that average-only comparison. The oracle gained less than 1% over the national average at every dose. See the [budget sweep](continuous_budget_summary.csv).

## Reproduce

From the repository root, prepare the historical inputs and surveys using the [data setup guide](../us_weekly_information/REPRODUCE.md). Prepare the September tweet aggregates with the existing TBCOV pipeline. It requires the September 2020 archive and may take time.

```bash
python -m epidemic.tbcov_pipeline --archive data/raw/tbcov_september2020.zip
python -m epidemic.us_weekly_case_calibration
python -m epidemic.us_case_calibrated_proxy
python -m epidemic.us_case_calibrated_proxy --weeks 12
python -m epidemic.us_case_calibration_bridge
python -m epidemic.us_case_calibration_bridge --long-horizon
python -m epidemic.us_case_calibration_bridge --growth-grid
python -m epidemic.us_case_calibration_bridge --historical-budgets --continuous-calibration
```

The first command builds county tweet features from the archive. The next command writes calibration and initialization files to outputs/weekly_case_calibration. The final command writes policy schedules, prediction errors, and paired simulation results to outputs/case_calibrated_proxy.

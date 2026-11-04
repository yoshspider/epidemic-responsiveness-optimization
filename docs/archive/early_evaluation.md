# Behavioral information: evaluated results

Updated October 14, 2026. The evidence supports a substantial benefit when responsiveness is heterogeneous and learnable. Real survey prediction has some signal, but its downstream intervention benefit remains unproven.

## Fully simulated prediction and allocation

The new experiment trains a random forest on 4,000 synthetic counties with noisy response labels (noise SD 0.06). Six observable Gaussian features are generated; three determine latent response through a nonlinear logistic function. Training and test seeds are separate. The fixed model is evaluated on 16 fresh, independently generated 24-county epidemic worlds for 84 days. Each policy receives the same budget, epidemic conditions and execution response. The uninformed policy receives the training-label mean. All six scenario/budget combinations are retained.

| Responsiveness | Held-out prediction R² | Prediction MAE | Weekly budget fraction | Learned policy additional infections prevented | 95% paired interval | Relative benefit |
|---|---:|---:|---:|---:|---:|---:|
| Constant | Undefined | 0.00094 | 5% or 20% | 0 | 0–0 | 0% |
| Moderate variation | 0.982 | 0.0202 | 5% | 6,581 | 4,899–8,263 | 0.70% |
| Moderate variation | 0.982 | 0.0202 | 20% | 17,320 | 13,463–21,177 | 2.09% |
| Strong variation | 0.986 | 0.0324 | 5% | 20,318 | 16,731–23,906 | 2.13% |
| Strong variation | 0.986 | 0.0324 | 20% | 54,098 | 46,308–61,887 | 6.32% |

Budget fractions refer to the cost of one population-equivalent effort across all counties, released weekly. They are model effort budgets, not measured spending. Learned policies recover approximately 97–98% of the corresponding greedy oracle gain. Constant response produces no useful county information; for that control the predicted policy uses the common training mean.

Intervals span independent test worlds and paired epidemic noise, conditional on one fitted predictor and the specified generator. These are exploratory, unadjusted intervals. High prediction accuracy is possible here because the features were deliberately made informative. This establishes a mechanism, not comparable accuracy for real social media features.

Reproduce with `python -m epidemic.learnable_behavior --worlds 16` and `python -m epidemic.evaluation_figures`. Complete predictions, trials, configuration and summaries are in `outputs/learnable_behavior`.

## Corrected survey proxy and COVID replay

The brief's behavior summary is reproduced almost exactly by **ALWAYS + FREQUENTLY + 0.5 × SOMETIMES** in the NYT survey: 3,142 counties; mean 0.776477; SD 0.111308; minimum 0.3485; median 0.7845; maximum 0.994. Earlier experiments used ALWAYS + FREQUENTLY and remain separate sensitivity analyses. This correction weights occasional adherence rather than changing observed shares. Neither formula identifies causal responsiveness to additional spending.

The corrected-proxy replay retains 3,041 counties and starts August 31, 2020. It uses Census populations, NYT initialization and one fixed July GeoDS mobility snapshot. Weekly transmission rates were fitted contemporaneously to historical cases, with a fixed 15% mobility component. Hence this is a retrospective model replay, not an out-of-sample epidemic forecast. Intervention efficacy, persistence, reporting fraction and cost remain assumed. Both policies receive the same aggregate response mean; only the oracle receives county responses.

| Weekly budget fraction | Oracle additional infections prevented | Paired 95% interval | Relative benefit |
|---|---:|---:|---:|
| 0.5% | 37,944 | 33,569–42,320 | 0.160% |
| 2% | 53,934 | 50,365–57,503 | 0.257% |

Eight new paired epidemic seeds are used. Intervals quantify epidemic noise conditional on the model, not causal or parameter uncertainty. These positive results concern known survey proxies, not learned predictions. Reproduce with `python -m epidemic.graded_proxy_study`; complete outputs are in `outputs/graded_proxy_replay`.

## What real TBCOV prediction has established

The complete September TBCOV archive was downloaded and streamed: 88,931,846 records, including 8,937,269 records mapped to US resident county labels. No tweet hydration was needed. September features predict October CTIS descriptive mask estimates. State-blocked cross-validation uses 437 counties with sufficient survey coverage and sentiment features.

The combined forest achieved R² 0.259 and MAE 0.0418, versus population-only R² 0.027. Posting volume accounts for part of this improvement: population plus posting activity reached R² 0.208; adding sentiment beyond activity has an uncertain forest MAE improvement. These are retrospective geographic validation results, not verified real-time availability.

The completed downstream November comparison did not show a reliable gain from the combined prediction. At the smaller budget it saved approximately 55 infections, with an interval crossing zero; at the larger budget it performed worse than the uninformed policy. The oracle greedy policy also sometimes lost. Better mask prediction does not automatically improve a myopic epidemic allocation objective.

## Theory and practical interpretation

For an optimal policy under a correct model, free additional information cannot worsen expected performance: the policy may ignore it. The exact finite-plan experiment in `outputs/exact_information` verifies this principle within its enumerated plan class. National greedy and finite-horizon optimization are heuristics, so that guarantee does not apply to each observed comparison.

The defensible result is two-part: a learned behavior model can deliver a substantial intervention benefit when informative covariates exist; a historical COVID replay shows positive value for known county proxies, while real sentiment-based allocation still needs better target identification and validation. Current adherence and causal marginal responsiveness should be distinguished before claiming practical intervention effectiveness.

All 23 existing verification tests passed after these experiments. The figures `outputs/learnable_behavior/policy_comparison.png` and `prediction_comparison.png` show policy effects and held-out prediction respectively.

## Multi-month follow-up in progress

The authorized follow-up is downloading full TBCOV archives for August 2020–March 2021 and has obtained CTIS mask estimates through April 2021. Twitter aggregation and the multi-month Twitter evaluation are not finished; do not treat the single-month results above as the outcome of this expanded study. See `MULTI_MONTH_STUDY.md` and `outputs/multi_month/status.json`.

The completed survey-history baseline is already much more predictive than population alone: carrying forward previous masking gives MAE 1.69 percentage points and pooled R² 0.807 on 1,061 retained January–April observations. For January alone, a ridge model using previous masking and population gives MAE 1.27 points and R² 0.828. Its paired MAE reduction versus persistence is 0.253 points, with a conditional state-cluster interval of 0.055–0.441 points. This is a different validation setting from held-out states in the single-month study and does not establish incremental Twitter value. April behavior decline caused the fitted history models to lose to persistence.

The expanded analysis keeps the February 2021 CTIS question change explicit and reports stable-definition and per-month results. All 26 verification tests now pass, including resumable range assembly and temporal prediction safeguards.

# Behavioral responsiveness in epidemic resource allocation

Can county-level behavioral information improve the allocation of a fixed epidemic-intervention budget?

This research project compares policies with **aggregate information**, **measured county proxies**, and **predicted responsiveness** in a mobility-coupled SIR model. It includes controlled mechanism experiments, U.S. county counterfactuals, prediction checks, PPO experiments, and geographic figures.

**Interpretation:** synthetic mechanism results demonstrate potential. Assigning survey fractions as responsiveness is a modeling assumption; the empirical-input simulations do not identify causal effects of spending.

## Start here

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e .
python -m epidemic.ideal_information_benchmark --days 84
python -m epidemic.moderate_ideal_confirm
```

The synthetic experiments need no downloaded datasets. Run commands from the repository root. Generated files go under ignored `outputs/`; reviewed snapshots are in [`results/`](results/).

For the complete tests and PPO modules, install CPU PyTorch separately or use the `rl` extra:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
```

## Current evidence

- The fixed-budget synthetic grid shows substantial information value across budgets; all tested settings are retained, rather than reporting only a favorable cell.
- A separate 64-world confirmation illustrates a moderate-intervention regime: perfect information reduces new infections by **34.1% relative to the uninformed policy**; synthetic noisy measurements with SD 0.15 recover a **24.9% reduction**. The budget in this diagnostic was selected to target a moderate uninformed effect; the informed outcomes were not fixed.
- U.S.-geography simulations show double-digit benefits from measured mask information in some settings. Stronger population-only controls explain much of the apparent benefit of demographic predictors.
- Historical COVID replay effects are materially smaller. Neither the Twitter features nor the observational surveys have established causal intervention responsiveness.

See the exported tables, full parameter grids, and each result manifest for exact settings and negative findings. Do not mix results from the earlier capped-adoption model with the current exponential-response model.

## Model and comparisons

Daily infections follow a bounded SIR hazard, with recovery probability `1-exp(-1/7)`. County populations are conserved. Weekly incoming mobility matrices have destination rows and source columns; self edges are removed and incoming rows normalized.

The main intervention mechanism is:

```text
lambda_i(t) = exp(-p_i * d_i(t)) * lambda_i_without_intervention(t)
d_i(t) = kappa * a_i(t) / N_i
```

Here `a_i` is effort, `p_i` is responsiveness, and `kappa` converts effort into effect. Dose units are **not calibrated dollars**. Comparisons use identical budgets, disease assumptions and execution response surfaces.

The independent-county benchmark computes globally optimal **static allocations on a finite dose grid**. National studies include both a separable horizon heuristic and a network-aware constrained optimizer with an exact finite-difference-checked SIR adjoint. The latter is a local numerical optimizer, not a proven global optimum. Static-horizon experiments are distinct from the weekly PPO experiments retained in `proposal_ppo.py`.

## U.S. data and reproduction

Sources include NYT COVID reports, GeoDS weekly mobility, Census populations, NYT/Dynata masking, HHS county vaccine-hesitancy estimates, CTIS, and TBCOV. NYT masking and HHS hesitancy each cover the same 3,142 Census county identifiers; HHS estimates are modeled small-area predictions, not independent direct surveys of every county.

```bash
python -m epidemic.bootstrap_us --mobility
python -m epidemic.national_mask_prediction
python -m epidemic.us_proxy_prepare
python -m epidemic.us_proxy_study
python -m epidemic.us_proxy_figures
```

Acquisition downloads substantial public datasets. The bootstrap does not download the very large TBCOV archives; their separate pipeline is retained for the social-media experiments. Raw datasets, tweet records, local dependencies and temporary publishing files are excluded from Git.

For network-aware evaluation, for example:

```bash
python -m epidemic.us_proxy_network_study --metric mask --dose 1 --beta 0.3
```

This is a scenario, not a fitted real-world intervention dose. The moderate-budget calibration and all other scenario definitions are documented in the result manifests and [`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md).

## Geographic outputs

The results snapshot includes county input maps, week 4/8/12 spread comparisons, allocation-benefit maps, and parameter sensitivity figures. County losses as well as gains are displayed; national improvements do not imply every county improves.

## Research limitations

1. Mask use, vaccine willingness, and sentiment measure behavior or attitudes, not the marginal response to intervention spending.
2. Controlled U.S.-geography outbreaks use synthetic initial epidemics on actual populations and networks. Historical replay is separately labeled and uses retrospectively fitted transmission.
3. Reported Monte Carlo intervals quantify epidemic noise conditional on the assumed proxy and model. They do not include proxy validity, survey error, reporting or parameter uncertainty.
4. Case-table aggregates and invalid initial states are explicitly excluded in historical replay. Synthetic geographic experiments include all 3,142 counties.
5. A large synthetic information effect does not guarantee a double-digit gain for empirical proxies or learned policies. Population-only and uniform-spending controls are retained.

[Data sources and attribution](DATA_SOURCES.md) · [Experiment index](docs/EXPERIMENT_INDEX.md) · [Historical notes](docs/archive/)

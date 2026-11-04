"""Network-aware policy evaluation; all source/model assumptions stay explicit."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
from .us_proxy_study import load_surveys, rank_map, save_snapshots
from .us_proxy_core import local_response_curves, interpolate_curves, cost_tables, horizon_allocation, simulate
from .us_proxy_network import checks, optimize, loss_gradient
from .us_proxy_confirmation import stochastic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['geographic', 'real_fall'], default='geographic')
    parser.add_argument('--metric', choices=['mask', 'vaccine'], default='mask')
    parser.add_argument('--dose', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=.3)
    parser.add_argument('--iterations', type=int, default=60)
    args = parser.parse_args()
    checks()
    print('Network gradient and budget checks passed', flush=True)
    root = Path('outputs/us_proxy_study')
    folder = root / args.kind
    name = f'{args.kind}_{args.metric}_{args.dose}_{args.beta}'
    out = root / 'network_aware' / name
    out.mkdir(parents=True, exist_ok=True)
    initial = np.load(folder / 'initial.npz')
    n = initial['n']
    s = initial['s']
    i = initial['i']
    r = initial['r']
    counties = pd.read_csv(folder / 'counties.csv', dtype={'fips': str})
    surveys = load_surveys(counties)
    p = surveys[args.metric]
    matrices = [sparse.load_npz(folder / f'mobility_{w}.npz') for w in range(12)]
    rates = np.full(
        12, args.beta) if args.kind == 'geographic' else pd.read_csv(
        folder / 'rates.csv').beta_total.to_numpy()
    doses = np.arange(0, 8.0001, .1)
    zgrid = np.arange(0, 8.0001, .025)
    curves = local_response_curves(n, s, i, rates, zgrid, 84)
    known, prior = cost_tables(curves, zgrid, doses, p, n)
    uniform = np.full(len(n), args.dose)
    prior_local = horizon_allocation(prior, n, doses, args.dose)
    mean = np.full(len(n), np.average(p, weights=n))
    diagnostics = json.loads((out / 'optimization.json').read_text()
                             ) if (out / 'optimization.json').exists() else {}
    plans = {'uniform': uniform, 'none': np.zeros(len(n))}
    prior_path = out / 'prior_candidate.npz'
    if prior_path.exists():
        prior_network = np.load(prior_path)['dose']
    else:
        prior_network, diag = optimize(n, s, i, [mean], matrices, rates, .15, args.dose, [
                                       prior_local], maxiter=args.iterations)
        np.savez_compressed(prior_path, dose=prior_network)
        diagnostics['prior_network'] = diag
        (out / 'optimization.json').write_text(json.dumps(diagnostics, indent=2))
    # Baseline selection sees only an aggregate response distribution and
    # independent synthetic prior validation maps, never target county labels.
    order = np.argsort(p)
    cdf = np.cumsum(n[order]) / n.sum()
    validation = []
    for draw in range(8):
        u = np.random.default_rng(998000 + draw).uniform(0, 1, len(n))
        validation.append(np.interp(u, cdf, p[order]))
    candidate_plans = {
        'uniform': uniform,
        'independent_prior': prior_local,
        'network_mean_prior': prior_network}
    prior_scores = {label: float(np.mean([loss_gradient(dose, n, s, i, world, matrices, rates, .15, False)
                                 for world in validation])) for label, dose in candidate_plans.items()}
    selected = min(prior_scores, key=prior_scores.get)
    plans['prior'] = candidate_plans[selected]
    print(name, 'prior selected', selected, flush=True)
    forecasts = {}
    if args.metric == 'mask':
        predictions = pd.read_csv(
            'outputs/nyt_alternative/state_heldout_predictions.csv',
            dtype={
                'fips': str}).set_index('fips').reindex(
            counties.fips)
        forecasts = {'population_prediction': predictions.forest_population.to_numpy(),
                     'demographic_prediction': predictions.forest_demographics.to_numpy()}
    else:
        forecasts = {
            'past_mask_raw': surveys['mask'],
            'past_mask_rank': rank_map(
                surveys['mask'],
                p,
                n)}
        from sklearn.model_selection import GroupKFold
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        x = np.log(n)[:, None]
        q = np.zeros(len(n))
        for train, test in GroupKFold(5).split(x, groups=counties.fips.str[:2]):
            learner = RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=1).fit(
                x[train],
                p[train])
            q[test] = learner.predict(x[test])
        forecasts['population_prediction'] = q
        (out / 'population_predictor_quality.json').write_text(json.dumps(dict(mae=mean_absolute_error(p, q), r2=r2_score(p, q),
                                                                               design='Five held-out state groups; geographic reconstruction of HHS modeled labels, not prospective forecasting'), indent=2))
    for policy, estimate in forecasts.items():
        path = out / f'{policy}_plan.npz'
        if path.exists():
            plans[policy] = np.load(path)['dose']
            continue
        local_cost = interpolate_curves(curves, zgrid, estimate[:, None] * doses[None, :])
        local_plan = horizon_allocation(local_cost, n, doses, args.dose)
        dose, diag = optimize(n, s, i, [estimate], matrices, rates, .15, args.dose, [
                              plans['prior'], local_plan], maxiter=args.iterations)
        plans[policy] = dose
        diagnostics[policy] = diag
        np.savez_compressed(path, dose=dose)
        (out / 'optimization.json').write_text(json.dumps(diagnostics, indent=2))
        print(name, 'network plan complete', policy, flush=True)
    path = out / 'proxy_plan.npz'
    if path.exists():
        plans['proxy'] = np.load(path)['dose']
    else:
        local_plan = horizon_allocation(known, n, doses, args.dose)
        candidates = [dose for label, dose in plans.items() if label != 'none'] + [local_plan]
        dose, diag = optimize(n, s, i, [p], matrices, rates, .15,
                              args.dose, candidates, maxiter=args.iterations)
        plans['proxy'] = dose
        diagnostics['proxy'] = diag
        np.savez_compressed(path, dose=dose)
        (out / 'optimization.json').write_text(json.dumps(diagnostics, indent=2))
        print(name, 'network plan complete proxy', flush=True)
    expected = []
    for policy, dose in plans.items():
        result = simulate(n, s, i, r, p, dose, matrices, rates, .15)
        save_snapshots(out, policy, result, dose)
        expected.append(
            dict(
                policy=policy,
                infections=result['total'],
                peak_day=int(
                    result['daily'].argmax() +
                    1)))
    table = pd.DataFrame(expected).set_index('policy')
    baseline = table.loc['prior', 'infections']
    table['gain_vs_prior'] = baseline - table.infections
    table['percent_vs_prior'] = 100 * table.gain_vs_prior / baseline
    table['gain_vs_uniform'] = table.loc['uniform', 'infections'] - table.infections
    table['gain_vs_population'] = table.loc['population_prediction', 'infections'] - table.infections
    table.to_csv(out / 'expected.csv')
    # Target true p is available only to reference planner and to evaluation.
    rows = []
    for draw in range(8):
        seed = 999000 + draw
        infected = np.minimum(
            np.random.default_rng(seed).poisson(
                n * .0001),
            n) if args.kind == 'geographic' else i
        for policy, dose in plans.items():
            total, peak = stochastic(n, n - infected - r, infected, r,
                                     p, dose, matrices, rates, seed)
            rows.append(dict(seed=draw, policy=policy, infections=total, peak_day=peak))
        pd.DataFrame(rows).to_csv(out / 'trials.csv', index=False)
    wide = pd.DataFrame(rows).pivot(index='seed', columns='policy', values='infections')
    summary = []
    for policy in wide:
        for baseline in ['prior', 'uniform', 'population_prediction']:
            delta = wide[baseline] - wide[policy]
            half = t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
            summary.append(
                dict(
                    policy=policy,
                    baseline=baseline,
                    infections=wide[policy].mean(),
                    gain=delta.mean(),
                    percent=100 *
                    delta.mean() /
                    wide[baseline].mean(),
                    ci_low=delta.mean() -
                    half,
                    ci_high=delta.mean() +
                    half))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(kind=args.kind, metric=args.metric, average_dose=args.dose, beta=args.beta,
                                                       planner='Network-aware static horizon L-BFGS-B with exact checked daily SIR adjoint; finite iteration cap, not proven global optimum or PPO',
                                                       optimization_iterations=args.iterations, prior_selection=selected, prior_validation_scores=prior_scores,
                                                       priors='Mean-response network plan, independent-prior plan, and uniform spending compared on eight fresh synthetic prior maps drawn from known global response histogram; target county labels excluded from baseline selection',
                                                       oracle='Exact assigned survey-derived response surface; starts from best feasible prior/forecast/uniform plans under its own information, so reference cannot be worse in deterministic model',
                                                       forecasts='Mask: five-state-fold pre-pandemic demographics/population. Vaccine: prior NYT mask raw/rank given shared histogram; added population-only five-state-fold reconstruction control',
                                                       uncertainty='Eight paired epidemic seeds999000–999007; intervals conditional on model/proxy assumptions only',
                                                       interpretation='Proxy-as-responsiveness counterfactual; geographic model uses synthetic initial outbreak; real_fall uses retrospectively calibrated actual-case initial state'), indent=2))
    print(table.to_string(), flush=True)


if __name__ == '__main__':
    main()

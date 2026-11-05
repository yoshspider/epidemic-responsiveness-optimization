"""Pure U.S.-network mechanism development: no proxies or measured p used."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
from .responsiveness_mechanisms import checks, objective, optimize
from .us_proxy_core import local_response_curves, cost_tables, horizon_allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mechanism', choices=['recipient', 'both_ends'], default='both_ends')
    parser.add_argument('--dose', type=float, default=.6)
    parser.add_argument('--beta', type=float, default=.36)
    parser.add_argument('--mobility', type=float, default=.15)
    parser.add_argument('--worlds', type=int, default=3)
    parser.add_argument('--seed-start', type=int, default=1010000)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--prior-scheme', choices=['iid', 'stratified'], default='iid')
    parser.add_argument('--phase', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--prior-from', type=Path)
    parser.add_argument('--warmstart-from', type=Path)
    parser.add_argument('--schedule', choices=['static', 'weekly'], default='static')
    args = parser.parse_args()
    checks()
    print('Mechanism normalization and ensemble-adjoint checks passed', flush=True)
    source = Path('outputs/us_proxy_study/geographic')
    initial = np.load(source / 'initial.npz')
    n, s, i = initial['n'], initial['s'], initial['i']
    matrices = [sparse.load_npz(source / f'mobility_{week}.npz') for week in range(12)]
    out = Path('outputs/us_perfect_information') / \
        f'{args.mechanism}_{args.beta}_{args.dose}_{args.mobility}_{args.seed_start}'
    if args.prior_scheme != 'iid':
        out = out.with_name(out.name + '_' + args.prior_scheme)
    if args.schedule == 'weekly':
        out = out.with_name(out.name + '_weekly')
    out.mkdir(parents=True, exist_ok=True)
    # Fixed prior simulations have no relationship to evaluation county labels.
    prior = np.column_stack([np.random.default_rng(
        1015000 + k).uniform(0, 1, len(n)) for k in range(8)])
    validation = np.column_stack([np.random.default_rng(
        1016000 + k).uniform(0, 1, len(n)) for k in range(8)])
    if args.prior_scheme == 'stratified':
        def stratified(seed):
            rng = np.random.default_rng(seed)
            strata = np.argsort(rng.random((len(n), 8)), axis=1)
            return (strata + rng.random((len(n), 8))) / 8
        prior, validation = stratified(1015000), stratified(1016000)
    doses = np.arange(0, 8.00001, .1)
    zgrid = np.arange(0, 8.00001, .025)
    curves = local_response_curves(n, s, i, np.full(12, args.beta), zgrid, 84)
    # An independent-county prior plan is an initializer, not the comparator.
    _, prior_cost = cost_tables(curves, zgrid, doses, prior[:, 0], n)
    local = horizon_allocation(prior_cost, n, doses, args.dose)
    uniform = np.full(len(n), args.dose)
    prior_path = out / 'prior.npz'
    if args.prior_from is not None:
        frozen = json.loads((args.prior_from / 'manifest.json').read_text())
        for key, value in [('mechanism', args.mechanism), ('beta', args.beta),
                           ('mobility', args.mobility), ('average_dose', args.dose)]:
            if frozen[key] != value:
                raise ValueError('Frozen comparator settings do not match this confirmation')
        baseline = np.load(args.prior_from / 'selected_baseline.npz')['dose']
        expected_shape = (12, len(n)) if args.schedule == 'weekly' else n.shape
        if baseline.shape != expected_shape:
            raise ValueError('Frozen comparator geography mismatch')
        name, scores = frozen['validation_selection'], frozen['validation_scores']
        planned = baseline
    elif prior_path.exists():
        planned = np.load(prior_path)['dose']
    else:
        starts = [local]
        if args.warmstart_from:
            starts.append(np.load(args.warmstart_from / 'selected_baseline.npz')['dose'])
        planned, diagnostic = optimize(n, s, i, prior, matrices, args.beta, args.mobility,
                                       args.mechanism, args.dose, starts, args.iterations, schedule=args.schedule)
        np.savez_compressed(prior_path, dose=planned)
        (out / 'prior_optimization.json').write_text(json.dumps(diagnostic, indent=2))
    if args.prior_from is None:
        candidates = {'uniform': uniform, 'independent': local, 'bayesian_network': planned}
        if args.warmstart_from:
            candidates['frozen_static'] = np.load(
                args.warmstart_from / 'selected_baseline.npz')['dose']
        scores = {name: objective(dose, n, s, i, validation, matrices, args.beta, args.mobility, args.mechanism, False)
                  for name, dose in candidates.items()}
        name = min(scores, key=scores.get)
        baseline = candidates[name]
    if args.schedule == 'weekly' and baseline.ndim == 1:
        baseline = np.tile(baseline, (12, 1))
    np.savez_compressed(out / 'selected_baseline.npz', dose=baseline)
    print('Uninformed comparator selected on independent prior validation:', name, flush=True)
    records = []
    none = objective(np.zeros(len(n)), n, s, i,
                     prior[:, 0], matrices, args.beta, args.mobility, args.mechanism, False)
    for draw in range(args.worlds):
        p = np.random.default_rng(args.seed_start + draw).uniform(0, 1, len(n))
        path = out / f'oracle_{draw}.npz'
        if path.exists():
            informed = np.load(path)['dose']
        else:
            known, _ = cost_tables(curves, zgrid, doses, p, n)
            local_oracle = horizon_allocation(known, n, doses, args.dose)
            starts = [baseline, local_oracle]
            if args.warmstart_from:
                warm_meta = json.loads((args.warmstart_from / 'manifest.json').read_text())
                warm_oracle = args.warmstart_from / f'oracle_{draw}.npz'
                if warm_meta['seed_start'] == args.seed_start and warm_oracle.exists():
                    starts.append(np.load(warm_oracle)['dose'])
            if args.schedule == 'weekly' and not any(np.asarray(candidate).ndim == 1 and
                                                     not np.allclose(candidate, candidate[0])
                                                     for candidate in starts[2:]):
                # Give every fresh response world the same static→weekly
                # optimization pipeline used during development. Do not make
                # confirmation depend on an unavailable development oracle.
                static_path = out / f'oracle_{draw}_static.npz'
                if static_path.exists():
                    static_plan = np.load(static_path)['dose']
                else:
                    static_starts = [baseline.mean(axis=0) if baseline.ndim == 2 else baseline,
                                     local_oracle]
                    static_plan, static_diagnostic = optimize(n, s, i, p, matrices, args.beta,
                                                              args.mobility, args.mechanism, args.dose, static_starts, args.iterations)
                    np.savez_compressed(static_path, dose=static_plan)
                    (
                        out /
                        f'oracle_{draw}_static_optimization.json').write_text(
                        json.dumps(
                            static_diagnostic,
                            indent=2))
                starts.append(static_plan)
            informed, diagnostic = optimize(n, s, i, p, matrices, args.beta, args.mobility,
                                            args.mechanism, args.dose, starts, args.iterations, schedule=args.schedule)
            np.savez_compressed(path, dose=informed, p=p)
            (out / f'oracle_{draw}_optimization.json').write_text(json.dumps(diagnostic, indent=2))
        for policy, dose in [('none', np.zeros(len(n))), ('uniform', uniform),
                             ('uninformed', baseline), ('perfect', informed)]:
            value = objective(dose, n, s, i, p, matrices, args.beta,
                              args.mobility, args.mechanism, False)
            records.append(dict(draw=draw, policy=policy, infections=value,
                           prevention_percent=100 * (none - value) / none))
        pd.DataFrame(records).to_csv(out / 'trials.csv', index=False)
        print(f'U.S. pure information {args.phase} world {draw + 1} complete', flush=True)
    wide = pd.DataFrame(records).pivot(index='draw', columns='policy', values='infections')
    rows = []
    for policy in wide:
        gain = wide.uninformed - wide[policy]
        half = t.ppf(.975, len(gain) - 1) * gain.std(ddof=1) / \
            np.sqrt(len(gain)) if len(gain) > 1 else np.nan
        rows.append(dict(policy=policy, infections=wide[policy].mean(), prevention_percent=100 * (none - wide[policy].mean()) / none,
                         gain=gain.mean(), relative_gain_percent=100 * gain.mean() / wide.uninformed.mean(), ci_low=gain.mean() - half, ci_high=gain.mean() + half))
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(phase=args.phase,
                                                       geographic_inputs='3,142 actual Census populations and twelve weekly GeoDS networks; synthetic initial prevalence .0001, no prior immunity',
                                                       response='Independent Uniform(0,1) per county, fixed over time; NO survey or proxy is used',
                                                       mechanism=args.mechanism, beta=args.beta, mobility=args.mobility, average_dose=args.dose,
                                                       schedule=args.schedule, warmstart_from=str(
                                                           args.warmstart_from) if args.warmstart_from else None,
                                                       prior='Eight uniform prior training maps; eight disjoint prior validation maps select baseline without evaluation county p',
                                                       prior_scheme=args.prior_scheme, frozen_prior_from=str(
                                                           args.prior_from) if args.prior_from else None,
                                                       validation_selection=name, validation_scores=scores, worlds=args.worlds, seed_start=args.seed_start,
                                                       numerical='Network-aware constrained expected-loss optimization; matched budget; finite iterations, not proven global optimum',
                                                       interpretation='Synthetic response mechanism study on U.S. geography, not real intervention effectiveness'), indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

"""Exact-grid static allocation benchmark for epidemic threshold effects.

Fully synthetic, equally sized/risky counties. Budget funds a sustained weekly
intensity for 24 weeks. Perfect and prior-only planners optimize the same
finite allocation grid, preventing a weak mean-plug-in baseline comparison.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t

DAYS = 168


def outcomes(p, dose, days=None, beta=.3):
    days = DAYS if days is None else days
    p = np.asarray(p)
    dose = np.asarray(dose)
    s = np.full(np.broadcast_shapes(p.shape, dose.shape), 99990.)
    i = np.full(s.shape, 10.)
    start = s.copy()
    factor = np.exp(-p * dose)
    recovery = -np.expm1(-1 / 7)
    for _ in range(days):
        new = s * (-np.expm1(-beta * factor * i / 100000))
        removed = i * recovery
        s -= new
        i += new - removed
    return start - s


def optimize(cost, budget):
    """Global optimum on finite dose grid, exact integer budget equality."""
    counties, choices = cost.shape
    value = np.full(budget + 1, np.inf)
    value[0] = 0
    decisions = np.full((counties, budget + 1), -1, dtype=np.int16)
    for county in range(counties):
        new = np.full_like(value, np.inf)
        for dose in range(min(choices - 1, budget) + 1):
            candidate = value[:budget + 1 - dose] + cost[county, dose]
            improve = candidate < new[dose:]
            new[dose:][improve] = candidate[improve]
            decisions[county, dose:][improve] = dose
        value = new
    allocation = np.zeros(counties, dtype=int)
    remaining = budget
    for county in reversed(range(counties)):
        allocation[county] = decisions[county, remaining]
        remaining -= allocation[county]
    if remaining != 0 or allocation.sum() != budget:
        raise ValueError('DP budget failure')
    return allocation, float(value[budget])


def network_outcomes(p, dose, mobility):
    n = len(p)
    s = np.full(n, 99990.)
    i = np.full(n, 10.)
    start = s.copy()
    factor = np.exp(-p * dose)
    recovery = -np.expm1(-1 / 7)
    for _ in range(DAYS):
        imported = (np.roll(i, 1) + np.roll(i, -1)) / 2
        h = .3 * ((1 - mobility) * i + mobility * imported) / 100000
        new = s * (-np.expm1(-factor * h))
        removed = i * recovery
        s -= new
        i += new - removed
    return (start - s).sum()


def main():
    global DAYS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, choices=[84, 168], default=168)
    DAYS = parser.parse_args().days
    out = Path('outputs/ideal_information_benchmark' if DAYS ==
               168 else 'outputs/ideal_information_benchmark_84d')
    out.mkdir(parents=True, exist_ok=True)
    # Verify global grid optimizer against brute force, independent tiny problem.
    import itertools
    test = np.array([[9., 5., 4.], [8., 6., 1.], [7., 4., 3.]])
    allocation, value = optimize(test, 3)
    exact = min(sum(test[j, a] for j, a in enumerate(assignment))
                for assignment in itertools.product(range(3), repeat=3) if sum(assignment) == 3)
    assert np.isclose(value, exact)
    count = 32
    step = .1
    doses = np.arange(0, 8 + step / 2, step)
    quadrature = (np.arange(1000) + .5) / 1000
    prior_curve = outcomes(quadrature[:, None], doses[None, :])
    prior_loss = prior_curve.mean(0)
    none = float(outcomes(np.array([.5]), np.array([0.]))[0] * count)
    records = []
    for average in [.25, .5, 1., 1.5, 2.]:
        units = round(count * average / step)
        blind, _ = optimize(np.tile(prior_loss, (count, 1)), units)
        for draw in range(24):
            rng = np.random.default_rng(993000 + draw)
            p = rng.uniform(0, 1, count)
            cost = outcomes(p[:, None], doses[None, :])
            informed, _ = optimize(cost, units)
            choices = {'prior_optimal': blind, 'perfect_information': informed}
            # Synthetic information quality, not observed survey/Twitter data.
            for noise in [.05, .15, .30]:
                measured = p + rng.normal(0, noise, count)
                weights = np.exp(-.5 * ((measured[:, None] - quadrature[None, :]) / noise)**2)
                weights /= weights.sum(axis=1)[:, None]
                posterior_cost = weights @ prior_curve
                estimated, _ = optimize(posterior_cost, units)
                choices[f'noisy_measurement_sd{noise}'] = estimated
            for policy, allocation in choices.items():
                for mobility in [0., .02, .10]:
                    total = network_outcomes(p, doses[allocation], mobility)
                    records.append(
                        dict(
                            average_dose=average,
                            draw=draw,
                            policy=policy,
                            mobility=mobility,
                            infections=total))
            if average == 1 and draw == 0:
                pd.DataFrame(
                    dict(
                        county=np.arange(count),
                        p=p,
                        prior_dose=doses[blind],
                        oracle_dose=doses[informed])).to_csv(
                    out /
                    'example_allocation.csv',
                    index=False)
            pd.DataFrame(records).to_csv(out / 'trials.csv', index=False)
        print('Idealized benchmark budget complete', average, flush=True)
    summary = []
    for (average, mobility), group in pd.DataFrame(records).groupby(['average_dose', 'mobility']):
        wide = group.pivot(index='draw', columns='policy', values='infections')
        for policy in wide:
            delta = wide['prior_optimal'] - wide[policy]
            half = t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
            summary.append(dict(average_dose=average, mobility=mobility, policy=policy, infections=wide[policy].mean(),
                                prevented_vs_none=none - wide[policy].mean(), prevention_percent=100 * (none - wide[policy].mean()) / none,
                                extra_prevented=delta.mean(), improvement_percent=100 * delta.mean() / wide['prior_optimal'].mean(), ci_low=delta.mean() - half, ci_high=delta.mean() + half))
    result = pd.DataFrame(summary)
    result.to_csv(out / 'summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(kind='FULLY SYNTHETIC theoretical benchmark, not empirical evidence',
                                                       counties=32, population_per_county=100000, initial_infected_per_county=10, initial_removed=0, days=DAYS, beta=.3, gamma='1/7 daily hazard',
                                                       responsiveness='Independent Uniform(0,1), fixed per county; 24 fresh random worlds 993000–993023',
                                                       intervention='Transmission multiplier exp(-p*dose); dose=kappa*a/N; sustained constant weekly allocation over 24 weeks',
                                                       outbreak_threshold='Initial effective reproduction approximately .3*exp(-p*dose)/(1-exp(-1/7)); intervention can push it below 1',
                                                       budget='Same total sustained resource intensity under every policy; average doses [.25,.5,1,1.5,2]',
                                                       allocation='Dynamic programming gives globally optimal static plan on dose grid 0:.1:8 for independent counties',
                                                       strong_baseline='Knows complete Uniform prior, optimizes expected epidemic loss using 1000-point quadrature; does not simply plug in mean p',
                                                       information='Perfect county p, or noisy p measurements with correctly specified Bayesian posterior under Uniform prior',
                                                       mobility='Ring mixing 0%,2%,10%; allocations optimized for independent counties then evaluated with mixing, so positive-mobility plans are not claimed optimal',
                                                       limitation='Fixed sustained schedules, not adaptive weekly control; synthetic efficacy units; not a fitted real intervention or evidence validating survey proxies',
                                                       no_intervention_infections=none), indent=2))
    print(result[(result.mobility == 0) & result.policy.isin(
        ['prior_optimal', 'perfect_information'])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

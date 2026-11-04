"""Population-weighted horizon allocation and weekly metapopulation SIR."""
import numpy as np


def local_response_curves(n, s, i, rates, zgrid, days):
    """Independent-county horizon incidence for effective doses z=p*d."""
    ss = np.broadcast_to(s[:, None], (len(n), len(zgrid))).copy()
    ii = np.broadcast_to(i[:, None], ss.shape).copy()
    start = ss.copy()
    multiplier = np.exp(-zgrid)[None, :]
    recovery = -np.expm1(-1 / 7)
    for day in range(days):
        beta = rates[min(day // 7, len(rates) - 1)]
        new = ss * (-np.expm1(-beta * multiplier * ii / n[:, None]))
        removed = ii * recovery
        ss -= new
        ii += new - removed
    return start - ss


def interpolate_curves(curves, zgrid, z):
    z = np.broadcast_to(z, (len(curves), np.asarray(z).shape[-1]))
    position = np.clip(z / (zgrid[1] - zgrid[0]), 0, len(zgrid) - 1)
    left = np.minimum(position.astype(int), len(zgrid) - 2)
    fraction = position - left
    row = np.arange(len(curves))[:, None]
    return curves[row, left] * (1 - fraction) + curves[row, left + 1] * fraction


def cost_tables(curves, zgrid, doses, p, weights, quantiles=64):
    observed = interpolate_curves(curves, zgrid, p[:, None] * doses[None, :])
    order = np.argsort(p)
    cdf = np.cumsum(weights[order]) / weights.sum()
    support = np.interp((np.arange(quantiles) + .5) / quantiles, cdf, p[order])
    prior = np.zeros_like(observed)
    for value in support:
        prior += interpolate_curves(curves, zgrid, value * doses[None, :]) / quantiles
    return observed, prior


def horizon_allocation(cost, n, doses, average, seed=42001):
    """Lower-convex-envelope resource allocation, with one fractional segment.

    This solves the population-weighted piecewise-linear relaxation. Actual
    SIR outcomes, not interpolated costs, determine reported policy results.
    Coupled mobility effects are evaluated after allocating, not optimized.
    """
    if average == 0:
        return np.zeros(len(n))
    if average > doses[-1]:
        raise ValueError('Budget exceeds dose range')
    priority = np.random.default_rng(seed).random(len(n))
    segments = []
    for county, values in enumerate(cost):
        hull = []
        for point in range(len(doses)):
            while len(hull) >= 2:
                a, b = hull[-2:]
                cross = (doses[b] - doses[a]) * (values[point] - values[b]) - \
                    (values[b] - values[a]) * (doses[point] - doses[b])
                if cross > 1e-12:
                    break
                hull.pop()
            hull.append(point)
        for a, b in zip(hull[:-1], hull[1:]):
            resource = n[county] * (doses[b] - doses[a])
            benefit = (values[a] - values[b]) / resource
            segments.append((-round(float(benefit), 12), priority[county], a, county, resource))
    segments.sort()
    budget = average * n.sum()
    remaining = budget
    allocation = np.zeros(len(n))
    for _, _, _, county, resource in segments:
        use = min(resource, remaining)
        allocation[county] += use / n[county]
        remaining -= use
        if remaining <= budget * 1e-12:
            break
    if not np.isclose(allocation @ n, budget, rtol=1e-9) or (allocation <
                                                             0).any() or (allocation > doses[-1] + 1e-9).any():
        raise ValueError('Horizon allocation budget/bounds failure')
    return allocation


def simulate(n, s, i, r, p, dose, matrices, rates, mobility=.15, days=84, stop_day=None):
    s = s.copy()
    i = i.copy()
    r = r.copy()
    total = np.zeros(len(n))
    weeks = []
    daily = []
    recovery = -np.expm1(-1 / 7)
    effect = np.exp(-p * dose)
    for day in range(days):
        week = min(day // 7, len(matrices) - 1)
        beta = rates[min(day // 7, len(rates) - 1)]
        factor = effect if stop_day is None or day < stop_day else 1.
        h = beta * ((1 - mobility) * i / n + mobility * (matrices[week] @ (i / n))) * factor
        new = s * (-np.expm1(-h))
        removed = i * recovery
        s -= new
        i += new - removed
        r += removed
        total += new
        daily.append(new.sum())
        if day % 7 == 6:
            weeks.append(dict(cumulative=total.copy(), active=i.copy(), daily_incidence=new.copy()))
    if not np.allclose(s + i + r, n, rtol=1e-9) or min(s.min(), i.min(), r.min()) < -1e-8:
        raise ValueError('SIR conservation/nonnegativity failure')
    return dict(total=float(total.sum()), county=total, daily=np.asarray(daily), weeks=weeks)


def checks():
    # A convex separable toy problem has known balanced resource optimum.
    doses = np.array([0., 1., 2.])
    cost = np.array([[10., 4., 2.], [10., 4., 2.]])
    a = horizon_allocation(cost, np.ones(2), doses, 1.)
    np.testing.assert_allclose(a, [1., 1.])
    # Threshold returns should fund a complete project, not split below threshold.
    cost = np.array([[10., 9., 0.], [10., 9., 0.]])
    a = horizon_allocation(cost, np.ones(2), doses, 1.)
    np.testing.assert_allclose(np.sort(a), [0., 2.])
    # Unequal populations still respect a fixed per-capita resource budget.
    a = horizon_allocation(cost, np.array([1., 3.]), doses, .7)
    np.testing.assert_allclose(a @ np.array([1., 3.]), 2.8)

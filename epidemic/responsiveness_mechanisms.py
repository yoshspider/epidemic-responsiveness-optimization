"""Matched-strength one-sided and two-sided behavioral contact mechanisms.

No survey is used here. Response inputs are synthetic and supplied explicitly.
The two-sided mechanism matches the original when effective county doses are
homogeneous: local attenuation is exp(-p*d) in both formulations.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

RECOVERY_PROBABILITY = -np.expm1(-1 / 7)


def force(i, n, p, dose, matrix, beta, mobility, mechanism):
    if mechanism == 'recipient':
        g = np.exp(-p * dose)
        return beta * g * ((1 - mobility) * i / n + mobility * (matrix @ (i / n)))
    if mechanism != 'both_ends':
        raise ValueError('Unknown intervention mechanism')
    g = np.exp(-p * dose / 2)
    return beta * g * ((1 - mobility) * g * i / n + mobility * (matrix @ (g * i / n)))


def objective(dose, n, s0, i0, responses, matrices, beta, mobility, mechanism, gradient=True):
    if mechanism not in ['recipient', 'both_ends']:
        raise ValueError('Unknown intervention mechanism')
    dose = np.asarray(dose)
    is_weekly = dose.ndim == 2
    weeks = len(matrices)
    weekly_beta = np.broadcast_to(np.asarray(beta, dtype=float), (weeks,))
    weekly_dose = dose if is_weekly else np.tile(dose, (weeks, 1))
    p = np.asarray(responses, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    ncol = n[:, None]
    worlds = p.shape[1]
    s = np.repeat(s0[:, None], worlds, axis=1)
    i = np.repeat(i0[:, None], worlds, axis=1)
    rho = RECOVERY_PROBABILITY
    protection = np.exp(-p[None, :, :] * weekly_dose[:, :, None] /
                        (2 if mechanism == 'both_ends' else 1))
    initial_s = s.copy()
    history = []
    for day in range(7 * weeks):
        beta = weekly_beta[day // 7]
        matrix = matrices[day // 7]
        g = protection[day // 7]
        if mechanism == 'recipient':
            h = beta * g * ((1 - mobility) * i / ncol + mobility * (matrix @ (i / ncol)))
        else:
            h = beta * g * ((1 - mobility) * g * i / ncol + mobility * (matrix @ (g * i / ncol)))
        if gradient:
            history.append((s.copy(), i.copy()))
        new = s * (-np.expm1(-h))
        s -= new
        i += new - rho * i
    loss = float((initial_s - s).sum() / worlds)
    if not gradient:
        return loss
    adj_s = np.zeros_like(s)
    adj_i = np.zeros_like(i)
    grad = np.zeros((weeks, *s.shape))
    for day in reversed(range(7 * weeks)):
        beta = weekly_beta[day // 7]
        s, i = history[day]
        matrix = matrices[day // 7]
        g = protection[day // 7]
        v = i / ncol
        if mechanism == 'recipient':
            base = beta * ((1 - mobility) * v + mobility * (matrix @ v))
            h = g * base
        else:
            base = beta * ((1 - mobility) * g * v + mobility * (matrix @ (g * v)))
            h = g * base
        survival = np.exp(-h)
        q = 1 - survival
        multiplier = 1 - adj_s + adj_i
        weight = multiplier * s * survival
        if mechanism == 'recipient':
            grad[day // 7] -= weight * base * p * g
            d = weight * g
            derivative_i = beta * ((1 - mobility) * d + mobility * (matrix.T @ d)) / ncol
        else:
            remote = matrix.T @ (weight * g)
            derivative_g = weight * base + beta * \
                ((1 - mobility) * weight * g * v + mobility * v * remote)
            grad[day // 7] -= .5 * p * g * derivative_g
            derivative_i = beta * ((1 - mobility) * weight * g * g + mobility * g * remote) / ncol
        adj_s = adj_s + multiplier * q
        adj_i = (1 - rho) * adj_i + derivative_i
    grad = grad.mean(axis=2)
    return loss, grad if is_weekly else grad.sum(axis=0)


def allocate(z, n, average, cap=8.):
    z = np.asarray(z)
    is_weekly = z.ndim == 2
    rows = z if is_weekly else z[None, :]
    low, high = np.full(len(rows), -60.), np.full(len(rows), 60.)
    for _ in range(60):
        mid = (low + high) / 2
        below = (cap * expit(rows + mid[:, None])) @ n < average * n.sum()
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    dose = cap * expit(rows + ((low + high) / 2)[:, None])
    return dose if is_weekly else dose[0]


def optimize(n, s, i, responses, matrices, beta, mobility, mechanism,
             average, starts=(), iterations=50, schedule='static'):
    weeks = len(matrices)
    shape = (weeks, len(n)) if schedule == 'weekly' else (len(n),)
    candidates = [np.full(shape, average)] + [np.tile(candidate, (weeks, 1)) if schedule ==
                                              'weekly' and np.asarray(candidate).ndim == 1 else candidate for candidate in starts]
    best_start = min(
        candidates,
        key=lambda d: objective(
            d,
            n,
            s,
            i,
            responses,
            matrices,
            beta,
            mobility,
            mechanism,
            False))
    best = [
        objective(
            best_start,
            n,
            s,
            i,
            responses,
            matrices,
            beta,
            mobility,
            mechanism,
            False),
        best_start.copy()]
    history = []

    def value_gradient(z):
        dose = allocate(z.reshape(shape), n, average)
        value, grad = objective(dose, n, s, i, responses, matrices, beta, mobility, mechanism)
        if value < best[0]:
            best[:] = [value, dose.copy()]
        tangent = dose * (1 - dose / 8)
        if schedule == 'weekly':
            derivative = tangent * (grad - n[None, :] * (grad * tangent).sum(axis=1)[:, None] /
                                    (n[None, :] * tangent).sum(axis=1)[:, None])
        else:
            derivative = tangent * (grad - n * (grad * tangent).sum() / (n * tangent).sum())
        history.append(value)
        return value / 1e6, derivative.ravel() / 1e6

    result = minimize(value_gradient, logit(np.clip(best_start / 8, 1e-7, 1 - 1e-7)).ravel(), jac=True,
                      method='L-BFGS-B', bounds=[(-18, 18)] * int(np.prod(shape)),
                      options={'maxiter': iterations, 'maxls': 20, 'maxcor': 8, 'ftol': 1e-10, 'gtol': 1e-7})
    if not np.all(np.isclose(best[1] @ n, average * n.sum(), rtol=1e-8)):
        raise ValueError('Budget mismatch')
    return best[1], dict(loss=best[0], converged=bool(result.success), iterations=int(result.nit),
                         evaluations=int(result.nfev), history=history)


def checks():
    from scipy import sparse
    n = np.array([1000., 1500., 2000.])
    matrix = sparse.csr_matrix([[0., .6, .4], [.4, 0., .6], [.7, .3, 0.]])
    matrices = [matrix] * 12
    p = np.array([[.2, .4], [.7, .5], [.9, .8]])
    dose = np.array([.3, .8, .6])
    for mechanism in ['recipient', 'both_ends']:
        value, grad = objective(dose, n, n * .99, n * .01, p, matrices, .3, .15, mechanism)
        numerical = []
        for coordinate in range(3):
            plus = dose.copy()
            minus = dose.copy()
            plus[coordinate] += 1e-5
            minus[coordinate] -= 1e-5
            a = objective(plus, n, n * .99, n * .01, p, matrices, .3, .15, mechanism, False)
            b = objective(minus, n, n * .99, n * .01, p, matrices, .3, .15, mechanism, False)
            numerical.append((a - b) / 2e-5)
        np.testing.assert_allclose(grad, numerical, rtol=2e-5, atol=1e-5)
    # Normalization isolates network placement instead of doubling efficacy.
    shared = np.full(3, .5)
    constant = np.full(3, .6)
    infected = n * np.array([.01, .02, .03])
    left = force(infected, n, shared, constant, matrix, .3, .25, 'recipient')
    right = force(infected, n, shared, constant, matrix, .3, .25, 'both_ends')
    np.testing.assert_allclose(left, right)
    np.testing.assert_allclose(force(infected, n, shared, np.zeros(3), matrix, .3, .25, 'both_ends'),
                               force(infected, n, shared, np.zeros(3), matrix, .3, .25, 'recipient'))

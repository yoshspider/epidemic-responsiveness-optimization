"""Network-aware sustained-dose optimization with checked exact SIR adjoint."""
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize


def budget_dose(z, n, average, cap=8.):
    low, high = -60., 60.
    budget = average * n.sum()
    for _ in range(65):
        shift = (low + high) / 2
        dose = cap * expit(z + shift)
        if dose @ n < budget:
            low = shift
        else:
            high = shift
    return cap * expit(z + (low + high) / 2)


def loss_gradient(dose, n, s0, i0, p, matrices, rates, mobility, gradient=True):
    s = s0.copy()
    i = i0.copy()
    g = np.exp(-p * dose)
    rho = -np.expm1(-1 / 7)
    history = []
    total = 0.
    for day in range(84):
        week = day // 7
        beta = rates[week]
        matrix = matrices[week]
        h = beta * ((1 - mobility) * i / n + mobility * (matrix @ (i / n)))
        survival = np.exp(-g * h)
        q = 1 - survival
        if gradient:
            history.append((week, s.copy(), h, survival, q))
        new = s * q
        total += new.sum()
        s -= new
        i += new - rho * i
    if not gradient:
        return total
    adj_s = np.zeros(len(n))
    adj_i = np.zeros(len(n))
    grad = np.zeros(len(n))
    for week, s, h, survival, q in reversed(history):
        multiplier = 1 - adj_s + adj_i
        d = multiplier * s * survival * g
        grad -= multiplier * s * survival * h * p * g
        beta = rates[week]
        new_adj_i = (1 - rho) * adj_i + beta * ((1 - mobility)
                                                * d + mobility * (matrices[week].T @ d)) / n
        adj_s = adj_s + multiplier * q
        adj_i = new_adj_i
    return total, grad


def optimize(n, s, i, responses, matrices, rates, mobility, average, candidates, maxiter=60):
    """Optimize expected loss over supplied response worlds, same fixed budget.

    Unknown-response planner receives synthetic prior draws, not the county
    survey labels. Proxy/predictor planners receive their own response inputs.
    """
    def evaluate(dose): return np.mean(
        [loss_gradient(dose, n, s, i, p, matrices, rates, mobility, False) for p in responses])
    candidates = [np.full(len(n), average)] + list(candidates)
    start = min(candidates, key=evaluate)
    start_loss = float(evaluate(start))
    z = logit(np.clip(start / 8, 1e-7, 1 - 1e-7))
    best = [start_loss, start.copy()]
    history = []

    def objective(z):
        dose = budget_dose(z, n, average)
        values = []
        gradients = []
        for p in responses:
            value, grad = loss_gradient(dose, n, s, i, p, matrices, rates, mobility)
            values.append(value)
            gradients.append(grad)
        value = float(np.mean(values))
        grad = np.mean(gradients, axis=0)
        if value < best[0]:
            best[:] = [value, dose.copy()]
        tangent = dose * (1 - dose / 8)
        grad_z = tangent * (grad - n * np.sum(grad * tangent) / np.sum(n * tangent))
        history.append(value)
        return value / 1e6, grad_z / 1e6
    fit = minimize(objective, z, jac=True, method='L-BFGS-B', bounds=[(-18, 18)] * len(n),
                   options={'maxiter': maxiter, 'maxls': 20, 'maxcor': 8, 'ftol': 1e-10, 'gtol': 1e-7})
    dose = best[1]
    if not np.isclose(n @ dose, average * n.sum(), rtol=1e-8):
        raise ValueError('Network planner violated budget')
    return dose, dict(start_loss=start_loss, selected_loss=best[0], iterations=int(fit.nit), evaluations=int(fit.nfev),
                      converged=bool(fit.success), message=str(fit.message), history=history)


def checks():
    from scipy import sparse
    n = np.array([1000., 1500., 2000.])
    s = n * .99
    i = n * .01
    matrix = sparse.csr_matrix([[0, .6, .4], [.4, 0, .6], [.7, .3, 0]])
    matrices = [matrix] * 12
    rates = np.full(12, .3)
    p = np.array([.3, .6, .9])
    z = np.array([-.4, .2, .5])
    dose = budget_dose(z, n, 1.)
    value, grad = loss_gradient(dose, n, s, i, p, matrices, rates, .15)
    tangent = dose * (1 - dose / 8)
    analytic = tangent * (grad - n * np.sum(grad * tangent) / np.sum(n * tangent))
    numerical = []
    for coordinate in range(3):
        plus = z.copy()
        minus = z.copy()
        plus[coordinate] += 1e-5
        minus[coordinate] -= 1e-5
        a = loss_gradient(budget_dose(plus, n, 1.), n, s, i, p, matrices, rates, .15, False)
        b = loss_gradient(budget_dose(minus, n, 1.), n, s, i, p, matrices, rates, .15, False)
        numerical.append((a - b) / 2e-5)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(dose @ n, n.sum())

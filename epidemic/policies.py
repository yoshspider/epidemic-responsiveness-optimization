import numpy as np
from .model import State


def allocate(model, state, budget, policy, predicted=None, steps_remaining=None):
    """Deterministic marginal benefit/cost planning; actual execution always uses true p."""
    n = len(model.n)
    allocation = np.zeros(n)
    decisions = []
    if policy == "none":
        return allocation, decisions
    if policy in ("epidemic", "lookahead_epidemic", "mpc_epidemic"):
        p = np.full(n, model.cfg.prior)  # Explicit prior; never estimate from model.p.
    elif policy == "epidemic_mean":
        p = np.full(n, model.cfg.mean_prior)  # Declared aggregate-information comparison.
    elif policy in ("oracle", "lookahead", "mpc"):
        p = model.p
    elif policy == "predicted":
        p = np.asarray(predicted, dtype=float)
        if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Predictions must be finite county probabilities")
    else:
        raise ValueError("Unknown policy")
    if policy in ("mpc", "mpc_epidemic"):
        from .planning import plan
        return plan(model,state,budget,p,steps_remaining)
    work = state.copy()
    horizon = max(1, round(model.cfg.horizon_days / model.cfg.dt)) if policy in ("lookahead", "lookahead_epidemic") else 1
    if steps_remaining is not None:
        horizon = min(horizon, steps_remaining)
    spent = 0.0
    # Infection pressure and compartments do not change within one allocation decision.
    cached_hazard = model.hazard(work.i, work.day) if horizon==1 else None
    # At most one final partial chunk per decision. Full chunks retain identical action units.
    while budget - spent > 1e-9:
        remaining = budget - spent
        chunks = np.minimum(model.cfg.chunk, remaining / model.cost)
        prices = chunks * model.cost
        if horizon == 1:
            before_adopt = model.adoption(work.effort,p)
            before = work.s*(-np.expm1(-cached_hazard*(1-model.cfg.efficacy*before_adopt[:,None])*model.cfg.dt))
            after_effort = work.effort + chunks
            after_adopt = model.adoption(after_effort, p)
            after = work.s * (-np.expm1(-cached_hazard *
                             (1 - model.cfg.efficacy * after_adopt[:, None]) * model.cfg.dt))
            gain = ((before - after) * model.weights).sum(1)
        else:
            base = model.rollout(work.copy(), p, horizon)
            gain = np.empty(n)
            for start in range(0, n, model.cfg.candidate_batch):
                ids = np.arange(start, min(n, start + model.cfg.candidate_batch))
                b = len(ids)
                candidates = State(*(np.broadcast_to(x, (b,) + x.shape).copy()
                                     for x in (work.s, work.i, work.r, work.effort)), day=work.day)
                candidates.effort[np.arange(b), ids] += chunks[ids]
                gain[ids] = base - model.rollout(candidates, p, horizon)
        score = gain / prices
        score[~model.eligible]=-np.inf
        chosen = int(np.argmax(score))
        if score[chosen] <= 1e-12:
            break
        work.effort[chosen] += chunks[chosen]
        allocation[chosen] += chunks[chosen]
        spent += prices[chosen]
        decisions.append({"county_index": chosen, "people_equivalent": float(chunks[chosen]),
                          "cost": float(prices[chosen]), "estimated_gain": float(gain[chosen]),
                          "estimated_gain_per_cost": float(score[chosen])})
    return allocation, decisions

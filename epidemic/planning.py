"""Budget-aware receding-horizon planning with an analytic network adjoint.

The finite-horizon problem is nonconvex. Frank-Wolfe is a heuristic, not a
globally optimal value policy. Both information levels use identical machinery.
"""
import numpy as np
from .model import State


def objective_gradient(model, initial, p, controls, steps):
    """controls is daily currency spending [steps,counties]. Exact deterministic gradient."""
    cfg=model.cfg
    if controls.shape!=(steps,len(model.n)):
        raise ValueError("Controls must be daily county spending")
    s,i,effort=initial.s.copy(),initial.i.copy(),initial.effort.copy()
    recovery=-np.expm1(-cfg.recovery_rate*cfg.dt)
    retention=cfg.retention_daily**cfg.dt
    total=0.
    tape=[]
    for step in range(steps):
        effort=effort+controls[step]/model.cost
        h=model.hazard(i,initial.day)
        adoption=model.adoption(effort,p)
        z=1-cfg.efficacy*adoption[:,None]
        survival=np.exp(-h*z*cfg.dt)
        q=1-survival
        new=s*q
        tape.append((s,h,z,survival,q,effort))
        total+=float((new*model.weights).sum())
        s,i,effort=s-new,i*(1-recovery)+new,effort*retention
    adj_s=np.zeros_like(initial.s)
    adj_i=np.zeros_like(initial.i)
    adj_effort=np.zeros(len(model.n))
    gradient=np.zeros_like(controls)
    mixing=np.asarray(cfg.mixing)
    population=model.n.sum(1)
    transpose=model.mobility_at(initial.day).T.tocsr()
    beta_local,beta_mobility=model.transmission_at(initial.day)
    for step in range(steps-1,-1,-1):
        s,h,z,survival,q,effort=tape[step]
        coefficient=model.weights+adj_i-adj_s
        grad_h=coefficient*s*survival*cfg.dt*z
        grad_i=(beta_local*(grad_h@mixing)+beta_mobility*(transpose@(grad_h@mixing)))/population[:,None]
        adoption_derivative=p*cfg.saturation/population*np.exp(-cfg.saturation*effort/population)
        current_effort=retention*adj_effort-cfg.efficacy*adoption_derivative*(coefficient*s*survival*cfg.dt*h).sum(1)
        gradient[step]=current_effort/model.cost
        adj_s=adj_s+coefficient*q
        adj_i=adj_i*(1-recovery)+grad_i
        adj_effort=current_effort
    return total,gradient


def objective(model, initial, p, controls, steps):
    state=initial.copy()
    total=0.
    for step in range(steps):
        state.effort+=controls[step]/model.cost
        state,new=model.step(state,p,mobility_day=initial.day)
        total+=float((new*model.weights).sum())
    return total


def plan(model, state, remaining, p, steps_remaining=None):
    cfg=model.cfg
    steps=round(cfg.horizon_days/cfg.dt)
    if steps_remaining is not None:
        steps=min(steps,steps_remaining)
    interval=round(cfg.decision_days/cfg.dt)
    if cfg.budget_mode=="weekly" and not np.isclose(cfg.decision_days,7):
        raise ValueError("MPC weekly planning currently requires seven-day decisions")
    decision_steps=np.arange(0,steps,interval)
    controls=np.zeros((steps,len(model.n)))
    # Equal person-equivalent effort, respecting heterogeneous cost.
    share=model.n.sum(1)*model.cost
    share[~model.eligible]=0
    share=share/share.sum()
    if cfg.budget_mode=="weekly":
        budgets=np.full(len(decision_steps),cfg.budget)
        budgets[0]=remaining
        controls[decision_steps]=budgets[:,None]*share
    else:
        controls[0]=remaining*share
        budgets=None
    # Warm start with the same myopic algorithm under the supplied information.
    from .policies import allocate
    temporary=state.copy()
    greedy=np.zeros_like(controls)
    for step in range(steps):
        if step in decision_steps:
            available=(remaining if step==0 else cfg.budget) if cfg.budget_mode=="weekly" else (remaining if step==0 else 0)
            action,_=allocate(model,temporary,available,"predicted",p,steps-step)
            greedy[step]=action*model.cost
            temporary.effort+=action
        temporary,_=model.step(temporary,p,mobility_day=state.day)
    value=objective(model,state,p,controls,steps)
    greedy_value=objective(model,state,p,greedy,steps)
    if greedy_value<value:
        controls,value=greedy,greedy_value
    initial_value=value
    history=[]
    for iteration in range(cfg.planning_iterations):
        value,gradient=objective_gradient(model,state,p,controls,steps)
        gradient[:,~model.eligible]=np.inf
        corner=np.zeros_like(controls)
        if cfg.budget_mode=="weekly":
            for row,budget in zip(decision_steps,budgets):
                county=int(np.argmin(gradient[row]))
                if gradient[row,county]<0:
                    corner[row,county]=budget
        else:
            sub=gradient[decision_steps]
            row,county=np.unravel_index(np.argmin(sub),sub.shape)
            if sub[row,county]<0:
                corner[decision_steps[row],county]=remaining
        direction=corner-controls
        gap=-float((direction[:,model.eligible]*gradient[:,model.eligible]).sum())
        if gap<=1e-8:
            break
        accepted=False
        # Backtracking descent, same acceptance criterion for both information levels.
        for power in range(16):
            alpha=.5**power
            candidate=controls+alpha*direction
            candidate_value=objective(model,state,p,candidate,steps)
            if candidate_value<value-1e-4*alpha*gap:
                controls,value=candidate,candidate_value
                accepted=True
                history.append({"iteration":iteration,"alpha":alpha,"forecast_objective":value,"fw_gap":gap})
                break
        if not accepted:
            break
    current=controls[0]/model.cost
    # The lump-sum optimizer may reserve funds for later decisions.
    decisions=[{"county_index":int(j),"people_equivalent":float(current[j]),
                "cost":float(controls[0,j]),"estimated_gain":None,"estimated_gain_per_cost":None,
                "planner":"receding-horizon Frank-Wolfe","forecast_before":float(initial_value),
                "forecast_after":float(value),"planning_iterations":len(history)}
               for j in np.flatnonzero(current>0)]
    return current,decisions

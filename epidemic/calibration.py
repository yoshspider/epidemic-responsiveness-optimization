"""Fit transmission rates with explicit observation assumptions and holdout scoring."""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from .model import Model


def calibrate(model, initial, reports, train_steps, reporting_fraction, reporting_delay_steps=0):
    """reports is time x county incident reports, NaN for excluded revisions/gaps.

    Reporting fraction and delay are fixed externally; pre-window delay history is
    unavailable and first delay steps are excluded. Recovery/contact/adoption
    parameters are fixed. Observed epidemic intervention effort must be represented
    in initial state or this no-additional-spending calibration is misspecified.
    """
    y=np.asarray(reports,dtype=float)
    if y.ndim!=2 or y.shape[1]!=len(model.n) or np.any(np.isinf(y)) or np.any(y[np.isfinite(y)]<0):
        raise ValueError("Reports must be nonnegative daily counts or explicit NaN exclusions")
    if not 0<reporting_fraction<=1 or not 1<train_steps<len(y) or not 0<=reporting_delay_steps<train_steps:
        raise ValueError("Invalid training/observation parameters")
    model.validate_state(initial)
    valid=np.isfinite(y[:train_steps])
    valid[:reporting_delay_steps]=False
    if valid.sum()<3:
        raise ValueError("Insufficient training observations")
    def simulate(rates):
        fitted=Model(model.n,model.p,model.cost,model.mobility_schedule if model.dynamic_mobility else model.mobility,
                     replace(model.cfg,beta_local=rates[0],beta_mobility=rates[1],stochastic=False,stochastic_adoption=False))
        state=initial.copy()
        incidence=[]
        for _ in range(len(y)):
            state,new=fitted.step(state)
            incidence.append(new.sum(1)*reporting_fraction)
        prediction=np.asarray(incidence)
        if reporting_delay_steps:
            prediction=np.vstack([np.full((reporting_delay_steps,len(model.n)),np.nan),prediction[:-reporting_delay_steps]])
        return prediction
    def residual(rates):
        predicted=simulate(rates)[:train_steps]
        return (np.sqrt(predicted[valid]+.375)-np.sqrt(y[:train_steps][valid]+.375))
    result=least_squares(residual,[model.cfg.beta_local,model.cfg.beta_mobility],bounds=(0,2))
    prediction=simulate(result.x)
    test=np.isfinite(y[train_steps:])&np.isfinite(prediction[train_steps:])
    rmse=np.sqrt(np.mean((prediction[train_steps:][test]-y[train_steps:][test])**2)) if test.any() else np.nan
    singular=np.linalg.svd(result.jac,compute_uv=False)
    condition=singular[0]/singular[-1] if singular[-1]>0 else np.inf
    return {"beta_local":float(result.x[0]),"beta_mobility":float(result.x[1]),
            "success":bool(result.success),"message":result.message,"heldout_rmse":float(rmse),
            "jacobian_condition":float(condition),"reporting_fraction":reporting_fraction,
            "reporting_delay_steps":reporting_delay_steps,"train_steps":train_steps},prediction

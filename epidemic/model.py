from dataclasses import dataclass, field
import numpy as np
from scipy import sparse
from scipy.stats import binom


@dataclass
class Config:
    beta_local: float = 0.22
    beta_mobility: float = 0.04
    recovery_rate: float = 1 / 7
    dt: float = 1.0
    efficacy: float = 0.5
    saturation: float = 1.0
    retention_daily: float = 0.95
    decision_days: float = 7
    horizon_days: float = 28
    budget: float = 12000
    budget_mode: str = "weekly"
    chunk: float = 25000
    prior: float = 0.776
    mean_prior: float = 0.716
    objective: str = "infections"
    stochastic: bool = False
    stochastic_adoption: bool = False
    candidate_batch: int = 32
    ifr: tuple = (0.001, 0.05)
    mixing: tuple = ((1.0, 1.0), (1.0, 1.0))
    start_date: str = "2020-07-06"
    rollout_backend: str = "numpy"
    planning_iterations: int = 12
    beta_schedule: tuple = ()
    eligible_counties: tuple = ()

    def validate(self):
        vals = [self.beta_local, self.beta_mobility, self.recovery_rate,
                self.dt, self.efficacy, self.saturation, self.retention_daily,
                self.decision_days, self.horizon_days, self.budget, self.chunk, self.prior, self.mean_prior]
        if not np.isfinite(vals).all() or min(vals) < 0 or min(self.dt, self.chunk, self.saturation) <= 0:
            raise ValueError("Invalid configuration values")
        if not 0 <= self.efficacy <= 1 or not 0 <= self.prior <= 1 or not 0 <= self.mean_prior <= 1 or not 0 <= self.retention_daily <= 1:
            raise ValueError("Probabilities must lie in [0,1]")
        if self.budget_mode not in ("weekly", "lump") or self.objective not in ("infections", "vulnerable", "deaths"):
            raise ValueError("Unknown budget mode or objective")
        for days in (7, self.decision_days, self.horizon_days):
            if not np.isclose(days / self.dt, round(days / self.dt)):
                raise ValueError("dt must divide weekly, decision, and horizon durations")
        c = np.asarray(self.mixing)
        if c.shape != (2, 2) or not np.isfinite(c).all() or np.any(c < 0):
            raise ValueError("mixing must be a nonnegative 2x2 matrix")
        if len(self.ifr) != 2 or not np.isfinite(self.ifr).all() or min(self.ifr) < 0 or max(self.ifr) > 1:
            raise ValueError("Invalid IFR")
        if self.candidate_batch < 1:
            raise ValueError("candidate_batch must be positive")
        if self.planning_iterations<0 or int(self.planning_iterations)!=self.planning_iterations:
            raise ValueError("planning_iterations must be a nonnegative integer")
        if self.rollout_backend not in ("numpy","auto","cuda","torch_cpu"):
            raise ValueError("Unknown rollout backend")
        if self.beta_schedule:
            schedule=np.asarray(self.beta_schedule,dtype=float)
            if schedule.ndim!=2 or schedule.shape[1]!=3 or not np.isfinite(schedule).all() or np.any(schedule[:,1:]<0) or schedule[0,0]>0 or np.any(np.diff(schedule[:,0])<=0):
                raise ValueError("beta_schedule must have increasing [day,beta_local,beta_mobility] rows covering day zero")


@dataclass
class State:
    s: np.ndarray
    i: np.ndarray
    r: np.ndarray
    effort: np.ndarray
    day: float = 0.0

    def copy(self):
        return State(*(x.copy() for x in (self.s, self.i, self.r, self.effort)), day=self.day)


class Model:
    def __init__(self, population, responsiveness, cost, mobility, config):
        config.validate()
        self.cfg = config
        self.n = np.asarray(population, dtype=float)
        self.p = np.asarray(responsiveness, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        if self.n.ndim != 2 or self.n.shape[1] != 2 or not np.isfinite(self.n).all() or np.any(self.n < 0) or np.any(self.n.sum(1) <= 0):
            raise ValueError("Population must be finite nonnegative county x [general,vulnerable]")
        size = len(self.n)
        self.eligible=np.ones(size,dtype=bool)
        if config.eligible_counties:
            ids=np.asarray(config.eligible_counties,dtype=int)
            if np.any(ids<0) or np.any(ids>=size) or len(np.unique(ids))!=len(ids):
                raise ValueError("Invalid eligible county indices")
            self.eligible[:]=False
            self.eligible[ids]=True
        if self.p.shape != (size,) or self.cost.shape != (size,) or not np.isfinite(self.p).all() or not np.isfinite(self.cost).all() or np.any((self.p < 0) | (self.p > 1)) or np.any(self.cost <= 0):
            raise ValueError("Invalid responsiveness or per-person costs")
        # Rows are destinations, columns are sources. Zero diagonal; each nonempty row sums to 1.
        schedule = mobility if isinstance(mobility, dict) else {0.: mobility}
        self.mobility_schedule = {}
        for day, matrix in schedule.items():
            m = sparse.csr_matrix(matrix, dtype=float)
            if m.shape != (size, size) or not np.isfinite(m.data).all() or np.any(m.data < 0):
                raise ValueError("Invalid mobility")
            m.setdiag(0)
            m.eliminate_zeros()
            sums = np.asarray(m.sum(1)).ravel()
            self.mobility_schedule[float(day)] = sparse.diags(np.divide(1, sums, out=np.zeros_like(sums), where=sums > 0)) @ m
        self.mobility_days = sorted(self.mobility_schedule)
        if not self.mobility_days or self.mobility_days[0] > 0 or not np.isfinite(self.mobility_days).all():
            raise ValueError("Mobility schedule must cover initialization")
        self.dynamic_mobility = isinstance(mobility, dict)
        self.mobility = self.mobility_at(0)
        self.weights = np.array([1, 1] if config.objective == "infections" else
                                [0, 1] if config.objective == "vulnerable" else config.ifr)

    def validate_state(self, state):
        for x in (state.s, state.i, state.r):
            if x.shape != self.n.shape or not np.isfinite(x).all() or np.any(x < 0):
                raise ValueError("Invalid compartment")
        if not np.allclose(state.s + state.i + state.r, self.n, atol=1e-7, rtol=1e-12):
            raise ValueError("Compartments must conserve population")
        if state.effort.shape != (len(self.n),) or not np.isfinite(state.effort).all() or np.any(state.effort < 0):
            raise ValueError("Invalid intervention effort")
        if self.cfg.stochastic and any(not np.allclose(x, np.round(x)) for x in (state.s, state.i, state.r)):
            raise ValueError("Stochastic compartments must be integer counts")

    def adoption(self, effort, p):
        return p * (-np.expm1(-self.cfg.saturation * effort / self.n.sum(1)))

    def mobility_at(self, day):
        available = [d for d in self.mobility_days if d <= day]
        if not available or (self.dynamic_mobility and day-available[-1] >= 7):
            raise ValueError(f"Missing weekly mobility coverage for day {day}")
        return self.mobility_schedule[available[-1]]

    def hazard(self, infected, day=0):
        # Group prevalence weighted by source-group population share.
        pressure = infected / self.n.sum(1)[:, None]
        local = pressure @ np.asarray(self.cfg.mixing).T
        if infected.ndim == 2:
            remote = (self.mobility_at(day) @ pressure) @ np.asarray(self.cfg.mixing).T
        else:
            b, n, g = pressure.shape
            flat = pressure.transpose(1, 0, 2).reshape(n, b * g)
            remote = (self.mobility_at(day) @ flat).reshape(n, b, g).transpose(1, 0, 2) @ np.asarray(self.cfg.mixing).T
        local_rate,mobility_rate=self.transmission_at(day)
        return local_rate * local + mobility_rate * remote

    def transmission_at(self,day):
        if not self.cfg.beta_schedule:
            return self.cfg.beta_local,self.cfg.beta_mobility
        schedule=np.asarray(self.cfg.beta_schedule,dtype=float)
        at=np.searchsorted(schedule[:,0],day,side='right')-1
        if at<0:
            raise ValueError("Missing transmission schedule at requested day")
        return tuple(schedule[at,1:])

    def incidence(self, state, p):
        adoption = self.adoption(state.effort, p)
        hazard = self.hazard(state.i, state.day) * (1 - self.cfg.efficacy * adoption[..., None])
        return state.s * (-np.expm1(-hazard * self.cfg.dt))

    def step(self, state, p=None, uniforms=None, mobility_day=None):
        p = self.p if p is None else p
        adoption = self.adoption(state.effort, p)
        if uniforms is not None and self.cfg.stochastic_adoption:
            counts = np.rint(self.n.sum(1)).astype(np.int64)
            adoption = binom.ppf(uniforms[2], counts, adoption) / counts
        probability = -np.expm1(-self.hazard(state.i, state.day if mobility_day is None else mobility_day) * (1 - self.cfg.efficacy * adoption[..., None]) * self.cfg.dt)
        recovery = -np.expm1(-self.cfg.recovery_rate * self.cfg.dt)
        if uniforms is None:
            new = state.s * probability
            recovered = state.i * recovery
        else:
            new = binom.ppf(uniforms[0], np.rint(state.s).astype(np.int64), probability) if self.cfg.stochastic else state.s * probability
            recovered = binom.ppf(uniforms[1], np.rint(state.i).astype(np.int64), recovery) if self.cfg.stochastic else state.i * recovery
        return State(state.s - new, state.i + new - recovered, state.r + recovered,
                     state.effort * self.cfg.retention_daily ** self.cfg.dt, day=state.day+self.cfg.dt), new

    def rollout(self, state, p, steps):
        if self.cfg.rollout_backend != "numpy":
            try:
                import torch
            except ImportError:
                if self.cfg.rollout_backend != "auto":
                    raise RuntimeError("Requested torch backend requires PyTorch")
            else:
                device = "cuda" if torch.cuda.is_available() and self.cfg.rollout_backend != "torch_cpu" else "cpu"
                if self.cfg.rollout_backend == "cuda" and device != "cuda":
                    raise RuntimeError("CUDA explicitly requested but unavailable")
                if self.cfg.rollout_backend != "auto" or device == "cuda":
                    from .tensor import rollout
                    return rollout(self,state,p,steps,device)
        total = np.zeros(state.s.shape[:-2])
        observed_day = state.day
        for _ in range(steps):
            state, new = self.step(state, p, mobility_day=observed_day)
            total += (new * self.weights).sum(axis=(-2, -1))
        return total

"""Optional double-precision torch candidate rollouts, CUDA or CPU."""
import numpy as np


def rollout(model, state, p, steps, device):
    import torch
    dtype=torch.float64
    def tensor(array):
        return torch.as_tensor(array,dtype=dtype,device=device)
    key=(state.day,device)
    cache=getattr(model,"_tensor_mobility_cache",{})
    if key not in cache:
        matrix=model.mobility_at(state.day).tocoo()
        indices=torch.as_tensor(np.vstack([matrix.row,matrix.col]),dtype=torch.int64,device=device)
        cache[key]=torch.sparse_coo_tensor(indices,tensor(matrix.data),matrix.shape,check_invariants=True).coalesce()
        model._tensor_mobility_cache=cache
    mobility=cache[key]
    s,i,r,effort=[tensor(x) for x in (state.s,state.i,state.r,state.effort)]
    pop=tensor(model.n.sum(1))
    mixing=tensor(model.cfg.mixing).T
    response=tensor(p)
    weights=tensor(model.weights)
    result=torch.zeros(s.shape[:-2],dtype=dtype,device=device)
    beta_local,beta_mobility=model.transmission_at(state.day)
    for _ in range(steps):
        pressure=i/pop[:,None]
        if pressure.ndim==2:
            imported=torch.sparse.mm(mobility,pressure)
        else:
            b,n,g=pressure.shape
            imported=torch.sparse.mm(mobility,pressure.permute(1,0,2).reshape(n,b*g)).reshape(n,b,g).permute(1,0,2)
        hazard=beta_local*(pressure@mixing)+beta_mobility*(imported@mixing)
        adoption=response*(-torch.expm1(-model.cfg.saturation*effort/pop))
        new=s*(-torch.expm1(-hazard*(1-model.cfg.efficacy*adoption[...,None])*model.cfg.dt))
        recovered=i*(-np.expm1(-model.cfg.recovery_rate*model.cfg.dt))
        s,i,r=s-new,i+new-recovered,r+recovered
        effort=effort*model.cfg.retention_daily**model.cfg.dt
        result+=(new*weights).sum(dim=(-2,-1))
    return result.detach().cpu().numpy()

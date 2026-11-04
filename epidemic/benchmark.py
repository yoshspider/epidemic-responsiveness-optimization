"""Measured CPU timings on explicitly synthetic county networks."""
import argparse
import json
import time
from pathlib import Path
from .data import synthetic
from .model import Model, Config
from .policies import allocate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--counties",type=int,default=3126)
    p.add_argument("--horizon",type=int,default=28)
    p.add_argument("--backend",default="numpy",choices=["numpy","auto","cuda","torch_cpu"])
    p.add_argument("--output",type=Path,default=Path("outputs/benchmark.json"))
    a=p.parse_args()
    _,n,responsiveness,cost,mobility,state,_=synthetic(a.counties)
    model=Model(n,responsiveness,cost,mobility,Config(horizon_days=a.horizon,budget=1,rollout_backend=a.backend))
    timings={"data_kind":"synthetic ring network, not empirical mobility","counties":a.counties,"horizon":a.horizon,"backend":a.backend}
    start=time.perf_counter()
    current=state.copy()
    for _ in range(84):
        current,_=model.step(current)
    timings["84_steps_seconds"]=time.perf_counter()-start
    for policy in ("epidemic","oracle","lookahead"):
        start=time.perf_counter()
        allocate(model,state,1,policy)
        timings[policy+"_one_partial_chunk_seconds"]=time.perf_counter()-start
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(timings,indent=2))
    print(json.dumps(timings,indent=2))


if __name__=="__main__":
    main()

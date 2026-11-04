"""Exact finite-plan value-of-information calculation on a small synthetic network.

This is a theoretical diagnostic, not a national policy result or adaptive-MDP
optimum. The same feasible plan set and latent-state prior are used on both sides.
"""
from itertools import product,permutations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from .model import Config,Model,State


def evaluate():
    population=np.column_stack(([100000.,70000.,140000.],[0.,0.,0.]))
    infected=np.column_stack(([1200.,500.,1600.],[0.,0.,0.]))
    initial=State(population-infected,infected,np.zeros_like(population),np.zeros(3))
    mobility=sparse.csr_matrix([[0,.8,.2],[.1,0,.9],[.7,.3,0]])
    plans=list(product(range(4),repeat=3))  # 0 = no spending; 1..3 = selected county.
    rows=[]
    for sd_values in ([.6,.6,.6],[.45,.6,.75],[.2,.6,1.]):
        worlds=list(dict.fromkeys(permutations(sd_values)))
        for weekly_effort in (5000.,25000.,100000.):
            outcomes=np.zeros((len(worlds),len(plans)))
            for world,p in enumerate(worlds):
                model=Model(population,p,np.ones(3),mobility,Config())
                b=len(plans)
                state=State(*(np.broadcast_to(x,(b,)+x.shape).copy() for x in (initial.s,initial.i,initial.r,initial.effort)))
                total=np.zeros(b)
                for day in range(42):
                    if day in (0,7,14):
                        choices=np.asarray([plan[day//7] for plan in plans])
                        active=np.flatnonzero(choices)
                        state.effort[active,choices[active]-1]+=weekly_effort
                    state,new=model.step(state,np.asarray(p))
                    total+=new.sum(axis=(1,2))
                outcomes[world]=total
            blind=int(np.argmin(outcomes.mean(0)))
            informed=np.argmin(outcomes,axis=1)
            blind_value=float(outcomes[:,blind].mean())
            informed_value=float(outcomes[np.arange(len(worlds)),informed].mean())
            rows.append({'p_values':json.dumps(sd_values),'latent_worlds':len(worlds),'plans':len(plans),
                'weekly_effort':weekly_effort,'blind_optimal_expected_infections':blind_value,
                'informed_optimal_expected_infections':informed_value,'exact_value_of_information':blind_value-informed_value,
                'relative_gain_percent':100*(blind_value-informed_value)/blind_value,
                'blind_plan':json.dumps(plans[blind]),'informed_plans':json.dumps([plans[k] for k in informed])})
    return pd.DataFrame(rows)


def main():
    out=Path('outputs/exact_information')
    out.mkdir(parents=True,exist_ok=True)
    table=evaluate()
    table.to_csv(out/'results.csv',index=False)
    (out/'METHOD.txt').write_text('Synthetic 3-county, 42-day epidemic. Exact evaluation of 64 open-loop plans with three weekly spending choices. Equal prior over permutations of county responsiveness. Compare min E[J(plan,p)] with E[min J(plan,p)]. No claim of global optimality for adaptive national policies.')
    print(table[['p_values','weekly_effort','exact_value_of_information','relative_gain_percent']].to_string(index=False))


if __name__=='__main__':
    main()

"""March–April 2021 network and retrospective epidemic calibration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import least_squares
from .prediction import predict


def main():
    out = Path('outputs/vaccine_experiment/real'); out.mkdir(parents=True, exist_ok=True)
    population = pd.read_csv('data/transformed/census_population_2019.csv', dtype={'fips':str}).sort_values('fips').reset_index(drop=True)
    reports = pd.read_csv('data/raw/vaccine_experiment/nyt_2021.csv', dtype={'fips':str}, parse_dates=['date'])
    reports=reports.dropna(subset=['fips']).sort_values(['fips','date'])
    reports['reported_incidence']=reports.groupby('fips').cases.diff()
    reports['date_gap']=reports.groupby('fips').date.diff().dt.days.ne(1)
    counts = reports.pivot(index='date', columns='fips', values='cases').reindex(columns=population.fips)
    infected = np.rint((counts.loc['2021-02-28']-counts.loc['2021-02-21']).to_numpy()/.2)
    removed = np.rint(counts.loc['2021-02-21'].to_numpy()/.2)
    n = population.population.to_numpy(float)
    valid = np.isfinite(infected)&np.isfinite(removed)&(infected>=0)&(removed>=0)&(infected+removed<=n)
    population = population[valid].reset_index(drop=True)
    n, infected, removed = n[valid],infected[valid],removed[valid]
    susceptible = n-infected-removed
    population.to_csv(out/'counties.csv', index=False)
    np.savez_compressed(out/'initial.npz', n=n,s=susceptible,i=infected,r=removed)
    index = dict(zip(population.fips, range(len(n))))
    mobility=[]
    for week,date in enumerate(pd.date_range('2021-03-01', periods=8, freq='7D')):
        path=out/f'mobility_{week}.npz'
        if path.exists():
            matrix=sparse.load_npz(path)
        else:
            raw=pd.read_csv(f'data/raw/midterm_mobility/{date:%Y_%m_%d}.csv',
                            dtype={'geoid_o':str,'geoid_d':str})
            for col in ['geoid_o','geoid_d']:
                raw[col]=raw[col].str.zfill(5)
            raw=raw[raw.geoid_o.isin(index)&raw.geoid_d.isin(index)&(raw.geoid_o!=raw.geoid_d)]
            matrix=sparse.coo_matrix((raw.pop_flows.to_numpy(float),
                (raw.geoid_d.map(index),raw.geoid_o.map(index))),shape=(len(n),len(n))).tocsr()
            row=np.asarray(matrix.sum(1)).ravel()
            matrix=sparse.diags(np.divide(1,row,out=np.zeros_like(row),where=row>0))@matrix
            sparse.save_npz(path,matrix)
        mobility.append(matrix)
        print(f'Actual mobility week {week+1}: {matrix.nnz:,} directed edges', flush=True)
    dates=pd.date_range('2021-03-01',periods=56)
    reports['target']=reports.reported_incidence.where((reports.reported_incidence>=0)&(~reports.date_gap))
    observed=reports.pivot(index='date',columns='fips',values='target').reindex(index=dates,columns=population.fips).to_numpy()
    s,i,r=susceptible.copy(),infected.copy(),removed.copy()
    rates=[]; fitting=[]
    for week,matrix in enumerate(mobility):
        data=observed[7*week:7*week+7]
        keep=np.isfinite(data)
        def simulate(total_rate):
            ss,ii,rr=s.copy(),i.copy(),r.copy(); values=[]
            for day in range(7):
                h=total_rate*(.85*ii/n+.15*(matrix@(ii/n)))
                new=ss*(-np.expm1(-h)); recovered=ii*(-np.expm1(-1/7))
                ss-=new; ii+=new-recovered; rr+=recovered
                values.append(new*.2)
            return np.asarray(values),ss,ii,rr
        def residual(rate):
            estimate,*_=simulate(rate[0])
            return np.sqrt(estimate[keep]+.375)-np.sqrt(data[keep]+.375)
        fit=least_squares(residual,[.20],bounds=(0,1.5),max_nfev=60)
        estimate,s,i,r=simulate(fit.x[0])
        rates.append({'week':week,'beta_local':.85*fit.x[0],'beta_mobility':.15*fit.x[0],
                      'in_sample_rmse':np.sqrt(((estimate[keep]-data[keep])**2).mean())})
        fitting.extend({'date':str(dates[7*week+day].date()),'observed':np.nansum(data[day]),
                        'fitted':np.where(np.isfinite(data[day]),estimate[day],0).sum()} for day in range(7))
        print(f'Retrospective rate week {week+1}: {fit.x[0]:.4f}',flush=True)
    pd.DataFrame(rates).to_csv(out/'weekly_rates.csv',index=False)
    pd.DataFrame(fitting).to_csv(out/'calibration_series.csv',index=False)


if __name__ == "__main__":
    main()

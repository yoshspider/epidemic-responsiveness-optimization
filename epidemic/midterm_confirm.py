"""Collect completed independent PPO seeds and evaluate fresh test worlds."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time
import pandas as pd


def main():
    output=Path('outputs/proposal_ppo_confirmation'); output.mkdir(exist_ok=True)
    sources={0:Path('outputs/proposal_ppo_main'),1:Path('outputs/proposal_ppo_seed1'),2:Path('outputs/proposal_ppo_seed2')}
    for _ in range(240):
        complete=True
        for seed,folder in sources.items():
            for information in ['blind','oracle','predicted']:
                training=folder/f'{information}_{seed}.training.csv'
                complete &= training.exists() and pd.read_csv(training)['update'].max()>=95
        if complete: break
        print('Waiting for the three fixed-budget PPO training seeds',flush=True)
        time.sleep(15)
    else:
        raise TimeoutError('Training did not complete within one hour')
    provenance=[]
    for seed,folder in sources.items():
        for information in ['blind','oracle','predicted']:
            for extension in ['pt','training.csv']:
                name=f'{information}_{seed}.{extension}'
                shutil.copy2(folder/name,output/name)
            provenance.append({'seed':seed,'information':information,'source':str(folder),
                'steps':24576,'sha256':hashlib.sha256((output/f'{information}_{seed}.pt').read_bytes()).hexdigest()})
    (output/'model_provenance.json').write_text(json.dumps(provenance,indent=2))
    command=[sys.executable,'tools/bundled_runner.py','epidemic.proposal_ppo','--evaluate-only',
        '--steps','24576','--training-seeds','3','--test-worlds','24','--test-seed-start','950000',
        '--output',str(output)]
    subprocess.run(command,check=True)
    trials=pd.read_csv(output/'trials.csv')
    rows=[]
    for (scenario,fraction,seed),group in trials.groupby(['scenario','fraction','training_seed']):
        wide=group.pivot(index='world',columns='policy',values='infections')
        for policy in ['ppo_oracle','ppo_predicted']:
            delta=wide.ppo_blind-wide[policy]
            rows.append({'scenario':scenario,'fraction':fraction,'training_seed':seed,'policy':policy,
                'mean_gain':delta.mean(),'relative_percent':100*delta.mean()/wide.ppo_blind.mean()})
    pd.DataFrame(rows).to_csv(output/'training_seed_summary.csv',index=False)
    (output/'complete.json').write_text(json.dumps({'complete':True,'independent_test_worlds':24,
        'test_seeds':'950000–950023; disjoint from development 900000–900007',
        'model_selection':'Fixed final training budget; no test-based checkpoint selection'},indent=2))


if __name__=='__main__': main()

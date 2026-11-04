"""Aggregate 2016 county election covariate from the official MIT public release."""
import json
from pathlib import Path
import pandas as pd
from .acquire import download


def main():
    folder=Path('data/raw/alternative_sources');folder.mkdir(exist_ok=True)
    url='https://dataverse.harvard.edu/api/access/datafile/13573089'
    path=folder/'mit_county_presidential_returns.tab'
    record=download(url,path)
    (folder/'mit_election_source.json').write_text(json.dumps({**record,'doi':'10.7910/DVN/VOQCHQ','publisher':'MIT Election Data and Science Lab / Harvard Dataverse','license':'CC0','feature_event_year':2016},indent=2))
    data=pd.read_csv(path,sep='\t',dtype={'county_fips':str})
    frame=data[data.year==2016].copy()
    if 'mode' in frame:
        print('2016 modes',frame['mode'].value_counts().to_dict(),flush=True)
        frame=frame[frame['mode'].str.upper()=='TOTAL']
    frame=frame[frame.county_fips.notna()]
    frame['fips']=pd.to_numeric(frame.county_fips,errors='raise').astype(int).astype(str).str.zfill(5)
    votes=frame.groupby(['fips','party']).candidatevotes.sum().unstack()
    votes['democratic_share_2016']=votes.DEMOCRAT/(votes.DEMOCRAT+votes.REPUBLICAN)
    if not votes.democratic_share_2016.between(0,1).all():raise ValueError('Invalid election shares')
    votes[['democratic_share_2016']].to_csv(folder/'election2016_county_share.csv')
    print('2016 county vote shares:',len(votes),flush=True)


if __name__=='__main__':main()

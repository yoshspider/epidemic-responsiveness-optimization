"""Documented public county compilation, with replicated Alaska values excluded."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .acquire import download


def main():
    folder=Path('data/raw/alternative_sources');folder.mkdir(exist_ok=True)
    url='https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-24/master/2016_US_County_Level_Presidential_Results.csv'
    path=folder/'tonmcg_county2016.csv'
    record=download(url,path)
    frame=pd.read_csv(path,dtype={'combined_fips':str})
    frame['fips']=pd.to_numeric(frame.combined_fips,errors='raise').astype(int).astype(str).str.zfill(5)
    frame['democratic_share_2016']=frame.votes_dem/(frame.votes_dem+frame.votes_gop)
    # The source repeats statewide Alaska totals in county rows. They are not
    # county measurements; use an explicit missing feature for those rows.
    invalid=frame.state_abbr.eq('AK')
    frame.loc[invalid,'democratic_share_2016']=np.nan
    if frame.fips.duplicated().any():raise ValueError('Duplicate county election rows')
    frame[['fips','democratic_share_2016']].to_csv(folder/'election2016_features.csv',index=False)
    (folder/'election2016_source.json').write_text(json.dumps({**record,
        'publisher':'tonmcg community county election compilation, NOT the official MIT release',
        'event_year':2016,'source_repository':'https://github.com/tonmcg/US_County_Level_Election_Results_08-24',
        'quality_rule':'Alaska statewide totals repeated across counties are excluded (missing feature); no future elections used',
        'rows':len(frame),'alaska_rows_excluded':int(invalid.sum())},indent=2))
    print(f'2016 political-context features: {len(frame)} rows, {invalid.sum()} Alaska aggregate rows excluded',flush=True)


if __name__=='__main__':main()

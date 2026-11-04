"""Pre-pandemic ACS covariates and an aggregate-only mask-entity pilot."""
import json
from pathlib import Path
from urllib.request import Request,urlopen
import zipfile
import pandas as pd
from .acquire import download


def main():
    folder=Path('data/raw/alternative_sources'); folder.mkdir(exist_ok=True)
    variables=['B01003_001E','B01002_001E','B19013_001E','B15003_001E',
               'B15003_022E','B15003_023E','B15003_024E','B15003_025E','B17001_001E','B17001_002E']
    metadata={}
    for variable in variables:
        with urlopen('https://api.census.gov/data/2019/acs/acs5/variables/'+variable+'.json',timeout=60) as source:
            metadata[variable]=json.load(source)
    (folder/'acs_variable_definitions.json').write_text(json.dumps(metadata,indent=2))
    url='https://api.census.gov/data/2019/acs/acs5?get=NAME,'+','.join(variables)+'&for=county:*'
    path=folder/'acs2019_counties.json'
    record=download(url,path)
    (folder/'acs2019_source.json').write_text(json.dumps(record,indent=2))
    raw=json.loads(path.read_text()); frame=pd.DataFrame(raw[1:],columns=raw[0])
    frame['fips']=frame.state+frame.county
    for field in variables:
        frame[field]=pd.to_numeric(frame[field],errors='coerce').where(lambda x:x>=0)
    frame['median_age']=frame.B01002_001E
    frame['median_income']=frame.B19013_001E
    frame['college_share']=frame[['B15003_022E','B15003_023E','B15003_024E','B15003_025E']].sum(axis=1,min_count=4)/frame.B15003_001E
    frame['poverty_share']=frame.B17001_002E/frame.B17001_001E
    frame.to_csv(folder/'acs2019_counties.csv',index=False)
    print(f'ACS2019: {len(frame)} counties and equivalents, source definitions saved',flush=True)


if __name__=='__main__':main()

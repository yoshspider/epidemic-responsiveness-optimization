"""Official pre-pandemic county socioeconomic covariates from CDC SVI 2018."""
import json
from pathlib import Path
import pandas as pd
from .acquire import download


def main():
    folder=Path('data/raw/alternative_sources'); folder.mkdir(exist_ok=True)
    url='https://svi2.cdc.gov/webapi/Documents/download?year=2018&type=csv&category=states_counties&name=SVI_2018_US_COUNTY'
    destination=folder/'cdc_svi2018_counties.csv'
    record=download(url,destination)
    frame=pd.read_csv(destination,dtype={'FIPS':str})
    if 'FIPS' not in frame: raise ValueError('Official CDC response is not county data')
    (folder/'cdc_svi2018_source.json').write_text(json.dumps(record,indent=2))
    print(len(frame),list(frame.columns),flush=True)


if __name__=='__main__':main()

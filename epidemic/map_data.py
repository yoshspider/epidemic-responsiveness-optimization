"""Build geographic data from official Census geometry and saved simulation snapshots."""
import base64
import gzip
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import shapefile


def simplify_ring(points,tolerance=.04):
    """Display-only Ramer-Douglas-Peucker simplification of sourced boundary vertices."""
    pts=np.asarray(points,dtype=float)
    if len(pts)<=5:
        return pts.tolist()
    def rdp(array):
        if len(array)<3:
            return array
        delta=array[-1]-array[0]
        length=float(delta@delta)
        if length:
            fraction=np.clip(((array-array[0])@delta)/length,0,1)
            distance=np.linalg.norm(array-(array[0]+fraction[:,None]*delta),axis=1)
        else:
            distance=np.linalg.norm(array-array[0],axis=1)
        at=int(np.argmax(distance))
        if distance[at]<=tolerance or at==0 or at==len(array)-1:
            return array[[0,-1]]
        return np.vstack((rdp(array[:at+1])[:-1],rdp(array[at:])))
    result=rdp(pts)
    if len(result)<4:
        # Keep a valid closed polygon when the ring is smaller than display tolerance.
        result=pts[np.unique(np.linspace(0,len(pts)-1,5,dtype=int))]
    return np.round(result,3).tolist()


def main():
    root=Path(__file__).resolve().parents[1]
    archive=root/"data/raw/census_counties_2020.zip"
    with zipfile.ZipFile(archive) as z:
        shp=next(name for name in z.namelist() if name.endswith('.shp'))
        dbf=next(name for name in z.namelist() if name.endswith('.dbf'))
        reader=shapefile.Reader(shp=io.BytesIO(z.read(shp)),dbf=io.BytesIO(z.read(dbf)))
        names=[field[0] for field in reader.fields[1:]]
        features=[]
        for item in reader.iterShapeRecords():
            record=dict(zip(names,item.record))
            if str(record["STATEFP"]) in ('60','66','69','72','78'):
                continue
            geometry=item.shape.__geo_interface__
            if geometry['type']=='Polygon':
                geometry['coordinates']=[simplify_ring(ring) for ring in geometry['coordinates']]
            else:
                geometry['coordinates']=[[simplify_ring(ring) for ring in polygon] for polygon in geometry['coordinates']]
            features.append({"type":"Feature","id":str(record["GEOID"]),
                             "properties":{"name":record["NAME"],"state":record["STATEFP"]},"geometry":geometry})
    population=pd.read_csv(root/"data/transformed/census_population_2019.csv",dtype={"fips":str}).set_index("fips").population
    study=root/"outputs/planner_national_1pct"
    series={}
    county_order=None
    weeks=None
    for policy in ('none','epidemic','oracle'):
        table=pd.read_csv(study/(policy+"_weekly.csv"),dtype={"fips":str})
        pivot=table.pivot(index="week",columns="fips",values="incidence").sort_index(axis=1)
        if county_order is None:
            county_order=pivot.columns.tolist()
            weeks=pivot.index.tolist()
        rates=pivot.to_numpy()/population.reindex(county_order).to_numpy()[None,:]*100000
        series[policy]={"weekly_rate":np.round(rates,2).tolist(),"national_weekly":np.round(pivot.sum(1),0).tolist()}
    payload={"county_order":county_order,"weeks":weeks,"series":series,"budget_fraction":.01,
             "policy_start":"2020-08-31","counties":len(county_order),
             "data_kind":"Empirical-input counterfactual; fixed July mobility, assumed costs and efficacy",
             "geometry_source":"US Census cb_2020_us_county_500k.zip; display-only RDP tolerance 0.04 degrees and rounding to 0.001 degrees; metrics use source FIPS, not simplified area",
             "features":features}
    out=root/"outputs/visuals"
    out.mkdir(parents=True,exist_ok=True)
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':')).encode())).decode()
    (out/"county_map_payload.b64").write_text(packed)
    print(f"{len(features)} county features, {len(county_order)} simulated counties; compressed payload {len(packed):,} bytes")


if __name__=="__main__":
    main()

"""Read archive metadata before allocating large local storage."""
from urllib.request import Request,urlopen
import json

urls={"tbcov_september":"https://crisisnlp.qcri.org/data/tbcov/monthly_files/september_2020_tbcov.zip",
      "census_counties":"https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"}
for name,url in urls.items():
    try:
        with urlopen(Request(url,method="HEAD",headers={"User-Agent":"EpidemicResearch/0.1"}),timeout=60) as response:
            print(json.dumps({"name":name,"url":response.url,"status":response.status,"length":response.headers.get("Content-Length"),"type":response.headers.get("Content-Type")}))
    except Exception as exc:
        print(json.dumps({"name":name,"error":str(exc)}))

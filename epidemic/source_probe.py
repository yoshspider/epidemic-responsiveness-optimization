"""Resolve official release links without downloading large tweet archives."""
import json
import re
from urllib.request import Request, urlopen
from pathlib import Path


def main():
    folder=Path("data/source_metadata")
    folder.mkdir(parents=True,exist_ok=True)
    urls={"tbcov":"https://crisisnlp.qcri.org/tbcov",
          "geods_tree":"https://api.github.com/repos/GeoDS/COVID19USFlows/git/trees/master?recursive=1"}
    for name,url in urls.items():
        with urlopen(Request(url,headers={"User-Agent":"EpidemicResearch/0.1"}),timeout=60) as response:
            text=response.read().decode()
        (folder/(name+".txt")).write_text(text,encoding="utf-8")
        if name=="tbcov":
            for line in text.splitlines():
                if ("United States" in line or "2020" in line) and ("option" in line or "http" in line):
                    print(line.strip()[:500])
        else:
            paths=[v["path"] for v in json.loads(text)["tree"] if "county2county" in v["path"] and "2020_07" in v["path"]]
            print(json.dumps(paths))


if __name__=="__main__":
    main()

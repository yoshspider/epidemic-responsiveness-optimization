"""Download modest public source tables with hashes; no hydration or synthetic fallback."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    next_progress = 256*1024*1024
    with urlopen(Request(url, headers={"User-Agent":"EpidemicResearch/0.1"}), timeout=120) as response, temp.open("wb") as out:
        while block := response.read(1024*1024):
            size += len(block)
            digest.update(block)
            out.write(block)
            if size>=next_progress:
                print(f"{destination.name}: {size/(1024**3):.2f} GiB downloaded",flush=True)
                next_progress+=256*1024*1024
    temp.replace(destination)
    return {"url":url,"path":str(destination),"bytes":size,"sha256":digest.hexdigest(),
            "retrieved_utc":datetime.now(timezone.utc).isoformat()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("data/raw"))
    p.add_argument("--nyt",action="store_true")
    p.add_argument("--url", help="An explicitly selected additional source URL")
    p.add_argument("--name", help="Local filename for --url")
    a=p.parse_args()
    if not a.nyt and not a.url:
        p.error("Choose --nyt and/or --url")
    sources=[]
    if a.nyt:
        base="https://raw.githubusercontent.com/nytimes/covid-19-data/master/"
        sources=[(base+"us-counties-2020.csv","nyt_cases_2020.csv"),
                 (base+"mask-use/mask-use-by-county.csv","nyt_mask_use.csv"),
                 (base+"mask-use/README.md","nyt_mask_readme.md"),
                 (base+"LICENSE","nyt_LICENSE.txt")]
    if a.url:
        if not a.name or Path(a.name).name != a.name:
            p.error("--name must be a simple filename")
        sources.append((a.url,a.name))
    a.output.mkdir(parents=True,exist_ok=True)
    manifest_path=a.output/"downloads.json"
    records=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for url,name in sources:
        print(f"Downloading {name}",flush=True)
        records.append(download(url,a.output/name))
        manifest_path.write_text(json.dumps(records,indent=2))


if __name__=="__main__":
    main()

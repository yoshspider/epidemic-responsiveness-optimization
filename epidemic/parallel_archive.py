"""Range-based recovery of a large public archive; keep original partial intact."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen
from datetime import datetime,timezone


def main():
    p=argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("destination",type=Path)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    if a.destination.exists():
        print("Complete archive already exists",flush=True)
        return
    with urlopen(Request(a.url,method="HEAD"),timeout=45) as response:
        size=int(response.headers["Content-Length"])
    folder=a.destination.parent/(a.destination.stem+"_segments")
    folder.mkdir(exist_ok=True)
    prefix=folder/"prefix.bin"
    old=a.destination.with_suffix(a.destination.suffix+".part")
    if not prefix.exists():
        length=old.stat().st_size if old.exists() else 0
        with prefix.open("wb") as out:
            if length:
                with old.open("rb") as source:
                    while length:
                        block=source.read(min(4*1024*1024,length))
                        if not block:
                            raise OSError("Partial source changed unexpectedly")
                        out.write(block)
                        length-=len(block)
    offset=prefix.stat().st_size
    span=(size-offset+a.workers-1)//a.workers
    ranges=[(offset+k*span,min(size,offset+(k+1)*span)-1,folder/f"segment_{k:02d}.bin") for k in range(a.workers)]
    ranges=[r for r in ranges if r[0]<=r[1]]
    def retrieve(item):
        start,end,path=item
        for attempt in range(10):
            present=path.stat().st_size if path.exists() else 0
            current=start+present
            if current>end:
                return
            try:
                request=Request(a.url,headers={"User-Agent":"EpidemicResearch/0.1","Range":f"bytes={current}-{end}"})
                with urlopen(request,timeout=45) as response:
                    content=response.headers.get("Content-Range","")
                    if response.status!=206 or not content.startswith(f"bytes {current}-"):
                        raise ValueError("Source does not honor requested range")
                    with path.open("ab") as out:
                        while current<=end:
                            block=response.read(min(1024*1024,end-current+1))
                            if not block:
                                raise OSError("Truncated source range")
                            out.write(block)
                            current+=len(block)
                return
            except (OSError,TimeoutError) as error:
                print(f"{path.name}: retry {attempt+1}: {error}",flush=True)
        raise OSError(f"Retry limit reached for {path}")
    print(f"Preserved {offset:,} bytes; downloading {len(ranges)} independent source ranges",flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(retrieve,item) for item in ranges}
        while pending:
            done,pending=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            have=offset+sum(path.stat().st_size if path.exists() else 0 for _,_,path in ranges)
            print(f"Parallel archive {have/1e9:.3f}/{size/1e9:.3f} GB",flush=True)
    temporary=a.destination.with_suffix(".parallel.part")
    digest=hashlib.sha256()
    written=0
    with temporary.open("wb") as out:
        for source_path in [prefix]+[path for _,_,path in ranges]:
            with source_path.open("rb") as source:
                while block:=source.read(4*1024*1024):
                    out.write(block)
                    digest.update(block)
                    written+=len(block)
    if written!=size:
        raise OSError("Archive assembly length mismatch")
    temporary.replace(a.destination)
    manifest=a.destination.parent/"downloads.json"
    records=json.loads(manifest.read_text())
    records.append({"url":a.url,"path":str(a.destination),"bytes":written,"sha256":digest.hexdigest(),
                    "retrieved_utc":datetime.now(timezone.utc).isoformat(),"transport":"resumed HTTP ranges"})
    manifest.write_text(json.dumps(records,indent=2))
    print("Archive assembled and hashed; source parts retained",flush=True)


if __name__=="__main__":
    main()

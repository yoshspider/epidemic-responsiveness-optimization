"""Resumable bounded retries for large official archives; preserve partial data."""
import argparse
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("destination",type=Path)
    a=p.parse_args()
    a.destination.parent.mkdir(parents=True,exist_ok=True)
    part=a.destination.with_suffix(a.destination.suffix+".part")
    if a.destination.exists():
        print("Archive already exists; preserving it",flush=True)
        return
    expected=None
    for attempt in range(10):
        offset=part.stat().st_size if part.exists() else 0
        headers={"User-Agent":"EpidemicResearch/0.1"}
        if offset:
            headers["Range"]=f"bytes={offset}-"
        print(f"Attempt {attempt+1}: resuming at {offset:,} bytes",flush=True)
        try:
            with urlopen(Request(a.url,headers=headers),timeout=45) as response:
                if offset and response.status!=206:
                    raise ValueError("Server ignored resume range; partial file preserved")
                if offset:
                    content_range=response.headers.get("Content-Range","")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("Mismatched range response")
                    expected=int(content_range.split("/")[-1])
                else:
                    expected=int(response.headers["Content-Length"])
                size=offset
                last=time.monotonic()
                with part.open("ab" if offset else "wb") as out:
                    while block:=response.read(1024*1024):
                        out.write(block)
                        size+=len(block)
                        if time.monotonic()-last>20:
                            print(f"Downloaded {size/1e9:.3f}/{expected/1e9:.3f} GB",flush=True)
                            last=time.monotonic()
                if size!=expected:
                    raise OSError(f"Incomplete response: {size}/{expected}")
            digest=hashlib.sha256()
            with part.open("rb") as source:
                while block:=source.read(4*1024*1024):
                    digest.update(block)
            part.replace(a.destination)
            manifest=a.destination.parent/"downloads.json"
            records=json.loads(manifest.read_text()) if manifest.exists() else []
            records.append({"url":a.url,"path":str(a.destination),"bytes":expected,
                "sha256":digest.hexdigest(),"retrieved_utc":datetime.now(timezone.utc).isoformat()})
            manifest.write_text(json.dumps(records,indent=2))
            print("Download complete and hashed",flush=True)
            return
        except (TimeoutError,OSError) as error:
            print(f"Recoverable download error: {error}",flush=True)
    raise RuntimeError("Retry limit reached; partial archive retained for later resume")


if __name__=="__main__":
    main()

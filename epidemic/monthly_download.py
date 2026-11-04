"""Resumable, bounded-memory HTTP range download without duplicate archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


def download(url, destination, workers=6, block_size=32*1024*1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = destination.with_suffix('.download.json')
    with urlopen(Request(url, method='HEAD'), timeout=60) as response:
        size = int(response.headers['Content-Length'])
        modified = response.headers.get('Last-Modified')
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError('Existing archive length differs from source')
        return
    partial = destination.with_suffix('.ranged.part')
    checkpoint = destination.with_suffix('.ranges.json')
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        'url': url, 'bytes': size, 'last_modified': modified,
        'block_size': block_size, 'completed': []}
    if (state['url'], state['bytes'], state['last_modified'], state['block_size']) != (url, size, modified, block_size):
        raise ValueError('Download checkpoint/source mismatch')
    if not partial.exists():
        if state['completed']:
            raise ValueError('Checkpoint refers to absent partial archive')
        if shutil.disk_usage(destination.parent).free < size + 10*1024**3:
            raise OSError('Insufficient disk space with 10 GiB reserve')
        with partial.open('wb') as out:
            out.truncate(size)
    if partial.stat().st_size != size:
        raise ValueError('Unexpected partial size')
    total_blocks = (size+block_size-1)//block_size
    done = set(state['completed'])
    def retrieve(index):
        first, last = index*block_size, min(size, (index+1)*block_size)-1
        current = first
        for attempt in range(20):
            try:
                with urlopen(Request(url, headers={'Range': f'bytes={current}-{last}',
                        'User-Agent': 'EpidemicResearch/0.2'}), timeout=60) as source:
                    if source.status != 206 or not source.headers.get('Content-Range', '').startswith(f'bytes {current}-'):
                        raise ValueError('Source did not honor range')
                    with partial.open('r+b') as out:
                        out.seek(current)
                        while current <= last:
                            data = source.read(min(1024*1024, last-current+1))
                            if not data:
                                raise OSError('Truncated range')
                            out.write(data)
                            current += len(data)
                if current == last+1:
                    return index
            except (OSError, TimeoutError) as error:
                if attempt == 19:
                    raise OSError(f'Failed block {index}: {error}') from error
                time.sleep(min(5, attempt+1))
        raise OSError('Incomplete block')
    print(f'{destination.name}: {len(done)}/{total_blocks} blocks already complete', flush=True)
    remaining = iter(index for index in range(total_blocks) if index not in done)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for _ in range(workers):
            index = next(remaining, None)
            if index is not None:
                pending.add(pool.submit(retrieve, index))
        last_print = time.monotonic()
        while pending:
            finished, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in finished:
                done.add(future.result())
                index = next(remaining, None)
                if index is not None:
                    pending.add(pool.submit(retrieve, index))
            state['completed'] = sorted(done)
            temp = checkpoint.with_suffix('.json.tmp')
            temp.write_text(json.dumps(state))
            temp.replace(checkpoint)
            if time.monotonic()-last_print >= 20 or not pending:
                downloaded = sum(min(block_size, size-index*block_size) for index in done)
                print(f'{destination.name}: {downloaded/1e9:.2f}/{size/1e9:.2f} GB', flush=True)
                last_print = time.monotonic()
    digest = hashlib.sha256()
    with partial.open('rb') as source:
        while chunk := source.read(8*1024*1024):
            digest.update(chunk)
    record = {'url': url, 'bytes': size, 'sha256': digest.hexdigest(),
              'retrieved_utc': datetime.now(timezone.utc).isoformat(),
              'last_modified': modified, 'transport': 'checkpointed direct HTTP ranges'}
    metadata.write_text(json.dumps(record, indent=2))
    partial.replace(destination)
    print(f'{destination.name}: complete and hashed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('months', nargs='+', help='YYYY-MM')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    for month in args.months:
        date = datetime.strptime(month, '%Y-%m')
        name = date.strftime('%B_%Y').lower()+'_tbcov.zip'
        download('https://crisisnlp.qcri.org/data/tbcov/monthly_files/'+name,
                 Path('data/raw')/('tbcov_'+date.strftime('%B%Y').lower()+'.zip'), args.workers)


if __name__ == '__main__':
    main()

"""Inspect only base-file headers from an incomplete ZIP, never tweet text or user IDs."""
from pathlib import Path
import struct
import zlib
import io
import pandas as pd

path=Path('data/raw/tbcov_september2020.zip.part')
with path.open('rb') as source:
    while True:
        header=source.read(30)
        if header[:4]!=b'PK\x03\x04':
            raise ValueError('Expected ZIP local header')
        _,version,flags,method,mtime,mdate,crc,compressed,uncompressed,name_length,extra_length=struct.unpack('<IHHHHHIIIHH',header)
        name=source.read(name_length).decode('utf-8')
        source.read(extra_length)
        print('Member:',name,'method',method,'compressed size field',compressed)
        if name.endswith('/'):
            source.seek(compressed,1)
            continue
        block=source.read(1024*1024)
        if method==8:
            decoded=zlib.decompressobj(-15).decompress(block,2*1024*1024)
        elif method==0:
            decoded=block
        else:
            raise ValueError('Unsupported compression method')
        first=decoded.splitlines()[0].decode('utf-8')
        print('Column header:',first)
        sample=pd.read_csv(io.BytesIO(decoded),sep='\t',dtype=str,nrows=500)
        print('Country-code counts in schema sample:',sample.user_loc_country_code.value_counts().head(6).to_dict())
        print('Timestamp format sample:',sample.date_time.iloc[0])
        us=sample[sample.user_loc_country_code.eq('US')]
        print('US county-label counts in schema sample:',us.user_loc_county.value_counts().head(5).to_dict())
        break

"""Retrieve only frozen DIV2K source members from the official ZIPs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import re
import struct
import zipfile
import zlib

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'runs/div2k_known_kernel'
BASE='https://data.vision.ee.ethz.ch/cvl/DIV2K/'


class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url
        self.length=int(requests.head(url,allow_redirects=True,timeout=25).headers['Content-Length'])
        self.pos=0

    def read(self,n=-1):
        end=min(self.length,self.pos+n) if n>=0 else self.length
        if end<=self.pos:
            return b''
        data=fetch(self.url,self.pos,end-1)
        self.pos=end
        return data

    def seek(self,offset,whence=0):
        self.pos={0:0,1:self.pos,2:self.length}[whence]+offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True


def fetch(url,start,end):
    for attempt in range(3):
        try:
            r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,45),stream=True)
            with r:
                assert r.status_code==206,(r.status_code,start,end)
                assert re.fullmatch(fr'bytes {start}-{end}/\d+',r.headers['Content-Range']),r.headers
                return r.content
        except (requests.RequestException,AssertionError):
            if attempt==2:
                raise


def run(part,rows):
    if (CACHE/f'DIV2K_{part}_HR.zip').exists():
        return
    url=BASE+f'DIV2K_{part}_HR.zip'
    output=CACHE/part
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(RemoteZip(url)) as z:
        items=[(z.getinfo(f'DIV2K_{part}_HR/{int(r.original_id):04d}.png'),r.original_sha256)
               for r in rows.itertuples()]

    def get(item):
        info,expected=item
        path=output/Path(info.filename).name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
            return
        start=info.header_offset
        head=fetch(url,start,start+29)
        assert head[:4]==b'PK\x03\x04'
        name_len,extra_len=struct.unpack_from('<HH',head,26)
        start+=30+name_len+extra_len
        compressed=fetch(url,start,start+info.compress_size-1)
        raw=zlib.decompress(compressed,-15) if info.compress_type==zipfile.ZIP_DEFLATED else compressed
        assert len(raw)==info.file_size and zlib.crc32(raw)==info.CRC
        assert hashlib.sha256(raw).hexdigest()==expected,info.filename
        temp=path.with_suffix('.partial')
        temp.write_bytes(raw)
        temp.replace(path)

    with ThreadPoolExecutor(max_workers=12) as pool:
        for i,_ in enumerate(pool.map(get,items),1):
            if i%10==0 or i==len(items):print(f'{part} {i}/{len(items)} verified',flush=True)


with zipfile.ZipFile(ROOT/'followup_v9/input_audit/Paper2_Focus_Input_Tables.zip') as z:
    dev=pd.read_csv(z.open('inventories/development.csv')).query('dataset=="div2k"')
    test=pd.read_csv(z.open('inventories/div2k.csv'))
run('train',dev)
run('valid',test)

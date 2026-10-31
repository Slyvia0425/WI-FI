"""Download only selected FallDetection train/val members via checked HTTP ranges."""
import argparse,json,struct,subprocess,tempfile,time,zlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
URL='https://www.kaggle.com/api/v1/datasets/download/guozhenjennzhu/csi-bench?datasetVersionNumber=12'

def resolve_download_url():
 """Reuse the official public archive redirect instead of reopening Kaggle per block."""
 for attempt in range(5):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d)
   r=subprocess.run(['curl','-sS','-L','--fail','--max-time','120','--max-filesize','1024','-H','Range: bytes=0-0','-w','%{url_effective}',URL,'-o',str(d/'probe')],capture_output=True,text=True)
   if not r.returncode and (d/'probe').read_bytes()==b'P':return r.stdout.strip()
  if attempt<4:time.sleep(5*(attempt+1))
 raise IOError('Could not resolve the official public archive; no records skipped')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',default='FallDetection');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 wanted={r['remote_path'] for r in json.loads(a.selection.read_text())['records'] if r['source']==a.source}
 entries=[r for r in json.loads(a.index.read_text()) if r['name'] in wanted];assert {r['name'] for r in entries}==wanted
 entries.sort(key=lambda r:r['offset']);blocks=[]
 for e in entries:
  if not blocks or e['offset']-blocks[-1][0]['offset']>16*1024**2:blocks.append([])
  blocks[-1].append(e)
 resolved_url=None
 def valid(e):
  p=a.root/e['name'];return p.exists() and p.stat().st_size==e['size'] and zlib.crc32(p.read_bytes())==e['crc']
 def one(block):
  if all(valid(e) for e in block):return len(block)
  begin=block[0]['offset'];last=block[-1];end=last['offset']+30+len(last['name'].encode())+65535+last['compressed']-1
  for attempt in range(5):
   try:
    with tempfile.TemporaryDirectory() as tmpdir:
     tmpdir=Path(tmpdir)
     request=subprocess.run(['curl','-sS','-L','--fail','--max-time','120','--max-filesize','40000000','-H',f'Range: bytes={begin}-{end}','-D',str(tmpdir/'headers'),resolved_url,'-o',str(tmpdir/'body')],capture_output=True,text=True)
     if request.returncode:raise IOError(request.stderr)
     ranges=[v.split(':',1)[1].strip() for v in (tmpdir/'headers').read_text().splitlines() if v.lower().startswith('content-range:')]
     assert ranges and ranges[-1].startswith(f'bytes {begin}-'),'Refusing complete archive download'
     raw=(tmpdir/'body').read_bytes();assert len(raw)<=40000000
    for e in block:
     pos=e['offset']-begin;assert raw[pos:pos+4]==b'PK\x03\x04'
     n,x=struct.unpack_from('<HH',raw,pos+26);i=pos+30+n+x;payload=raw[i:i+e['compressed']]
     if e['method']==8:payload=zlib.decompress(payload,-15)
     else:assert e['method']==0
     assert len(payload)==e['size'] and zlib.crc32(payload)==e['crc']
     p=(a.root/e['name']).resolve();assert p.is_relative_to(a.root.resolve());p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial');t.write_bytes(payload);t.replace(p)
    return len(block)
   except Exception:
    if attempt==4:raise
    time.sleep(2**attempt)
 if all(valid(e) for e in entries):print('SELECTED_CRC_VERIFIED',len(entries),'/',len(entries),flush=True);return
 resolved_url=resolve_download_url()
 count=0
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(one,b) for b in blocks]
  for f in as_completed(futures):count+=f.result();print('SELECTED_CRC_VERIFIED',count,'/',len(entries),flush=True)
 assert all(valid(e) for e in entries)

if __name__=='__main__':main()

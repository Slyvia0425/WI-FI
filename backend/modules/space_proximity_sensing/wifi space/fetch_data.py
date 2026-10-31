import hashlib,json,os,shutil,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 spec=json.loads((ROOT/'assets.json').read_text(encoding='utf-8'))
 cache=ROOT/'.downloads';cache.mkdir(exist_ok=True)
 complete=cache/'complete.json'
 if complete.exists() and json.loads(complete.read_text()).get('sha256')==spec['archive_sha256'] and all((ROOT/p).is_file() and digest(ROOT/p)==value for p,value in spec['models'].items()) and (ROOT/'multi_object/data/home_v5/manifest.json').exists():return
 for part in spec['parts']:
  path=cache/part['name']
  if path.exists() and path.stat().st_size==part['bytes'] and digest(path)==part['sha256']:continue
  temp=path.with_suffix(path.suffix+'.partial')
  url=f"https://github.com/{spec['repository']}/releases/download/{spec['tag']}/{part['name']}"
  print(part['name'],flush=True)
  offset=temp.stat().st_size if temp.exists() else 0
  headers={'User-Agent':'wifi-space'}
  if offset:headers['Range']=f'bytes={offset}-'
  with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
   if response.status!=206:offset=0
   with temp.open('ab' if offset else 'wb') as f:shutil.copyfileobj(response,f,4*1024*1024)
  if temp.stat().st_size!=part['bytes'] or digest(temp)!=part['sha256']:raise RuntimeError('Download checksum mismatch')
  temp.replace(path)
 archive=cache/'wifi-space.zip'
 if not archive.exists() or archive.stat().st_size!=spec['archive_bytes'] or digest(archive)!=spec['archive_sha256']:
  temp=cache/'wifi-space.zip.partial'
  with temp.open('wb') as out:
   for part in spec['parts']:
    with (cache/part['name']).open('rb') as source:shutil.copyfileobj(source,out,8*1024*1024)
  if digest(temp)!=spec['archive_sha256']:raise RuntimeError('Archive checksum mismatch')
  temp.replace(archive)
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   rel=Path(info.filename)
   if not rel.parts or rel.parts[0]!='wifi space' or '..' in rel.parts:raise RuntimeError('Invalid archive path')
   if len(rel.parts)==1:continue
   target=ROOT.joinpath(*rel.parts[1:]).resolve()
   if not target.is_relative_to(ROOT):raise RuntimeError('Invalid archive target')
   if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
   parts=rel.parts[1:]
   payload=parts[0] in {'data','outputs'} or len(parts)>1 and parts[0]=='multi_object' and parts[1] in {'data','outputs'}
   if target.exists() and not payload:continue
   if target.exists() and target.stat().st_size==info.file_size:continue
   target.parent.mkdir(parents=True,exist_ok=True)
   temp=target.with_name(target.name+'.partial')
   with z.open(info) as source,temp.open('wb') as out:shutil.copyfileobj(source,out,4*1024*1024)
   temp.replace(target)
 if not all(digest(ROOT/p)==value for p,value in spec['models'].items()):raise RuntimeError('Model checksum mismatch')
 temp=cache/'complete.json.partial';temp.write_text(json.dumps({'sha256':spec['archive_sha256']}),encoding='utf-8');temp.replace(complete)

if __name__=='__main__':main()

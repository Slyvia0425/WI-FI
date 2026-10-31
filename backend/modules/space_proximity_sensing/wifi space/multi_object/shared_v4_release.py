
import argparse,json
from shared_v4_common import O4,read,sha,now

def activate(version):
    if version==4:
        r=read(O4/'results.json');assert r['release_check']['eligible'],'Final release check failed'
        v=read(O4/'verification.json');assert v['passed'] and v['model_sha256']==sha(O4/'model.joblib')
        a=read(O4/'data_audit_full.json');assert a['files_passed'] and a['clean_split_audit'],'Check raw signal overlap/ambiguity before activation'
        assert read(O4/'placement_verification.json')['passed']
    root=O4.parent;p=root/'active_profile.json';history=root/'profile_history.json';old=read(p) if p.exists() else dict(version=3);changes=read(history) if history.exists() else [];changes.append(dict(at=now(),previous=old,version=version))
    for path,data in [(history,changes),(p,dict(version=version,activated_at=now()))]:
        tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)
    print('Active model v'+str(version))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['activate','rollback']);a=p.parse_args();activate(4 if a.action=='activate' else 3)

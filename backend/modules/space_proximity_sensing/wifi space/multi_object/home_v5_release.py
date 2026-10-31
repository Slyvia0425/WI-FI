
import argparse,json
from home_v5_common import O5,read,sha,now

def activate(version):
    if version==5:
        r=read(O5/'results.json');assert r['release_check']['eligible'],'Final release check failed'
        v=read(O5/'verification.json');assert v['passed'] and v['model_sha256']==sha(O5/'model.joblib')
        a=read(O5/'data_audit_full.json');assert a['files_passed'] and a['clean_split_audit'],'Check raw signal overlap/ambiguity before activation'
        assert read(O5/'placement_verification.json')['passed']
        assert read(O5/'ui_verification.json')['passed'],'Candidate browser checks required before activation'
    root=O5.parent;p=root/'active_profile.json';history=root/'profile_history.json';old=read(p) if p.exists() else dict(version=3);changes=read(history) if history.exists() else [];changes.append(dict(at=now(),previous=old,version=version))
    for path,data in [(history,changes),(p,dict(version=version,activated_at=now()))]:
        tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)
    print('Active model v'+str(version))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['activate','rollback']);a=p.parse_args();activate(5 if a.action=='activate' else 4)

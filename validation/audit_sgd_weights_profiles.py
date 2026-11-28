"""Verify frozen Signed weights and retained Perceptron/SGD evidence and retained nested/flat outputs."""
from pathlib import Path
import argparse,ast,hashlib,json,re
import numpy as np
import sklearn


def normalized_source_message(message):
    # Python hash randomization changes the display order of valid-choice sets.
    # Keep all message text while comparing those unordered values canonically.
    def choices(match):
        try:
            values=ast.literal_eval(match.group(0))
        except (ValueError,SyntaxError):
            return match.group(0)
        return repr(sorted(values)) if isinstance(values,set) else match.group(0)
    return re.sub(r"\{[^{}]*\}",choices,message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checked',type=Path,required=True);parser.add_argument('--optimized',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    profiles=[json.loads(p.read_text()) for p in [args.checked,args.optimized]]
    assert [p['profile'] for p in profiles]==['checked','optimized']
    for key in ['source_sha256','archive_sha256','compiler_sha256']:assert profiles[0][key]==profiles[1][key],key
    for path,digest in profiles[0]['source_sha256'].items():assert sha(path)==digest,path
    reports=[]
    for profile in profiles:
        assert all(p['passed'] for p in profile['native'])
        evidence={}
        for key,item in profile['evidence'].items():
            if isinstance(item,str):continue
            assert sha(item['report'])==item['sha256'],key
            data=json.loads(Path(item['report']).read_text())
            assert all(c['passed'] for c in data.get('checks',[])),key
            assert not data.get('gaps',[]) and data.get('passed',True) and data.get('strict_parity',True),key
            for field in ['hashes','source_sha256']:
                for path,digest in data.get(field,{}).items():
                    if Path(path).suffix.lower()!='.exe':
                        target=Path(path)
                        if not target.exists():target=Path(sklearn.__file__).parent/target
                        assert sha(target)==digest,(key,path)
            evidence[key]=data
        reports.append(evidence)
    assert reports[0].keys()==reports[1].keys()
    summary={}
    for key,left in reports[0].items():
        right=reports[1][key]
        for field in ['cases','expected','fixtures','stages','rejections','nonfinite_reference_values','source_sha256']:
            if field in left:assert left[field]==right[field],(key,field)
        assert len(left.get('checks',[]))==len(right.get('checks',[]))
        for a,b in zip(left.get('checks',[]),right.get('checks',[])):
            for field in ['case','step','fixture','stage','field','passed']:
                if field in a:assert a[field]==b[field],(key,field)
        if 'actual' in left:
            assert len(left['actual'])==len(right['actual'])==len(left['cases'])
            for case,(a,b) in enumerate(zip(left['actual'],right['actual'])):
                if isinstance(a,dict):a=[a];b=[b];expected=[left['expected'][case]]
                else:expected=left['expected'][case]
                assert len(a)==len(b)==len(expected)
                for x,y in zip(a,b):
                    assert x.keys()==y.keys()
                    for field in x:
                        exact=field in ['mask','iterations','t','classes','features','preserved'] or (field=='prediction' and left['cases'][case]['family']!=0)
                        if exact:np.testing.assert_array_equal(x[field],y[field],err_msg=key+':'+field)
                        else:np.testing.assert_allclose(x[field],y[field],atol=3e-9,rtol=3e-9,err_msg=key+':'+field)
        if 'guards' in left:
            assert len(left['guards'])==len(right['guards'])
            for a,b in zip(left['guards'],right['guards']):
                assert (a['mode'],a.get('oracle','source'),normalized_source_message(a['source']),a['passed'])==(b['mode'],b.get('oracle','source'),normalized_source_message(b['source']),b['passed'])
        summary[key]=dict(groups=len(left.get('checks',[])),fixtures=left.get('fixtures',len(left.get('cases',[]))),
                          diagnostics=len(left.get('guards',left.get('rejections',[]))))
    result=dict(selected_source_output_parity=True,cross_profile_outputs=True,project_strict_source_parity=False,
                archive_sha256=profiles[0]['archive_sha256'],evidence=summary,native_programs=[len(p['native']) for p in profiles],
                input_sha256={str(p):sha(p) for p in [args.checked,args.optimized,Path(__file__)]})
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

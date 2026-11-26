"""Inject truncated native batches into the four neighbor reference readers.

Only subprocess output is injected. Small real sklearn fixtures still populate
the reference expectations; this checks failure reporting, not numeric parity.
"""
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
import importlib,itertools,json,subprocess,sys,tempfile


def main():
    checks=[]
    names=['compare_functional_neighbor_graphs','compare_neighbor_outputs',
           'compare_neighbor_classification_outputs','compare_radius_classification_outputs']
    with tempfile.TemporaryDirectory(prefix='fsklearn-frame-reader-') as directory:
        for name in names:
            module=importlib.import_module(name)
            fixture_name='graph_cases' if name==names[0] else 'fixtures'
            fixtures=list(itertools.islice(getattr(module,fixture_name)(True),3))
            width=len(module.FIELDS)+1
            for mode in ['partial_failure','partial_success','full_failure']:
                report=Path(directory)/(name+'_'+mode+'.json')
                frames=3 if mode=='full_failure' else 1
                output=''.join(f'CASE {j+1}\n'+'0\n'*(width-1) for j in range(frames))
                if frames==1:output+='CASE 2\n0\n'
                returncode=0 if mode=='partial_success' else 3
                def native(args,**kwargs):
                    assert len(args)==1 and Path(args[0])==Path(__file__).resolve()
                    kwargs['stdout'].write(output)
                    return subprocess.CompletedProcess(args,returncode,stderr='SIGSEGV injected failure')
                argv=[name,'--executable',str(Path(__file__).resolve()),'--report',str(report),'--smoke']
                with patch.object(module,fixture_name,return_value=iter(fixtures)), \
                     patch.object(sys,'argv',argv),patch.object(subprocess,'run',side_effect=native):
                    sort_patch=patch.object(module,'sort_cases',return_value=iter([])) if name==names[0] else nullcontext()
                    with sort_patch:
                        try:module.main()
                        except (AssertionError,RuntimeError) as error:
                            if mode=='partial_failure':assert "'index': 1" in str(error),str(error)
                            if mode=='full_failure':assert 'native failure after full output' in str(error),str(error)
                        else:raise AssertionError('reader accepted an injected incomplete/failing batch')
                artifact=report.with_name(report.stem+'_batch_0')
                metadata=json.loads(Path(str(artifact)+'.json').read_text())
                assert metadata['requested']==3 and metadata['complete_frames']==frames
                assert metadata['returncode']==returncode and metadata['lines']==len(output.splitlines())
                assert metadata['case_headers']==(3 if frames==3 else 2)
                assert Path(str(artifact)+'.stdout').read_text()==output
                assert Path(str(artifact)+'.stderr').read_text()=='SIGSEGV injected failure'
                assert Path(str(artifact)+'.in').read_text()
                assert not report.exists()
                checks.append(dict(reader=name,mode=mode,passed=True))
    print(json.dumps(dict(checks=checks,count=len(checks)),indent=2))


if __name__=='__main__':main()

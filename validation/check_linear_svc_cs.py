"""Crammer-Singer control rejections and bounded sparse memory."""
from pathlib import Path
import argparse,hashlib,json,subprocess
from sklearn.svm import LinearSVC
from check_minibatch_sparse_memory import peak_memory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    before=sha(args.executable);guards=[]
    for name,options in [('multi_class',dict(multi_class='invalid')),('inactive_binary',dict(multi_class='crammer_singer',sample_weight=[0.,1.]))]:
        try:
            options=dict(options);weights=options.pop('sample_weight',None)
            LinearSVC(**options).fit([[-1.],[1.]],[-3,9],sample_weight=weights)
        except (ValueError,TypeError,IndexError) as error:source=type(error).__name__
        else:raise AssertionError(('source accepted',name))
        run=subprocess.run([str(args.executable),name],capture_output=True,text=True);message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(name,message)
        assert not any(s in message for s in ['SIGFPE','Fortran runtime error']),(name,message)
        guards.append(dict(name=name,source=source,passed=True))
    process=subprocess.Popen([str(args.executable),'memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate();peak=peak_memory(process)
    assert process.returncode==0 and 'Large Crammer-Singer memory case: PASS' in stdout,(stdout,stderr)
    assert 0<peak<128*1024**2,peak
    assert before==sha(args.executable)
    result=dict(passed=True,guards=guards,peak_working_set_bytes=peak,limit_bytes=128*1024**2,
        dense_design_bytes=128*200000*8,executable_sha256=before)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(len(guards),'guards; peak memory',peak,'bytes PASS')


if __name__=='__main__':main()

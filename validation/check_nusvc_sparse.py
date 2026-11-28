"""CSR NuSVC guards and bounded feature-memory checks."""
from pathlib import Path
import argparse,hashlib,json,subprocess
from check_minibatch_sparse_memory import peak_memory
GUARDS=['unallocated','pointer','column','nan','empty','zero_width','target_count','weight_count','kernel','precomputed','unfitted','dense_fit_query','query_width','query_nan']

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before=sha(args.executable);guards=[]
    for mode in GUARDS:
        run=subprocess.run([str(args.executable),mode],capture_output=True,text=True);message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'Fortran runtime error' not in message and 'SIGFPE' not in message,(mode,message)
        guards.append(dict(mode=mode,passed=True))
    process=subprocess.Popen([str(args.executable),'memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate();peak=peak_memory(process)
    assert process.returncode==0 and 'Large CSR NuSVC memory case: PASS' in stdout,(stdout,stderr)
    assert 0<peak<128*1024**2,peak
    assert before==sha(args.executable)
    result=dict(passed=True,guards=guards,peak_working_set_bytes=peak,limit_bytes=128*1024**2,
        dense_design_bytes=128*200000*8,executable_sha256=before)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(len(guards),'guards; peak memory',peak,'bytes PASS')

if __name__=='__main__':main()

"""NuSVC native guard diagnostics."""
from pathlib import Path
import argparse,hashlib,json,subprocess
GUARDS=['nu_zero','nu_high','nu_nan','tol_zero','cap_zero','kernel','weight_count','weight_negative','weight_nan',
        'weight_zero','missing_class','infeasible','constant','endpoint','fit_nan','unfitted','query_width','query_nan']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    before=sha(args.executable);records=[]
    for mode in GUARDS:
        run=subprocess.run([str(args.executable),mode],capture_output=True,text=True);message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'Fortran runtime error' not in message and 'SIGFPE' not in message,(mode,message)
        records.append(dict(mode=mode,passed=True))
    assert before==sha(args.executable);args.report.write_text(json.dumps(dict(passed=True,guards=records,executable_sha256=before),indent=2)+'\n')
    print(len(records),'NuSVC guards PASS')

if __name__=='__main__':main()

"""Isolated Ward tree guard checks; includes native typed/domain contracts."""
import argparse, subprocess
parser=argparse.ArgumentParser()
parser.add_argument('--executable',required=True)
args=parser.parse_args()
for case in range(1,16):
    result=subprocess.run([args.executable,str(case)],capture_output=True,text=True)
    assert result.returncode!=0,(case,result.stdout,result.stderr)
    assert 'ERROR STOP' in result.stderr and 'SIGFPE' not in result.stderr,(case,result.stderr)
print('15 Ward tree input guards passed')

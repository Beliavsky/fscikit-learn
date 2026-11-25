"""KernelRidge typed/domain checks, including source singular-branch failures."""
import argparse,subprocess
parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True)
args=parser.parse_args()
for case in range(1,31):
    result=subprocess.run([args.executable,str(case)],capture_output=True,text=True)
    assert result.returncode!=0,(case,result.stdout,result.stderr)
    assert 'ERROR STOP' in result.stderr and 'SIGFPE' not in result.stderr,(case,result.stderr)
print('30 extended KernelRidge input guards passed')

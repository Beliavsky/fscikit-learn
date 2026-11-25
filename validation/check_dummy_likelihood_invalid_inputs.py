import argparse,subprocess
ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);args=ap.parse_args()
for case in range(1,21):
 p=subprocess.run([args.executable,str(case)],text=True,capture_output=True)
 assert p.returncode and 'ERROR STOP'in p.stderr and 'SIGFPE'not in p.stderr,(case,p.stdout,p.stderr)
print('20 dummy-feature/likelihood input guards passed')

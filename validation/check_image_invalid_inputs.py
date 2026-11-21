"""Isolated rejection checks for image patch helpers."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_image import ROOT,FLAGS

CASES={
    'negative_size':'patch size must be positive','mixed_auto':'patch size must be positive',
    'negative_count':'max_patches must be nonnegative','negative_fraction':'patch_fraction must be in',
    'unit_fraction':'patch_fraction must be in','large_fraction':'patch_fraction must be in',
    'nan_fraction':'patch_fraction must be finite','inf_fraction':'patch_fraction must be finite',
    'both_limits':'choose max_patches or patch_fraction','negative_seed':'random_state must be nonnegative',
    'tall':'patch size exceeds','wide':'patch size exceeds','auto_zero':'automatic patch size is zero',
    'nan_image':'patch images must be finite','inf_image':'patch images must be finite',
    'empty_batch':'image dimensions must be nonempty','empty_height':'image dimensions must be nonempty',
    'empty_width':'image dimensions must be nonempty','zero_extract':'patch size must be positive',
    'bad_recon_image':'reconstruction image dimensions must be positive',
    'bad_recon_patch':'reconstruction image dimensions must be positive',
    'large_recon_patch':'invalid reconstruction patch size','recon_channels':'channel count mismatch',
    'zero_recon_patch':'invalid reconstruction patch size',
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        options=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        r=subprocess.run(['fpm','run',*options,'--target','image_invalid_driver','--runner'],cwd=ROOT,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
               if s.strip().strip('"').endswith(('image_invalid_driver','image_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout;exe=paths[0]
    checks=[]
    for case,message in CASES.items():
        r=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode!=0 and 'fscikit-learn:' in r.stdout and message in r.stdout,(case,r.stdout)
        checks.append(dict(name=case,expected=message,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results'/f'image_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated image invalid-input checks passed')


if __name__=='__main__':main()

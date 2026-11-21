"""Isolated rejection checks for structured feature maps and hash controls."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_feature_mapping import ROOT,FLAGS

CASES={
    'unfitted_transform':'DictVectorizer is not fitted','unfitted_names':'DictVectorizer is not fitted',
    'unfitted_index':'DictVectorizer is not fitted','unfitted_restrict':'DictVectorizer is not fitted',
    'unfitted_inverse':'DictVectorizer is not fitted','dict_empty_transform':'sample sequence must be nonempty',
    'dict_empty_fit_transform':'sample sequence must be nonempty','unallocated_key':'feature key is unallocated',
    'two_categories':'choose a category or category list','none_category':'None feature value cannot also be categorical',
    'none_list':'None feature value cannot also be categorical','unallocated_category':'feature category is unallocated',
    'dict_tokens':'mapping feature input must use items','hasher_zero_features':'n_features must be positive',
    'hasher_negative_features':'n_features must be positive','hasher_mode':'unknown input_type',
    'hasher_empty':'sample sequence must be nonempty','hasher_none':'does not accept None values',
    'hasher_list':'does not accept category lists','hasher_key':'feature key is unallocated',
    'hasher_tokens':'mapping feature input must use items','string_items':'string feature input must use tokens',
    'unallocated_token':'feature token is unallocated','negative_seed':'seed must fit uint32','large_seed':'seed must fit uint32',
    'mask_width':'support mask length mismatch','index_zero':'support index out of bounds',
    'index_large':'support index out of bounds','index_negative':'support index out of bounds',
    'index_duplicate':'support indices must be unique','inverse_empty_rows':'inverse matrix must be nonempty',
    'inverse_empty_columns':'inverse matrix must be nonempty','inverse_width':'inverse feature width mismatch',
    'inverse_nan':'inverse values must be finite','inverse_inf':'inverse values must be finite',
    'csr_inverse_nan':'inverse values must be finite','csr_inverse_inf':'inverse values must be finite',
    'csr_inverse_width':'inverse feature width mismatch','csr_inverse_empty':'inverse matrix must be nonempty',
    'csr_inverse_invalid':'CSR pointer endpoints are invalid',
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        options=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        r=subprocess.run(['fpm','run',*options,'--target','feature_mapping_invalid_driver','--runner'],cwd=ROOT,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
               if s.strip().strip('"').endswith(('feature_mapping_invalid_driver','feature_mapping_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout;exe=paths[0]
    checks=[]
    for case,message in CASES.items():
        r=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode!=0 and 'fscikit-learn:' in r.stdout and message in r.stdout,(case,r.stdout)
        checks.append(dict(name=case,expected=message,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results'/f'feature_mapping_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated structured-feature invalid-input checks passed')


if __name__=='__main__':main()

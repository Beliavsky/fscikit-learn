"""CSR LDA rejected-input contracts and a large sparse input memory check."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

from check_minibatch_sparse_memory import peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    binary = args.executable.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    records = []
    messages = dict(negative='LDA X must be nonnegative', nan='LDA X must be finite',
                    zero_rows='LDA X must have samples and features',
                    zero_cols='LDA X must have samples and features',
                    unfitted='LDA is not fitted', features='LDA feature count mismatch',
                    bad_pointer='CSR pointer endpoints are invalid',
                    bad_index='CSR column index is out of bounds',
                    changed_components='LDA n_components changed; use fit to reset')
    for mode in ['negative', 'nan', 'zero_rows', 'zero_cols', 'unfitted', 'features']:
        x = csr_matrix(np.ones((6, 4)))
        if mode == 'negative':
            x.data[0] = -1
        elif mode == 'nan':
            x.data[0] = np.nan
        elif mode == 'zero_rows':
            x = csr_matrix((0, 4))
        elif mode == 'zero_cols':
            x = csr_matrix((1, 0))
        source = LatentDirichletAllocation(n_components=3, max_iter=1)
        rejected = False
        try:
            if mode == 'unfitted':
                source.transform(x)
            elif mode == 'features':
                source.fit(x).transform(x[:, :3])
            else:
                source.fit(x)
        except ValueError:
            rejected = True
        process = subprocess.run([str(binary), mode], capture_output=True, text=True)
        records.append(dict(mode=mode, source_rejected=rejected,
                            native_returncode=process.returncode,
                            passed=rejected and process.returncode != 0
                            and messages[mode] in process.stdout+process.stderr))
    for mode in ['bad_pointer', 'bad_index', 'changed_components']:
        process = subprocess.run([str(binary), mode], capture_output=True, text=True)
        records.append(dict(mode=mode, native_returncode=process.returncode,
                            passed=process.returncode != 0
                            and messages[mode] in process.stdout+process.stderr,
                            oracle='native CSR/model invariant'))
    process = subprocess.Popen([str(binary), 'memory'], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    peak = peak_memory(process)
    memory = dict(peak_working_set_bytes=peak, limit_bytes=128*1024**2,
                  n_samples=2048, n_features=50000, n_components=3,
                  dense_design_bytes=2048*50000*8, returncode=process.returncode,
                  stdout=stdout, stderr=stderr,
                  passed=process.returncode == 0 and 0 < peak < 128*1024**2
                  and 'Large sparse LDA memory case: PASS' in stdout)
    assert digest == hashlib.sha256(binary.read_bytes()).hexdigest()
    passed = all(record['passed'] for record in records) and memory['passed']
    args.report.write_text(json.dumps(dict(passed=passed, guards=records, memory=memory,
                                          executable_sha256=digest), indent=2)+'\n')
    print('CSR LDA guards/memory:', passed, peak)
    raise SystemExit(not passed)


if __name__ == '__main__':
    main()

"""RBM CSR shape/value guards and large sparse training/inference memory."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neural_network import BernoulliRBM

from check_minibatch_sparse_memory import peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    binary = args.executable.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    messages = dict(nan='RBM CSR input must be finite', infinity='RBM CSR input must be finite',
                    zero_rows='RBM CSR input must have samples and features',
                    zero_cols='RBM CSR input must have samples and features',
                    unfitted='RBM is not fitted', unfitted_score='RBM is not fitted',
                    features='RBM feature count mismatch',
                    changed_components='RBM n_components changed; use fit to reset',
                    bad_pointer='CSR pointer endpoints are invalid',
                    bad_index='CSR column index is out of bounds')
    records = []
    for mode, message in messages.items():
        rejected = None
        if mode not in ['bad_pointer', 'bad_index']:
            x = csr_matrix(np.ones((13, 5)))
            if mode in ['nan', 'infinity']:
                x.data[0] = np.nan if mode == 'nan' else np.inf
            elif mode == 'zero_rows':
                x = csr_matrix((0, 5))
            elif mode == 'zero_cols':
                x = csr_matrix((1, 0))
            source = BernoulliRBM(n_components=3, batch_size=4, n_iter=3, random_state=7)
            rejected = False
            try:
                if mode == 'unfitted':
                    source.transform(x)
                elif mode == 'unfitted_score':
                    source.score_samples(x)
                elif mode == 'features':
                    source.fit(x).transform(x[:, :4])
                elif mode == 'changed_components':
                    source.fit(x).set_params(n_components=4).partial_fit(x)
                else:
                    source.fit(x)
            except ValueError:
                rejected = True
        native = subprocess.run([str(binary), mode], capture_output=True, text=True)
        expected = rejected is True or mode in ['bad_pointer', 'bad_index', 'changed_components']
        records.append(dict(mode=mode, source_rejected=rejected, native_returncode=native.returncode,
                            kind='native fitted-shape invariant' if mode == 'changed_components'
                            else 'native CSR invariant' if rejected is None else 'Source rejection',
                            passed=expected and native.returncode != 0
                            and message in native.stdout+native.stderr))
    process = subprocess.Popen([str(binary), 'memory'], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    peak = peak_memory(process)
    sizes = re.findall(r'dense_input_bytes\s+(\d+)', stdout)
    memory = dict(peak_working_set_bytes=peak, limit_bytes=128*1024**2,
                  n_samples=8192, n_features=100000, n_components=3,
                  dense_input_bytes=int(sizes[0]) if sizes else None,
                  returncode=process.returncode, stdout=stdout, stderr=stderr,
                  operations='zero-epoch fit, full-CSR partial update, transform, score, two-row dense Gibbs output',
                  passed=process.returncode == 0 and 0 < peak < 128*1024**2
                  and sizes == ['6553600000'] and 'Large sparse RBM memory case: PASS' in stdout)
    assert digest == hashlib.sha256(binary.read_bytes()).hexdigest()
    passed = all(item['passed'] for item in records) and memory['passed']
    args.report.write_text(json.dumps(dict(passed=passed, guards=records, memory=memory,
                                          executable_sha256=digest), indent=2)+'\n')
    print('CSR RBM guards/memory:', passed, peak)
    raise SystemExit(not passed)


if __name__ == '__main__':
    main()

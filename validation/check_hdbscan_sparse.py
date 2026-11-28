"""CSR HDBSCAN input rejection, native distance guards and sparse memory."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import HDBSCAN

from check_minibatch_sparse_memory import peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    binary = args.executable.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    records = []
    messages = dict(asymmetric='sparse precomputed values must be symmetric',
                    pattern='sparse precomputed structure must be symmetric',
                    few_neighbors='fewer than min_samples stored neighbors',
                    few_neighbors_max='fewer than min_samples stored neighbors',
                    disconnected='must be connected', nan='sparse distances must be finite',
                    infinity='sparse distances must be finite',
                    negative='precomputed distances must be nonnegative',
                    zero_mst='zero-weight MST cannot define a complete hierarchy',
                    single='requires at least two finite samples', shape='must be square',
                    min_samples='min_samples exceeds finite sample count',
                    centers='cannot store precomputed centers',
                    bad_pointer='CSR pointer endpoints are invalid',
                    bad_index='CSR column index is out of bounds')
    for mode in messages:
        source_rejected = None
        if mode not in ['bad_pointer', 'bad_index']:
            x = np.ones((12, 12))-np.eye(12)
            options = dict(metric='precomputed', min_samples=3, min_cluster_size=3)
            if mode == 'asymmetric':
                x[0, 1] += .1
            elif mode == 'pattern':
                x = np.array([[0., 1., 0.], [0., 0., 0.], [1., 0., 0.]])
            elif mode in ['few_neighbors','few_neighbors_max']:
                x = np.eye(3)
                if mode == 'few_neighbors_max': options['metric_params'] = dict(max_distance=10.)
            elif mode == 'disconnected':
                x = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.],
                              [0., 0., 0., 1.], [0., 0., 1., 0.]])
                options['min_samples'] = 1
            elif mode in ['nan', 'infinity']:
                x[0, 0] = np.nan if mode == 'nan' else np.inf
            elif mode == 'negative':
                x[:] = -1
            elif mode == 'zero_mst':
                x[:] = 0
                options['min_samples'] = 1
            elif mode == 'single':
                x = np.zeros((1, 1))
            elif mode == 'shape':
                x = np.ones((2, 3))
                options['min_samples'] = 1
            elif mode == 'min_samples':
                options['min_samples'] = 13
            elif mode == 'centers':
                options['store_centers'] = 'both'
            # Retain stored zeros: their structural presence affects sparse validation.
            matrix = csr_matrix((x.ravel(), np.tile(np.arange(x.shape[1]), x.shape[0]),
                                 np.arange(0, x.size+1, x.shape[1])), shape=x.shape)
            if mode in ['pattern', 'disconnected', 'few_neighbors','few_neighbors_max']:
                matrix.eliminate_zeros()
            source_rejected = False
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore')
                try:
                    HDBSCAN(**options).fit(matrix)
                except (ValueError, KeyError):
                    source_rejected = True
        native = subprocess.run([str(binary), mode], capture_output=True, text=True)
        expected = source_rejected is True or mode in ['negative', 'bad_pointer', 'bad_index']
        records.append(dict(mode=mode, source_rejected=source_rejected,
                            native_returncode=native.returncode,
                            kind='native nonnegative-distance correction' if mode == 'negative'
                            else 'native CSR invariant' if source_rejected is None else 'Source rejection',
                            passed=expected and native.returncode != 0
                            and messages[mode] in native.stdout+native.stderr))
    process = subprocess.Popen([str(binary), 'memory'], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    peak = peak_memory(process)
    sizes = re.findall(r'dense_distance_bytes\s+(\d+)', stdout)
    memory = dict(peak_working_set_bytes=peak, limit_bytes=128*1024**2,
                  n_samples=12000, dense_distance_bytes=int(sizes[0]) if sizes else None,
                  returncode=process.returncode, stdout=stdout, stderr=stderr,
                  passed=process.returncode == 0 and 0 < peak < 128*1024**2
                  and sizes == ['1152000000'] and 'Large sparse HDBSCAN memory case: PASS' in stdout)
    assert digest == hashlib.sha256(binary.read_bytes()).hexdigest()
    passed = all(item['passed'] for item in records) and memory['passed']
    args.report.write_text(json.dumps(dict(passed=passed, guards=records, memory=memory,
                                          executable_sha256=digest), indent=2)+'\n')
    print('CSR HDBSCAN guards/memory:', passed, peak)
    raise SystemExit(not passed)


if __name__ == '__main__':
    main()

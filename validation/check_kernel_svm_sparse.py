"""Source validation and bounded-memory checks for the native CSR kernel SVMs."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import scipy.sparse as sp
from sklearn.svm import SVR, NuSVR
from check_minibatch_sparse_memory import peak_memory


def reject_source(name):
    x = sp.csr_matrix((np.sin(np.arange(1, 25)) + .8).reshape(8, 3, order='F'))
    y = np.sin(x.toarray()[:, 0])
    if name == 'bad_index':
        x.indices[0] = -1
        x.check_format(full_check=True)
    elif name == 'bad_ptr':
        x.indptr[0] = -1
        x.check_format(full_check=True)
    elif name == 'unfitted':
        SVR().predict(x)
    elif name in ['nan', 'inf']:
        x.data[0] = np.nan if name == 'nan' else np.inf
        SVR().fit(x, y)
    elif name == 'shape':
        SVR().fit(x, y).predict(sp.csr_matrix((8, 4)))
    elif name == 'precomputed':
        SVR(kernel='precomputed').fit(x, y)
    elif name == 'dense_fit_csr_query':
        SVR().fit(x.toarray(), y).predict(x)
    elif name == 'bad_y':
        SVR().fit(x, y[:7])
    elif name == 'bad_nu':
        NuSVR(nu=0).fit(x, y)
    elif name == 'bad_epsilon':
        SVR(epsilon=-1).fit(x, y)
    else:
        raise AssertionError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    before = sha(args.executable)
    guards = []
    for name in ['unfitted', 'nan', 'inf', 'shape', 'precomputed', 'dense_fit_csr_query',
                 'bad_index', 'bad_ptr', 'negative_weight', 'bad_y', 'bad_nu', 'bad_epsilon']:
        if name == 'negative_weight':
            source = 'existing_native_nonnegative_weight_policy'
        else:
            try:
                reject_source(name)
            except (ValueError, TypeError) as error:
                source = type(error).__name__
            else:
                raise AssertionError(('source accepted', name))
        run = subprocess.run([str(args.executable), name], capture_output=True, text=True)
        message = run.stdout + run.stderr
        assert run.returncode and 'fscikit-learn:' in message, (name, message)
        assert not any(s in message for s in ['SIGFPE', 'Fortran runtime error']), (name, message)
        guards.append(dict(name=name, source=source, passed=True))
    memory = []
    for mode in ['memory', 'memory_linear']:
        process = subprocess.Popen([str(args.executable), mode], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        peak = peak_memory(process)
        assert process.returncode == 0 and 'Large CSR kernel SVM memory case: PASS' in stdout, (stdout, stderr)
        assert 0 < peak < 128 * 1024**2, (mode, peak)
        memory.append(dict(mode=mode, peak_working_set_bytes=peak, passed=True))
    peak = max(item['peak_working_set_bytes'] for item in memory)
    assert sha(args.executable) == before
    result = dict(passed=True, guards=guards, memory=memory, peak_working_set_bytes=peak,
                  limit_bytes=128 * 1024**2, dense_design_bytes=64 * 200000000 * 8,
                  dense_kernel_bytes=64 * 64 * 8, executable_sha256=before)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(len(guards), 'guards and peak memory', peak, 'bytes PASS')


if __name__ == '__main__':
    main()

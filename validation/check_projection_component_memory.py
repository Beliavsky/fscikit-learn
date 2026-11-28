"""Bound peak memory for fitting and querying CSR projection components."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from check_minibatch_sparse_memory import peak_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    before = digest(args.executable)
    process = subprocess.Popen([str(args.executable), 'memory'], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    peak = peak_memory(process)
    limit = 128 * 1024**2
    passed = process.returncode == 0 and 0 < peak < limit and \
        'Large sparse projection components memory case: PASS' in stdout
    assert digest(args.executable) == before
    result = dict(passed=passed, peak_working_set_bytes=peak, limit_bytes=limit,
                  n_samples=4096, n_features=2000000, n_components=1024,
                  dense_component_bytes=1024 * 2000000 * 8,
                  dense_input_bytes=4096 * 2000000 * 8,
                  returncode=process.returncode, stdout=stdout, stderr=stderr,
                  executable_sha256=before)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print('CSR projection components peak memory:', peak, 'bytes;', 'PASS' if passed else 'FAIL')
    raise SystemExit(not passed)


if __name__ == '__main__':
    main()

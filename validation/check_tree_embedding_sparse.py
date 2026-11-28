"""RandomTreesEmbedding control guards and sparse output peak-memory check."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

import numpy as np
from sklearn.ensemble import RandomTreesEmbedding

from check_minibatch_sparse_memory import peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    binary = args.executable.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    guards = []
    controls = dict(leaf_budget=dict(max_leaf_nodes=1), negative_budget=dict(max_leaf_nodes=-1),
                    weight_low=dict(min_weight_fraction_leaf=-.1),
                    weight_high=dict(min_weight_fraction_leaf=.51),
                    weight_nan=dict(min_weight_fraction_leaf=np.nan),
                    leaf_zero=dict(min_samples_leaf=0.), leaf_one=dict(min_samples_leaf=1.),
                    split_zero=dict(min_samples_split=0.), split_high=dict(min_samples_split=1.1),
                    unfitted={}, unfitted_path={}, features={}, infinite_path={})
    for mode, settings in controls.items():
        source = RandomTreesEmbedding(n_estimators=5, max_depth=4, random_state=7, **settings)
        x = np.arange(180, dtype=float).reshape(60, 3)
        rejected = False
        try:
            if mode == 'unfitted':
                source.transform(x)
            elif mode == 'unfitted_path':
                source.decision_path(x)
            elif mode == 'features':
                source.fit(x).transform(x[:, :2])
            elif mode == 'infinite_path':
                source.fit(x)
                x[0, 0] = np.inf
                source.decision_path(x)
            else:
                source.fit(x)
        except ValueError:
            rejected = True
        native = subprocess.run([str(binary), mode], capture_output=True, text=True)
        guards.append(dict(mode=mode, source_rejected=rejected, native_returncode=native.returncode,
                           passed=rejected and native.returncode != 0
                           and 'ERROR STOP' in native.stdout+native.stderr))
    process = subprocess.Popen([str(binary), 'memory'], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    peak = peak_memory(process)
    sizes = {name: int(value) for name, value in re.findall(r'(dense_\w+_bytes)\s+(\d+)', stdout)}
    memory = dict(peak_working_set_bytes=peak, limit_bytes=128*1024**2,
                  n_samples=2048, n_features=3, n_estimators=16, dense_equivalent_bytes=sizes,
                  returncode=process.returncode, stdout=stdout, stderr=stderr,
                  passed=process.returncode == 0 and 0 < peak < 128*1024**2
                  and len(sizes) == 2 and min(sizes.values()) > 256*1024**2
                  and 'Large sparse tree embedding memory case: PASS' in stdout)
    assert digest == hashlib.sha256(binary.read_bytes()).hexdigest()
    passed = all(item['passed'] for item in guards) and memory['passed']
    args.report.write_text(json.dumps(dict(passed=passed, guards=guards, memory=memory,
                                          executable_sha256=digest), indent=2)+'\n')
    print('Tree embedding CSR guards/memory:', passed, peak, sizes)
    raise SystemExit(not passed)


if __name__ == '__main__':
    main()

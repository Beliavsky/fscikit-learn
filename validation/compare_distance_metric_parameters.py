"""Probe variance/precision edge cases in isolated sklearn processes.

Source's reserved Cython reduced return -1 can crash the process; retain its
exit status and output, and check our controlled error separately. Crashes
never count as numerical parity. Every defined result uses the usual 2e-12
tolerance, including repeated calls to the stateful sparse Mahalanobis metric.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import warnings

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np


def inputs(p):
    x = np.vstack([np.eye(p), np.zeros(p)])
    y = np.zeros((2, p))
    y[1, :2] = [-1, 2]
    return x, y


def source(case):
    if os.name == "nt":
        # Report the child's crash status without a Windows Error Reporting UI.
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x8003)
    from scipy.sparse import csr_matrix
    from sklearn.metrics import DistanceMetric
    parameter = np.asarray(case["parameter"])
    x, y = inputs(parameter.shape[0])
    if case["storage"] & 1:
        x = csr_matrix(x)
    if case["storage"] & 2:
        y = csr_matrix(y)
    print("START", flush=True)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            key = "V" if case["params"] in [2, 4] else "VI"
            obj = DistanceMetric.get_metric(case["metric"], **{key: parameter})
            result = dict(values=obj.pairwise(x, None if case["own"] else y).ravel().tolist(),
                          repeat_values=obj.pairwise(x, None if case["own"] else y).ravel().tolist())
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            result = dict(error_type=type(error).__name__, error=str(error))
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-case", type=Path)
    ap.add_argument("--source-batch", type=Path)
    ap.add_argument("--executable", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    if args.source_case:
        source(json.loads(args.source_case.read_text()))
        return
    if args.source_batch:
        for case in json.loads(args.source_batch.read_text()):
            source(case)
        return
    if not args.executable or not args.report:
        ap.error("--executable and --report are required")
    folder = args.report.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    specs = [("seuclidean", 2, np.array(v)) for v in
             [[0., 1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.], [1e-300, 1e300]]]
    matrices = [np.array(a) for a in [
        [[1., 2.], [3., 4.]], [[-1., 0.], [0., 1.]],
        [[np.nan, 0.], [0., 1.]], [[np.inf, 0.], [0., 1.]],
        [[1., 0.], [0., 1e-100]], [[1., 1.], [1., 1. + 1e-12]],
        [[1., 1.], [1., 1.]]]]
    rng = np.random.RandomState(5831)
    for p in [3, 4, 7, 12]:
        a = rng.normal(size=(p, p))
        matrices.append(a @ a.T + np.eye(p))
    specs += [("mahalanobis", kind, a) for a in matrices for kind in [3, 4]]
    cases = []
    for index, (metric, kind, parameter) in enumerate(specs):
        for storage in range(4):
            for own in [False, True]:
                case = dict(index=index, metric=metric, params=kind,
                            parameter=parameter.tolist(), storage=storage, own=own)
                path = folder / f"case_{len(cases):03d}.json"
                path.write_text(json.dumps(case) + "\n")
                cases.append((case, path))

    def run(pair, src=None, source_batch=None):
        case, path = pair
        if src is None:
            src = subprocess.run([sys.executable, "-u", __file__, "--source-case", str(path)],
                                 capture_output=True, text=True, timeout=60)
        path.with_suffix(".source.stdout").write_text(src.stdout)
        path.with_suffix(".source.stderr").write_text(src.stderr)
        parameter = np.asarray(case["parameter"])
        x, y = inputs(parameter.shape[0])
        text = (f'{len(x)} {x.shape[1]} {len(y)} {case["storage"]} {int(case["own"])} '
                f'{case["metric"]} 2 {case["params"]}\n')
        for a in [x, y, np.atleast_2d(parameter)]:
            text += "\n".join(" ".join(map(str, row)) for row in a) + "\n"
        native = subprocess.run([str(args.executable)], input=text,
                                capture_output=True, text=True, timeout=60)
        path.with_suffix(".native.in").write_text(text)
        path.with_suffix(".native.stdout").write_text(native.stdout)
        path.with_suffix(".native.stderr").write_text(native.stderr)
        out = native.stdout + native.stderr
        differences = []
        expected = None
        if src.returncode:
            classification = "source_process_failure"
            if not (native.returncode and "reserved Cython reduced error value -1" in out):
                differences.append("Source process failure lacks controlled native adapter")
        else:
            expected = json.loads(src.stdout.splitlines()[-1])
            classification = "source_rejection" if "error_type" in expected else "source_success"
            if classification == "source_rejection":
                if not native.returncode or "fscikit-learn:" not in out:
                    differences.append("Source rejection not matched")
            elif native.returncode or len(native.stdout.splitlines()) != 7:
                differences.append("native failure")
            else:
                for label, line, want in zip(["first", "repeat"], native.stdout.splitlines()[1:4:2],
                                             [expected["values"], expected["repeat_values"]]):
                    actual = np.fromstring(line, sep=" ")
                    try:
                        np.testing.assert_allclose(actual, want, rtol=2e-12, atol=2e-12, equal_nan=True)
                    except AssertionError:
                        differences.append(dict(field=label, actual=actual.tolist(), source=want))
        if any(t in out for t in ["SIGFPE", "SIGSEGV", "Fortran runtime error"]):
            differences.append("native runtime failure")
        return dict(case=case, classification=classification, source_returncode=src.returncode,
                    native_returncode=native.returncode, expected=expected,
                    differences=differences, artifact_prefix=str(path.with_suffix("")),
                    source_batch=source_batch)

    def run_group(group):
        # Amortize imports across independent metric instances. Any crashing
        # batch is retried one case per process, preserving each crash status.
        batch = folder / f'batch_{group[0][0]["index"]:03d}.json'
        batch.write_text(json.dumps([case for case, path in group]) + "\n")
        result = subprocess.run([sys.executable, "-u", __file__, "--source-batch", str(batch)],
                                capture_output=True, text=True, timeout=60)
        batch.with_suffix(".source.stdout").write_text(result.stdout)
        batch.with_suffix(".source.stderr").write_text(result.stderr)
        if result.returncode:
            return [run(pair) for pair in group]
        values = [line for line in result.stdout.splitlines() if line.startswith("{")]
        assert len(values) == len(group), (batch, result.stdout)
        return [run(pair, subprocess.CompletedProcess(result.args, 0, "START\n" + value + "\n", ""),
                    str(batch)) for pair, value in zip(group, values)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(run_group, [cases[i:i+8] for i in range(0, len(cases), 8)]))
        records = [record for group in groups for record in group]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = dict(configurations=len(records), records=records,
                  failed_configurations=sum(bool(c["differences"]) for c in records),
                  source_process_failures=sum(c["classification"] == "source_process_failure" for c in records),
                  strict_parity=not any(c["classification"] == "source_process_failure" for c in records),
                  sha256={str(p): sha(p) for p in [Path(__file__), args.executable,
                                                 Path("src/sklearn_distance_metrics.f90")]})
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k not in ["records", "sha256"]})
    assert report["failed_configurations"] == 0


if __name__ == "__main__":
    main()

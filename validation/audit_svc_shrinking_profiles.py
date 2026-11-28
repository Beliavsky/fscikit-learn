"""Require complete SVC output parity and compare frozen profile evidence."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checked', type=Path, required=True)
    parser.add_argument('--optimized', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    profiles = [json.loads(path.read_text()) for path in [args.checked, args.optimized]]
    assert [p['profile'] for p in profiles] == ['checked', 'optimized']
    assert profiles[0]['archive_sha256'] == profiles[1]['archive_sha256']
    assert profiles[0]['compiler_sha256'] == profiles[1]['compiler_sha256']
    assert profiles[0]['source_sha256'] == profiles[1]['source_sha256']
    for path, digest in profiles[0]['source_sha256'].items():
        assert sha(path) == digest, path
    evidence = []
    for profile in profiles:
        assert all(item['passed'] for item in profile['native'])
        reports = {}
        for key, item in profile['evidence'].items():
            if isinstance(item, str):  # Optimized object digests accompany report records.
                continue
            assert sha(item['report']) == item['sha256'], key
            data = json.loads(Path(item['report']).read_text())
            assert all(check['passed'] for check in data.get('checks', [])), key
            assert not data.get('gaps', []), key  # No retained asymmetric exceptions.
            assert data.get('passed', True), key
            assert data.get('strict_parity', True), key
            reports[key] = data
        evidence.append(reports)
    summary = {}
    for key, checked in evidence[0].items():
        optimized = evidence[1][key]
        for field in ['cases', 'expected', 'source_iterations', 'shrinking']:
            if field in checked:
                assert checked[field] == optimized[field], (key, field)
        if 'source_rejections' in checked:
            def rejection_outcomes(data):
                return [(r['original_case'],r['case'],r['source'],r['passed']) for r in data['source_rejections']]
            assert rejection_outcomes(checked)==rejection_outcomes(optimized),key
            assert all(r['passed'] and r['native_returncode'] for r in checked['source_rejections']),key
        if 'actual' in checked:
            assert len(checked['actual']) == len(optimized['actual'])
            for left, right in zip(checked['actual'], optimized['actual']):
                assert left.keys() == right.keys()
                for field in left:
                    np.testing.assert_allclose(left[field], right[field], atol=3e-6, rtol=3e-6,
                                               err_msg=key + ':' + field)
        summary[key] = dict(groups=len(checked.get('checks', [])),
                            fits=len(checked.get('cases', [])),
                            iteration_disagreements=len(checked.get('iteration_disagreements', [])))
    result = dict(selected_source_output_parity=True, cross_profile_outputs=True,
                  archive_sha256=profiles[0]['archive_sha256'], evidence=summary,
                  native_programs=[len(p['native']) for p in profiles],
                  input_sha256={str(p): sha(p) for p in [args.checked, args.optimized, Path(__file__)]})
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

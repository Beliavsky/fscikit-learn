"""Require selected dense/CSR PA Source parity and compare frozen profile evidence."""
from pathlib import Path
import argparse, ast, hashlib, json, re
import numpy as np


def normalized_source_message(message):
    # Python hash randomization changes the display order of valid-choice sets.
    # Keep all message text while comparing those unordered values canonically.
    def choices(match):
        try:
            values=ast.literal_eval(match.group(0))
        except (ValueError,SyntaxError):
            return match.group(0)
        return repr(sorted(values)) if isinstance(values,set) else match.group(0)
    return re.sub(r"\{[^{}]*\}",choices,message)


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
            assert not data.get('gaps', []), key
            assert data.get('passed', True), key
            assert all(g['passed'] for g in data.get('guards',[])),key
            assert data.get('strict_parity', True), key
            reports[key] = data
        evidence.append(reports)
    summary = {}
    for key, checked in evidence[0].items():
        optimized = evidence[1][key]
        for field in ['cases', 'expected', 'source_iterations', 'shrinking']:
            if field in checked:
                assert checked[field] == optimized[field], (key, field)
        if 'guards' in checked:
            outcomes=lambda d:[(g['mode'],g['oracle'],normalized_source_message(g['source']),g['passed']) for g in d['guards']]
            assert outcomes(checked)==outcomes(optimized),key
        if 'actual' in checked:
            assert len(checked['actual']) == len(optimized['actual'])
            for left, right in zip(checked['actual'], optimized['actual']):
                assert left.keys() == right.keys()
                for field in left:
                    np.testing.assert_allclose(left[field], right[field], atol=3e-8, rtol=3e-8,
                                               err_msg=key + ':' + field)
        summary[key] = dict(groups=len(checked.get('checks', [])),
                            fits=len(checked.get('cases', [])),
                            guards=len(checked.get('guards',[])),
                            iteration_disagreements=len(checked.get('iteration_disagreements', [])))
    result = dict(selected_source_output_parity=True, cross_profile_outputs=True,project_strict_source_parity=False,
                  archive_sha256=profiles[0]['archive_sha256'], evidence=summary,
                  native_programs=[len(p['native']) for p in profiles],
                  input_sha256={str(p): sha(p) for p in [args.checked, args.optimized, Path(__file__)]})
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

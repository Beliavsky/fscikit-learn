"""Audit seeded projection sampling with checked or optimized library code.

Run only after fpm has finished. The optimized audit keeps checked harnesses
and substitutes only sklearn_random_projection in a frozen checked archive.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=['checked', 'optimized'], required=True)
    parser.add_argument('--all-native', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    objects = args.build_dir / 'gfortran.exe_F638FECF362AB298'
    binaries = args.build_dir / 'gfortran.exe_E1FE13DB8881E18E'
    archive = objects / 'fscikit-learn/libfscikit-learn.a'
    compiler = Path('C:/Programs/Octave-11.1.0/mingw64/bin/gfortran.exe')
    shim = Path(tempfile.gettempdir()) / 'fsklearn-locale-probe/windows_locale_copy.o'
    os.environ['PATH'] = str(compiler.parent) + os.pathsep + os.environ['PATH']
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    archive_hash = digest(archive)
    files = [Path(__file__), *[root / name for name in [
        'src/sklearn_random_projection.f90', 'test/test_seeded_projections.f90',
        'test/test_sparse_projection_components.f90', 'app/reference_seeded_projection_driver.f90',
        'app/reference_csr_projection_driver.f90', 'validation/compare_seeded_projections.py',
        'validation/compare_csr_projections.py', 'validation/check_projection_component_memory.py']]]
    source_hashes = {str(p): digest(p) for p in files}
    selected = ['test_seeded_projections', 'test_csr_projections', 'test_factor_projection',
                'test_sparse_projection_components']
    native = sorted(p for p in (root / 'test').glob('*.f90')
                    if re.search(r'^\s*program\s', p.read_text(), re.I | re.M))
    programs = native if args.all_native else [root / 'test' / (n + '.f90') for n in selected]
    drivers = [root / 'app' / (n + '.f90') for n in
               ['reference_seeded_projection_driver', 'reference_csr_projection_driver']]
    evidence = {}
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='fsklearn-projection-sampling-') as temporary:
        work = Path(temporary)
        if args.profile == 'optimized':
            frozen = work / 'checked.a'
            shutil.copyfile(archive, frozen)
            assert digest(frozen) == archive_hash == digest(archive)
            common = ['-ffree-line-length-none', '-J' + str(work), '-I' + str(work), '-I' + str(objects)]
            module = work / 'projection.o'
            subprocess.run([str(compiler), '-O3', '-funroll-loops', *common, '-c',
                            str(root / 'src/sklearn_random_projection.f90'), '-o', str(module)], check=True)
            evidence['optimized_object_sha256'] = digest(module)
            for p in programs + drivers:
                subprocess.run([str(compiler), '-fcheck=all', '-fbacktrace',
                                '-ffpe-trap=invalid,zero,overflow', *common, str(p),
                                str(module), str(frozen), '-static-libgfortran', str(shim),
                                '-Wl,--wrap=setlocale', '-o', str(work / (p.stem + '.exe'))], check=True)

        def executable(p):
            return work / (p.stem + '.exe') if args.profile == 'optimized' else binaries / p.relative_to(root).with_suffix('.exe')

        def run_native(p):
            exe = executable(p)
            before = digest(exe)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            assert digest(exe) == before
            assert run.returncode == 0, (p, run.stdout[-1500:], run.stderr[-1500:])
            print(p.stem, 'PASS', flush=True)
            return dict(program=p.stem, binary_sha256=before, exit_code=0)

        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(run_native, programs))
        for driver, script in zip(drivers, ['compare_seeded_projections.py', 'compare_csr_projections.py']):
            for sparse_components in [False, True]:
                key = script.removeprefix('compare_').removesuffix('.py') + ('_csr_components' if sparse_components else '')
                report = args.report.with_name(args.report.stem + '_' + key + '.json')
                subprocess.run([sys.executable, str(root / 'validation' / script), '--executable',
                                str(executable(driver)), '--profile', args.profile, '--report', str(report),
                                *(['--sparse-components'] if sparse_components else [])], check=True)
                evidence[key] = dict(report=str(report), sha256=digest(report))
        report = args.report.with_name(args.report.stem + '_memory.json')
        subprocess.run([sys.executable, str(root / 'validation/check_projection_component_memory.py'),
                        '--executable', str(executable(root / 'test/test_sparse_projection_components.f90')),
                        '--report', str(report)], check=True)
        evidence['memory'] = dict(report=str(report), sha256=digest(report))
    assert digest(archive) == archive_hash
    assert source_hashes == {str(p): digest(p) for p in files}
    result = dict(profile=args.profile, native=outcomes, evidence=evidence,
                  archive_sha256=archive_hash, compiler_sha256=digest(compiler),
                  source_sha256=source_hashes)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(args.profile, len(outcomes), 'native programs, four reference runs and memory bound PASS')


if __name__ == '__main__':
    main()

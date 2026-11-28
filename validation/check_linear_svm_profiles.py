"""Checked/O3 primal linear SVM audits, retaining extended Source differences."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--profile',choices=['checked','optimized'],required=True)
    parser.add_argument('--all-native',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    objects=args.build_dir/'gfortran.exe_F638FECF362AB298'
    binaries=args.build_dir/'gfortran.exe_E1FE13DB8881E18E'
    archive=objects/'fscikit-learn/libfscikit-learn.a'
    compiler=Path('C:/Programs/Octave-11.1.0/mingw64/bin/gfortran.exe')
    shim=Path(tempfile.gettempdir())/'fsklearn-locale-probe/windows_locale_copy.o'
    os.environ['PATH']=str(compiler.parent)+os.pathsep+os.environ['PATH']
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();archive_hash=sha(archive)
    files=[Path(__file__),*[root/name for name in ['src/sklearn_linear_svm.f90',
        'test/test_linear_svm.f90','test/test_linear_svm_sparse.f90','app/reference_linear_svm_driver.f90',
        'validation/compare_linear_svm.py','validation/check_linear_svm_sparse.py']]]
    hashes={str(p):sha(p) for p in files}
    selected=[root/'test'/(name+'.f90') for name in ['test_linear_svm','test_linear_svm_sparse']]
    programs=sorted(p for p in (root/'test').glob('*.f90')
        if re.search(r'^\s*program\s',p.read_text(),re.I|re.M)) if args.all_native else selected
    driver=root/'app/reference_linear_svm_driver.f90';evidence={}
    with tempfile.TemporaryDirectory(prefix='fsklearn-linear-svm-audit-') as temporary:
        work=Path(temporary)
        if args.profile=='optimized':
            frozen=work/'checked.a';shutil.copyfile(archive,frozen)
            assert sha(frozen)==archive_hash==sha(archive)
            common=['-ffree-line-length-none','-J'+str(work),'-I'+str(work),'-I'+str(objects)]
            module=work/'linear_svm.o'
            subprocess.run([str(compiler),'-O3','-funroll-loops',*common,'-c',
                str(root/'src/sklearn_linear_svm.f90'),'-o',str(module)],check=True)
            evidence['optimized_object_sha256']=sha(module)
            for p in programs+[driver]:
                subprocess.run([str(compiler),'-fcheck=all','-fbacktrace','-ffpe-trap=invalid,zero,overflow',
                    *common,str(p),str(module),str(frozen),'-static-libgfortran',str(shim),
                    '-Wl,--wrap=setlocale','-o',str(work/(p.stem+'.exe'))],check=True)

        def executable(p):
            return work/(p.stem+'.exe') if args.profile=='optimized' else binaries/p.relative_to(root).with_suffix('.exe')

        def native(p):
            binary=executable(p);before=sha(binary)
            run=subprocess.run([str(binary)],capture_output=True,text=True)
            assert sha(binary)==before
            assert run.returncode==0,(p,run.stdout[-1000:],run.stderr[-1500:])
            print(p.stem,'PASS',flush=True)
            return dict(program=p.stem,binary_sha256=before,passed=True)

        with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(native,programs))
        for extended in [False,True]:
            for layout in (['csr','reverse','duplicates','zeros'] if extended else ['dense','csr','reverse','duplicates','zeros']):
                key=layout+('_extended' if extended else '')
                report=args.report.with_name(args.report.stem+'_'+key+'.json')
                subprocess.run([sys.executable,str(root/'validation/compare_linear_svm.py'),
                    '--executable',str(executable(driver)),'--layout',layout,'--report',str(report),
                    *(['--extended','--diagnostic'] if extended else [])],check=True)
                evidence[key]=dict(report=str(report),sha256=sha(report),kind='diagnostic' if extended else 'strict')
        report=args.report.with_name(args.report.stem+'_guards_memory.json')
        subprocess.run([sys.executable,str(root/'validation/check_linear_svm_sparse.py'),
            '--executable',str(executable(root/'test/test_linear_svm_sparse.f90')),'--report',str(report)],check=True)
        evidence['guards_memory']=dict(report=str(report),sha256=sha(report),kind='strict')
    assert archive_hash==sha(archive)
    assert hashes=={str(p):sha(p) for p in files}
    result=dict(profile=args.profile,native=outcomes,evidence=evidence,archive_sha256=archive_hash,
        source_sha256=hashes,compiler_sha256=sha(compiler),project_strict_source_parity=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(args.profile,len(outcomes),'native programs, strict references, extended diagnostics and guards/memory COMPLETE')


if __name__=='__main__':main()

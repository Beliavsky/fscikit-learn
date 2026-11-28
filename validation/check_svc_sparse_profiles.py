"""Checked and scoped-O3 dense/CSR SVC and shared kernel audits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--profile',choices=['checked','optimized'],required=True);parser.add_argument('--all-native',action='store_true')
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    objects=args.build_dir/'gfortran.exe_F638FECF362AB298';binaries=args.build_dir/'gfortran.exe_E1FE13DB8881E18E'
    archive=objects/'fscikit-learn/libfscikit-learn.a';compiler=Path('C:/Programs/Octave-11.1.0/mingw64/bin/gfortran.exe')
    shim=Path(tempfile.gettempdir())/'fsklearn-locale-probe/windows_locale_copy.o'
    os.environ['PATH']=str(compiler.parent)+os.pathsep+os.environ['PATH'];os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();archive_hash=sha(archive)
    files=[Path(__file__),*[root/p for p in ['src/sklearn_svm.f90','src/sklearn_kernel_svm.f90','src/sklearn_libsvm_sparse.f90',
        'test/test_svm.f90','test/test_svc_weighted.f90','test/test_svc_sparse.f90','test/test_svm_regression.f90',
        'test/test_one_class_svm.f90','test/test_kernel_svm_sparse.f90','app/reference_svc_weighted_driver.f90',
        'app/reference_svc_sparse_driver.f90','app/reference_kernel_svm_driver.f90','app/reference_kernel_svm_sparse_driver.f90',
        'validation/compare_svc_weighted.py','validation/compare_svc_sparse.py','validation/check_svc_sparse.py',
        'validation/check_svc_weighted.py','validation/compare_kernel_svm.py','validation/compare_kernel_svm_sparse.py',
        'validation/check_kernel_svm_sparse.py']]]
    hashes={str(p):sha(p) for p in files}
    selected=[root/'test'/(p+'.f90') for p in ['test_svm','test_svc_weighted','test_svc_sparse',
        'test_svm_regression','test_one_class_svm','test_kernel_svm_sparse']]
    programs=sorted(p for p in (root/'test').glob('*.f90') if re.search(r'^\s*program\s',p.read_text(),re.I|re.M)) if args.all_native else selected
    drivers=[root/'app'/(p+'.f90') for p in ['reference_svc_weighted_driver','reference_svc_sparse_driver',
        'reference_kernel_svm_driver','reference_kernel_svm_sparse_driver']];evidence={}
    with tempfile.TemporaryDirectory(prefix='fsklearn-svc-audit-') as temporary:
        work=Path(temporary)
        if args.profile=='optimized':
            frozen=work/'checked.a';shutil.copyfile(archive,frozen);assert sha(frozen)==archive_hash==sha(archive)
            common=['-ffree-line-length-none','-J'+str(work),'-I'+str(work),'-I'+str(objects)];modules=[]
            for name in ['sklearn_libsvm_sparse','sklearn_kernel_svm','sklearn_svm']:
                module=work/(name+'.o');modules.append(module)
                subprocess.run([str(compiler),'-O3','-funroll-loops',*common,'-c',str(root/'src'/(name+'.f90')),'-o',str(module)],check=True)
                evidence[name+'_optimized_sha256']=sha(module)
            for p in programs+drivers:
                subprocess.run([str(compiler),'-fcheck=all','-fbacktrace','-ffpe-trap=invalid,zero,overflow',*common,
                    str(p),*[str(m) for m in reversed(modules)],str(frozen),'-static-libgfortran',str(shim),'-Wl,--wrap=setlocale','-o',str(work/(p.stem+'.exe'))],check=True)
        def executable(p):return work/(p.stem+'.exe') if args.profile=='optimized' else binaries/p.relative_to(root).with_suffix('.exe')
        def native(p):
            binary=executable(p);before=sha(binary);run=subprocess.run([str(binary)],capture_output=True,text=True)
            assert sha(binary)==before;assert run.returncode==0,(p,run.stdout[-1000:],run.stderr[-1500:])
            print(p.stem,'PASS',flush=True);return dict(program=p.stem,binary_sha256=before,passed=True)
        with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(native,programs))
        for key,script,program in [('weighted_dense','compare_svc_weighted.py',drivers[0]),
                ('weighted_csr','compare_svc_sparse.py',drivers[1]),('dense_guards','check_svc_weighted.py',selected[1]),
                ('csr_guards_memory','check_svc_sparse.py',selected[2]),('legacy_dense','compare_kernel_svm.py',drivers[2]),
                ('regression_csr','compare_kernel_svm_sparse.py',drivers[3]),
                ('regression_guards_memory','check_kernel_svm_sparse.py',selected[5])]:
            report=args.report.with_name(args.report.stem+'_'+key+'.json')
            subprocess.run([sys.executable,str(root/'validation'/script),'--executable',str(executable(program)),'--report',str(report)],check=True)
            evidence[key]=dict(report=str(report),sha256=sha(report),kind='strict')
    assert archive_hash==sha(archive);assert hashes=={str(p):sha(p) for p in files}
    result=dict(profile=args.profile,native=outcomes,evidence=evidence,archive_sha256=archive_hash,source_sha256=hashes,
        compiler_sha256=sha(compiler),project_strict_source_parity=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(args.profile,len(outcomes),'native programs and strict dense/CSR kernel references COMPLETE')

if __name__=='__main__':main()

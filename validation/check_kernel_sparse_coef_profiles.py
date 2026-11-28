"""Frozen checked/scoped-O3 sparse linear coefficients and retained SVR/NuSVR/OneClassSVM references."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True);parser.add_argument('--profile',choices=['checked','optimized'],required=True)
    parser.add_argument('--all-native',action='store_true');parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    objects=args.build_dir/'gfortran.exe_F638FECF362AB298';binaries=args.build_dir/'gfortran.exe_E1FE13DB8881E18E'
    archive=objects/'fscikit-learn/libfscikit-learn.a';compiler=Path('C:/Programs/Octave-11.1.0/mingw64/bin/gfortran.exe')
    shim=Path(tempfile.gettempdir())/'fsklearn-locale-probe/windows_locale_copy.o'
    os.environ['PATH']=str(compiler.parent)+os.pathsep+os.environ['PATH'];os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();archive_hash=sha(archive)
    selected=[root/'test'/(name+'.f90') for name in ['test_one_class_svm','test_oneclass_libsvm','test_svm_regression','test_kernel_svm_sparse','test_svc_weighted','test_nusvc_precomputed','test_svr_libsvm','test_nusvr_libsvm']]
    drivers=[root/'app'/(name+'.f90') for name in ['reference_oneclass_libsvm_driver','reference_kernel_svm_driver','reference_kernel_svm_sparse_driver','reference_svc_weighted_driver','reference_nusvc_libsvm_driver','reference_svr_libsvm_driver','reference_nusvr_libsvm_driver']]
    files=[Path(__file__),*[root/'src'/(name+'.f90') for name in ['sklearn_libsvm_solver','sklearn_kernel_svm','sklearn_libsvm_sparse','sklearn_svm','sklearn_nu_classification']],*selected,*drivers,
           *[root/'validation'/name for name in ['compare_oneclass_libsvm.py','check_oneclass_libsvm.py','compare_kernel_svm.py','compare_kernel_svm_sparse.py','check_kernel_svm_sparse.py','compare_svc_weighted.py','compare_nusvc_libsvm.py','compare_svc_sparse.py','check_minibatch_sparse_memory.py','compare_svr_libsvm.py','check_svr_libsvm.py','compare_nusvr_libsvm.py','check_nusvr_libsvm.py']]]
    hashes={str(p):sha(p) for p in files};evidence={}
    programs=sorted(p for p in (root/'test').glob('*.f90') if re.search(r'^\s*program\s',p.read_text(),re.I|re.M)) if args.all_native else selected
    args.report.resolve().parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fsklearn-nusvc-audit-',dir=args.report.resolve().parent) as temporary:
        work=Path(temporary)
        if args.profile=='optimized':
            frozen=work/'checked.a';shutil.copyfile(archive,frozen);assert sha(frozen)==archive_hash==sha(archive)
            common=['-ffree-line-length-none','-J'+str(work),'-I'+str(work),'-I'+str(objects)]
            modules=[]
            for name in ['sklearn_libsvm_solver','sklearn_kernel_svm']:
                module=work/(name+'.o');modules.append(module)
                subprocess.run([str(compiler),'-O3','-funroll-loops',*common,'-c',str(root/'src'/(name+'.f90')),'-o',str(module)],check=True)
                evidence[name+'_optimized_sha256']=sha(module)
            for p in programs+drivers:
                subprocess.run([str(compiler),'-fcheck=all','-fbacktrace','-ffpe-trap=invalid,zero,overflow',*common,
                                str(p),*[str(m) for m in reversed(modules)],str(frozen),'-static-libgfortran',str(shim),'-Wl,--wrap=setlocale','-o',str(work/(p.stem+'.exe'))],check=True)
        def executable(p):return work/(p.stem+'.exe') if args.profile=='optimized' else binaries/p.relative_to(root).with_suffix('.exe')
        def native(p):
            binary=executable(p);digest=sha(binary);run=subprocess.run([str(binary)],capture_output=True,text=True)
            assert sha(binary)==digest;assert run.returncode==0,(p,run.stdout[-1000:],run.stderr[-1500:])
            print(p.stem,'PASS',flush=True);return dict(program=p.stem,binary_sha256=digest,passed=True)
        with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(native,programs))
        for key,script,program in [('nusvr','compare_nusvr_libsvm.py',drivers[6]),('nusvr_guards','check_nusvr_libsvm.py',selected[7]),
                                   ('svr','compare_svr_libsvm.py',drivers[5]),('svr_guards','check_svr_libsvm.py',selected[6]),
                                   ('oneclass','compare_oneclass_libsvm.py',drivers[0]),('legacy_dense','compare_kernel_svm.py',drivers[1]),
                                   ('legacy_csr','compare_kernel_svm_sparse.py',drivers[2]),('guards','check_oneclass_libsvm.py',selected[1]),
                                   ('csr_guards_memory','check_kernel_svm_sparse.py',selected[3])]:
            report=args.report.with_name(args.report.stem+'_'+key+'.json')
            command=[sys.executable,str(root/'validation'/script),'--executable',str(executable(program)),'--report',str(report)]
            subprocess.run(command,check=True)
            data=json.loads(report.read_text());assert not data.get('gaps',[]) and data.get('strict_parity',True)
            evidence[key]=dict(report=str(report),sha256=sha(report),kind='strict')
    assert archive_hash==sha(archive);assert hashes=={str(p):sha(p) for p in files}
    result=dict(profile=args.profile,native=outcomes,evidence=evidence,archive_sha256=archive_hash,source_sha256=hashes,
                compiler_sha256=sha(compiler),project_strict_source_parity=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(args.profile,len(outcomes),'native programs and sparse-linear and retained regression/one-class references COMPLETE')


if __name__=='__main__':main()

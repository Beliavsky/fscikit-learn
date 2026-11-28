"""Frozen checked/scoped-O3 passive-aggressive streaming and retained dense/CSR PA contracts."""
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
    selected=[root/'test'/(name+'.f90') for name in ['test_pa_partial','test_pa_csr','test_pa_seeded','test_passive_aggressive','test_linear','test_positive_linear',
        'test_theil_sen','test_bayes_neighbors','test_huber_controls','test_ransac_controls','test_bayesian_ridge_controls',
        'test_sgd_validation','test_sgd_clone','test_cv_clone','test_pipeline','test_multioutput']]
    drivers=[root/'app'/(name+'.f90') for name in ['reference_pa_seeded_driver','reference_passive_aggressive_driver','reference_pa_csr_driver','reference_pa_partial_driver']]
    files=[Path(__file__),root/'src/sklearn_linear_model.f90',root/'src/sklearn_legacy_rng.f90',root/'src/sklearn_sparse.f90',*selected,*drivers,
           *[root/'validation'/name for name in ['compare_pa_seeded.py','check_pa_seeded.py','compare_passive_aggressive.py','compare_pa_csr.py','check_pa_csr.py','compare_kernel_svm_sparse.py','check_minibatch_sparse_memory.py','compare_pa_partial.py','check_pa_partial.py']]]
    hashes={str(p):sha(p) for p in files};evidence={}
    programs=sorted(p for p in (root/'test').glob('*.f90') if re.search(r'^\s*program\s',p.read_text(),re.I|re.M)) if args.all_native else selected
    args.report.resolve().parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fsklearn-pa-audit-',dir=args.report.resolve().parent) as temporary:
        work=Path(temporary)
        if args.profile=='optimized':
            frozen=work/'checked.a';shutil.copyfile(archive,frozen);assert sha(frozen)==archive_hash==sha(archive)
            common=['-ffree-line-length-none','-J'+str(work),'-I'+str(work),'-I'+str(objects)]
            module=work/'linear.o'
            subprocess.run([str(compiler),'-O3','-funroll-loops',*common,'-c',str(root/'src/sklearn_linear_model.f90'),'-o',str(module)],check=True)
            evidence['optimized_linear_model_sha256']=sha(module)
            for p in programs+drivers:
                subprocess.run([str(compiler),'-fcheck=all','-fbacktrace','-ffpe-trap=invalid,zero,overflow',*common,
                    str(p),str(module),str(frozen),'-static-libgfortran',str(shim),'-Wl,--wrap=setlocale',
                    '-o',str(work/(p.stem+'.exe'))],check=True)
        def executable(p):return work/(p.stem+'.exe') if args.profile=='optimized' else binaries/p.relative_to(root).with_suffix('.exe')
        def native(p):
            binary=executable(p);digest=sha(binary);run=subprocess.run([str(binary)],capture_output=True,text=True)
            assert sha(binary)==digest;assert run.returncode==0,(p,run.stdout[-1000:],run.stderr[-1500:])
            print(p.stem,'PASS',flush=True);return dict(program=p.stem,binary_sha256=digest,passed=True)
        with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(native,programs))
        for key,script,program in [('partial','compare_pa_partial.py',drivers[3]),('partial_guards_memory','check_pa_partial.py',selected[0]),
                                   ('csr','compare_pa_csr.py',drivers[2]),('csr_guards_memory','check_pa_csr.py',selected[1]),
                                   ('seeded','compare_pa_seeded.py',drivers[0]),('guards','check_pa_seeded.py',selected[2]),
                                   ('legacy','compare_passive_aggressive.py',drivers[1])]:
            report=args.report.with_name(args.report.stem+'_'+key+'.json')
            subprocess.run([sys.executable,str(root/'validation'/script),'--executable',str(executable(program)),'--report',str(report)],check=True)
            data=json.loads(report.read_text());assert not data.get('gaps',[]) and data.get('strict_parity',True)
            assert all(c['passed'] for c in data.get('checks',[]))
            evidence[key]=dict(report=str(report),sha256=sha(report),kind='strict')
    assert archive_hash==sha(archive);assert hashes=={str(p):sha(p) for p in files}
    result=dict(profile=args.profile,native=outcomes,evidence=evidence,archive_sha256=archive_hash,source_sha256=hashes,
                compiler_sha256=sha(compiler),project_strict_source_parity=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(args.profile,len(outcomes),'native programs and PA reference audits COMPLETE')


if __name__=='__main__':main()

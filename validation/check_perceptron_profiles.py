"""Frozen checked/scoped-O3 Perceptron and retained full SGD corpora."""
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
    selected=[root/'test'/(name+'.f90') for name in ['test_sgd_validation','test_sgd','test_sgd_classifier','test_sgd_regularization',
        'test_sgd_fit','test_sgd_csr','test_sgd_clone','test_cv_clone','test_pipeline','test_multioutput','test_perceptron_controls','test_perceptron_search','test_linear','test_multiclass','test_pa_warm_clone','test_pa_validation_search']]
    drivers=[root/'app'/(name+'.f90') for name in ['reference_sgd_validation_driver','reference_sgd_csr_driver',
        'reference_sgd_fit_driver','reference_sgd_regularization_driver','reference_perceptron_driver']]
    scripts=['compare_sgd_validation.py','check_sgd_validation.py','compare_sgd_csr.py','compare_sgd_fit.py',
             'compare_sgd_regularization.py','compare_image_graph.py','compare_perceptron.py','check_perceptron.py','check_minibatch_sparse_memory.py']
    files=[Path(__file__),*[root/'src'/(name+'.f90') for name in ['sklearn_sgd','sklearn_sgd_clone','sklearn_search_adapters','sklearn_resampling','sklearn_legacy_rng','sklearn_perceptron','sklearn_linear_model','sklearn_base_dispatch','sklearn_multiclass']],
           *selected,*drivers,*[root/'validation'/name for name in scripts]]
    hashes={str(p):sha(p) for p in files};evidence={}
    programs=sorted(p for p in (root/'test').glob('*.f90') if re.search(r'^\s*program\s',p.read_text(),re.I|re.M)) if args.all_native else selected
    args.report.resolve().parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fsklearn-sgd-validation-',dir=args.report.resolve().parent) as temporary:
        work=Path(temporary)
        if args.profile=='optimized':
            frozen=work/'checked.a';shutil.copyfile(archive,frozen);assert sha(frozen)==archive_hash==sha(archive)
            common=['-ffree-line-length-none','-J'+str(work),'-I'+str(work),'-I'+str(objects)];modules=[]
            for name in ['sklearn_sgd','sklearn_perceptron','sklearn_linear_model','sklearn_sgd_clone','sklearn_base_dispatch','sklearn_multiclass','sklearn_search_adapters']:
                module=work/(name+'.o');modules.append(module)
                subprocess.run([str(compiler),'-O3','-funroll-loops',*common,'-c',str(root/'src'/(name+'.f90')),'-o',str(module)],check=True)
                evidence[name+'_optimized_sha256']=sha(module)
            for p in programs+drivers:
                subprocess.run([str(compiler),'-fcheck=all','-fbacktrace','-ffpe-trap=invalid,zero,overflow',*common,
                    str(p),*[str(m) for m in reversed(modules)],str(frozen),'-static-libgfortran',str(shim),'-Wl,--wrap=setlocale',
                    '-o',str(work/(p.stem+'.exe'))],check=True)
        def executable(p):return work/(p.stem+'.exe') if args.profile=='optimized' else binaries/p.relative_to(root).with_suffix('.exe')
        def native(p):
            binary=executable(p);digest=sha(binary);run=subprocess.run([str(binary)],capture_output=True,text=True)
            assert sha(binary)==digest;assert run.returncode==0,(p,run.stdout[-1000:],run.stderr[-1500:])
            print(p.stem,'PASS',flush=True);return dict(program=p.stem,binary_sha256=digest,passed=True)
        with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(native,programs))
        for key,script,program in [('perceptron','compare_perceptron.py',drivers[4]),('perceptron_guards_memory','check_perceptron.py',selected[10]),
            ('validation','compare_sgd_validation.py',drivers[0]),('guards','check_sgd_validation.py',selected[0]),
            ('legacy_csr','compare_sgd_csr.py',drivers[1]),('legacy_fit','compare_sgd_fit.py',drivers[2]),
            ('legacy_regularization','compare_sgd_regularization.py',drivers[3])]:
            report=args.report.with_name(args.report.stem+'_'+key+'.json')
            command=[sys.executable,str(root/'validation'/script),'--executable',str(executable(program)),'--report',str(report)]
            if key.startswith('legacy_'):command+=['--profile',args.profile]
            subprocess.run(command,check=True)
            data=json.loads(report.read_text());assert not data.get('gaps',[]) and data.get('strict_parity',True)
            assert all(c['passed'] for c in data.get('checks',[]))
            evidence[key]=dict(report=str(report),sha256=sha(report),kind='strict')
    assert archive_hash==sha(archive);assert hashes=={str(p):sha(p) for p in files}
    result=dict(profile=args.profile,native=outcomes,evidence=evidence,archive_sha256=archive_hash,source_sha256=hashes,
                compiler_sha256=sha(compiler),project_strict_source_parity=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(args.profile,len(outcomes),'native programs and SGD reference audits COMPLETE')


if __name__=='__main__':main()

"""MiniBatchKMeans: seeded fit, weighted updates and step-by-step streaming state."""
from pathlib import Path
import argparse
import json
import os
import subprocess
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def snapshot(model, prefix=""):
    result = dict(centers=model.cluster_centers_.copy(), counts=model._counts.copy(),
                  meta=[model.n_features_in_, model.n_steps_])
    if hasattr(model, "n_iter_"): result["n_iter"] = [model.n_iter_]
    if hasattr(model, "labels_"): result["labels"] = model.labels_.copy()
    if hasattr(model, "inertia_"): result["inertia"] = [model.inertia_]
    return {prefix+name: value for name, value in result.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--executable", type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        proc = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_minibatch_driver", "--runner"],
                              cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError(proc.stdout)
        paths = [ROOT / s.strip().strip(chr(34)) for s in proc.stdout.splitlines()
                 if s.strip().strip(chr(34)).endswith(("reference_minibatch_driver", "reference_minibatch_driver.exe"))]
        paths = [p for p in paths if p.is_file()]; assert len(paths) == 1, proc.stdout
        executable = paths[0].resolve()
    checks = []
    for seed,n,p in ((7,65,3),(42,88,4),(0,301,2),(2026,49,5)):
        rng = np.random.default_rng(seed); original = rng.normal(size=(n,p))
        w0 = rng.uniform(.2,2,n); w0[::7] = 0
        for case in range(34):
            n=len(original)
            x,w = original.copy(),np.ones(n)
            options = dict(n_clusters=4, max_iter=25, batch_size=17, random_state=seed)
            operation,chunk,initial = 1,19,None
            if case == 1: options.update(init="random")
            if case == 2: options.update(init="random",n_init=5)
            if case == 3: options.update(n_init=4,init_size=21)
            if case == 4: w=w0
            if case == 5: options.update(reassignment_ratio=.5); w=w0
            if case == 6: options.update(tol=.02,max_no_improvement=None)
            if case == 7: options.update(max_no_improvement=0)
            if case == 8: options.update(max_iter=1,batch_size=n)
            if case == 9: options.update(compute_labels=False)
            if case == 10: options.update(max_no_improvement=None,reassignment_ratio=0)
            # Test init-size fallback without an ambiguous equal-count cap tie.
            if case == 11: options.update(init_size=2,batch_size=3,reassignment_ratio=0)
            if case == 12: options.update(n_clusters=1); w=w0
            if case == 13: initial=x[[1,5,10,15]].copy(); w=w0
            if case == 14: operation=3
            if case == 15: operation=4
            if case == 16: operation=2
            if case == 17: operation=2; w=w0; options.update(init="random")
            if case == 18: operation=2; options.update(compute_labels=False)
            if case == 19: operation=2; initial=x[[1,5,10,15]].copy(); w=w0
            if case == 20: operation=2; options.update(reassignment_ratio=.8); w=w0
            if case == 21: operation=5; w=w0
            if case == 22: operation=6; w=w0
            if case == 23: options.update(batch_size=1024,n_clusters=8)
            if case == 24: options.update(n_clusters=3,init_size=7,batch_size=2,reassignment_ratio=0)
            if case == 25: x[:,0]=1; options.update(tol=.001)
            # Exact zero centers avoid BLAS-dependent near-ties after averaging.
            if case == 26: x[:]=0; options.update(n_clusters=3,reassignment_ratio=0)
            if case == 27: x+=100; options.update(n_init=3)
            if case == 28: operation=2; options.update(n_clusters=1,batch_size=1); w=w0
            if case == 29: options.update(init="random",init_size=20); w=w0
            if case == 30: initial=x[[1,5,10,15]].copy(); w[:]=0
            if case == 31: operation=2; initial=x[[1,5,10,15]].copy(); w[:]=0
            # Distinct weights force an unambiguous reassignment cap ranking.
            if case == 32:
                n=4; x=12*np.eye(4,p); w=np.arange(1,5,dtype=float)
                initial=x.copy(); operation=2; chunk=4; options.update(reassignment_ratio=.9)
            # A huge budget must not overflow before two-step early stopping.
            if case == 33: options.update(n_clusters=1,max_iter=1000000000,batch_size=n,max_no_improvement=0)
            if initial is not None: options["init"]=initial
            query=np.vstack([x+.137,rng.normal(size=(13,p))]); qw=rng.uniform(.2,2,len(query))
            model=MiniBatchKMeans(**options)
            data,output=RESULTS/"minibatch_input.txt",RESULTS/"minibatch_actual.txt"
            with data.open("w") as stream:
                stream.write(f"{n} {p} {len(query)} {operation} {chunk} {int(initial is not None)}\n")
                stream.write(f"{model.n_clusters} {0 if model.n_init=='auto' else model.n_init} {model.max_iter} {seed} "
                             f"{model.batch_size} {model.init_size or 0}\n")
                stream.write(f"{model.tol} {model.reassignment_ratio} "
                             f"{-1 if model.max_no_improvement is None else model.max_no_improvement} "
                             f"{model.compute_labels} {model.init if isinstance(model.init,str) else 'k-means++'}\n")
                np.savetxt(stream,np.column_stack([x,w]),fmt="%.17g")
                np.savetxt(stream,np.column_stack([query,qw]),fmt="%.17g")
                if initial is not None: np.savetxt(stream,initial,fmt="%.17g")
            subprocess.run([str(executable),str(data),str(output)],cwd=ROOT,check=True)
            expected={}
            if operation in (2,5,6):
                if operation==5: model.fit(x,sample_weight=w)
                for step,i in enumerate(range(0,n,chunk),1):
                    model.partial_fit(x[i:i+chunk],sample_weight=w[i:i+chunk])
                    expected.update(snapshot(model,f"step{step}_"))
                if operation==6: model.fit(x,sample_weight=w)
            elif operation==3: expected["fit_predict"]=model.fit_predict(x,sample_weight=w)
            elif operation==4: expected["fit_transform"]=model.fit_transform(x,sample_weight=w)
            else: model.fit(x,sample_weight=w)
            expected.update(snapshot(model))
            expected.update(predict=model.predict(query),transform=model.transform(query),
                            score=[model.score(x),model.score(x,sample_weight=w),model.score(query,sample_weight=qw)])
            actual={}
            for line in output.read_text().splitlines():
                name,*values=line.split(); actual[name]=np.array(values,dtype=float)
            assert actual.keys()==expected.keys(),(seed,case,actual.keys()^expected.keys())
            for name,value in expected.items():
                tol=0 if name.endswith(("labels","meta","n_iter")) or name in ("predict","fit_predict") else 3e-8
                np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order="F"),atol=tol,rtol=tol,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed,case=case,name=name,passed=True))
    (RESULTS/"minibatch_kmeans.json").write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+"\n")
    print(f"{len(checks)} MiniBatchKMeans result groups passed")


if __name__ == "__main__":
    main()

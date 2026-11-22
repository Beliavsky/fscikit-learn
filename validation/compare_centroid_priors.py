"""NearestCentroid 1.6 priors, discriminants, shrinkage and degenerate diagnostics."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import numpy as np
import sklearn
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
from compare_function_transformer import ROOT, resolve
from compare_gp_kernels import block

REJECTIONS = {
 'unfitted_predict':'nearest centroid is not fitted','unfitted_decision':'nearest centroid is not fitted',
 'unfitted_proba':'nearest centroid is not fitted','one_class':'requires multiple classes',
 'metric':'unknown centroid metric','prior_mode':'unknown centroid prior_mode',
 'negative_shrink':'shrink_threshold must be nonnegative','nan_shrink':'shrink_threshold must be finite',
 'inf_shrink':'shrink_threshold must be finite','prior_size':'priors must match class count',
 'prior_negative':'priors must be nonnegative','prior_nan':'priors must be finite','prior_inf':'priors must be finite',
 'prior_range':'prior total exceeds numeric range','data_range':'data exceeds working range',
 'data_nan':'X contains NaN or infinity','constant':'all features have zero variance',
 'empty':'X must be nonempty','target_size':'X and y sample counts differ',
 'width':'query feature count mismatch','query_nan':'X contains NaN or infinity','query_empty':'X must be nonempty',
 'query_range':'query exceeds working range','manhattan_decision':'methods require euclidean metric',
 'manhattan_proba':'methods require euclidean metric','manhattan_log_proba':'methods require euclidean metric',
 'singleton_decision':'require finite within-class deviations','singleton_nonuniform':'require finite within-class deviations',
 'undefined_shrink':'requires finite fitted centroids','standardized_range':'standardized values exceed working range'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--profile', choices=['checked','release'], default='checked')
    parser.add_argument('--executable', type=Path); args = parser.parse_args()
    exe = args.executable.resolve() if args.executable else resolve(args.profile, 'reference_centroid_priors_driver')
    fixtures = []
    for seed in [0,42,2026]:
        rng = np.random.RandomState(seed)
        for n,p in [(9,1),(24,3),(11,8)]:
            for k in [2,3,4]:
                labels = np.arange(n)%k
                normal = rng.normal(size=(n,p))*np.geomspace(.03,7,p)+labels[:,None]*.7
                for variant in range(3):
                    x = normal.copy()
                    if variant == 1:
                        if p == 1: x[:,0] = labels*3
                        else: x[:,1:] = 7
                    if variant == 2: x = (labels[:,None]*3+np.arange(p)).astype(float)
                    q = rng.normal(size=(5,p))*np.geomspace(.03,7,p); q[0] = 0; q[1] = x[0]
                    q[2] = (x[labels==0].mean(axis=0)+x[labels==k-1].mean(axis=0))/2
                    for metric in ['euclidean','manhattan']:
                        for threshold in [None,.7]:
                            for prior in range(7): fixtures.append((seed,x,labels*5-2,q,metric,threshold,prior,variant))
        for k in [2,3,4]:
            x = np.arange(k*3,dtype=float).reshape(k,3); q = rng.normal(size=(5,3))
            for metric in ['euclidean','manhattan']:
                for threshold in [None,.7]:
                    for prior in range(7):fixtures.append((seed,x,np.arange(k)*5-2,q,metric,threshold,prior,3))
    checks,undefined,ties,errors,identities = [],[],[],[],[]
    for index,(seed,x,y,q,metric,threshold,prior,variant) in enumerate(fixtures):
        n,p = x.shape; classes = np.unique(y); k = len(classes)
        priors = 'uniform' if prior == 0 else 'empirical'
        if prior >= 2:
            values = np.arange(1,k+1,dtype=float)
            if prior == 3: values[0] = 0
            if prior == 4: values[:] = 0
            if prior == 5: values = np.full(k,(1+1e-7)/k)
            if prior == 6: values = values/values.sum()*(1+1e-7)
            priors = values.tolist()
        with warnings.catch_warnings(record=True), np.errstate(divide='ignore',invalid='ignore'):
            model = NearestCentroid(metric=metric,shrink_threshold=threshold,priors=priors).fit(x,y)
            expected = dict(classes=model.classes_,centroids=model.centroids_,std=model.within_class_std_dev_,
                            deviations=model.deviations_,priors=model.class_prior_,features=[model.n_features_in_])
            actions = 0; failures = []
            try:
                predicted = model.predict(q); trained = model.predict(x)
                expected.update(predict=predicted,train_predict=trained,score=[model.score(x,y)]); actions |= 1
                uniform = np.isclose(model.class_prior_,1/k).all()
                query_scores = -pairwise_distances(q,model.centroids_,metric=metric)**2 if uniform else model._decision_function(q)
                train_scores = -pairwise_distances(x,model.centroids_,metric=metric)**2 if uniform else model._decision_function(x)
            except ValueError:
                failures.append((1,'predict'))
            if metric == 'euclidean':
                try:
                    expected.update(decision=model.decision_function(q),logp=model.predict_log_proba(q),proba=model.predict_proba(q))
                    actions |= 2
                except ValueError:
                    failures.append((2,'decision'))
        def payload(request):
            data = f'{n} {p} {len(q)} {int(metric=="manhattan")} {min(prior,2)} {k} {request} {threshold or 0:.17g}\n'
            for values in [x.ravel(order='F'),y,q.ravel(order='F')]:data += ' '.join(f'{v:.17g}' for v in values)+'\n'
            if prior >= 2:data += ' '.join(f'{v:.17g}' for v in priors)+'\n'
            return data
        run = subprocess.run([str(exe)],input=payload(actions),text=True,capture_output=True,cwd=ROOT,check=True)
        lines = iter(run.stdout.splitlines()); actual = {}
        for name,value in expected.items():
            got = block(lines); want = np.asarray(value)
            if want.ndim == 1: want = want[:,None]
            assert got.shape == want.shape,(index,name,got.shape,want.shape)
            actual[name] = got
            if name in ['predict','train_predict']:
                scores = query_scores if name == 'predict' else train_scores
                for row,(a,b) in enumerate(zip(got[:,0],want[:,0])):
                    assert a in classes
                    if a == b:continue
                    assert not np.isnan(scores[row]).any(),(index,name,row)
                    finite = scores[row,np.isfinite(scores[row])]
                    points = q if name == 'predict' else x
                    center_scale = model.centroids_.copy(); point_scale = points[row].copy()
                    if not uniform:
                        mask = model.within_class_std_dev_ != 0
                        center_scale[:,mask] /= model.within_class_std_dev_[mask]
                        point_scale[mask] /= model.within_class_std_dev_[mask]
                    # Norm-identity cancellation in sklearn's Euclidean distances
                    # depends on coordinate magnitudes, not only the final distances.
                    if metric == 'euclidean':
                        magnitude = (np.linalg.norm(point_scale)+np.linalg.norm(center_scale,axis=1).max())**2
                    else:
                        magnitude = (np.abs(point_scale).sum()+np.abs(center_scale).sum(axis=1).max())**2
                    bound = 32*(n+p+2)*np.finfo(float).eps*max(1.,float(np.max(np.abs(finite))),float(magnitude))
                    gap = float(scores[row].max()-scores[row,np.searchsorted(classes,a)])
                    assert np.isfinite(gap) and 0<=gap<=bound,(index,name,row,gap,bound)
                    ties.append(dict(fixture=index,name=name,row=row,gap=gap,bound=bound,coordinate_magnitude=float(magnitude),passed=True))
            elif name == 'score':
                np.testing.assert_allclose(got[0,0],np.mean(actual['train_predict'][:,0]==y),atol=0,rtol=0)
            else:
                np.testing.assert_allclose(got,want,atol=3e-9,rtol=3e-9,equal_nan=True,err_msg=f'{index} {name}')
            if np.any(~np.isfinite(want)):
                np.testing.assert_equal(np.isnan(got),np.isnan(want))
                np.testing.assert_equal(np.isposinf(got),np.isposinf(want));np.testing.assert_equal(np.isneginf(got),np.isneginf(want))
                undefined.append(dict(fixture=index,name=name,passed=True))
            checks.append(dict(fixture=index,name=name,passed=True))
        assert next(lines,None) is None
        for request,name in failures:
            run = subprocess.run([str(exe)],input=payload(request),text=True,capture_output=True,cwd=ROOT)
            assert run.returncode!=0 and ('finite fitted centroids' in run.stdout+run.stderr or 'finite within-class deviations' in run.stdout+run.stderr),(index,name,run.stdout,run.stderr)
            errors.append(dict(fixture=index,name=name,source_value_error=True,native_error=True,passed=True))
        if 'proba' in actual:
            finite = np.isfinite(actual['proba']).all(axis=1)
            np.testing.assert_allclose(actual['proba'][finite].sum(axis=1),1,atol=3e-12,rtol=0)
            np.testing.assert_allclose(np.exp(actual['logp']),actual['proba'],atol=3e-12,rtol=3e-12)
            if k == 2:
                both = np.isfinite(actual['logp']).all(axis=1)
                np.testing.assert_allclose(actual['logp'][both,1]-actual['logp'][both,0],actual['decision'][both,0],atol=3e-9,rtol=3e-9)
            identities.append(dict(fixture=index,passed=True))
        if (index+1)%84==0:print(index+1,'nearest centroid fixtures passed',flush=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(profile=args.profile,sklearn_version=sklearn.__version__,native_sha256=sha(exe),
                    validator_sha256=sha(Path(__file__)),source_sha256={rel:sha(Path(sklearn.__file__).parent/rel) for rel in ['neighbors/_nearest_centroid.py','discriminant_analysis.py']})
    results = ROOT/'validation/results';results.mkdir(exist_ok=True)
    report = dict(**metadata,checks=checks,fixtures=len(fixtures),undefined_certificates=undefined,label_certificates=ties,
                  error_certificates=errors,probability_identities=identities)
    (results/f'centroid_priors_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    rejects = []
    for case,message in REJECTIONS.items():
        run = subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
        assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
        rejects.append(dict(case=case,passed=True))
    (results/f'centroid_priors_invalid_{args.profile}.json').write_text(json.dumps(dict(**metadata,checks=rejects),indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups, {len(fixtures)} fixtures, {len(undefined)} nonfinite fields, '
          f'{len(errors)} source/native error certificates, {len(identities)} probability identities, '
          f'{len(ties)} tied labels and {len(rejects)} rejections PASS')


if __name__ == '__main__':main()

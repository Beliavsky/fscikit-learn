"""Normalizer/Binarizer inside sklearn pipelines and feature compositions."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import Normalizer, Binarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from compare_nmf_cd import fields
from compare_stateless_preprocessing import assert_close, names_matrix, NORMS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--profile', choices=['development', 'checked', 'release'], default='development')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve()
    checks, inventories = [], []
    fixtures = []
    for seed, shape, norm, mode, named, threshold in itertools.product(
            [0, 42], [(17, 4), (23, 7)], range(3), range(9), [False, True], [-.2, .25]):
        if mode>=7 and norm!=0:
            continue
        for weighted in ([False, True] if mode in [3, 4, 6, 7, 8] else [False]):
            for nb_kind in (range(3) if mode >= 6 else [0]):
                fixtures.append((seed, shape, norm, mode, named, threshold, weighted, nb_kind))
    for index, (seed, shape, norm, mode, named, threshold, weighted, nb_kind) in enumerate(fixtures):
        rng = np.random.RandomState(seed)
        n, p = shape
        x = rng.normal(size=shape);q = rng.normal(size=(5, p));x[0] = 0;q[0] = 0
        y = x@rng.normal(size=p)+.3*rng.normal(size=n)+1.7
        labels = np.array([-2, 3, 7])[np.arange(n) % 3]
        weights = rng.uniform(.1, 2., size=n);weights[1] = 0
        names = [f'axis {j}' for j in range(p)]
        train = pd.DataFrame(x, columns=names) if named else x
        query = pd.DataFrame(q, columns=names) if named else q
        unit, binary = Normalizer(norm=NORMS[norm]), Binarizer(threshold=threshold)
        expected = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if mode in [0, 3, 4, 6]:
                first = binary if mode in [4, 6] else unit
                last = binary if mode == 0 else ([MultinomialNB, BernoulliNB, ComplementNB][nb_kind]()
                                                if mode == 6 else Ridge(alpha=.75, solver='svd'))
                last_name = 'binary' if mode == 0 else 'model'
                model = Pipeline([('map', first), (last_name, last)])
                if mode == 0:
                    z = model.fit_transform(train)
                    expected.extend([('fit_transform', z), ('query', model.transform(query)),
                                     ('names', names_matrix(model.get_feature_names_out())),
                                     ('metadata', np.array([p, p])[:, None])])
                else:
                    kwargs = {'model__sample_weight': weights} if weighted else {}
                    model.fit(train, labels if mode == 6 else y, **kwargs)
                    expected.extend([('train_predict', model.predict(train)[:, None]),
                                     ('query_predict', model.predict(query)[:, None])])
                    if mode == 6:
                        expected.extend([('query_proba', model.predict_proba(query)), ('classes', model.classes_[:, None])])
                    else:
                        expected.extend([('coef', model[-1].coef_[:, None]), ('intercept', [[model[-1].intercept_]])])
                    expected.append(('names', names_matrix(model[:-1].get_feature_names_out())))
                expected.append(('can_inverse', [[int(hasattr(model, 'inverse_transform'))]]))
            elif mode == 1:
                model = FeatureUnion([('unit', unit), ('binary', binary)],
                                     transformer_weights={'unit': .7, 'binary': 1.3})
                z = model.fit_transform(train)
                expected.extend([('fit_transform', z), ('query', model.transform(query)),
                                 ('names', names_matrix(model.get_feature_names_out())),
                                 ('metadata', np.array([p, 2*p])[:, None])])
            elif mode in [2,5]:
                model = ColumnTransformer([('unit', unit, [2, 0]), ('binary', binary, [1])],
                                          remainder='passthrough' if mode == 2 else Normalizer(norm=NORMS[norm]))
                z = model.fit_transform(train)
                expected.extend([('fit_transform', z), ('query', model.transform(query)),
                                 ('names', names_matrix(model.get_feature_names_out())),
                                 ('metadata', np.array([p, p])[:, None])])
            else:
                a, b = binary.transform(x), binary.transform(q)
                base = [MultinomialNB, BernoulliNB, ComplementNB][nb_kind]
                if mode == 7:
                    model = BaggingClassifier(base(), n_estimators=5, max_samples=.7,
                                              max_features=.75, random_state=42)
                else:
                    model = VotingClassifier([('left',base()),('right',base())], voting='soft',weights=[1,2])
                model.fit(a, labels, **({'sample_weight':weights} if weighted else {}))
                expected.extend([('train_predict',model.predict(a)[:,None]),('query_predict',model.predict(b)[:,None]),
                                 ('query_proba',model.predict_proba(b)),('classes',model.classes_[:,None])])
        header = [mode, n, p, len(q), norm, int(named), int(weighted), nb_kind, threshold]
        payload = ' '.join(map(str, header))+'\n'+'\n'.join(
            ' '.join(format(v, '.17g') for v in a.ravel(order='F')) for a in [x, q, y, weights, labels])+'\n'
        proc = subprocess.run([str(exe)], input=payload, text=True, capture_output=True)
        assert proc.returncode == 0, (index, header, proc.stdout, proc.stderr)
        actual = fields(proc.stdout)
        assert len(actual) == len(expected), (index, header, len(actual), len(expected))
        field_names = []
        for value, (field, reference) in zip(actual, expected):
            reference = np.asarray(reference)
            error = assert_close(value, reference, index, field)
            if field in ['names', 'metadata', 'classes', 'can_inverse'] or mode >= 6 and field.endswith('predict'):
                np.testing.assert_array_equal(value, reference)
            checks.append(dict(fixture=index, field=field, max_abs_error=error))
            field_names.append(field)
        checks.append(dict(fixture=index, field='unfitted_prototypes_preserved', max_abs_error=0.))
        field_names.append('unfitted_prototypes_preserved')
        inventories.append(dict(fixture=index, mode=mode, weighted=weighted, named=named, nb_kind=nb_kind, fields=field_names))
        if index % 100 == 0:
            print(f'{index}/{len(fixtures)} fixtures passed', flush=True)
    report = dict(profile=args.profile, sklearn_version=sklearn.__version__, fixtures=len(fixtures),
                  checks=checks, inventories=inventories, executable=str(exe),
                  executable_sha256=digest(exe), validator_sha256=digest(__file__),
                  core_validator_sha256=digest(Path(__file__).with_name('compare_stateless_preprocessing.py')))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures, {len(checks)} composition groups')


if __name__ == '__main__':
    main()

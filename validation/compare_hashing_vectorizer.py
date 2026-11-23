"""ASCII HashingVectorizer against sklearn 1.6.1, including exact CSR/analyzers."""
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
from scipy.sparse import csr_matrix
import sklearn
from sklearn.feature_extraction import text as source_text
from sklearn.feature_extraction.text import HashingVectorizer, ENGLISH_STOP_WORDS
from sklearn.utils import murmurhash3_32

ANALYZERS = ['word', 'char', 'char_wb']
NORMS = [None, 'l1', 'l2']
REJECTIONS = {
    'features': 'n_features must be positive', 'min_gram': 'invalid ngram range',
    'gram_order': 'invalid ngram range', 'token_length': 'token_min_length must be positive',
    'analyzer': 'unknown analyzer', 'norm': 'unknown norm',
    'stop_conflict': 'choose English or custom stop words',
    'unicode': 'currently requires ASCII text', 'unicode_stop': 'currently requires ASCII text',
    'unallocated': 'document is unallocated', 'empty': 'document sequence must be nonempty',
    'fit_control': 'n_features must be positive', 'partial_control': 'unknown norm',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(text):
    return str(len(text))+'\n'+(' '.join(str(ord(c)) for c in text)+'\n' if text else '')


def parse(output):
    tokens = iter(output.split())
    nr, nc, nnz = (int(next(tokens)) for _ in range(3))
    pointers = np.array([int(next(tokens))-1 for _ in range(nr+1)])
    indices = np.array([int(next(tokens))-1 for _ in range(nnz)])
    values = np.array([float(next(tokens)) for _ in range(nnz)])
    matrix = csr_matrix((values, indices, pointers), shape=(nr, nc))
    count = int(next(tokens))
    documents = []
    for _ in range(count):
        words = []
        for _ in range(int(next(tokens))):
            words.append(''.join(chr(int(next(tokens))) for _ in range(int(next(tokens)))))
        documents.append(words)
    assert next(tokens, None) is None
    return matrix, documents


def reconstruct(analyzed, width, sign, binary, norm):
    pointers = [0]
    indices, data = [], []
    for words in analyzed:
        counts = {}
        for word in words:
            h = murmurhash3_32(word, seed=0, positive=False)
            col = abs(h) % width
            counts[col] = counts.get(col, 0) + (-1 if sign and h < 0 else 1)
        row_indices = sorted(counts)
        values = np.array([1. if binary else counts[c] for c in row_indices], dtype=float)
        scale = np.sum(abs(values)) if norm == 'l1' else (np.sqrt(np.sum(values**2)) if norm == 'l2' else 0.)
        if scale > 0:
            values /= scale
        indices.extend(row_indices)
        data.extend(values)
        pointers.append(len(data))
    return csr_matrix((data, indices, pointers), shape=(len(analyzed), width))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--profile', choices=['development', 'checked', 'release'], default='development')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve()
    checks, fingerprints = [], []
    fixtures = []
    for analyzer, gram, norm, sign, binary, lower, stop, minimum in itertools.product(
            range(3), [(1, 1), (1, 3), (2, 4), (5, 7)], range(3), [False, True],
            [False, True], [False, True], range(3), [1, 2]):
        i = len(fixtures)
        width = [1, 2, 7, 31, 1024, 1048576][(i*5+i//6) % 6]
        padded = bool((i//6) % 2)
        rng = np.random.RandomState(i % 3)
        docs = [' The QUICK brown fox and 99 _ x __ 42 foo foo bar  ',
                'a\tb \t C!!_D E-F 00', ' \t\n \x1c\x1d'+ ' '.join(rng.choice(
                    ['alpha', 'BETA', 'the', 'AND', 'one', 'x', '_', '99'], size=8)), '', '_ ! ? a I 9']
        fixtures.append((width, analyzer, gram, norm, sign, binary, lower, stop, minimum, padded, docs, 'grid'))
    for analyzer, norm, sign, binary in itertools.product(range(3), range(3), [False, True], [False, True]):
        fixtures.append((1, analyzer, (1, 3), norm, sign, binary, True, 0, 2, False,
                         ['foo bar', 'foo foo bar', 'bar foo', ''], 'signed_collision'))
    for analyzer, special in itertools.product(range(3), ['ascii', 'long_token', 'english', 'short_word', 'huge_ngram']):
        docs = {'ascii': [''.join(map(chr, range(128)))], 'long_token': ['a'*2048+' BETA'],
                'english': [' '.join(sorted(ENGLISH_STOP_WORDS))+' useful document'],
                'short_word': [' a xy  z '], 'huge_ngram': ['a xy longword']}[special]
        gram = (2147483647, 2147483647) if special == 'huge_ngram' else ((4, 6) if special == 'short_word' else (1, 2))
        fixtures.append((37, analyzer, gram, 2, True, False, True, int(special == 'english'), 2, False, docs, special))
    for index, (width, analyzer, gram, norm, sign, binary, lower, stop, minimum, padded, docs, special) in enumerate(fixtures):
        custom = ['the', 'AND', 'is', '99'] if stop == 2 else []
        stop_words = 'english' if stop == 1 else (custom if stop == 2 else None)
        source_docs = [d.rstrip(' ') for d in docs] if padded else docs
        model = HashingVectorizer(n_features=width, analyzer=ANALYZERS[analyzer], ngram_range=gram,
                                  norm=NORMS[norm], alternate_sign=sign, binary=binary, lowercase=lower,
                                  stop_words=stop_words, token_pattern=rf'(?u)\b\w{{{minimum},}}\b')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reference = model.transform(source_docs)
            source_analyzer = model.build_analyzer()
            analyzed = [list(source_analyzer(d)) for d in source_docs]
        header = [len(docs), width, analyzer, *gram, minimum, int(lower), int(sign), int(binary),
                  int(stop == 1), norm, int(padded), len(custom)]
        payload = ' '.join(map(str, header))+'\n'+''.join(encode(d) for d in custom+docs)
        proc = subprocess.run([str(exe)], input=payload, text=True, capture_output=True)
        assert proc.returncode == 0, (index, header, special, proc.stdout, proc.stderr)
        actual, native_analyzed = parse(proc.stdout)
        assert actual.shape == reference.shape
        np.testing.assert_array_equal(actual.indptr, reference.indptr)
        np.testing.assert_array_equal(actual.indices, reference.indices)
        np.testing.assert_allclose(actual.data, reference.data, rtol=2.e-14, atol=1.e-15,
                                   err_msg=f'{index} {header} {special}')
        assert native_analyzed == analyzed, (index, header, special, native_analyzed, analyzed)
        independent = reconstruct(analyzed, width, sign, binary, NORMS[norm])
        np.testing.assert_array_equal(actual.indptr, independent.indptr)
        np.testing.assert_array_equal(actual.indices, independent.indices)
        np.testing.assert_allclose(actual.data, independent.data, rtol=2.e-14, atol=1.e-15)
        norm_error = 0.
        if norm:
            for row in range(len(docs)):
                values = actual.data[actual.indptr[row]:actual.indptr[row+1]]
                length = np.sum(abs(values)) if norm == 1 else np.sqrt(np.sum(values**2))
                want = float(np.any(values != 0))
                norm_error = max(norm_error, abs(length-want))
                np.testing.assert_allclose(length, want, rtol=2.e-14, atol=1.e-15)
        error = float(np.max(abs(actual.data-reference.data), initial=0.))
        for field, value in [('shape', 0.), ('indptr', 0.), ('indices', 0.), ('data', error),
                             ('analyzer_tokens', 0.), ('independent_signed_hash', 0.),
                             ('row_norms', float(norm_error)), ('stateless_dense_invariants', 0.)]:
            checks.append(dict(fixture=index, field=field, max_abs_error=value))
        fingerprint = lambda words: hashlib.sha256(json.dumps(words, ensure_ascii=True).encode()).hexdigest()
        fingerprints.append(dict(fixture=index, source_sha256=fingerprint(analyzed),
                                 native_sha256=fingerprint(native_analyzed), tokens=sum(map(len, analyzed))))
        if index % 200 == 0:
            print(f'{index}/{len(fixtures)} fixtures passed', flush=True)
    for name, message in REJECTIONS.items():
        proc = subprocess.run([str(exe), name], text=True, capture_output=True)
        assert proc.returncode != 0 and message in proc.stdout+proc.stderr, (name, proc.stdout, proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr, (name, proc.stderr)
    report = dict(profile=args.profile, sklearn_version=sklearn.__version__, fixtures=len(fixtures),
                  checks=checks, analyzer_fingerprints=fingerprints, rejections=list(REJECTIONS),
                  executable=str(exe), executable_sha256=digest(exe), validator_sha256=digest(__file__),
                  source_sha256={'feature_extraction/text.py': digest(source_text.__file__),
                    'feature_extraction/_stop_words.py': digest(Path(source_text.__file__).with_name('_stop_words.py'))},
                  limitations=['ASCII analysis only; Unicode input and Unicode stop words receive diagnostics.',
                               'Character-array inputs discard right-padding spaces; feature_string inputs preserve all bytes.',
                               'token_min_length encodes a word regex; custom callables/regex and accent decoding are absent.'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures, {len(checks)} groups, {len(REJECTIONS)} rejections')


if __name__ == '__main__':
    main()

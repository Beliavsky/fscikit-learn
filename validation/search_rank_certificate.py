"""Source ranking on native scores; certify any source-fit ordering roundoff."""
import numpy as np
from scipy.stats import rankdata


def source_ranks(means):
    ranks = np.ones(means.shape, dtype=int)
    for k, column in enumerate(means.T):
        if np.isnan(column).all():
            continue
        with np.errstate(all='ignore'):
            mapped = np.nan_to_num(column, nan=np.nanmin(column)-1)
        ranks[:,k] = rankdata(-mapped, method='min').astype(int)
    return ranks


def certify(native_means, source_means, native_ranks, source_rank, n_folds):
    expected = source_ranks(native_means)
    np.testing.assert_array_equal(native_ranks, expected)
    certificates = []
    for k in range(native_means.shape[1]):
        if np.array_equal(expected[:,k], source_rank[:,k]):
            continue
        a, b = native_means[:,k], source_means[:,k]
        bound = 128*n_folds*np.finfo(float).eps*max(1., np.max(abs(b[np.isfinite(b)]), initial=0))
        changed = []
        for i in range(len(a)):
            for j in range(i):
                if not all(np.isfinite([a[i],a[j],b[i],b[j]])):
                    # Nonfinite scores must retain the source's exact ranking.
                    assert expected[i,k] == source_rank[i,k] and expected[j,k] == source_rank[j,k]
                    continue
                if np.sign(a[i]-a[j]) != np.sign(b[i]-b[j]):
                    gap = max(abs(a[i]-a[j]),abs(b[i]-b[j]),abs(a[i]-b[i]),abs(a[j]-b[j]))
                    assert gap <= bound, (k,i,j,gap,bound)
                    changed.append(dict(first=i,second=j,max_gap=float(gap)))
        assert changed
        certificates.append(dict(metric=k,absolute_bound=float(bound),changed_pairs=changed))
    return expected, certificates

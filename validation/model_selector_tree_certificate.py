"""Certify native CART splits and importances when source tie traversal differs."""
import numpy as np


def certify(x, y, weights, nodes, importance, classification):
    x = x.astype(np.float32).astype(float)
    labels = np.unique(y)
    reconstructed = np.zeros(x.shape[1])
    visited = set()
    worst_gap = 0.

    def impurity(ids):
        mass = weights[ids].sum()
        if classification:
            probability = np.array([weights[ids][y[ids] == label].sum()/mass for label in labels])
            return 1. - probability @ probability
        mean = np.average(y[ids], weights=weights[ids])
        return np.average((y[ids]-mean)**2, weights=weights[ids])

    def walk(index, ids, depth):
        nonlocal worst_gap
        assert index not in visited and 0 <= index < len(nodes)
        visited.add(index)
        left, right, feature, threshold, stored_impurity, mass = nodes[index]
        left, right, feature = int(left), int(right), int(feature)
        actual_mass = weights[ids].sum(); actual_impurity = impurity(ids)
        np.testing.assert_allclose([mass, stored_impurity], [actual_mass, actual_impurity], rtol=2e-10, atol=2e-10)
        if not left:
            assert not right
            assert depth == 3 or len(ids) < 2 or actual_impurity <= 2e-12
            return
        assert depth < 3 and 1 <= feature <= x.shape[1]
        a = ids[x[ids, feature-1] <= threshold]; b = ids[x[ids, feature-1] > threshold]
        assert len(a) and len(b)
        loss = weights[a].sum()*impurity(a) + weights[b].sum()*impurity(b)
        best = actual_mass*actual_impurity
        for j in range(x.shape[1]):
            unique = np.unique(x[ids, j])
            for cut in unique[:-1]/2 + unique[1:]/2:
                first = ids[x[ids, j] <= cut]; second = ids[x[ids, j] > cut]
                candidate = weights[first].sum()*impurity(first) + weights[second].sum()*impurity(second)
                best = min(best, candidate)
        gap = (loss-best)/max(1., actual_mass*actual_impurity)
        assert gap < 2e-10, ('nonoptimal split', index, gap)
        worst_gap = max(worst_gap, gap)
        reconstructed[feature-1] += max(0., actual_mass*actual_impurity-loss)
        walk(left-1, a, depth+1); walk(right-1, b, depth+1)

    walk(0, np.flatnonzero(weights > 0), 0)
    assert len(visited) == len(nodes)
    if reconstructed.sum():
        reconstructed /= reconstructed.sum()
    np.testing.assert_allclose(importance, reconstructed, rtol=2e-10, atol=2e-10)
    return dict(nodes=len(nodes), worst_relative_split_gap=float(worst_gap),
                max_importance_error=float(np.max(abs(importance-reconstructed), initial=0)))

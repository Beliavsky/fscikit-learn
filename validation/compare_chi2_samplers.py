"""Additive and skewed chi2 feature maps; sklearn receives the native uniform draws."""
from pathlib import Path
from unittest.mock import patch
import json
import subprocess
import numpy as np
import sklearn
from sklearn.kernel_approximation import AdditiveChi2Sampler, SkewedChi2Sampler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 30, 2), (42, 36, 3), (2026, 42, 4)):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 4, size=(n, p)); x[0] = 0; x[1] = 1; x[2, 0] = 0
        data, output = RESULTS/f"chi2_{seed}.txt", RESULTS/f"chi2_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n"); np.savetxt(stream, x, fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_chi2_samplers_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        expected = {}
        for j in range(1, 9):
            steps, interval = j, None
            if j >= 4: steps, interval = {4: (5, .3), 5: (2, 0), 6: (4, 1.2), 7: (8, .25), 8: (2, 300)}[j]
            model = AdditiveChi2Sampler(sample_steps=steps, sample_interval=interval)
            with np.errstate(over="ignore"):
                z = model.fit_transform(x); query = model.transform(x+.031)
            values = dict(train=z, query=query, gram=z@z.T, metadata=[p, z.shape[1]])
            expected.update({f"additive{j}_{name}": value for name, value in values.items()})
        draw_keys = set()
        for j in range(1, 7):
            skew, count, train, query = 1., 64, x, x+.031
            if j == 2: skew, count = .01, 127
            if j == 3: skew, train, query = 0., x+.2, x+.231
            if j == 4: skew, train, query = -.3, x+.5, x+.531
            if j == 5: skew, count, train, query = 2., 1, x-.5, x-.531
            if j == 6: skew, count, query = .5, 256, x-.1
            prefix = f"skewed{j}_"
            u = actual[prefix+"draws_u"].reshape((p, count), order="F")
            v = actual[prefix+"draws_v"]
            assert np.all((u > 0) & (u < 1)) and np.all((v >= 0) & (v < 1))
            draw_keys.update((prefix+"draws_u", prefix+"draws_v"))
            class FixedDraws:
                def uniform(self, low=0., high=1., size=None):
                    values = u if isinstance(size, tuple) else v
                    assert values.shape == (size if isinstance(size, tuple) else (size,))
                    return low+(high-low)*values.copy()
            model = SkewedChi2Sampler(skewedness=skew, n_components=count)
            with patch("sklearn.kernel_approximation.check_random_state", return_value=FixedDraws()):
                z = model.fit_transform(train)
            values = dict(weights=model.random_weights_, offset=model.random_offset_, train=z,
                          query=model.transform(query), gram=z@z.T, metadata=[p, count])
            expected.update({prefix+name: value for name, value in values.items()})
        assert actual.keys() == expected.keys() | draw_keys
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F"); got = actual[name]
            assert got.shape == want.shape, (seed, name, got.shape, want.shape)
            np.testing.assert_allclose(got, want, atol=3e-12, rtol=3e-12, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"chi2_samplers.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} chi2 sampler result groups passed")

if __name__ == "__main__":
    main()

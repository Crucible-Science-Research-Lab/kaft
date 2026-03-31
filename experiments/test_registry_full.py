# test_registry_full.py
import numpy as np
from kaft.geometry import get, list_metrics

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])
x = np.array([1.0, 2.0])
y = np.array([4.0, 6.0])

# classical metrics take vectors, information metrics take distributions
classical = {"euclidean", "minkowski", "gaussian_curved"}

print(f"Registry has {len(list_metrics())} metrics\n")

all_pass = True
for name in sorted(list_metrics()):
    try:
        m = get(name)
        a, b = (x, y) if name in classical else (p, q)
        
        d_pq  = m.compute(a, b)
        d_pp  = m.compute(a, a)
        nonneg = "✓" if d_pq >= -1e-9 else "✗"
        ident  = "✓" if d_pp < 1e-5  else "✗"
        
        # matrix
        vecs = np.array([a, b, (a + b) / 2])
        D = m.distances(vecs)
        matrix = "✓" if D.shape == (3, 3) else "✗"

        print(f"{name:<30} compute={d_pq:.6f}  identity={ident}  non-neg={nonneg}  matrix={matrix}")
        
        if "✗" in [nonneg, ident, matrix]:
            all_pass = False

    except Exception as e:
        print(f"{name:<30} ERROR: {e}")
        all_pass = False

print(f"\n{'✓ All 16 metrics green' if all_pass else '✗ Some metrics need attention'}")
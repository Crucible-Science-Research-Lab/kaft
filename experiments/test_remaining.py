# test_remaining.py
import numpy as np
from kaft.geometry import get

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])
vecs = np.array([p, q, np.array([0.1, 0.1, 0.8])])

metrics_to_test = [
    "kl_divergence",
    "jensen_shannon",
    "fisher_rao",
    "wasserstein_1",
    "wasserstein_2",
]

for name in metrics_to_test:
    m = get(name)
    d_pq = m.compute(p, q)
    d_pp = m.compute(p, p)
    D    = m.distances(vecs)
    
    non_neg = "✓" if d_pq >= 0 else "✗"
    identity = "✓" if abs(d_pp) < 1e-6 else "✗"
    matrix   = "✓" if D.shape == (3, 3) else "✗"
    
    print(f"{name}: compute(p,q)={d_pq:.6f}  compute(p,p)={d_pp:.8f}  "
          f"non-neg={non_neg}  identity={identity}  matrix={matrix}")
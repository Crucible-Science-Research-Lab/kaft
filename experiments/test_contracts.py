# test_contracts.py
import numpy as np
from kaft.geometry import get

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])
r = np.array([0.1, 0.2, 0.7])

SYMMETRIC_METRICS = [
    "euclidean", "minkowski", "gaussian_curved",
    "fisher_rao", "jensen_shannon",
    "alpha_hellinger",        # α=0 symmetric 
    "wasserstein_1", "wasserstein_2",
]

ASYMMETRIC_DIVERGENCES = [
    "kl_divergence",
    "bregman_kl", "bregman_euclidean",
    "bregman_itakura_saito", "bregman_tsallis_2",
    "alpha_bhattacharyya",   # α=0.5  asymmetric, correct by design
    "alpha_reverse",         # α=2.0  asymmetric, correct by design
]

# --- Classical: must be symmetric + satisfy triangle inequality ---
for name in ["euclidean", "minkowski", "gaussian_curved"]:
    m = get(name)
    x, y, z = np.array([0.,0.]), np.array([3.,4.]), np.array([6.,0.])
    assert abs(m.compute(x,y) - m.compute(y,x)) < 1e-9, f"{name} not symmetric!"
    assert m.compute(x,z) <= m.compute(x,y) + m.compute(y,z) + 1e-9, f"{name} triangle violated!"
    print(f"✓ {name}: symmetric + triangle ✓")

# --- Divergences: NOT symmetric — document this as correct behaviour ---
for name in ["kl_divergence", "bregman_kl", "bregman_itakura_saito"]:
    m = get(name)
    d_pq = m.compute(p, q)
    d_qp = m.compute(q, p)
    is_asymmetric = abs(d_pq - d_qp) > 1e-6
    print(f"{'✓' if is_asymmetric else '!'} {name}: D(p||q)={d_pq:.4f}  D(q||p)={d_qp:.4f}  asymmetric={is_asymmetric}")

# --- Statistical distances: symmetric (Hellinger, Wasserstein, Jensen-Shannon) ---
for name in ["alpha_hellinger", "wasserstein_1", "wasserstein_2", "jensen_shannon"]:
    m = get(name)
    d_pq = m.compute(p, q)
    d_qp = m.compute(q, p)
    assert abs(d_pq - d_qp) < 1e-6, f"{name} should be symmetric! {d_pq} vs {d_qp}"
    print(f"✓ {name}: symmetric ✓  D={d_pq:.4f}")
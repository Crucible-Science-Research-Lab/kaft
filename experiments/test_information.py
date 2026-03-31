# test_information_three.py
import numpy as np
from kaft.geometry import get

p = np.array([0.1, 0.4, 0.3, 0.2])
q = np.array([0.25, 0.25, 0.25, 0.25])  # uniform

# --- FisherRao ---
fr = get("fisher_rao")
d = fr.compute(p, q)
assert d >= 0.0, f"Fisher-Rao must be non-negative, got {d}"
assert fr.compute(p, p) < 1e-9, "Fisher-Rao(p,p) must be 0"
assert abs(fr.compute(p, q) - fr.compute(q, p)) < 1e-9, "Fisher-Rao must be symmetric"
print(f"✓ FisherRao   compute(p,q) = {d:.6f}")

# --- KL ---
kl = get("kl_divergence")
d_pq = kl.compute(p, q)
d_qp = kl.compute(q, p)
assert d_pq >= 0.0, f"KL must be non-negative, got {d_pq}"
assert kl.compute(p, p) < 1e-9, "KL(p,p) must be 0"
assert abs(d_pq - d_qp) > 1e-6, f"KL should be ASYMMETRIC — KL(p||q)={d_pq:.4f}, KL(q||p)={d_qp:.4f}"
print(f"✓ KL          compute(p||q) = {d_pq:.6f}  compute(q||p) = {d_qp:.6f}  (asymmetric ✓)")

# --- JensenShannon ---
js = get("jensen_shannon")
d = js.compute(p, q)
assert d >= 0.0, f"JS must be non-negative, got {d}"
assert js.compute(p, p) < 1e-9, "JS(p,p) must be 0"
assert abs(js.compute(p, q) - js.compute(q, p)) < 1e-9, "JS must be symmetric"
assert d <= np.log(2) + 1e-9, f"JS must be bounded by log(2)≈0.693, got {d}"
print(f"✓ JensenShannon compute(p,q) = {d:.6f}  (bounded by log2={np.log(2):.4f} ✓)")

# --- consistency check: compute vs distances matrix diagonal ---
vecs = np.array([p, q])
D_fr = fr.distances(vecs)
assert abs(D_fr[0, 1] - fr.compute(p, q)) < 1e-9, "FisherRao: compute vs distances mismatch"
print("✓ FisherRao   compute consistent with distances matrix")

# test_information.py
import numpy as np
from kaft.geometry import get

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])

# AlphaDivergence scalar
alpha_h = get("alpha_hellinger")
d = alpha_h.compute(p, q)
assert d >= 0, f"Alpha divergence must be non-negative, got {d}"
assert alpha_h.compute(p, p) < 1e-9, "D(p,p) must be 0"
print(f"✓ alpha_hellinger compute: {d:.6f}")

# AlphaDivergence still has fast distances()
vecs = np.array([p, q, np.array([0.1, 0.1, 0.8])])
D = alpha_h.distances(vecs)
assert D.shape == (3, 3)
print(f"✓ alpha_hellinger distances matrix shape: {D.shape}")

# BregmanDivergence scalar
bkl = get("bregman_kl")
d_bkl = bkl.compute(p, q)
assert d_bkl >= 0
assert bkl.compute(p, p) < 1e-6, "Bregman D(p,p) must be 0"
print(f"✓ bregman_kl compute: {d_bkl:.6f}")

# BregmanDivergence gets distances() loop for free from base
D_bkl = bkl.distances(vecs)
assert D_bkl.shape == (3, 3)
print(f"✓ bregman_kl distances matrix shape: {D_bkl.shape}")

# test_wasserstein.py
import numpy as np
from kaft.geometry import get

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])

w1 = get("wasserstein_1")
w2 = get("wasserstein_2")

# scalar
d1 = w1.compute(p, q)
d2 = w2.compute(p, q)
assert d1 >= 0 and d2 >= 0
assert w1.compute(p, p) < 1e-9,  "W(p,p) must be 0"
assert w2.compute(p, p) < 1e-9,  "W(p,p) must be 0"

# symmetry — Wasserstein IS a true metric
assert abs(w1.compute(p, q) - w1.compute(q, p)) < 1e-9, "W1 must be symmetric"
assert abs(w2.compute(p, q) - w2.compute(q, p)) < 1e-9, "W2 must be symmetric"

# matrix via base class loop
vecs = np.array([p, q, np.array([0.1, 0.1, 0.8])])
D1 = w1.distances(vecs)
D2 = w2.distances(vecs)
assert D1.shape == (3, 3) and D2.shape == (3, 3)

print(f"✓ wasserstein_1 compute: {d1:.6f}")
print(f"✓ wasserstein_2 compute: {d2:.6f}")
print(f"✓ symmetry holds for W1 and W2")
print(f"✓ distance matrices: {D1.shape}")
print("\n✓ WassersteinMetric fully verified")

print("\n✓ All information metrics verified")

print("\n✓ All three information metrics verified")
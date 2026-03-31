# import numpy as np
# from kaft.geometry import DivergenceRegistry   # your actual import path
# r = DivergenceRegistry()
# p = np.array([0.5, 0.3, 0.2])
# q = np.array([0.4, 0.35, 0.25])
# for name in ["bregman_kl", "bregman_tsallis_2", "bregman_euclidean", "bregman_itakura_saito"]:
#     d = r.get(name).distances(p, q)
#     print(f"{name:28s}  D(p||q) = {d:.6f}")

# add to your test file or run separately
import numpy as np
from kaft.geometry import get

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])
vecs = np.array([p, q, np.array([0.1, 0.1, 0.8])])

for name in ["alpha_hellinger", "alpha_bhattacharyya", "alpha_reverse",
             "bregman_kl", "bregman_euclidean", "bregman_itakura_saito",
             "bregman_tsallis_2", "bregman_tsallis_0.5"]:
    m = get(name)
    d_pq = m.compute(p, q)
    d_pp = m.compute(p, p)
    D    = m.distances(vecs)
    flag_nn  = "✓" if d_pq >= 0       else "✗ NEGATIVE"
    flag_id  = "✓" if d_pp < 1e-6     else "✗ IDENTITY FAIL"
    flag_mat = "✓" if D.shape==(3,3)  else "✗ MATRIX SHAPE"
    print(f"{name}: compute(p,q)={d_pq:.6f}  compute(p,p)={d_pp:.8f}  non-neg={flag_nn}  identity={flag_id}  matrix={flag_mat}")

# class BregmanDivergence:
#     def __init__(self, F, grad_F, hess_F=None, name="bregman"):
#         self.F = F
#         self.grad_F = grad_F
#         self.hess_F = hess_F
#         self._name = name

#     def distances(self, p: np.ndarray, q: np.ndarray) -> float:
#         p, q = np.asarray(p, float), np.asarray(q, float)
#         return float(self.F(p) - self.F(q) - np.dot(self.grad_F(q), p - q))

#     def geometry_type(self):
#         return f"bregman_{self._name}"


# # --- KL as Bregman ---
# kl = BregmanDivergence(
#     F      = lambda p: np.sum(p * np.log(p + 1e-12)),
#     grad_F = lambda p: np.log(p + 1e-12) + 1.0,
#     hess_F = lambda p: 1.0 / (p + 1e-12),
#     name   = "kl"
# )

# p = np.array([0.5, 0.3, 0.2])
# q = np.array([0.4, 0.35, 0.25])

# d_pq = kl.distances(p, q)
# d_qp = kl.distances(q, p)

# print(f"Bregman(KL gen): D(p||q) = {d_pq:.6f}")
# print(f"Bregman(KL gen): D(q||p) = {d_qp:.6f}")
# print(f"Asymmetric (expected): {d_pq != d_qp}")
# print(f"Non-negative (expected): {d_pq >= 0 and d_qp >= 0}")


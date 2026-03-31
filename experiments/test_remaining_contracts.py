
import numpy as np
from kaft.geometry import get

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])

remaining = [
    "fisher_rao",
    "alpha_bhattacharyya",
    "alpha_reverse",
    "bregman_euclidean",
    "bregman_tsallis_2",
    "bregman_tsallis_0.5",
]

for name in remaining:
    m = get(name)
    dpq = m.compute(p, q)
    dqp = m.compute(q, p)
    dpp = m.compute(p, p)
    is_symmetric = abs(dpq - dqp) < 1e-9
    identity_holds = dpp < 1e-9
    print(f"{name}:")
    print(f"  D(p||q)={dpq:.4f}  D(q||p)={dqp:.4f}  symmetric={is_symmetric}  D(p,p)≈0={identity_holds}")
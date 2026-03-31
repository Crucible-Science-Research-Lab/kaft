# test_phase4_information.py
import numpy as np
from kaft.geometry import get

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])

for name in ["kl_divergence", "jensen_shannon", "fisher_rao"]:
    m = get(name)
    d = m.compute(p, q)
    d_self = m.compute(p, p)
    print(f"{name}: compute(p,q)={d:.6f}  compute(p,p)={d_self:.6f}  "
          f"non-negative={'✓' if d >= 0 else '✗'}  "
          f"identity={'✓' if d_self < 1e-6 else '✗'}")
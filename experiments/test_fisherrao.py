from kaft.geometry import get
import numpy as np

p = np.array([0.5, 0.3, 0.2])
fr = get("fisher_rao")

assert fr.compute(p, p) < 1e-9,  f"D(p,p) should be 0, got {fr.compute(p, p)}"
assert fr.compute(p, p) >= 0,    "non-negative"

q = np.array([0.2, 0.5, 0.3])
assert abs(fr.compute(p, q) - fr.compute(q, p)) < 1e-9, "Fisher-Rao must be symmetric"

print(f"✓ fisher_rao D(p,p) = {fr.compute(p, p):.2e}")
print(f"✓ fisher_rao D(p,q) = {fr.compute(p, q):.6f}")
print(f"✓ symmetric: {abs(fr.compute(p,q) - fr.compute(q,p)) < 1e-9}")
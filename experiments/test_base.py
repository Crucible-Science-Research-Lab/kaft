# test_base.py
import numpy as np
from kaft.geometry.base import AbstractMetric, DivergenceRegistry

# --- verify the abstract contract ---
class DummyMetric(AbstractMetric):
    def compute(self, a, b):
        return float(np.sqrt(np.sum((a - b) ** 2)))
    def geometry_type(self):
        return "dummy_euclidean"

m = DummyMetric()

# compute works
d = m.compute(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
assert abs(d - 5.0) < 1e-9, f"Expected 5.0, got {d}"

# distances() derived for free — no override needed
vecs = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 0.0]])
D = m.distances(vecs)
assert D.shape == (3, 3)
assert abs(D[0, 1] - 5.0) < 1e-9
assert abs(D[1, 0] - 5.0) < 1e-9   # symmetry
assert D[0, 0] == 0.0               # diagonal

# registry stores instances cleanly
reg = DivergenceRegistry()
reg.register("dummy", m)
assert reg.get("dummy") is m
assert "dummy" in reg.available()

print("✓ base.py contract verified")
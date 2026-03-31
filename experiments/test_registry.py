# test_registry.py
import numpy as np
from kaft.geometry import get, list_metrics

p = np.array([0.4, 0.35, 0.25])
q = np.array([0.2, 0.5, 0.3])
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

classical = {"euclidean", "minkowski", "gaussian_curved"}

print(f"Registry has {len(list_metrics())} metrics:\n")

for name in list_metrics():
    metric = get(name)
    try:
        a, b = (x, y) if name in classical else (p, q)
        d = metric.compute(a, b)
        self_d = metric.compute(a, a)
        assert self_d < 1e-6, f"{name}: D(x,x)={self_d} should be ~0"
        print(f"  ✓ {name:<30} compute={d:.6f}  geometry='{metric.geometry_type()}'")
    except Exception as e:
        print(f"  ✗ {name:<30} FAILED: {e}")

print(f"\n✓ Registry end-to-end complete")
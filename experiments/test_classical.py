# test_classical.py
import numpy as np
from kaft.geometry.base import AbstractMetric

# bootstrap runs on import — if this passes, all 3 classes instantiate cleanly
import kaft.geometry

# pull metrics from registry
euclid  = kaft.geometry.get("euclidean")
mink    = kaft.geometry.get("minkowski")
curved  = kaft.geometry.get("gaussian_curved")

a = np.array([0.0, 0.0])
b = np.array([3.0, 4.0])

# compute() works on all three
assert abs(euclid.compute(a, b) - 5.0) < 1e-9
assert abs(mink.compute(a, b) - 5.0) < 1e-9
assert abs(curved.compute(a, b) - 5.0) < 1e-9

# distances() matrix still works
vecs = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 0.0]])
D = euclid.distances(vecs)
assert D.shape == (3, 3)
assert abs(D[0, 1] - 5.0) < 1e-9
assert abs(D[1, 0] - D[0, 1]) < 1e-9  # symmetry

# speed_field untouched
assert mink.speed_field(10).mean() == 0.6
sf = curved.speed_field(100)
assert sf[50, 50] < 0.2  # centre is slow — curvature bump
assert sf[0, 0] > 0.9    # corners are fast — flat

print("✓ classical.py verified — all 3 metrics clean")
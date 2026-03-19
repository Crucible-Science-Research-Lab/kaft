# save as experiments/exp_geomstats_compare.py
import numpy as np
import geomstats

from geomstats.geometry.hypersphere import Hypersphere
from kaft.core import build_manifold
from kaft.core.metric import FisherRaoMetric

# Build our manifold
records = [
    {"text": "CRISPR-Cas9 gene editing mechanism"},
    {"text": "protein folding molecular dynamics"},
    {"text": "Fisher-Rao information geometry manifold"},
    {"text": "Riemannian curvature tensor geodesic"},
]
state  = build_manifold(records)
metric = FisherRaoMetric(state)
D_kaft = metric.compute()

# Normalise embeddings onto the unit hypersphere S^383
E = state.embeddings
E_normalised = E / np.linalg.norm(E, axis=1, keepdims=True)

# geomstats hypersphere S^383
sphere = Hypersphere(dim=383)

print("── Our FisherRaoMetric (arccos) vs geomstats Hypersphere ──\n")
pairs = [(0,1), (0,2), (0,3), (1,2), (2,3)]
for i, j in pairs:
    d_ours = D_kaft[i, j]
    d_geom = sphere.metric.dist(E_normalised[i], E_normalised[j])
    diff   = abs(d_ours - d_geom)
    match  = "✓ match" if diff < 1e-4 else f"Δ = {diff:.6f}"
    print(f"  [{i}]↔[{j}]  ours={d_ours:.4f}  geomstats={d_geom:.4f}  {match}")

print()
print("── What geomstats gives us for FREE ──")
# Logarithmic map: tangent vector from point i toward point j
log = sphere.metric.log(E_normalised[1], E_normalised[0])
print(f"  log map (tangent vector) shape : {log.shape}")
print(f"  → this IS the direction of parallel transport P(V)")

# Exponential map: travel distance dt along a geodesic
exp = sphere.metric.exp(log * 0.1, E_normalised[0])
print(f"  exp map (step along geodesic)  shape: {exp.shape}")
print(f"  → this IS how KAFTSimulator will evolve K along ∇^FR")

print()
print("geomstats experiment ✓")

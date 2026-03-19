import sys
import numpy as np
from kaft.core.manifold import build_manifold
from kaft.core.topology import KDensity
from kaft.navigate.geodesic import GeodesicNavigator
from kaft.navigate.seeder import DomainSeeder
from kaft.core.metric import FisherRaoMetric

CORPUS = [
    {"text": "Riemannian tensor curvature manifold geometry"},
    {"text": "Fisher information matrix statistical manifold"},
    {"text": "geodesic distance information geometry"},
    {"text": "Gaussian distribution probability density function"},
    {"text": "softmax attention transformer neural network"},
    {"text": "CRISPR Cas9 gene editing DNA repair"},
    {"text": "protein folding molecular dynamics simulation"},
    {"text": "mRNA vaccine immune response spike protein"},
    {"text": "drug target interaction binding affinity"},
    {"text": "statistical mechanics entropy thermodynamics"},
    {"text": "topology boundary detection algebraic homology"},
    {"text": "Kullback Leibler divergence information theory"},
]

print("=" * 60)
print("KAFT SESSION 3 — GeodesicPath Experiment")
print("=" * 60)

# Build manifold — same as Shadow Principle corpus
state = build_manifold(CORPUS)
print(f"\n[Manifold] {state.embeddings.shape[0]} points in {state.embeddings.shape[1]}-dim Fisher-Rao space")

# Attach texts to state for navigator labels
# state.texts = [d["text"] for d in CORPUS]

# Compute and attach K-density to state
metric = FisherRaoMetric(state)
metric.compute()
kdensity = KDensity(state, metric)
state.k_density = kdensity.compute()

# Find the two most distant points — maximize geodesic length
# Rthe max_dist search loop:
print("\n[Search] Finding maximal Fisher-Rao pair...")
D = metric.distances
max_dist = 0
src, tgt = 0, 1
for i in range(len(CORPUS)):
    for j in range(i + 1, len(CORPUS)):
        if D[i, j] > max_dist:
            max_dist = D[i, j]
            src, tgt = i, j


print(f"  Source [{src}]: {state.records[src]}")
print(f"  Target [{tgt}]: {state.records[tgt]}")
print(f"  Fisher-Rao distance: {max_dist:.4f}")

# Trace geodesic
navigator = GeodesicNavigator(state, metric, n_steps=15)
path = navigator.trace(src, tgt)

print(f"\n[GeodesicPath] Total FR length: {path.total_fr_length:.4f}")
print(f"  Waypoints: {len(path.waypoints)}")
print(f"  Boundary crossings detected: {len(path.boundary_crossings)}")

print("\n[K-Terrain along geodesic]")
print(f"  {'t':>5}  {'K-density':>10}  Nearest concept")
print("  " + "-" * 55)
for wp in path.waypoints:
    marker = " ←←← BOUNDARY" if any(abs(wp.t - bc[0]) < 0.05 for bc in path.boundary_crossings) else ""
    print(f"  {wp.t:>5.2f}  {wp.k_density:>10.4f}  {wp.nearest_text[:40]}{marker}")

if path.boundary_crossings:
    print("\n[Jordan Boundary Crossings]")
    for t_val, label in path.boundary_crossings:
        print(f"  t={t_val:.2f} — {label}")

# Save manifold state for Wonder navigation layer
DomainSeeder.save(state, "kaft/experiments/geometric_state.json")

print("\n[Done] Geodesic traced. Manifold saved.")
print("  Next: Wonder Navigation layer can load geometric_state.json")
print("        and render this terrain as the live 2D topology map.")

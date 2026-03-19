"""
Session 4 — Real corpus experiment.
arXiv papers → manifold → geodesic → Jordan boundaries.
Same pipeline as exp_geodesic.py but on actual research literature.
"""
from kaft.core.manifold import build_manifold
from kaft.core.metric import FisherRaoMetric
from kaft.core.topology import KDensity
from kaft.navigate.geodesic import GeodesicNavigator
from kaft.navigate.seeder import DomainSeeder
from kaft.ingest.router import ArxivRouter
import time
import os

print("=" * 60)
print("KAFT SESSION 4 — Real arXiv Corpus")
print("=" * 60)

router = ArxivRouter()

# Two semantically distant domains — same logic as synthetic corpus
# but now real papers, real abstracts, real geometry
# math_papers   = router.fetch("Fisher information geometry Riemannian manifold", max_results=8)
# bio_papers    = router.fetch("CRISPR Cas9 gene editing drug discovery", max_results=8)

# corpus = math_papers + bio_papers
DOMAINS = [
    {"query": "Fisher information geometry Riemannian manifold", "n": 6, "label": "math"},
    {"query": "CRISPR Cas9 gene editing drug discovery",         "n": 6, "label": "biology"},
    {"query": "tokamak plasma confinement fusion energy",        "n": 6, "label": "fusion"},
    {"query": "transformer attention mechanism neural network",  "n": 6, "label": "ai"},
]
corpus = []
for domain in DOMAINS:
    papers = router.fetch(domain["query"], max_results=domain["n"])
    for p in papers:
        p["domain_label"] = domain["label"]  # tag each record with its domain
    corpus.extend(papers)
    time.sleep(5)

print(f"\n[Corpus] {len(corpus)} real papers across {len(DOMAINS)} domains")


# Build manifold — identical call to synthetic experiment
state = build_manifold(corpus)
print(f"[Manifold] {state.embeddings.shape[0]} points in {state.embeddings.shape[1]}-dim Fisher-Rao space")

# Metric + density — same pipeline
metric = FisherRaoMetric(state)
metric.compute()
kdensity = KDensity(state, metric)
state.k_density = kdensity.compute()

# Find maximal Fisher-Rao pair
print("\n[Search] Finding maximal Fisher-Rao pair...")
D = metric.distances
max_dist, src, tgt = 0, 0, 1
for i in range(len(corpus)):
    for j in range(i + 1, len(corpus)):
        if D[i, j] > max_dist:
            max_dist = D[i, j]
            src, tgt = i, j

print(f"  Source [{src}]: {corpus[src]['title'][:60]}")
print(f"  Target [{tgt}]: {corpus[tgt]['title'][:60]}")
print(f"  Fisher-Rao distance: {max_dist:.4f}")

# Trace geodesic
navigator = GeodesicNavigator(state, metric, n_steps=15)
path = navigator.trace(src, tgt)

print(f"\n[GeodesicPath] Total FR length: {path.total_fr_length:.4f}")
print(f"  Boundary crossings: {len(path.boundary_crossings)}")

print("\n[K-Terrain along geodesic]")
print(f"  {'t':>5}  {'K-density':>9}  Nearest paper")
print("  " + "-" * 65)
for wp in path.waypoints:
    marker = " ←←← BOUNDARY" if any(abs(wp.t - bc[0]) < 0.05 for bc in path.boundary_crossings) else ""
    print(f"  {wp.t:>5.2f}  {wp.k_density:>9.4f}  {wp.nearest_text[:45]}{marker}")

if path.boundary_crossings:
    print("\n[Jordan Boundaries on real corpus]")
    for t_val, label in path.boundary_crossings:
        print(f"  t={t_val:.2f} — {label}")

# Save for Wonder navigation layer
BASE = os.path.dirname(os.path.abspath(__file__))
DomainSeeder.save(state, os.path.join(BASE, "geometric_state_arxiv.json"))
print("\n[Done] Real corpus manifold saved.")

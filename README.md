
# kaft

**Geometric dynamics on information manifolds.**

[![PyPI version](https://badge.fury.io/py/kaft.svg)](https://badge.fury.io/py/kaft)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

```bash
pip install kaft
```

---

## What kaft does

`kaft` implements geometric dynamics on Riemannian information manifolds.

Standard approaches treat knowledge as accumulation in flat Euclidean space like
dot product similarity, cosine distance, softmax attention.
Čencov's uniqueness theorem (1981) proves this is mathematically forced to be wrong:
Fisher-Rao is the **only** Riemannian metric invariant under sufficient-statistic
transformations on probability manifolds.

`kaft` builds on that foundation:

- Embeds any corpus into a unified Fisher-Rao metric space
- Computes K-density fields via inverse square law interaction dynamics
- Detects Jordan-Brouwer domain boundaries as emergent geometry, not imposed constraints
- Evolves knowledge state via the master field equation across timesteps
- Traces geodesic paths between concepts on the curved manifold
- Persists and reloads manifold state for session continuity

---

## Core concepts

**Fisher-Rao metric**
The unique invariant Riemannian metric on statistical manifolds (Čencov 1981).
Geodesic distance = arccos of cosine similarity on the hypersphere S^383.
Validated against geomstats to 4 decimal places.

**K-density field**
Inverse square law interaction: `K ∝ I₁·I₂ / d²`
where d is geodesic distance on the Fisher-Rao manifold, not Euclidean distance.
High K = dense semantic cluster. Low K = open conceptual space.

**Jordan boundaries**
Emergent domain separators.
They crystallise as level sets of K-density gradients from the geometry itself.
Hallucination prevention: crossing a boundary requires discrete energy accumulation,
not smooth interpolation.

**Knowledge evolution**
Master field equation:
`∂K/∂t = c·v²_cog·K² + P(V)·∇^FR(K) + ξ_J(t)`
K² term: self-amplifying dynamics — understanding compounds.
Fisher-Rao gradient: geodesic flow along information-theoretically optimal paths.
Jordan noise: topologically constrained stochastic exploration within domains.

---

## Quickstart

### Build a manifold

```python
from kaft.core.manifold import build_manifold

corpus = [
    {"text": "Fisher-Rao information geometry manifold"},
    {"text": "Riemannian curvature tensor geodesic"},
    {"text": "CRISPR-Cas9 gene editing mechanism"},
    {"text": "mRNA vaccine immunological response"},
    {"text": "statistical mechanics entropy thermodynamics"},
]

state = build_manifold(corpus)
print(state.embeddings.shape)  # (5, 384)
```

### Compute Fisher-Rao geometry

```python
from kaft.core.metric import FisherRaoMetric
from kaft.core.topology import KDensity, JordanBoundary

metric = FisherRaoMetric(state)
metric.compute()

kdensity = KDensity(state, metric)
K = kdensity.compute()
# K[i] ∈  — normalised interaction density at each point [actualisedaily](https://actualisedaily.com/transformation/transformative-crucibles-and-the-adventure-of-life/)

boundaries = JordanBoundary(state, kdensity).detect()
# Boundaries emerge from K-density gradients — not imposed
for b in boundaries:
    print(f"Boundary at: {b['text']} | energy barrier: {b['energy_barrier']:.4f}")
```

### Evolve knowledge dynamics

```python
from kaft.simulate.kaft import KAFTSimulator
from kaft.simulate.softmax import SoftmaxSimulator
from kaft.simulate.compare import compare

result = compare(KAFTSimulator(), SoftmaxSimulator(), state, n_steps=50)
print(f"Divergence: {result['divergence']:.4f}")
# KAFT sees domain structure — varied K across clusters
# Softmax sees uniform K=1.000 across all points
```

### Navigate geodesics

```python
from kaft.navigate.geodesic import GeodesicNavigator

navigator = GeodesicNavigator(state, metric, n_steps=15)
path = navigator.trace(source_idx=0, target_idx=2)

print(f"Fisher-Rao path length: {path.total_fr_length:.4f}")
for wp in path.waypoints:
    print(f"t={wp.t:.2f}  K={wp.k_density:.4f}  {wp.nearest_text[:40]}")

# Jordan boundary crossings detected automatically
for t_val, label in path.boundary_crossings:
    print(f"t={t_val:.2f} — {label}")
```

### Real corpus via arXiv

```python
from kaft.ingest.router import ArxivRouter

router = ArxivRouter()
records = router.fetch("Fisher information geometry", max_results=10)
state = build_manifold(records)
```

### Persist manifold state

```python
from kaft.navigate.seeder import DomainSeeder

DomainSeeder.save(state, "domain.json")          # compute once
state = DomainSeeder.load("domain.json")          # reload instantly
```

---

## Validation

| Experiment | Result |
|---|---|
| Fisher-Rao vs geomstats | Exact match to 4 decimal places |
| Jordan boundaries | Emerge on real 24-paper arXiv corpus (4 domains) |
| K² soliton | Topology locks at step 10, stable through step 50 |
| Geometric vs flat dynamics | Divergence D=0.2230 on real corpus |
| 3-SAT phase transition α=5.0 | 95.73% ±1.03% satisfaction, O(N^1.17) scaling |

---

## Installation

```bash
pip install kaft
```

**Development:**
```bash
git clone https://github.com/Crucible-Science-Research-Lab/kaft
cd kaft
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Background

`kaft` is the reference implementation of
**Knowledge Attention Field Theory** -
a geometric framework for knowledge dynamics on Riemannian information manifolds.

Built at [Crucible Science](https://cruciblescience.com).
arXiv paper: *forthcoming — endorsements in progress*

---

## License

MIT — see [LICENSE](LICENSE)

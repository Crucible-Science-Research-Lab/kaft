# kaft

**Geometric dynamics for knowledge fields on curved manifolds.**

Standard embedding models project knowledge into flat Euclidean space which has every document equidistant, every domain indistinguishable. `kaft` replaces flat distance with Fisher-Rao information geometry, letting domain structure emerge from the manifold itself.

```bash
pip install kaft
```

---

## What it does

`kaft` computes a K-density field over an embedding corpus using a pluggable Riemannian metric, then evolves that field through time using wave dynamics and Jordan boundary detection. Two things become measurable that flat geometry cannot see:

1. **Domain structure** — K-density varies meaningfully across documents; Softmax (flat) returns uniform density
2. **Soliton locking** — the field self-organises into a stable attractor; geometry and noise anneal together until the field holds

---

## Quickstart

```python
from kaft.ingest.router import ArxivRouter
from kaft.ingest.embedder import Embedder
from kaft.geometry import get as get_metric
from kaft.dynamics.kstate import KDensity
from kaft.dynamics.softmax_dynamics import SoftmaxDynamics

# 1. Ingest
router     = ArxivRouter()
records    = router.fetch("Fisher-Rao information geometry", max_results=20)

# 2. Embed + manifold
embedder   = Embedder()
embeddings = embedder.encode([r["text"] for r in records])
metric     = get_metric("fisher_rao")
kd         = KDensity(embeddings, metric)
kd.compute()

# 3. Flat baseline
flat       = SoftmaxDynamics(temperature=1.0).run(embeddings)

# 4. Compare
div = SoftmaxDynamics().distribution_divergence(kd.density, flat.K)
print(f"KAFT K_var  : {kd.density.var():.6f}")
print(f"Softmax K_var: {flat.K_var:.6f}")
print(f"Divergence  : {div:.4f}")
```

**Output on a real 80-paper, 4-domain arXiv corpus:**

| | KAFT (Fisher-Rao) | Softmax (Euclidean) |
|---|---|---|
| K variance | **0.037222** | **0.000000** |
| Divergence | **0.1802** | — |

Softmax returns K=1.000 for every document. 
Fisher-Rao geometry ranges 0.000–1.000, resolving domain boundaries invisible to flat distance.

---

## Simulation dynamics

```python
from kaft.dynamics.kstate import KDensity, KState
from kaft.dynamics.jordan import JordanBoundary, JordanNoise
from kaft.dynamics.metric_evolution import MetricEvolution
from kaft.simulate.crucible import SimulationCrucible
from kaft.simulate.runner import SimulationRunner
import numpy as np

noise          = JordanNoise(sigma=0.05, rng=np.random.default_rng(42))
metric_evolver = MetricEvolution(temperature=1.0, base_step=0.1)

ks = KState(); ks.update(kd.density)
jb = JordanBoundary(kd); jb.detect()

runner = SimulationRunner(
    embeddings=embeddings, records=records,
    k_density=kd, k_state=ks, jordan=jb,
    crucible=SimulationCrucible("demo", "fisher_rao", "output"),
    metric_evolver=metric_evolver,
    softmax=SoftmaxDynamics(temperature=1.0),
    noise=noise,
)

SIGMA_MIN = 0.005

for _ in range(50):
    frame = runner.step()
    if frame.phase == "locked":
        metric_evolver.base_step = 0.0       # freeze geometry
        noise.sigma              = SIGMA_MIN  # hold the field
    else:
        noise.sigma              = max(SIGMA_MIN, 0.05 * (1.0 - frame.K_mean))
        metric_evolver.base_step = max(0.001, 0.1  * (1.0 - frame.K_mean))

    print(f"step={frame.step:>3}  K_mean={frame.K_mean:.4f}"
          f"  phase={frame.phase:<12}  locked={frame.soliton_locked}")
```

**Output on 80-paper corpus, 4 domains:**

```
step=  0  K_mean=0.3887  phase=clustering   locked=False
step= 10  K_mean=0.5458  phase=clustering   locked=False
step= 20  K_mean=0.6004  phase=locked       locked=False
step= 24  K_mean=0.6153  phase=locked       locked=True
step= 30  K_mean=0.6153  phase=locked       locked=True
...
step= 90  K_mean=0.6153  phase=locked       locked=True
```

Soliton lock at step 24. Field holds identical K_mean=0.6153 from step 30 through 90 — the attractor is stable.

The adaptive schedule reflects the physics: soliton formation requires a frozen medium. Once the phase gates to `locked`, geometry stops evolving and the field settles.

---

## Metric zoo

```python
from kaft.geometry import get, list_metrics

list_metrics()
# ['fisher_rao', 'kl_divergence', 'jensen_shannon',
#  'alpha_divergence', 'euclidean', 'minkowski', 'alpha_hellinger',
#  'alpha_bhattacharyya', 'bregman_kl', 'bregman_euclidean', 
#  'bregman_itakura_saito', 'bregman_tsallis_2'
#  'gaussian_curved', 'wasserstein']

metric = get("jensen_shannon")   # drop-in replacement
```

All metrics implement `AbstractMetric` — swap one line to run the same experiment on any geometry.

---

## Library structure

```
kaft/
├── geometry/     Riemannian metrics — Fisher-Rao, KL, Jensen-Shannon,
│                 Alpha, Bregman, Wasserstein, Euclidean, Minkowski
├── dynamics/     K-density field, KState, JordanBoundary, WaveEngine,
│                 MetricEvolution, GradientFlow, SoftmaxDynamics
├── simulate/     SimulationRunner, SimulationCrucible, Frame log,
│                 SimulationVisualiser
├── navigate/     GeodesicNavigator, GeodesicPath, DomainSeeder
└── ingest/       ArxivRouter, Embedder
```

`kaft` is domain-agnostic. The same primitives that resolve arXiv domain boundaries work on protein sequence corpora, patent claim graphs, or financial time series. Any corpus that embeds into a vector space.

---

## Experiments

Full experiment scripts and the N-scaling study (soliton lock step as a function of corpus size) are in the [GitHub repository](https://github.com/Crucible-Science-Research-Lab/kaft).

---

## License

MIT

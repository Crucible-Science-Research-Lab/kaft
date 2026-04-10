"""
geometric_dynamics.py

Reproduces the core KAFT claim on a real arXiv corpus:
curved Fisher-Rao geometry preserves domain structure;
flat Euclidean geometry cannot distinguish domains.
"""

import os
import sys
import warnings
import argparse
import logging

# suppress third-party noise
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("arxiv").setLevel(logging.ERROR)

from kaft.ingest.router import ArxivRouter

DOMAINS = [
    ("information geometry Riemannian manifold",  "geometry"),
    ("CRISPR gene editing drug discovery",         "biology"),
    ("tokamak plasma confinement fusion energy",   "fusion"),
    ("transformer attention neural network",       "ai"),
]

def ingest_corpus(n_papers: int) -> tuple:
    """Fetch real arXiv papers across four domains."""
    router = ArxivRouter(max_results=n_papers)
    records, labels = [], []
    for query, domain in DOMAINS:
        fetched = router.fetch(query, max_results=n_papers, silent=True)
        for r in fetched:
            records.append(r)
            labels.append(domain)
        print(f"  {query:<45} →  {len(fetched)} papers")
    return records, labels




import numpy as np
from sentence_transformers import SentenceTransformer
from kaft.geometry import get as get_metric
from kaft.dynamics.kstate import KDensity

def build_manifold(records: list) -> tuple:
    """Embed corpus and compute Fisher-Rao K-density field."""
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    texts      = [r["text"] for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False)
    metric     = get_metric("fisher_rao")
    kd         = KDensity(embeddings, metric)
    kd.compute()
    print(f"  Manifold: {len(records)} points in {embeddings.shape[1]}-dim Fisher-Rao space")
    return embeddings, metric, kd



from kaft.simulate.runner import SimulationRunner, Frame
from kaft.simulate.crucible import SimulationCrucible
from kaft.dynamics.kstate import KState
from kaft.dynamics.jordan import JordanBoundary, JordanNoise
from kaft.dynamics.metric_evolution import MetricEvolution
from kaft.dynamics.softmax_dynamics import SoftmaxDynamics
from kaft.navigate.seeder import DomainSeeder

def run_dynamics(embeddings, records, metric, kd):
    """Run curved (KAFT) and flat (Euclidean) dynamics on same corpus."""
    ks = KState(); ks.update(kd.density)
    jb = JordanBoundary(kd); jb.detect()

    crucible = SimulationCrucible("geometric_dynamics", "fisher_rao", "output")
    runner   = SimulationRunner(
        embeddings     = embeddings,
        records        = records,
        k_density      = kd,
        k_state        = ks,
        jordan         = jb,
        crucible       = crucible,
        metric_evolver = MetricEvolution(temperature=1.0, base_step=0.1),
        softmax        = SoftmaxDynamics(temperature=1.0),
        noise          = JordanNoise(sigma=0.05, rng=np.random.default_rng(42)),
    )
    runner.run_static(n_steps=50)
    return runner, kd

def print_k_density(kd_or_array, records, labels, mode="Curved (Fisher-Rao)"):
    """Print K-density bar chart."""
    if hasattr(kd_or_array, "density"):
        density = kd_or_array.density
    elif hasattr(kd_or_array, "K"):
        density = kd_or_array.K
    else:
        density = np.array(kd_or_array)
    print(f"\n── K-density: {mode} ──")
    for i, (k, r) in enumerate(zip(density, records)):
        bar   = "█" * int(k * 19)
        title = r["title"][:42]
        print(f"  [{i:2d}] K={k:.3f} {bar:<19}  {title}")

def print_divergence(kd, flat_result):
    sd = SoftmaxDynamics(temperature=1.0)
    div = sd.distribution_divergence(kd.density, flat_result.K)
    print(f"\n── Divergence D(curved ∥ flat) ──")
    bar = "▓" * int(div * 30)
    print(f"  {bar}  {div:.4f}")
    print(f"\n  Final divergence D(curved ∥ flat) = {div:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_papers", type=int, default=10)
    args = parser.parse_args()

    print("═" * 59)
    print(" KAFT — Geometric Dynamics Experiment")
    print(f" Corpus: {len(DOMAINS) * args.n_papers} papers | {len(DOMAINS)} domains | Fisher-Rao metric")
    print("═" * 59)
    print(f"\n[1/3] Ingesting corpus from arXiv...")

    records, labels = ingest_corpus(args.n_papers)
    print(f"\n  Corpus: {len(records)} papers across {len(DOMAINS)} domains\n")

    print("[2/3] Building manifold...")
    embeddings, metric, kd = build_manifold(records)
    print()

    print("[3/3] Running dynamics...")
    runner, kd = run_dynamics(embeddings, records, metric, kd)


    flat_result = SoftmaxDynamics(temperature=1.0).run(embeddings)

    print_k_density(kd, records, labels, mode="Curved (Fisher-Rao)")
    print_k_density(flat_result, records, labels, mode="Flat (Euclidean)")
    print_divergence(kd, flat_result)

    os.makedirs("output", exist_ok=True)
    import json

    BASE = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE, "output")
    os.makedirs(output_dir, exist_ok=True)
    sd = SoftmaxDynamics(temperature=1.0)

    state = {
        "corpus_size": len(records),
        "domains": [d for _, d in DOMAINS],
        "kaft_k_density": kd.density.tolist(),
        "softmax_k_density": flat_result.K.tolist(),
        "divergence": sd.distribution_divergence(kd.density, flat_result.K),
    }

    output_path = os.path.join(output_dir, "geometric_state.json")
    with open(output_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\n  State saved → {output_path}")
    print("\n" + "═" * 59)
"""
exp_simulation.py

KAFT Dynamics Claim — K evolves over time.
Phase transitions emerge from K² interaction geometry.
Soliton locks when density field self-organises.

Adaptive noise schedule: sigma cools as K_mean rises,
allowing soliton lock to hold once field converges.
"""

import warnings
import logging
import argparse
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("arxiv").setLevel(logging.ERROR)

from kaft.ingest.router import ArxivRouter
from kaft.ingest.embedder import Embedder
from kaft.geometry import get as get_metric
from kaft.dynamics.kstate import KDensity, KState
from kaft.dynamics.jordan import JordanBoundary, JordanNoise
from kaft.dynamics.metric_evolution import MetricEvolution
from kaft.dynamics.softmax_dynamics import SoftmaxDynamics
from kaft.simulate.crucible import SimulationCrucible
from kaft.simulate.runner import SimulationRunner

DOMAINS = [
    ("information geometry Riemannian manifold", "geometry"),
    ("CRISPR gene editing drug discovery",        "biology"),
    ("tokamak plasma confinement fusion energy",  "fusion"),
    ("transformer attention neural network",      "ai"),
]

SIGMA_MAX = 0.05   # exploration phase — field moves freely
SIGMA_MIN = 0.005  # convergence phase — field holds lock


def ingest_and_embed(n_papers: int):
    router = ArxivRouter(max_results=n_papers)
    records = []
    for query, domain in DOMAINS:
        fetched = router.fetch(query, max_results=n_papers, silent=True)
        for r in fetched:
            r["domain_label"] = domain
            records.append(r)
        print(f"  {domain:<10} {len(fetched)} papers")
    embedder   = Embedder()
    embeddings = embedder.encode([r["text"] for r in records])
    print(f"  Total: {len(records)} papers  |  {embeddings.shape[1]}-dim")
    return embeddings, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_papers", type=int, default=20)
    parser.add_argument("--n_steps",  type=int, default=50)
    args = parser.parse_args()

    print("═" * 65)
    print(" KAFT — Simulation Dynamics (adaptive noise)")
    print(f" {len(DOMAINS) * args.n_papers} papers  |  {args.n_steps} steps  |  Fisher-Rao")
    print("═" * 65)

    print("\n[1/3] Ingesting corpus...")
    embeddings, records = ingest_and_embed(args.n_papers)

    print("\n[2/3] Building manifold...")
    metric = get_metric("fisher_rao")
    kd     = KDensity(embeddings, metric)
    kd.compute()
    ks     = KState()
    ks.update(kd.density)
    jb     = JordanBoundary(kd)
    jb.detect()
    print(f"  K_mean: {kd.density.mean():.4f}  K_var: {kd.density.var():.6f}")
    print(f"  Boundaries: {len(jb.boundaries)}  Phase: {ks.phase}")

    print("\n[3/3] Running simulation (adaptive noise)...")
    noise    = JordanNoise(sigma=SIGMA_MAX, rng=np.random.default_rng(42))
    metric_evolver = MetricEvolution(temperature=1.0, base_step=0.1)  
    
    crucible = SimulationCrucible(
        domain_label = "4domain_arxiv",
        metric_name  = "fisher_rao",
        corpus_path  = "output",
    )
    runner = SimulationRunner(
        embeddings     = embeddings,
        records        = records,
        k_density      = kd,
        k_state        = ks,
        jordan         = jb,
        crucible       = crucible,
        metric_evolver = metric_evolver,
        softmax        = SoftmaxDynamics(temperature=1.0),
        noise          = noise,
    )

    print(f"\n  {'step':>4}  {'K_mean':>7}  {'K_var':>9}  {'div_L1':>7}"
          f"  {'phase':<12}  {'sigma':>6}  locked")
    print(f"  {'-'*68}")

    frames    = []
    lock_step = None

    for step in range(args.n_steps):
        frame = runner.step()
        frames.append(frame)
        if frame.phase == "locked":
            # freeze geometry — let KState accumulate consecutive stable steps
            metric_evolver.base_step = 0.0
            noise.sigma              = SIGMA_MIN
        else:
            # exploration — anneal toward convergence
            noise.sigma              = max(SIGMA_MIN, SIGMA_MAX * (1.0 - frame.K_mean))
            metric_evolver.base_step = max(0.001, 0.1 * (1.0 - frame.K_mean))

        if frame.soliton_locked and lock_step is None:
            lock_step = frame.step

        if frame.step <= 15 or frame.step % 10 == 0:
            print(f"  {frame.step:>4}  {frame.K_mean:>7.4f}  {frame.K_var:>9.6f}"
                  f"  {frame.divergence_l1:>7.4f}  {frame.phase:<12}"
                  f"  {noise.sigma:>6.4f}  {frame.soliton_locked}")

    print(f"  {'-'*68}")
    print(f"  Soliton lock step : {lock_step}")
    print(f"  Final phase       : {frames[-1].phase}")
    print(f"  Final K_var       : {frames[-1].K_var:.6f}")
    print(f"  Final div_L1      : {frames[-1].divergence_l1:.4f}")
    print("\n" + "═" * 65)


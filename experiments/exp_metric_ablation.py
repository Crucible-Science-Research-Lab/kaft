# experiments/exp_metric_ablation.py
"""
Multi-metric ablation — same corpus, same geodesic, three metrics side by side.

FisherRao vs Euclidean vs Wasserstein on the seeded 200-paper corpus.
Čencov's theorem made empirically visible: metric choice changes what the
geometry sees, and FisherRao finds the most structure.
"""
import numpy as np
from kaft.navigate.seeder import DomainSeeder
from kaft.navigate.geodesic import GeodesicNavigator
from kaft.dynamics.kstate import KDensity, KState
from kaft.dynamics.jordan import JordanBoundary
from kaft.geometry.information import FisherRaoMetric
from kaft.geometry.classical import EuclideanMetric
from kaft.geometry.information import WassersteinMetric

SEED_PATH = "experiments/seeds/test_corpus.json"

METRICS = [
    ("FisherRao",  FisherRaoMetric()),
    ("Euclidean",  EuclideanMetric()),
    ("Wasserstein",WassersteinMetric()),
]

# math_geometry[0] → ai_transformer[15] — cross-domain path
SOURCE_IDX = 0
TARGET_IDX = 100


def ablation_for_metric(embeddings, records, metric, name):
    kd = KDensity(embeddings, metric)
    kd.compute()

    ks = KState()
    ks.update(kd.density)

    jb = JordanBoundary(kd)
    jb.detect()

    nav  = GeodesicNavigator(embeddings, records, metric, n_steps=20)
    path = nav.trace(SOURCE_IDX, TARGET_IDX)

    k_arr = np.array([wp.k_density for wp in path.waypoints])

    return {
        "name":          name,
        "K_mean":        float(ks.K),
        "K_var":         float(np.var(kd.density)),
        "boundaries":    len(jb.boundaries),
        "geodesic_len":  path.total_length,
        "path_k_min":    float(k_arr.min()),
        "path_k_max":    float(k_arr.max()),
        "path_k_range":  float(k_arr.max() - k_arr.min()),
        "crossings":     len(path.boundary_crossings),
        "crossing_types": [c[1] for c in path.boundary_crossings],
        "k_profile":     k_arr.tolist(),
        "waypoints":     path.waypoints,
    }


def main():
    print("\n" + "═" * 68)
    print("  Multi-metric ablation — FisherRao vs Euclidean vs Wasserstein")
    print("  exp_metric_ablation.py")
    print("═" * 68)

    # Load warm corpus — no arXiv calls, no embedding recompute
    base = DomainSeeder.load(SEED_PATH, metric=FisherRaoMetric())
    embeddings = base["embeddings"]
    records    = base["records"]

    print(f"\n  Corpus: {len(records)} records | Dim: {embeddings.shape[1]}")
    print(f"  Path  : [{records[SOURCE_IDX].get('domain','?')}][{SOURCE_IDX}]"
          f" → [{records[TARGET_IDX].get('domain','?')}][{TARGET_IDX}]")
    print(f"\n  SOURCE: {records[SOURCE_IDX].get('text','')[:80]}…")
    print(f"  TARGET: {records[TARGET_IDX].get('text','')[:80]}…")

    results = []
    for name, metric in METRICS:
        print(f"\n  Running {name}…")
        r = ablation_for_metric(embeddings, records, metric, name)
        results.append(r)

    # ── comparison table ──────────────────────────────────────────────────────
    print(f"\n\n{'═'*68}")
    print("  METRIC ABLATION — corpus topology")
    print(f"{'═'*68}")
    print(f"  {'Metric':>12}  {'K_mean':>8}  {'K_var':>10}  {'Boundaries':>12}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*12}")
    for r in results:
        print(f"  {r['name']:>12}  {r['K_mean']:>8.4f}  {r['K_var']:>10.6f}"
              f"  {r['boundaries']:>12}")

    print(f"\n{'═'*68}")
    print("  METRIC ABLATION — geodesic path")
    print(f"{'═'*68}")
    print(f"  {'Metric':>12}  {'length':>8}  {'K_range':>10}"
          f"  {'crossings':>10}  boundary types")
    print(f"  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*30}")
    for r in results:
        ctypes = " | ".join(
            "ridge" if "maximum" in t else "valley"
            for t in r["crossing_types"]
        ) or "none"
        print(f"  {r['name']:>12}  {r['geodesic_len']:>8.4f}"
              f"  {r['path_k_range']:>10.4f}  {r['crossings']:>10}  {ctypes}")

    # ── K profile comparison ──────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  K-DENSITY PROFILE ALONG GEODESIC  (t=0→1)")
    print(f"{'═'*68}")
    print(f"  {'t':>6}  {'FisherRao':>12}  {'Euclidean':>12}  {'Wasserstein':>12}  nearest domain (FR)")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*20}")
    n_steps = len(results[0]["k_profile"])
    for i in range(n_steps):
        t      = i / (n_steps - 1)
        domain = records[results[0]["waypoints"][i].nearest_idx].get("domain", "?")
        vals   = [r["k_profile"][i] for r in results]
        print(f"  {t:>6.2f}  {vals[0]:>12.4f}  {vals[1]:>12.4f}  {vals[2]:>12.4f}  {domain}")

    print(f"\n  Expected: FisherRao shows richest K variation along path.")
    print(f"  Expected: Euclidean K profile is flatter — less domain sensitivity.")
    print(f"  Čencov: FisherRao is the only information-invariant metric.")
    print("═" * 68)


if __name__ == "__main__":
    main()
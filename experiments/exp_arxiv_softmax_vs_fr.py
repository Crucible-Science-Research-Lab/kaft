#!/usr/bin/env python
"""exp_arxiv_softmax_vs_fr.py

KAFT vs Softmax on real arXiv corpus.

Pipeline:
    ArxivRouter.fetch_batch(query, sizes)  →  list[RouterRecord]
        ↓
    Embedder.encode([r["abstract"] for r in records])  →  (N, 384) raw embs
        ↓
    KDensity(raw_emb, FisherRaoMetric())   →  K_kaft   (N,)   ← geodesic gravity
    SoftmaxDynamics().run(raw_emb)         →  K_soft   (N,)   ← flat attention
        ↓
    Compare: variance, shadow divergence, N-scaling sweep

Expected smoking gun:
    - K_kaft.var  >>>  K_soft.var                  (KAFT sees domain structure)
    - shadow_divergence ≈ 0.22+ and soliton-locks  (matches Mar-19 result)
    - var_kaft grows with N;  var_soft near-zero    (geometry improves at scale)

Usage:
    cd /path/to/kaft
    PYTHONPATH=. python experiments/exp_arxiv_softmax_vs_fr.py
"""

from __future__ import annotations

import time
from typing import List, Dict

import numpy as np

# ── imports from kaft library ─────────────────────────────────────────────────
from kaft.ingest.router import ArxivRouter
from kaft.ingest.embedder import Embedder
from kaft.geometry.information import FisherRaoMetric
from kaft.dynamics.kstate import KDensity
from kaft.dynamics.softmax_dynamics import SoftmaxDynamics

# ── experiment config ─────────────────────────────────────────────────────────
DOMAINS: List[Dict] = [
    {"label": "math_geometry",     "query": "information geometry Fisher-Rao manifold"},
    {"label": "biology_crispr",    "query": "CRISPR gene editing biological systems"},
    {"label": "fusion_plasma",     "query": "tokamak plasma confinement fusion energy"},
    {"label": "ai_transformer",    "query": "transformer attention mechanism large language model"},
]

# Papers per domain for each sweep step
N_SWEEP: List[int] = [25, 50, 100, 200]    # total N; per-domain = N // 4

SEP = "─" * 68


# ── helpers ───────────────────────────────────────────────────────────────────

# def fetch_corpus(n_total: int, router: ArxivRouter) -> tuple[List[str], List[str]]:
#     """Fetch n_total papers evenly across DOMAINS.
#     Returns (abstracts, domain_labels) — labels are ground truth, never fed to geometry.
#     """
#     per_domain = max(1, n_total // len(DOMAINS))
#     abstracts, labels = [], []
#     for d in DOMAINS:
#         records = router.fetch_batch(d["query"], sizes=[per_domain])
#         for r in records:
#             abstracts.append(r["abstract"] or r["title"])
#             labels.append(d["label"])
#     return abstracts, labels
def fetch_corpus(
    router:  ArxivRouter,
    n_sweep: list[int],
) -> dict[int, tuple[list[str], list[str]]]:
    """One fetch_batch call per domain covers all sweep sizes.
    Returns dict[n_total -> (texts, labels)].
    labels = ground truth from config, never fed to geometry.
    """
    per_domain_sizes = sorted(set(max(1, n // len(DOMAINS)) for n in n_sweep))

    domain_cache: dict[str, dict[int, list]] = {}
    for d in DOMAINS:
        print(f"  [{d['label']}]  fetching sizes={per_domain_sizes}…")
        domain_cache[d["label"]] = router.fetch_batch(
            d["query"], sizes=per_domain_sizes
        )

    result: dict[int, tuple[list[str], list[str]]] = {}
    for n_total in n_sweep:
        per_n = max(1, n_total // len(DOMAINS))
        texts, labels = [], []
        for d in DOMAINS:
            for r in domain_cache[d["label"]].get(per_n, []):
                texts.append(r["text"])      # title + ". " + abstract
                labels.append(d["label"])
        result[n_total] = (texts, labels)

    return result


def run_comparison(
    raw_emb: np.ndarray,
    labels:  List[str],
    n_label: str = "",
) -> Dict:
    """One KAFT vs Softmax run on a fixed embedding matrix.

    Returns dict of stats for the N-sweep table.
    """
    metric  = FisherRaoMetric()

    # ── KAFT ──────────────────────────────────────────────────────────────────
    kd       = KDensity(raw_emb, metric)
    K_kaft   = kd.density        # (N,)  — if your KDensity uses .compute_K(), swap here

    # ── Softmax ───────────────────────────────────────────────────────────────
    sm       = SoftmaxDynamics(temperature=1.0)
    sm_res   = sm.run(raw_emb)
    K_soft   = sm_res.K      # (N,)

    # ── shadow divergence (soliton signal) ────────────────────────────────────
    divergence = sm.shadow_divergence(K_kaft, K_soft)

    # ── variance ratio (smoking gun) ─────────────────────────────────────────
    var_kaft = float(np.var(K_kaft))
    var_soft = float(np.var(K_soft))
    var_ratio = var_kaft / (var_soft + 1e-12)

    result = {
        "N":           len(raw_emb),
        "K_kaft_mean": float(K_kaft.mean()),
        "K_kaft_std":  float(K_kaft.std()),
        "K_kaft_var":  var_kaft,
        "K_soft_mean": sm_res.K_mean,
        "K_soft_std":  sm_res.K_std,
        "K_soft_var":  var_soft,
        "var_ratio":   var_ratio,
        "divergence":  divergence,
        "K_kaft":      K_kaft,
        "K_soft":      K_soft,
        "labels":      labels,
    }

    # ── print per-run summary ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  N = {len(raw_emb)}{' (' + n_label + ')' if n_label else ''}")
    print(SEP)
    print(f"  {'':30s}  {'KAFT':>10}  {'Softmax':>10}")
    print(f"  {'K mean':30s}  {result['K_kaft_mean']:>10.4f}  {result['K_soft_mean']:>10.4f}")
    print(f"  {'K std':30s}  {result['K_kaft_std']:>10.4f}  {result['K_soft_std']:>10.4f}")
    print(f"  {'K var  ← smoking gun':30s}  {var_kaft:>10.6f}  {var_soft:>10.6f}")
    print(f"  {'var ratio  (KAFT / Softmax)':30s}  {var_ratio:>10.2f}x")
    print(f"  {'shadow divergence':30s}  {divergence:>10.4f}")

    # top-5 KAFT attractors
    top5_idx = np.argsort(K_kaft)[::-1][:5]
    print(f"\n  Top-5 KAFT attractors:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    [{rank}] [{labels[idx]:20s}]  K={K_kaft[idx]:.4f}")

    # bottom-3 peripherals
    bot3_idx = np.argsort(K_kaft)[:3]
    print(f"  Bottom-3 peripherals:")
    for idx in bot3_idx:
        print(f"        [{labels[idx]:20s}]  K={K_kaft[idx]:.4f}")

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 68)
    print("  KAFT vs Softmax — arXiv corpus experiment")
    print("  exp_arxiv_softmax_vs_fr.py")
    print("═" * 68)
    print(f"  Domains : {[d['label'] for d in DOMAINS]}")
    print(f"  N sweep : {N_SWEEP}")

    router   = ArxivRouter(page_size=100, delay_seconds=5.0)
    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    # ── fetch ONCE for all sweep levels ──────────────────────────────────────
    print(f"\n  Fetching corpus ({len(DOMAINS)} domains × 1 call each)…")
    t0           = time.time()
    corpus_cache = fetch_corpus(router, N_SWEEP)
    print(f"  Done in {time.time() - t0:.1f}s")

    sweep_results = []

    # ── sweep ─────────────────────────────────────────────────────────────────
    for n_total in N_SWEEP:
        texts, labels = corpus_cache[n_total]
        actual_n      = len(texts)

        print(f"\n{'━'*68}")
        print(f"  N={n_total} → {actual_n} records after domain merge")

        print(f"  Embedding…")
        raw_emb = embedder.encode(texts)

        res = run_comparison(raw_emb, labels, n_label=f"N={actual_n}")
        sweep_results.append(res)

    # ── N-scaling summary table ───────────────────────────────────────────────
    print(f"\n\n{'═'*68}")
    print("  N-SCALING SWEEP — smoking gun table")
    print("═" * 68)
    print(f"  {'N':>6}  {'var_KAFT':>12}  {'var_Soft':>12}  {'ratio':>8}  {'div':>8}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
    for r in sweep_results:
        print(f"  {r['N']:>6}  {r['K_kaft_var']:>12.6f}  {r['K_soft_var']:>12.6f}"
              f"  {r['var_ratio']:>7.1f}x  {r['divergence']:>8.4f}")

    print(f"\n  Expected: var_KAFT grows with N, var_Soft near-zero.")
    print(f"  Expected: divergence plateaus (soliton lock) ≈ 0.22+")
    print("═" * 68)


if __name__ == "__main__":
    main()
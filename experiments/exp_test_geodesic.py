# experiments/test_geodesic_smoke.py
"""Smoke test — GeodesicNavigator on real arXiv corpus."""
import numpy as np
from kaft.ingest.router import ArxivRouter
from kaft.ingest.embedder import Embedder
from kaft.geometry.information import FisherRaoMetric
from kaft.navigate.geodesic import GeodesicNavigator

DOMAINS = [
    {"label": "math_geometry",  "query": "information geometry Fisher-Rao manifold"},
    {"label": "biology_crispr", "query": "CRISPR gene editing biological systems"},
    {"label": "fusion_plasma",  "query": "tokamak plasma confinement fusion energy"},
    {"label": "ai_transformer", "query": "transformer attention mechanism large language model"},
]
N_PER_DOMAIN = 50   #  total — enough for a clean trace

def main():
    # ── ingest ────────────────────────────────────────────────────────────────
    router   = ArxivRouter(page_size=100, delay_seconds=5.0)
    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    records = []
    for d in DOMAINS:
        batch = router.fetch_batch(d["query"], sizes=[N_PER_DOMAIN])
        for r in batch.get(N_PER_DOMAIN, []):
            r["domain"] = d["label"]
            records.append(r)

    print(f"\n  Corpus: {len(records)} records across {len(DOMAINS)} domains")

    texts     = [r["text"] for r in records]
    raw_emb   = embedder.encode(texts)          # (N, 384)

    # ── navigator ─────────────────────────────────────────────────────────────
    metric = FisherRaoMetric()
    nav    = GeodesicNavigator(raw_emb, records, metric, n_steps=20)
    print(f"  KDensity warmed — distance cache ready\n")

    # ── trace: math_geometry [0] → ai_transformer [36] ───────────────────────
    # indices 0–11  = math_geometry
    # indices 12–23 = biology_crispr
    # indices 24–35 = fusion_plasma
    # indices 36–47 = ai_transformer
    source_idx, target_idx = 0, 36

    path = nav.trace(source_idx, target_idx)

    print(f"  SOURCE  [{records[source_idx]['domain']}]")
    print(f"    {path.source_text[:80]}…")
    print(f"\n  TARGET  [{records[target_idx]['domain']}]")
    print(f"    {path.target_text[:80]}…")
    print(f"\n  Total geodesic length : {path.total_length:.4f}")
    print(f"  Waypoints             : {len(path.waypoints)}")

    # ── K profile ─────────────────────────────────────────────────────────────
    print(f"\n  K-density profile along path:")
    print(f"  {'t':>6}  {'K':>10}  {'nearest domain':>20}  nearest text")
    print(f"  {'─'*6}  {'─'*10}  {'─'*20}  {'─'*40}")
    for wp in path.waypoints:
        domain = records[wp.nearest_idx].get("domain", "?")
        print(f"  {wp.t:>6.2f}  {wp.k_density:>10.4f}  {domain:>20}  {wp.nearest_text[:40]}…")

    # ── boundary crossings ────────────────────────────────────────────────────
    print(f"\n  Boundary crossings detected: {len(path.boundary_crossings)}")
    for t_cross, label in path.boundary_crossings:
        print(f"    t={t_cross:.2f}  →  {label}")

if __name__ == "__main__":
    main()
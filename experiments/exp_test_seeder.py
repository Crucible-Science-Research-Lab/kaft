# experiments/test_seeder.py
"""Smoke test — DomainSeeder save + load round-trip on real arXiv corpus."""
import numpy as np
from kaft.ingest.router import ArxivRouter
from kaft.ingest.embedder import Embedder
from kaft.geometry.information import FisherRaoMetric
from kaft.dynamics.kstate import KDensity, KState
from kaft.dynamics.jordan import JordanBoundary
from kaft.navigate.seeder import DomainSeeder

DOMAINS = [
    {"label": "math_geometry",  "query": "information geometry Fisher-Rao manifold"},
    {"label": "ai_transformer", "query": "transformer attention mechanism large language model"},
]
N_PER_DOMAIN = 100
SAVE_PATH    = "experiments/seeds/test_corpus.json"

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

    print(f"\n  Corpus: {len(records)} records")
    raw_emb = embedder.encode([r["text"] for r in records])

    # ── build kaft objects ────────────────────────────────────────────────────
    metric = FisherRaoMetric()
    kd     = KDensity(raw_emb, metric)
    kd.compute()

    ks = KState()
    ks.update(kd.density)

    jb = JordanBoundary(kd)
    jb.detect()

    print(f"  K={ks.K:.3f} | Phase={ks.phase} | Boundaries={len(jb.boundaries)}")

    # ── save ──────────────────────────────────────────────────────────────────
    DomainSeeder.save(
        path=SAVE_PATH,
        embeddings=raw_emb,
        records=records,
        metric=metric,
        k_density=kd,
        k_state=ks,
        jordan=jb,
    )

    # ── load and verify round-trip ────────────────────────────────────────────
    print(f"\n  Loading from seed…")
    state = DomainSeeder.load(SAVE_PATH, metric=FisherRaoMetric())

    assert state["embeddings"].shape == raw_emb.shape,  "embeddings shape mismatch"
    assert len(state["records"]) == len(records),        "records count mismatch"
    assert len(state["k_density"].density) == len(records), "k_density length mismatch"
    assert state["k_density"]._distances is not None,   "distance cache not restored"
    assert len(state["boundaries"]) == len(jb.boundaries), "boundaries count mismatch"

    print(f"\n  Round-trip assertions: ALL GREEN ✓")
    print(f"  K restored  : {state['k_value']:.3f}")
    print(f"  Phase       : {state['phase']}")
    print(f"  Boundaries  : {len(state['boundaries'])}")
    print(f"\n  Sample boundary:")
    if state["boundaries"]:
        b = state["boundaries"][0]
        print(f"    energy={b['energy_barrier']:.4f} | {b['text'][:80]}…")

    # ── verify navigator works from loaded state (no recompute) ──────────────
    from kaft.navigate.geodesic import GeodesicNavigator
    nav  = GeodesicNavigator(
        state["embeddings"], state["records"],
        FisherRaoMetric(), n_steps=10
    )
    path = nav.trace(0, N_PER_DOMAIN)   # math_geometry[0] → ai_transformer[0]
    print(f"\n  Navigator from seed:")
    print(f"    Geodesic length : {path.total_length:.4f}")
    print(f"    Boundary crossings: {len(path.boundary_crossings)}")
    print(f"\n  Seeder smoke test complete 🎯")

if __name__ == "__main__":
    main()
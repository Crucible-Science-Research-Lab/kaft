# experiments/exp_2d_topology_map.py
"""
2D Topology Map — KAFT knowledge manifold visualised.

Loads seeded corpus, projects to 2D via UMAP, renders:
  - Nodes sized by K-density (gravitational mass)
  - Colored by domain
  - Jordan boundary points marked
  - FisherRao geodesic arc overlaid
  - Euclidean straight-line chord overlaid for contrast
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP

from kaft.navigate.seeder import DomainSeeder
from kaft.navigate.geodesic import GeodesicNavigator
from kaft.dynamics.kstate import KDensity, KState
from kaft.dynamics.jordan import JordanBoundary
from kaft.geometry.information import FisherRaoMetric
from kaft.geometry.classical import EuclideanMetric

SEED_PATH  = "experiments/seeds/test_corpus.json"
OUTPUT_PNG = "experiments/output/kaft_topology_map.png"
SOURCE_IDX = 0
TARGET_IDX = 100


def main():
    # ── load warm corpus ──────────────────────────────────────────────────────
    state      = DomainSeeder.load(SEED_PATH, metric=FisherRaoMetric())
    embeddings = state["embeddings"]
    records    = state["records"]
    kd         = state["k_density"]
    jb         = state["jordan"]
    density    = kd.density                     # (N,) normalised [0,1]

    print(f"  Corpus: {len(records)} | K={state['k_value']:.3f}"
          f" | Boundaries: {len(jb.boundaries)}")

    # ── UMAP projection ───────────────────────────────────────────────────────
    print("  Projecting to 2D via UMAP…")
    reducer = UMAP(n_components=2, random_state=42,
                n_neighbors=15, min_dist=0.1)
    coords  = reducer.fit_transform(embeddings)
    print(f"  UMAP done: {coords.shape}")

    # ── geodesic waypoints projected ─────────────────────────────────────────
    print("  Tracing FisherRao geodesic…")
    fr_nav  = GeodesicNavigator(embeddings, records, FisherRaoMetric(), n_steps=40)
    fr_path = fr_nav.trace(SOURCE_IDX, TARGET_IDX)

    # Project waypoint embeddings through UMAP transform
    # wp_embs = np.array([wp.embedding for wp in fr_path.waypoints])
    # wp_2d   = UMAP(n_components=2, random_state=42,
    #                n_neighbors=15, min_dist=0.1).fit_transform(
    #                    np.vstack([embeddings, wp_embs])
    #                )[len(embeddings):]
    
    # Project waypoint embeddings through same UMAP fit
    wp_embs = np.array([wp.embedding for wp in fr_path.waypoints])
    wp_2d   = reducer.transform(wp_embs)

    # ── domain colors ─────────────────────────────────────────────────────────
    domain_list   = [r.get("domain", "unknown") for r in records]
    unique_domains = sorted(set(domain_list))
    palette = {
        "math_geometry":  "#4f98a3",
        "biology_crispr": "#6daa45",
        "fusion_plasma":  "#fdab43",
        "ai_transformer": "#a86fdf",
        "unknown":        "#999999",
    }
    colors = [palette.get(d, "#999999") for d in domain_list]

    # ── boundary point indices ────────────────────────────────────────────────
    boundary_idxs = set(b["boundary_point"] for b in jb.boundaries)

    # ── node sizes from K-density ─────────────────────────────────────────────
    sizes = 20 + density * 180    # 20px min → 200px max

    # ── render ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0e0c")
    ax.set_facecolor("#171614")

    # Corpus nodes
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes, c=colors, alpha=0.75,
        linewidths=0.3, edgecolors="white",
        zorder=2
    )

    # Jordan boundary points — highlight as low-density ridge markers
    b_coords = coords[list(boundary_idxs)]
    ax.scatter(
        b_coords[:, 0], b_coords[:, 1],
        s=12, c="#ff4466", alpha=0.6,
        marker="x", linewidths=1.0,
        zorder=3, label="Jordan boundary"
    )

    # FisherRao geodesic arc
    ax.plot(
        wp_2d[:, 0], wp_2d[:, 1],
        color="#4f98a3", linewidth=2.2, alpha=0.9,
        zorder=4, label="FisherRao geodesic"
    )

    # Euclidean straight chord
    src_2d = coords[SOURCE_IDX]
    tgt_2d = coords[TARGET_IDX]
    ax.plot(
        [src_2d[0], tgt_2d[0]], [src_2d[1], tgt_2d[1]],
        color="#fdab43", linewidth=1.5, alpha=0.7,
        linestyle="--", zorder=4, label="Euclidean chord"
    )

    # Source / target markers
    ax.scatter(*src_2d, s=220, c="#ffffff", zorder=5, marker="*")
    ax.scatter(*tgt_2d, s=220, c="#ffffff", zorder=5, marker="*")
    ax.annotate("SOURCE", src_2d, color="#ffffff", fontsize=7,
                xytext=(6, 6), textcoords="offset points")
    ax.annotate("TARGET", tgt_2d, color="#ffffff", fontsize=7,
                xytext=(6, 6), textcoords="offset points")

    # ── legend ────────────────────────────────────────────────────────────────
    domain_patches = [
        mpatches.Patch(color=palette.get(d, "#999"), label=d.replace("_", " "))
        for d in unique_domains
    ]
    geo_line  = plt.Line2D([0], [0], color="#4f98a3", lw=2, label="FisherRao geodesic")
    euc_line  = plt.Line2D([0], [0], color="#fdab43", lw=1.5,
                           linestyle="--", label="Euclidean chord")
    bnd_mark  = plt.Line2D([0], [0], marker="x", color="#ff4466",
                           lw=0, markersize=6, label="Jordan boundary")

    ax.legend(
        handles=domain_patches + [geo_line, euc_line, bnd_mark],
        loc="lower right", fontsize=8,
        facecolor="#1c1b19", edgecolor="#393836",
        labelcolor="white", framealpha=0.85
    )

    # ── labels ────────────────────────────────────────────────────────────────
    ax.set_title(
        "KAFT Knowledge Manifold — arXiv Corpus Topology",
        color="white", fontsize=14, pad=14, fontweight="bold"
    )
    ax.set_xlabel("UMAP dim 1", color="#7a7974", fontsize=9)
    ax.set_ylabel("UMAP dim 2", color="#7a7974", fontsize=9)
    ax.tick_params(colors="#7a7974")
    for spine in ax.spines.values():
        spine.set_edgecolor("#393836")

    ax.text(
        0.01, 0.99,
        f"N={len(records)} | K={state['k_value']:.3f}"
        f" | Phase={state['phase']} | Boundaries={len(jb.boundaries)}",
        transform=ax.transAxes, color="#7a7974", fontsize=8,
        va="top", ha="left"
    )

    import os
    os.makedirs("experiments/output", exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Map saved → {OUTPUT_PNG}")
    print(f"  Boundary crossings on FR path: {len(fr_path.boundary_crossings)}")
    for t, label in fr_path.boundary_crossings:
        print(f"    t={t:.2f} → {label}")


if __name__ == "__main__":
    main()
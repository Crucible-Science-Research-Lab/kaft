"""
visualiser.py — SimulationVisualiser: FrameLog → publication-quality matplotlib figures.

Four plot types, one summary figure:
    plot_k_evolution()       — K_mean trajectory with 5F phase bands + soliton lock
    plot_variance()          — K_var KAFT vs Softmax over steps (primary paper signal)
    plot_k_distributions()   — K_field vs K_softmax histogram at a specific frame
    plot_divergence()        — L1 divergence growth over steps
    summary()                — 2×2 multi-panel combining all four (paper-ready)

Usage
-----
    vis = SimulationVisualiser(runner.log)
    vis.summary(save_path="figures/kaft_vs_softmax.pdf")
    vis.plot_variance()   # standalone
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from kaft.simulate.runner import Frame


# ── Style constants ───────────────────────────────────────────────────────────

KAFT_COLOR    = "#01696f"   # Teal  — KAFT geometric signal
SOFTMAX_COLOR = "#da7101"   # Amber — Softmax flat baseline
LOCK_COLOR    = "#a12c7b"   # Plum  — soliton lock marker

PHASE_COLORS = {
    "Holofractal":         "#e8f4f8",
    "Resonance_Spheres":   "#e8f5e9",
    "Fractal_Causality":   "#fff8e1",
    "Toroidal_Resonance":  "#fff3e0",
    "Fracture_Lines":      "#fce4ec",
}

PHASE_LABELS = {
    "Holofractal":         "H — Holofractal",
    "Resonance_Spheres":   "RS — Resonance",
    "Fractal_Causality":   "FC — Fractal Causality",
    "Toroidal_Resonance":  "TR — Toroidal",
    "Fracture_Lines":      "FL — Fracture Lines",
}

def _pub_style() -> None:
    """Apply publication-quality rcParams — call once per session."""
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
        "lines.linewidth":   1.8,
    })


def _annotate_phases(ax: plt.Axes, frames: List[Frame]) -> None:
    """Shade 5F phase regions as background bands."""
    if not frames:
        return

    phase_runs: list[tuple[int, int, str]] = []
    start = 0
    current = frames[0].phase

    for i, f in enumerate(frames[1:], 1):
        if f.phase != current:
            phase_runs.append((start, i - 1, current))
            start, current = i, f.phase
    phase_runs.append((start, len(frames) - 1, current))

    for x0, x1, phase in phase_runs:
        ax.axvspan(x0, x1, alpha=0.15,
                   color=PHASE_COLORS.get(phase, "#f5f5f5"),
                   label=PHASE_LABELS.get(phase, phase))


def _mark_soliton(ax: plt.Axes, frames: List[Frame]) -> None:
    """Vertical dashed line at first soliton lock step."""
    for f in frames:
        if f.soliton_locked:
            ax.axvline(f.step, color=LOCK_COLOR, linestyle="--",
                       linewidth=1.2, alpha=0.8, label=f"Soliton lock (step {f.step})")
            break


def plot_topology_map(
    self,
    embeddings:  np.ndarray,
    records:     list[dict],
    umap_reducer = None,
    source_idx:  int = 0,
    target_idx:  int = -1,
    metric = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    2D UMAP topology map — nodes sized by K-density, colored by domain,
    Jordan boundaries marked, FR geodesic arc overlaid.

    Parameters
    ----------
    embeddings   : (N, D) raw embeddings
    records      : parallel record list
    umap_reducer : fitted UMAP instance or None (fits fresh)
    source_idx   : geodesic source node index
    target_idx   : geodesic target node index (-1 = last node)
    metric       : AbstractMetric for geodesic tracing
    save_path    : optional PNG/PDF output path
    """
    from umap import UMAP
    from kaft.navigate.geodesic import GeodesicNavigator
    import matplotlib.patches as mpatches

    last_frame = self.log[-1]
    density    = last_frame.K_field
    N          = len(embeddings)
    target_idx = target_idx % N

    # UMAP projection — reuse existing fit if passed in
    if umap_reducer is None:
        umap_reducer = UMAP(n_components=2, random_state=42,
                            n_neighbors=15, min_dist=0.1)
        coords = umap_reducer.fit_transform(embeddings)
    else:
        coords = umap_reducer.transform(embeddings)

    # Geodesic arc (only if metric provided)
    wp_2d = None
    if metric is not None:
        nav    = GeodesicNavigator(embeddings, records, metric, n_steps=40)
        path   = nav.trace(source_idx, target_idx)
        wp_emb = np.array([wp.embedding for wp in path.waypoints])
        wp_2d  = umap_reducer.transform(wp_emb)

    # Domain colors
    palette = {
        "math_geometry":  "#4f98a3",
        "biology_crispr": "#6daa45",
        "fusion_plasma":  "#fdab43",
        "ai_transformer": "#a86fdf",
        "unknown":        "#999999",
    }
    domain_list    = [r.get("domain", "unknown") for r in records]
    unique_domains = sorted(set(domain_list))
    colors         = [palette.get(d, "#999999") for d in domain_list]
    sizes          = 20 + density * 180

    # Boundary indices from last frame
    boundary_idxs = set()
    for b in last_frame.__dict__.get("boundaries_raw", []):
        boundary_idxs.add(b["boundary_point"])

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0e0c")
    ax.set_facecolor("#171614")

    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=colors,
               alpha=0.75, linewidths=0.3, edgecolors="white", zorder=2)

    if boundary_idxs:
        b_coords = coords[list(boundary_idxs)]
        ax.scatter(b_coords[:, 0], b_coords[:, 1], s=12, c="#ff4466",
                   alpha=0.6, marker="x", linewidths=1.0, zorder=3,
                   label="Jordan boundary")

    if wp_2d is not None:
        ax.plot(wp_2d[:, 0], wp_2d[:, 1], color="#4f98a3",
                linewidth=2.2, alpha=0.9, zorder=4, label="FR geodesic")
        src, tgt = coords[source_idx], coords[target_idx]
        ax.plot([src[0], tgt[0]], [src[1], tgt[1]], color="#fdab43",
                linewidth=1.5, alpha=0.7, linestyle="--", zorder=4,
                label="Euclidean chord")
        ax.scatter(*src, s=220, c="#ffffff", zorder=5, marker="*")
        ax.scatter(*tgt, s=220, c="#ffffff", zorder=5, marker="*")

    domain_patches = [
        mpatches.Patch(color=palette.get(d, "#999"), label=d.replace("_", " "))
        for d in unique_domains
    ]
    ax.legend(handles=domain_patches, loc="lower right", fontsize=8,
              facecolor="#1c1b19", edgecolor="#393836",
              labelcolor="white", framealpha=0.85)

    ax.set_title("KAFT Knowledge Manifold — Topology Map",
                 color="white", fontsize=14, pad=14, fontweight="bold")
    ax.set_xlabel("UMAP dim 1", color="#7a7974", fontsize=9)
    ax.set_ylabel("UMAP dim 2", color="#7a7974", fontsize=9)
    ax.tick_params(colors="#7a7974")
    for spine in ax.spines.values():
        spine.set_edgecolor("#393836")

    ax.text(0.01, 0.99,
            f"N={N} | K={last_frame.K_mean:.3f} | "
            f"Phase={last_frame.phase} | Boundaries={last_frame.boundary_count}",
            transform=ax.transAxes, color="#7a7974", fontsize=8, va="top")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualiser] Topology map → {save_path}")

    return fig, ax

# ── Visualiser ────────────────────────────────────────────────────────────────

class SimulationVisualiser:
    """
    Publication-quality figures from a SimulationRunner FrameLog.

    Parameters
    ----------
    log : List[Frame]
        runner.log — captured from SimulationRunner after stepping.
    domain_label : str
        Used in figure titles. Defaults to "Simulation".
    """

    def __init__(self, log: List[Frame], domain_label: str = "Simulation"):
        if not log:
            raise ValueError("FrameLog is empty — run at least one step first.")
        self.log          = log
        self.domain_label = domain_label
        _pub_style()

    # ── Individual plots ──────────────────────────────────────────────────────

    def plot_k_evolution(
        self,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """K_mean trajectory over steps with 5F phase bands and soliton lock."""
        standalone = ax is None
        fig = None
        if standalone:
            fig, ax = plt.subplots(figsize=(7.0, 3.5))

        steps  = [f.step     for f in self.log]
        K_mean = [f.K_mean   for f in self.log]

        _annotate_phases(ax, self.log)
        ax.plot(steps, K_mean, color=KAFT_COLOR, label="K mean (KAFT)")
        _mark_soliton(ax, self.log)

        ax.set_xlabel("Step")
        ax.set_ylabel("K (mean)")
        ax.set_title(f"K Evolution — {self.domain_label}")
        ax.set_ylim(0, 1.05)
        self._dedupe_legend(ax)

        if standalone:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path)
        return fig or ax.figure, ax

    def plot_variance(
        self,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        K variance — KAFT vs Softmax over steps.
        Primary paper signal: KAFT variance high and stable, Softmax ≈ 0.
        """
        standalone = ax is None
        fig = None
        if standalone:
            fig, ax = plt.subplots(figsize=(7.0, 3.5))

        steps       = [f.step          for f in self.log]
        kaft_var    = [f.K_var         for f in self.log]
        softmax_var = [f.softmax_K_var for f in self.log]

        ax.plot(steps, kaft_var,    color=KAFT_COLOR,    label="Variance — KAFT (FR geometry)")
        ax.plot(steps, softmax_var, color=SOFTMAX_COLOR, label="Variance — Softmax (flat)")
        _mark_soliton(ax, self.log)

        ax.set_xlabel("Step")
        ax.set_ylabel("K Variance")
        ax.set_title(f"Variance Divergence — {self.domain_label}")
        self._dedupe_legend(ax)

        if standalone:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path)
        return fig or ax.figure, ax

    def plot_k_distributions(
        self,
        frame_idx: int = -1,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Histogram of per-node K values — KAFT vs Softmax at a given frame.
        Shows KAFT resolving attractor structure, Softmax collapsing to a spike.
        """
        standalone = ax is None
        fig = None
        if standalone:
            fig, ax = plt.subplots(figsize=(7.0, 3.5))

        frame = self.log[frame_idx]
        bins  = 30

        ax.hist(frame.K_field,   bins=30, alpha=0.7,
                    color=KAFT_COLOR,    label=f"KAFT  (var={frame.K_var:.4f})",
                    density=False)       # ← counts not density
        ax.hist(frame.K_softmax, bins=30, alpha=0.7,
                    color=SOFTMAX_COLOR, label=f"Softmax (var={frame.softmax_K_var:.6f})",
                    density=False)       # ← counts not density
        ax.set_ylabel("Count")

        ax.set_xlabel("K value per node")
        ax.set_ylabel("Density")
        ax.set_title(
            f"K Distributions at step {frame.step} "
            f"(N={frame.n_papers}) — {self.domain_label}"
        )
        ax.legend()

        if standalone:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path)
        return fig or ax.figure, ax

    def plot_divergence(
        self,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """L1 divergence between KAFT and Softmax K distributions over steps."""
        standalone = ax is None
        fig = None
        if standalone:
            fig, ax = plt.subplots(figsize=(7.0, 3.5))

        steps = [f.step         for f in self.log]
        div   = [f.divergence_l1 for f in self.log]

        ax.fill_between(steps, div, alpha=0.15, color=KAFT_COLOR)
        ax.plot(steps, div, color=KAFT_COLOR, label="L1 divergence (KAFT ‖ Softmax)")
        _mark_soliton(ax, self.log)

        ax.set_xlabel("Step")
        ax.set_ylabel("L1 Divergence")
        ax.set_title(f"KAFT vs Softmax Divergence — {self.domain_label}")
        ax.set_ylim(0, 1.0)
        self._dedupe_legend(ax)

        if standalone:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path)
        return fig or ax.figure, ax

    # ── Summary figure ────────────────────────────────────────────────────────

    def summary(
        self,
        save_path: Optional[str] = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        2×2 multi-panel summary figure — paper-ready.

        Panel layout:
            [K evolution]    [K variance]
            [K distributions][L1 divergence]
        """
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
        fig.suptitle(
            f"KAFT Geometric Dynamics vs Softmax Baseline\n{self.domain_label}",
            fontsize=13, y=1.01
        )

        self.plot_k_evolution(ax=axes[0, 0])
        self.plot_variance(ax=axes[0, 1])
        self.plot_k_distributions(ax=axes[1, 0])
        self.plot_divergence(ax=axes[1, 1])

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[Visualiser] Saved → {save_path}")

        return fig, axes

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe_legend(ax: plt.Axes) -> None:
        """Remove duplicate phase band labels from legend."""
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if unique:
            ax.legend(*zip(*unique), loc="best")
"""
crucible.py — SimulationCrucible: domain configuration for geometric dynamics.

Maps a scientific domain → geometric parameters → runner setup.
One crucible per experiment domain. Same runner, different crucible = different physics.

Mirrors Wonder's Crucible concept at the scientific layer:
    simulate/ Crucibles  → domain geometry experiments
    Wonder   Crucibles   → domain conversation sessions
Same idea, two zoom levels.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationCrucible:
    """
    Domain configuration object for SimulationRunner.

    Parameters
    ----------
    domain_label : str
        Human-readable domain name — used in plot titles and FrameLog metadata.
        e.g. "arxiv_information_geometry", "quantum_fubini", "protein_davydov"

    metric_name : str
        Key in DivergenceRegistry. Must be registered before runner is built.
        e.g. "fisher_rao", "euclidean", "kl_divergence", "fubini_study"

    corpus_path : str
        Path to DomainSeeder JSON seed file.
        Must exist and match metric_name geometry.

    v_cog : float
        Cognitive velocity — domain-specific knowledge propagation rate.
        physics         ≈ 0.8   (dense, constrained domain)
        general         = 1.0   (balanced, default)
        creative        ≈ 1.2   (open, generative)
        interdisciplinary ≈ 1.5 (high domain-crossing velocity)

    temperature : float
        MetricEvolution Boltzmann temperature T.
        Low  T (0.1) → sharp manifold contraction, fast convergence.
        High T (10.) → near-flat, slow metric evolution.
        Default 1.0 — balanced.

    sigma : float
        JordanNoise amplitude. 0.05 = 5% perturbation at K=0.
        Smaller for stable/dense corpora, larger for sparse exploration.

    boundary_threshold : float
        JordanBoundary detection threshold.
        Points with K-density below this = boundary candidates.
        0.35 default. Dense corpora → 0.25, sparse → 0.45.

    n_waypoints : int
        GeodesicNavigator resolution — interpolation steps per trace().
        20 default. Increase for smoother boundary crossing detection.

    seed : int
        RNG seed for JordanNoise reproducibility across runs.
    """
    domain_label:       str
    metric_name:        str
    corpus_path:        str
    v_cog:              float = 1.0
    temperature:        float = 1.0
    sigma:              float = 0.05
    boundary_threshold: float = 0.35
    n_waypoints:        int   = 20
    seed:               int   = 42

    def __post_init__(self) -> None:
        if self.v_cog <= 0:
            raise ValueError(f"v_cog must be > 0, got {self.v_cog}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not 0.0 < self.boundary_threshold < 1.0:
            raise ValueError(f"boundary_threshold must be in (0, 1), got {self.boundary_threshold}")


# ── Pre-built crucibles for common domains ────────────────────────────────────

ARXIV_KAFT = SimulationCrucible(
    domain_label   = "arxiv_information_geometry",
    metric_name    = "fisher_rao",
    corpus_path    = "experiments/seeds/test_corpus.json",
    v_cog          = 1.1,
    temperature    = 1.0,
    sigma          = 0.05,
    seed           = 42,
)

ARXIV_EUCLIDEAN = SimulationCrucible(
    domain_label   = "arxiv_euclidean_baseline",
    metric_name    = "euclidean",
    corpus_path    = "seeds/arxiv_kaft_seed.json",
    v_cog          = 1.1,
    temperature    = 1.0,
    sigma          = 0.05,
    seed           = 42,
)
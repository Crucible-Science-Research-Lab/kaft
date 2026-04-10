"""
runner.py — SimulationRunner: metric-agnostic geometric dynamics engine.

Sequential ingestion loop over any corpus + any metric from DivergenceRegistry.
Runs KAFT dynamics and Softmax baseline in parallel on identical embeddings.
Returns a FrameLog — one Frame per step, drives all downstream experiments.

Step sequence (per step):
    1. Optionally expand corpus with new embedding
    2. KDensity.compute() — geodesic field on current geometry
    3. SoftmaxDynamics.run() — flat baseline on same embeddings
    4. JordanBoundary.detect() — emergent domain structure
    5. KState.update() — accumulate scalar K, detect soliton
    6. MetricEvolution.step() — reshape geometry for next step
    7. Capture Frame

Wave v1 note: uses direct KDensity loop, NOT WaveEngine.
WaveEngine (RK4) is Wave v2 — when K_field and k_density.density unify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from kaft.dynamics.jordan          import JordanBoundary, JordanNoise
from kaft.dynamics.kstate          import KDensity, KState
from kaft.dynamics.metric_evolution import MetricEvolution
from kaft.dynamics.resonance       import ResonanceField
from kaft.dynamics.softmax_dynamics import SoftmaxDynamics
from kaft.dynamics.transport       import KAFTTerm2
from kaft.navigate.seeder          import DomainSeeder
from kaft.simulate.crucible        import SimulationCrucible


# ── Frame ─────────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """
    Snapshot of full manifold state at one simulation step.

    One FrameLog (list of Frames) drives all experiments:
        exp_k_evolution         — K_field sequence over steps
        exp_fr_vs_softmax       — K_field vs K_softmax + divergence_l1
        exp_n_scaling           — K_var vs softmax_K_var at different N
    """
    step:            int
    n_papers:        int
    K_field:         np.ndarray   # (N,) KAFT K-density, normalised [0, 1]
    K_softmax:       np.ndarray   # (N,) Softmax K per node
    K_mean:          float        # scalar mean of K_field
    K_var:           float        # scalar variance of K_field — primary signal
    softmax_K_mean:  float
    softmax_K_var:   float        # expected to diverge from K_var as N grows
    divergence_l1:   float        # L1 distance between normalised distributions ∈ [0, 1]
    phase:           str          # 5F phase label
    soliton_locked:  bool
    delta_K:         float        # K_mean change since last step
    boltzmann_dist:  np.ndarray   # (N,) p_i ∝ e^(-K_i/T) — geometry reshaping weights
    boundary_count:  int          # number of detected Jordan boundaries
    boundary_indices: list


# ── Runner ────────────────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Metric-agnostic simulation engine built from a SimulationCrucible.

    Usage
    -----
    Build once from a crucible, then call step() or run():

        runner = SimulationRunner.from_crucible(crucible, registry)
        frame  = runner.step()                      # static step (metric evolution only)
        frame  = runner.step(new_embedding)         # ingest one new paper
        log    = runner.run(embeddings_list)        # ingest a list sequentially
        log    = runner.run_static(n_steps=10)      # evolve geometry, no new papers
    """

    def __init__(
        self,
        embeddings:    np.ndarray,
        records:       list[dict],
        k_density:     KDensity,
        k_state:       KState,
        jordan:        JordanBoundary,
        crucible:      SimulationCrucible,
        metric_evolver: MetricEvolution,
        softmax:       SoftmaxDynamics,
        noise:         JordanNoise,
    ):
        self._embeddings    = embeddings
        self._records       = records
        self._kd            = k_density
        self._ks            = k_state
        self._jordan        = jordan
        self._crucible      = crucible
        self._evolver       = metric_evolver
        self._softmax       = softmax
        self._noise         = noise
        self._step_count    = 0
        self.log: List[Frame] = []

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_crucible(
        cls,
        crucible: SimulationCrucible,
        seeds_root: str | None = None,
    ) -> "SimulationRunner":
        """
        Build a fully-warmed runner from a SimulationCrucible.

        Loads corpus from DomainSeeder — no arXiv re-fetch, no metric recompute.
        """
        from kaft.geometry import get as    get_metric
        from pathlib import Path
        corpus_path = (
            Path(seeds_root) / crucible.corpus_path
            if seeds_root
            else Path(crucible.corpus_path)
        )
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Seed file not found: {corpus_path}\n"
                f"Pass seeds_root='/absolute/path/to/seeds/parent' to from_crucible()"
            )
        metric = get_metric(crucible.metric_name)
        corpus  = DomainSeeder.load(crucible.corpus_path, metric)

        embeddings = corpus["embeddings"]
        records    = corpus["records"]
        kd         = corpus["k_density"]    # _density + _distances pre-loaded
        ks         = corpus["k_state"]
        jordan     = corpus["jordan"]       # _boundaries pre-loaded

        evolver = MetricEvolution(
            temperature = crucible.temperature,
            base_step   = 0.1,
        )
        softmax = SoftmaxDynamics(temperature=1.0)
        noise   = JordanNoise(
            sigma = crucible.sigma,
            rng   = np.random.default_rng(crucible.seed),
        )

        return cls(
            embeddings     = embeddings,
            records        = records,
            k_density      = kd,
            k_state        = ks,
            jordan         = jordan,
            crucible       = crucible,
            metric_evolver = evolver,
            softmax        = softmax,
            noise          = noise,
        )

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self, new_embedding: Optional[np.ndarray] = None) -> Frame:
        """
        Run one simulation step.

        Parameters
        ----------
        new_embedding : np.ndarray (D,) or None
            If provided, expands corpus before computing this step.
            Triggers full KDensity recompute on grown (N+1, D) matrix.

        Returns
        -------
        Frame — full manifold snapshot at this step.
        """
        # 1. Expand corpus if new paper is arriving
        if new_embedding is not None:
            self._embeddings = np.vstack([self._embeddings, new_embedding])
            self._kd = KDensity(self._embeddings, self._kd.metric)

        # 2. KAFT K-density — full geodesic computation
        K_field  = self._kd.compute()

        # 3. Softmax baseline — same embeddings, flat geometry
        sm_result = self._softmax.run(self._embeddings)

        # 4. Jordan boundaries — emergent domain structure
        self._jordan = JordanBoundary(self._kd, threshold=self._crucible.boundary_threshold)
        self._jordan.detect()

        # 5. KState update — scalar K, phase, soliton
        K_scalar = self._ks.update(K_field)
        locked   = self._ks.soliton_locked   # call ONCE, store

        # 6. Boltzmann distribution — geometry reshaping weights
        boltzmann = self._evolver.distribution(K_field)

        # 7. Metric evolution — reshape geometry for next step
        self._evolver.step(self._kd, K_field)

        # 8. L1 divergence between KAFT and Softmax K distributions
        divergence = self._softmax.distribution_divergence(K_field, sm_result.K)

        # 9. Capture Frame
        frame = Frame(
            step           = self._step_count,
            n_papers       = len(self._embeddings),
            K_field        = K_field.copy(),
            K_softmax      = sm_result.K.copy(),
            K_mean         = float(K_field.mean()),
            K_var          = float(K_field.var()),
            softmax_K_mean = sm_result.K_mean,
            softmax_K_var  = sm_result.K_var,
            divergence_l1  = divergence,
            phase          = self._ks.phase,
            soliton_locked = locked,
            delta_K        = self._ks.delta_K,
            boltzmann_dist = boltzmann.copy(),
            boundary_count   = len(self._jordan.boundaries),
            boundary_indices = [b["boundary_point"] for b in self._jordan.boundaries],
        )

        self.log.append(frame)
        self._step_count += 1
        return frame

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def run(self, new_embeddings: List[np.ndarray]) -> List[Frame]:
        """
        Sequentially ingest a list of new embeddings, one per step.
        Returns the frames produced during this run.
        """
        frames = []
        for emb in new_embeddings:
            frames.append(self.step(emb))
        return frames

    def run_static(self, n_steps: int) -> List[Frame]:
        """
        Run n_steps of metric evolution on fixed corpus — no new papers.
        Watches geometry breathe under existing knowledge density.
        """
        return [self.step() for _ in range(n_steps)]

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def records(self) -> list[dict]:
        return self._records

    @property
    def current_phase(self) -> str:
        return self._ks.phase

    @property
    def K_history(self) -> list[float]:
        return self._ks.history
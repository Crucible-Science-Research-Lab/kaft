"""
kaft_source.py — KAFTSource: KAFT-specific source_fn for WaveEngine

Assembles the master evolution equation:
    ∂K/∂t = c·v²_cog·K²  +  P(V)·∇^FR(K)  +  ξ_J(t)
              Term 1            Term 2           Term 3

after_step protocol:
    MetricEvolution runs ONCE per full timestep here — not inside RK4 sub-steps.
    This keeps the geometry evolving at the correct rate.

NOTE on K_field vs k_density.density:
    K_field is the WaveEngine integration state.
    k_density.density is the live geometric field on the manifold.
    Wave v1: these are coupled — k_density drives geometry.
    Wave v2 will unify them into a single field state.
"""
from __future__ import annotations
import numpy as np
from kaft.dynamics.kstate           import KDensity, KState
from kaft.dynamics.resonance        import ResonanceField
from kaft.dynamics.transport        import KAFTTerm2
from kaft.dynamics.jordan           import JordanNoise
from kaft.dynamics.metric_evolution import MetricEvolution


class KAFTSource:
    """
    source_fn for WaveEngine — assembles all three KAFT terms.

    __call__(K, t)        → dK/dt (N,)   pure compute, stateless
    after_step(K_new, t)  → metric evolution + KState update, once per step
    """

    def __init__(
        self,
        k_density:      KDensity,
        resonance:      ResonanceField,
        term2:          KAFTTerm2,
        jordan_noise:   JordanNoise,
        metric_evolver: MetricEvolution,
        c:              float = 1.0,
    ):
        self.k_density      = k_density
        self.resonance      = resonance
        self.term2          = term2
        self.jordan_noise   = jordan_noise
        self.metric_evolver = metric_evolver
        self.c              = c
        self.kstate         = KState()

    def __call__(self, K_field: np.ndarray, t: float) -> np.ndarray:
        """Stateless: compute dK/dt = Term1 + Term2 + Term3."""
        K     = self.k_density.density
        term1 = self.c * self.resonance.compute()
        term2 = self.term2.compute(self.k_density)
        term3 = self.jordan_noise.sample(K)
        return term1 + term2 + term3

    def after_step(self, K_new: np.ndarray, t_new: float) -> None:
        """Called once per full step by WaveEngine."""
        self.metric_evolver.step(self.k_density, K_new)
        self.kstate.update(K_new)

    @property
    def phase(self) -> str:
        return self.kstate.phase

    @property
    def soliton_locked(self) -> bool:
        return self.kstate.soliton_locked

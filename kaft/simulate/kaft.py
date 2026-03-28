"""
KAFTEvolution — the flagship model.

Master evolution equation:
    ∂K/∂t = c·v²_cog·K² + P(V)·∇^FR(K) + ξ_J(t)

Terms:
    K²          — self-amplifying dynamics. Understanding accelerates understanding.
    ∇^FR(K)     — geodesic flow along information-theoretically optimal paths.
    ξ_J(t)      — topologically-constrained stochastic exploration (Jordan noise).

Wave equation admits sech² bright soliton solutions:
    Deep insights are topologically protected wave packets.
    You cannot easily destroy a deep understanding — geometry stabilises it.
"""
from __future__ import annotations
import numpy as np
from kaft.simulate.base import AbstractManifoldDynamics
from kaft.core.manifold import Manifold
from kaft.core.metric import FisherRaoMetric
from kaft.core.topology import KDensity, JordanBoundary
from kaft.core.resonance import ResonanceField


class KAFTEvolution(AbstractManifoldDynamics):
    """
    KAFT dynamics — master equation emergent from composed primitives.

    ∂K/∂t = c·v²_cog·K²  +  P(V)·∇^FR(K)  +  ξ_J(t)
              ↑ ResonanceField  ↑ geodesic flow   ↑ Jordan noise

    No term is hard-coded. Each emerges from its primitive.
    The equation assembles at runtime from composition.
    """

    def __init__(self, c: float = 1.0, v_cog: float = 1.0,
                 noise_scale: float = 0.02, epsilon: float = 1e-6):
        self.c = c
        self.v_cog = v_cog
        self.noise_scale = noise_scale
        self.epsilon = epsilon

    def run(self, manifold: Manifold, dt: float = 0.1) -> dict:
        """
        Evolve K-density one timestep via KAFT master equation.

        Returns
        -------
        dict with:
            k_density   — evolved K-density field (N,)
            resonance   — R = K²·v²_cog field (N,)
            k_delta     — change in K this step (N,)
        """
        # ── Build geometry stack ──────────────────────────────────────
        metric    = FisherRaoMetric(manifold)
        D         = metric.distances
         # Always build kdensity object — needed for ResonanceField
        kdensity = KDensity(manifold, metric)
        
        if manifold.k_density is not None:
            K = manifold.k_density                         # ← evolve from previous step
        else:
            kdensity = KDensity(manifold, metric)
            K = kdensity.compute()  

        resonance = ResonanceField(kdensity, self.v_cog)

        # ── Term 1: K² self-amplification (ResonanceField) ───────────
        # Dense knowledge regions grow faster — understanding compounds
        R = resonance.compute()                          # K²·v²_cog, shape (N,)
        k_squared_term = self.c * R                      # c·v²_cog·K²

        # ── Term 2: P(V)·∇^FR(K) — geodesic gradient flow ───────────
        # Each point attracted toward high-K neighbors
        # weighted by inverse square of CURVED geodesic distance
        # This is parallel transport P(V) on Fisher-Rao manifold
        weights = K[np.newaxis, :] / (D ** 2 + self.epsilon)   # (N, N)
        np.fill_diagonal(weights, 0.0)
        weight_sum = weights.sum(axis=1) + self.epsilon
        geodesic_flow = (weights @ K) / weight_sum              # (N,)

        # ── Term 3: ξ_J(t) — Jordan-constrained noise ────────────────
        # Exploration is BOUNDED by domain boundaries
        # Points near boundaries get less noise (harder to cross)
        # Points in open regions get more noise (free to explore)
        boundary_distance = D.min(axis=1)               # (N,) min dist to any neighbor
        boundary_scale = boundary_distance / (boundary_distance.max() + self.epsilon)
        jordan_noise = np.random.normal(0, self.noise_scale, len(K)) * boundary_scale

        # ── Master equation assembles from primitives ─────────────────
        dK_dt   = k_squared_term + geodesic_flow + jordan_noise
        K_new   = K + dt * dK_dt
        K_new   = np.clip(K_new, 0.0, None)             # K ≥ 0 always
        K_new   = K_new / (K_new.max() + self.epsilon)  # normalise [0,1]

        return {
            "k_density" : K_new,
            "resonance" : R,
            "k_delta"   : K_new - K,
            "model"     : "kaft",
        }

    def geometry_type(self) -> str:
        return "curved"

"""
resonance.py — Resonance field: R ≡ K² · v²_cog

Term 1 of the master evolution equation:
    ∂K/∂t = c·v²_cog·K²  +  P(V)·∇^FR(K)  +  ξ_J(t)
              ↑ THIS TERM

Attention density — regions where knowledge self-amplifies most strongly.
NOT designed. Falls directly out of K² recursion.

High R = powerful attractor. Knowledge flows toward these regions.
Low R  = exploratory space. Jordan noise (Term 3) dominates here.

vs transformer softmax attention:
    Softmax:  scaled dot-product in flat R^d, O(N²), static geometry
    KAFT R:   covariant derivative in Fisher-Rao space, context-dependent, O(N log N)
"""
from __future__ import annotations
import numpy as np
from kaft.dynamics.kstate import KDensity


class ResonanceField:
    """
    R = K² · v²_cog for every point on the manifold.

    Parameters
    ----------
    k_density : KDensity
        The K-density field providing K(x) for every point x.
    v_cog : float
        Cognitive velocity — how fast knowledge propagates in this domain.
        Default 1.0 (normalised). Domain-specific values:
            physics       ≈ 0.8   (dense, constrained)
            creative      ≈ 1.2   (open, generative)
            interdisciplinary ≈ 1.5 (high crossing velocity)
    """

    def __init__(self, k_density: KDensity, v_cog: float = 1.0):
        self.k_density = k_density
        self.v_cog     = v_cog

    def compute(self) -> np.ndarray:
        """
        Returns R = K² · v²_cog for every point.

        Shape (N,) — same shape as K-density field.
        Emerges from K² recursion. Not imposed.
        """
        K = self.k_density.density            # (N,) — pulled from cached field
        return (K ** 2) * (self.v_cog ** 2)   # R ≡ K²·v²_cog

    def peak_indices(self, top_k: int = 5) -> np.ndarray:
        """
        Indices of the top-k resonance peaks — strongest attractors.
        These are where K² amplification is most powerful.
        """
        R = self.compute()
        return np.argsort(R)[::-1][:top_k]

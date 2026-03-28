"""
EuclideanBaseline — flat Euclidean attention baseline.

Softmax attention = dot product in R^d_k = flat metric tensor g_uv = δ_uv.
Knowledge K = Σ(I_i). Static geometry, recomputed at every layer.

This is the SHADOW — run alongside KAFTSimulator on the same corpus.
Two topologies diverge live. Researcher sees it without needing Čencov.
"""
from __future__ import annotations
import numpy as np
from kaft.simulate.base import AbstractManifoldDynamics
from kaft.core.manifold import Manifold
from kaft.core.metric import FisherRaoMetric
from kaft.core.topology import KDensity


class EuclideanBaseline(AbstractManifoldDynamics):
    """
    Flat Euclidean attention — the shadow.

    Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V

    Flat metric tensor g_uv = δ_uv. No curvature. No K² amplification.
    No Jordan boundaries. No geodesic flow.
    K evolves as attention-weighted average — smoothing, not amplifying.

    This is what current transformers do.
    Run alongside KAFTSimulator on same corpus to see the divergence.
    """

    def __init__(self, temperature: float = 1.0, epsilon: float = 1e-6):
        self.temperature = temperature
        self.epsilon = epsilon

    def run(self, manifold: Manifold, dt: float = 0.1) -> dict:
        """
        Evolve K-density via flat softmax attention.
        No curvature. No amplification. Pure pattern matching.
        """
        E = manifold.embeddings                          # (N, 384)
        metric   = FisherRaoMetric(manifold)
        if manifold.k_density is not None:
                K = manifold.k_density                         # ← carry state forward
        else:
                kdensity = KDensity(manifold, metric)
                K = kdensity.compute()                   # (N,)

        # Scaled dot-product in flat Euclidean space
        scale   = np.sqrt(E.shape[1]) * self.temperature
        scores  = (E @ E.T) / scale                     # (N, N) flat similarity

        # Softmax — normalised flat weights
        scores  -= scores.max(axis=1, keepdims=True)    # numerical stability
        exp_s   = np.exp(scores)
        attention = exp_s / (exp_s.sum(axis=1, keepdims=True) + self.epsilon)

        # K_new = attention-weighted average — no K², no amplification
        # High-K and low-K regions blend toward the mean
        K_new   = attention @ K                         # (N,) smoothed

        # No growth. No boundary awareness. No self-amplification.
        K_new   = K_new / (K_new.max() + self.epsilon)

        return {
            "k_density" : K_new,
            "resonance" : K_new,                        # no K² — flat
            "k_delta"   : K_new - K,
            "model"     : "softmax",
        }

    def geometry_type(self) -> str:
        return "flat"

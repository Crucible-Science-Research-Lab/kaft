from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class SoftmaxResult(NamedTuple):
    """Output of SoftmaxDynamics.run().

    K       : (N,) activation density per node — softmax-weighted mass sum
    attn    : (N, N) row-stochastic attention matrix
    K_mean  : float
    K_std   : float
    K_var   : float — primary comparison signal vs geometric attention
    """
    K:       np.ndarray
    attn:    np.ndarray
    K_mean:  float
    K_std:   float
    K_var:   float


@dataclass
class SoftmaxDynamics:
    """Flat dot-product attention baseline.

    Comparison to geometric geodesic attention:

        Geometric: K_i = Σ_j  (I_i · I_j) / d²_FR(i,j)   ← curved space, geodesic d
        Softmax:   K_i = Σ_j  attn_ij  ·  I_j             ← flat space, cosine d

    Both aggregate semantic mass from all neighbours.
    Geometric attention uses Fisher-Rao geodesic distance.
    Softmax uses scaled cosine dot-product and normalises it into a flat weight.

    The geometry is the only independent variable.

    Expected result:
        - Geometric: K_i varies across nodes — high-density attractors, 
                     cross-domain bridges, and peripheral nodes are distinguishable
        - Softmax:   K_i ≈ mean(I) for all i — uniform, no domain structure visible

    Args:
        temperature: scales the sqrt(D) denominator.
                     1.0 = standard scaled dot-product attention.
                     Lower T → sharper attention, higher T → more uniform.
    """

    temperature: float = 1.0

    def run(self, embeddings: np.ndarray) -> SoftmaxResult:
        """Compute softmax activation density for all N nodes.

        Args:
            embeddings: (N, D) raw unnormalised float32 from Embedder.
                        Same array fed to KDensity — no preprocessing needed.

        Returns:
            SoftmaxResult with K (N,), attn (N, N), and scalar stats.
        """
        emb = embeddings.astype(np.float64)
        N, D = emb.shape

        # Semantic mass: raw L2 norm (mirrors KDensity.I_i)
        I = np.linalg.norm(emb, axis=1)                             # (N,)

        # Unit-normalise for cosine similarity in flat Euclidean space
        e_unit = emb / (I[:, None] + 1e-8)                          # (N, D)

        # Scaled dot-product scores: e_i · e_j / (√D · T)
        scale = float(np.sqrt(D)) * self.temperature
        logits = (e_unit @ e_unit.T) / scale                         # (N, N)

        # Numerically stable softmax over j for each query i
        logits -= logits.max(axis=1, keepdims=True)
        attn = np.exp(logits)
        attn /= attn.sum(axis=1, keepdims=True)                      # rows sum to 1

        # K_i = softmax-weighted sum of semantic masses
        K = attn @ I                                                  # (N,)

        return SoftmaxResult(
            K=K,
            attn=attn,
            K_mean=float(K.mean()),
            K_std=float(K.std()),
            K_var=float(K.var()),
        )

    def distribution_divergence(
        self, K_kaft: np.ndarray, K_softmax: np.ndarray
    ) -> float:
        """L1 divergence between normalised geometric and flat-space activation distributions.

        Measures structural difference between the two attention regimes:
            0.0 = identical distributions
            1.0 = maximally different

        A stable plateau in this value indicates the geometric activation
        distribution has reached a fixed point relative to the flat baseline.
        """
        def _normalise(x: np.ndarray) -> np.ndarray:
            s = x.sum()
            return x / s if s > 0 else x

        p = _normalise(K_kaft.astype(np.float64))
        q = _normalise(K_softmax.astype(np.float64))
        return float(np.abs(p - q).sum() / 2.0)   # symmetric L1 ∈ [0, 1]
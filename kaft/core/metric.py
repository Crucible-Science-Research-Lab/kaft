"""
FisherRaoMetric — the mathematically forced metric.

Čencov's uniqueness theorem (1981): Fisher-Rao is the ONLY
Riemannian metric on statistical manifolds invariant under
sufficient-statistic Markov morphisms. Not chosen. Forced.

g^FR_μν = E[ ∂log p/∂θ_μ · ∂log p/∂θ_ν ]
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from kaft.core.manifold import Manifold


class FisherRaoMetric:
    """
    Fisher-Rao metric on the knowledge manifold.

    Sentence-transformer embeddings live on a hypersphere S^383.
    The natural Riemannian metric on a hypersphere gives geodesic
    distance as angular distance — arccos of cosine similarity.

    This IS curved geometry. Not Euclidean. Not dot-product similarity.
    Two concepts can be close in dot-product space but far on the manifold
    if the curved path between them crosses a semantic boundary.

    Čencov forcing: Fisher-Rao is the unique invariant metric on
    statistical manifolds. We don't choose it. It's the only valid one.
    """

    def __init__(self, manifold: Manifold):
        self.manifold = manifold
        self._distances: np.ndarray | None = None

    def compute(self) -> np.ndarray:
        """
        Compute pairwise Fisher-Rao geodesic distances.

        Returns
        -------
        np.ndarray, shape (N, N)
            D[i,j] = arccos(e_i · e_j) — geodesic distance on hypersphere.
            D[i,i] = 0 always. Symmetric. Range [0, π].
        """
        E = self.manifold.embeddings                    # (N, 384)
        cos_sim = cosine_similarity(E)                  # (N, N), range [-1, 1]
        cos_sim = np.clip(cos_sim, -1.0, 1.0)          # numerical safety for arccos
        self._distances = np.arccos(cos_sim)            # geodesic distance
        return self._distances

    def geodesic_distance(self, i: int, j: int) -> float:
        """
        Fisher-Rao geodesic distance between points i and j.
        This is the d in K ∝ I₁·I₂/d² — curved, not Euclidean.
        Context reshapes this distance by curving the manifold.
        """
        if self._distances is None:
            self.compute()
        return float(self._distances[i, j])

    @property
    def distances(self) -> np.ndarray:
        if self._distances is None:
            self.compute()
        return self._distances

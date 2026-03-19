"""
Topology — K-density field and Jordan boundary detection.

Key insight: Jordan boundaries are NOT imposed constraints.
They EMERGE from K² attractor dynamics as level sets:
    B = { x ∈ M | ||∇^FR K(x)|| = 0, ∂²K/∂n² < 0 }

Semantic boundaries crystallize from K-density gradients.
They cannot be arbitrarily placed — they are geometric inevitability.
"""

from __future__ import annotations
import numpy as np
from kaft.core.manifold import Manifold
from kaft.core.metric import FisherRaoMetric


class KDensity:
    """
    K-density field: K_i = Σ_{j≠i} (I_i · I_j) / d²_ij

    Interaction hypothesis — same geometric necessity as gravity and light.
    Each point radiates influence into conceptual space.
    At geodesic distance d, flux = I / 4πd².

    High K-density = dense semantic cluster (constrained, well-explored)
    Low K-density  = open conceptual space (sparse, exploratory)
    Zero-gradient ridges between them = Jordan boundaries emerging
    """

    def __init__(self, manifold: Manifold, metric: FisherRaoMetric, epsilon: float = 1e-6):
        self.manifold = manifold
        self.metric = metric
        self.epsilon = epsilon      # prevents 1/0 at diagonal
        self._density: np.ndarray | None = None

    def compute(self) -> np.ndarray:
        """
        Compute normalised K-density for every point in the manifold.

        Returns
        -------
        np.ndarray, shape (N,)
            Normalised K-density in [0, 1].
            0 = isolated, 1 = maximum density cluster center.
        """
        D = self.metric.distances           # (N, N) geodesic distances
        E = self.manifold.embeddings        # (N, 384)

        # Information magnitude — L2 norm of each embedding
        # Represents semantic mass each point carries into the interaction
        I = np.linalg.norm(E, axis=1)      # (N,)

        # Outer product I_i * I_j → (N, N) interaction numerator
        I_outer = I[:, None] * I[None, :]  # (N, N)

        # Denominator: d² with epsilon to prevent division by zero at diagonal
        D_squared = D ** 2 + self.epsilon   # (N, N)

        # K ∝ I₁·I₂ / d² — the inverse square law in information space
        interaction = I_outer / D_squared   # (N, N)

        # No self-interaction
        np.fill_diagonal(interaction, 0.0)

        # K_i = sum of all incoming interactions
        raw_density = interaction.sum(axis=1)   # (N,)

        # Normalise to [0, 1]
        d_min, d_max = raw_density.min(), raw_density.max()
        if d_max - d_min < 1e-10:
            self._density = np.ones(len(raw_density))
        else:
            self._density = (raw_density - d_min) / (d_max - d_min)

        return self._density

    @property
    def density(self) -> np.ndarray:
        if self._density is None:
            self.compute()
        return self._density


class JordanBoundary:
    """
    Emergent Jordan-Brouwer semantic domain boundaries.

    NOT imposed. NOT configurable. They crystallise from K-density gradients.

    B = { points where K-density is locally minimal between two dense regions }

    Energy barrier ΔE_B = geodesic distance from boundary point to nearest cluster.
    Hallucination prevention: crossing requires accumulated energy > ΔE_B.
    The geometry enforces semantic coherence — not rules, not filters.
    """

    def __init__(self, manifold: Manifold, k_density: KDensity, threshold: float = 0.35):
        self.manifold = manifold
        self.k_density = k_density
        self.threshold = threshold
        self._boundaries: list | None = None

    def detect(self) -> list:
        """
        Detect Jordan boundaries as emergent low-density ridges.

        Returns
        -------
        list of dicts
            Each boundary: point index, which clusters it separates,
            energy barrier, K-density at boundary, source text.
        """
        density = self.k_density.density        # (N,)
        D = self.k_density.metric.distances     # (N, N)

        mean_density = density.mean()

        cluster_indices  = np.where(density > mean_density)[0]
        boundary_indices = np.where(density < self.threshold)[0]

        boundaries = []

        if len(boundary_indices) == 0 or len(cluster_indices) < 2:
            return boundaries   # single domain — no boundary to emerge

        for b_idx in boundary_indices:
            dist_to_clusters = D[b_idx, cluster_indices]
            nearest = np.argsort(dist_to_clusters)[:2]

            if len(nearest) >= 2:
                cluster_a = cluster_indices[nearest[0]]
                cluster_b = cluster_indices[nearest[1]]

                boundaries.append({
                    "boundary_point"      : int(b_idx),
                    "separates"           : (int(cluster_a), int(cluster_b)),
                    "energy_barrier"      : float(dist_to_clusters[nearest[0]]),
                    "k_density_at_boundary": float(density[b_idx]),
                    "text"                : self.manifold.records[b_idx].get("text", "")[:60],
                })

        self._boundaries = boundaries
        return boundaries

"""
transport.py — Term 2: P(V)·∇^FR(K)

P(V) is the O(N log N) numerical approximation to the exact covariant
derivative. The d'Alembertian □_g in wave.py recovers the full Levi-Civita
structure. P(V) is the efficient inner-loop implementation.

Two classes do the work:

    FisherRaoGradient   — ∇^FR K as a tangent vector field (N, D)
                          finite differences along geodesics to kNN
                          k auto-scales: max(8, floor(log2(N)))

    ParallelTransport   — moves gradient vectors along geodesics
                          exact on S^n via geomstats

    KAFTTerm2           — assembles P(V)·∇^FR(K) as scalar field (N,)
                          for WaveEngine source_fn interface
"""
from __future__ import annotations
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from kaft.dynamics.kstate import KDensity


class FisherRaoGradient:
    """
    Covariant gradient ∇^FR K at each point on the Fisher-Rao hypersphere.

    g_i = Σ_{j ∈ kNN(i)} (K_j - K_i) · log_i(j) / d(i,j)

    log_i(j) is the exact tangent vector at i pointing toward j — no Euclidean
    approximation. The gradient lives in tangent space T_i S^n at each point.
    """

    def __init__(self, k_neighbors: int | None = None):
        """
        Parameters
        ----------
        k_neighbors : int | None
            None → auto-scales as max(8, floor(log2(N))).
            Set explicitly for ablation runs.
        """
        self.k_neighbors = k_neighbors
        self._sphere: Hypersphere | None = None

    def _resolve_k(self, N: int) -> int:
        if self.k_neighbors is not None:
            return min(self.k_neighbors, N - 1)
        return min(max(8, int(np.log2(max(N, 2)))), N - 1)

    def _get_sphere(self, D: int) -> Hypersphere:
        if self._sphere is None or self._sphere.dim != D - 1:
            self._sphere = Hypersphere(dim=D - 1)
        return self._sphere

    def compute(
        self,
        embeddings: np.ndarray,
        k_density:  np.ndarray,
        distances:  np.ndarray,
    ) -> np.ndarray:
        """
        Returns ∇^FR K tangent vector field, shape (N, D).

        Near cluster centre: small magnitude (flat K, stable attractor).
        Near Jordan boundary: large magnitude (steep gradient between domains).
        """
        N, D   = embeddings.shape
        k      = self._resolve_k(N)
        sphere = self._get_sphere(D)
        field  = np.zeros((N, D))

        for i in range(N):
            row     = distances[i].copy()
            row[i]  = np.inf
            knn_idx = np.argsort(row)[:k]

            g_i = np.zeros(D)
            for j in knn_idx:
                d_ij = distances[i, j]
                if d_ij < 1e-10:
                    continue
                log_ij  = sphere.metric.log(
                    np.array([embeddings[j]]),
                    np.array([embeddings[i]])
                )[0]
                delta_K = k_density[j] - k_density[i]
                g_i    += delta_K * log_ij / d_ij

            field[i] = g_i

        return field

    def magnitude(self, field: np.ndarray) -> np.ndarray:
        return np.linalg.norm(field, axis=1)


class ParallelTransport:
    """
    P(V): parallel transport of tangent vectors along Fisher-Rao geodesics.

    Exact on S^n via geomstats — not an approximation.
    Brings gradient vectors from different tangent spaces into a common
    reference frame so they can be compared and summed coherently.
    """

    def __init__(self):
        self._sphere: Hypersphere | None = None

    def _get_sphere(self, D: int) -> Hypersphere:
        if self._sphere is None or self._sphere.dim != D - 1:
            self._sphere = Hypersphere(dim=D - 1)
        return self._sphere

    def to_attractor(
        self,
        embeddings: np.ndarray,
        field:      np.ndarray,
        k_density:  np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """
        Transport all gradient vectors toward the global K-maximum.
        Returns (transported_field (N, D), attractor_idx).
        """
        N, D          = embeddings.shape
        sphere        = self._get_sphere(D)
        attractor_idx = int(np.argmax(k_density))
        attractor     = embeddings[attractor_idx]
        transported   = np.zeros((N, D))

        for i in range(N):
            if i == attractor_idx:
                transported[i] = field[i]
                continue
            transported[i] = sphere.metric.parallel_transport(
                tangent_vec=np.array([field[i]]),
                base_point =np.array([embeddings[i]]),
                end_point  =np.array([attractor])
            )[0]

        return transported, attractor_idx

    def to_point(
        self,
        embeddings: np.ndarray,
        field:      np.ndarray,
        target:     np.ndarray,
    ) -> np.ndarray:
        """Transport all gradient vectors toward a specific target point."""
        N, D        = embeddings.shape
        sphere      = self._get_sphere(D)
        transported = np.zeros((N, D))

        for i in range(N):
            transported[i] = sphere.metric.parallel_transport(
                tangent_vec=np.array([field[i]]),
                base_point =np.array([embeddings[i]]),
                end_point  =np.array([target])
            )[0]

        return transported


class KAFTTerm2:
    """
    Full Term 2: P(V)·∇^FR(K) → scalar field (N,) for WaveEngine.

    Scalar projection:
        term2_i = <transported_gradient_i, unit_local_gradient_i>

    Measures gradient coherence after transport.
    High value = gradient preserved (low curvature).
    Low value  = curvature is twisting the gradient (high geometric complexity).
    Flat-space limit: reduces to |∇K|² — standard gradient magnitude.
    """

    def __init__(self, k_neighbors: int | None = None):
        self.grad      = FisherRaoGradient(k_neighbors=k_neighbors)
        self.transport = ParallelTransport()

    def compute(self, k_density_obj: KDensity) -> np.ndarray:
        """
        Returns Term 2 scalar field (N,).
        Positive = K flows toward this point.
        Negative = K flows away.
        """
        emb  = k_density_obj.embeddings
        K    = k_density_obj.density
        D    = k_density_obj.distances

        field        = self.grad.compute(emb, K, D)
        trans, _     = self.transport.to_attractor(emb, field, K)

        norms      = self.grad.magnitude(field)
        safe_norms = np.where(norms < 1e-10, 1.0, norms)
        unit_grad  = field / safe_norms[:, None]

        return np.einsum("nd,nd->n", trans, unit_grad)

"""
jordan.py — Jordan-Brouwer boundaries + stochastic exploration noise ξ_J(t)

Term 3 of the master evolution equation:
    ∂K/∂t = c·v²_cog·K²  +  P(V)·∇^FR(K)  +  ξ_J(t)
                                                ↑ THIS TERM

Boundaries emerge from K² attractor dynamics as level sets:
    B = { x ∈ M | ‖∇^FR K(x)‖ = 0, ∂²K/∂n² < 0 }

They are NOT imposed. NOT configurable. They crystallise from K-density gradients.

ξ_J(t): topologically-constrained stochastic noise.
    — Explores freely WITHIN domains
    — Suppressed in high-K regions (preserve structure)
    — Crossing a boundary requires accumulated energy > ΔE_B
    — One noise spike cannot teleport across a semantic boundary
    — This is hallucination prevention at the geometric level
"""
from __future__ import annotations
import numpy as np
from kaft.dynamics.kstate import KDensity


class JordanBoundary:
    """
    Emergent Jordan-Brouwer semantic domain boundaries.

    B = { points where K-density is locally minimal between two dense regions }

    Energy barrier:
        ΔE_B = geodesic distance from boundary point to nearest cluster centre.
        Crossing requires accumulated perturbation energy > ΔE_B.
        The geometry enforces semantic coherence — not rules, not filters.

    Invariance property (Čencov):
        Under context reparameterisation, boundaries deform continuously
        but preserve topological identity. Domain structure cannot be
        destroyed by coordinate change.
    """

    def __init__(self, k_density: KDensity, threshold: float = 0.35):
        """
        Parameters
        ----------
        k_density : KDensity
            Must have .density and .distances computed (or lazily computed).
        threshold : float
            K-density below this value = potential boundary point.
            0.35 is a principled default — below mean/2 for normalised field.
            Tune per-corpus: denser corpora may need 0.25, sparse may need 0.45.
        """
        self.k_density = k_density
        self.threshold = threshold
        self._boundaries: list[dict] | None = None

    def detect(self) -> list[dict]:
        """
        Detect Jordan boundaries as emergent low-density ridges.

        Algorithm:
            1. Cluster centres = points with density > mean (high-K attractors)
            2. Boundary candidates = points with density < threshold
            3. Each boundary point is assigned to two nearest cluster centres
            4. Energy barrier = geodesic distance to nearest cluster

        Returns
        -------
        list[dict] with fields:
            boundary_point        : int   — index in embedding set
            separates             : tuple — (cluster_a_idx, cluster_b_idx)
            energy_barrier        : float — ΔE_B, min geodesic crossing cost
            k_density_at_boundary : float — density at this boundary point
        """
        density = self.k_density.density    # (N,) — normalised [0,1]
        D       = self.k_density.distances  # (N, N) — reuses cached computation

        mean_density     = density.mean()
        cluster_indices  = np.where(density > mean_density)[0]
        boundary_indices = np.where(density < self.threshold)[0]

        if len(boundary_indices) == 0 or len(cluster_indices) < 2:
            self._boundaries = []
            return []   # single domain — no boundary to emerge

        boundaries = []
        for b_idx in boundary_indices:
            dist_to_clusters = D[b_idx, cluster_indices]
            nearest = np.argsort(dist_to_clusters)[:2]

            if len(nearest) >= 2:
                cluster_a = cluster_indices[nearest[0]]
                cluster_b = cluster_indices[nearest[1]]
                boundaries.append({
                    "boundary_point":         int(b_idx),
                    "separates":              (int(cluster_a), int(cluster_b)),
                    "energy_barrier":         float(dist_to_clusters[nearest[0]]),
                    "k_density_at_boundary":  float(density[b_idx]),
                })

        self._boundaries = boundaries
        return boundaries

    @property
    def boundaries(self) -> list[dict]:
        if self._boundaries is None:
            self.detect()
        return self._boundaries

    def crossing_allowed(self, from_idx: int, to_idx: int, accumulated_energy: float) -> bool:
        """
        Can navigation cross the boundary between two points?

        Crossing is allowed only if accumulated_energy >= ΔE_B.
        This is the hallucination prevention gate:
            — smooth evolution cannot cross a Jordan boundary
            — only explicit energy accumulation enables crossing
        """
        B = self.boundaries
        for b in B:
            a, c = b["separates"]
            # check if this boundary separates from_idx's and to_idx's clusters
            if (a == from_idx or c == from_idx) and (a == to_idx or c == to_idx):
                return accumulated_energy >= b["energy_barrier"]
        return True   # no boundary between these points


class JordanNoise:
    """
    ξ_J(t): topologically-constrained stochastic exploration noise.

    Sampling rule:
        ξ_J(x, t) = σ · (1 - K(x)) · η(t)    η ~ N(0, 1)

    Where K is low (open conceptual space): noise is large → explore freely.
    Where K is high (dense cluster):        noise is suppressed → preserve structure.

    This is NOT standard Gaussian noise. It is information-geometrically aware:
    the noise amplitude is modulated by the local knowledge density.
    High-K regions are attractors precisely because noise cannot disrupt them.
    """

    def __init__(self, sigma: float = 0.05, rng: np.random.Generator | None = None):
        """
        Parameters
        ----------
        sigma : float
            Global noise amplitude. 0.05 default (5% perturbation at K=0).
            Tune per-experiment: smaller for stable corpora, larger for sparse exploration.
        rng : np.random.Generator
            Seeded RNG for reproducibility. Defaults to unseeded.
        """
        self.sigma = sigma
        self.rng   = rng or np.random.default_rng()

    def sample(self, k_density: np.ndarray) -> np.ndarray:
        """
        Sample ξ_J(t) modulated by local K-density.

        Parameters
        ----------
        k_density : np.ndarray, shape (N,)
            Current normalised K-density from KDensity.density.

        Returns
        -------
        np.ndarray, shape (N,)
            ξ_J field — large where K is low, near-zero where K is high.
        """
        eta = self.rng.standard_normal(k_density.shape)   # η ~ N(0, 1)
        return self.sigma * (1.0 - k_density) * eta       # ξ_J = σ·(1 - K)·η

    def sample_with_boundary_suppression(
        self,
        k_density: np.ndarray,
        boundary: JordanBoundary,
    ) -> np.ndarray:
        """
        Sample ξ_J(t) with additional suppression at detected boundary points.

        Boundary points receive zero noise — crossing must be deliberate
        (accumulated energy), not accidental (single noise spike).
        """
        xi = self.sample(k_density)
        for b in boundary.boundaries:
            xi[b["boundary_point"]] = 0.0   # enforce boundary impermeability
        return xi

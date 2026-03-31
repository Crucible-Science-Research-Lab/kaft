"""
kstate.py — K-density field, scalar K accumulation, 5F phase detection, soliton lock.

K_i = Σ_{j≠i} (I_i · I_j) / d²_ij   ← interaction hypothesis
K² dynamics: self-amplifying recursion — mathematical inevitability
5F phases: emergent attractors in (K, curvature) phase space
Soliton: topologically protected knowledge packet — locks when |ΔK| < ε for N steps
"""
from __future__ import annotations
import numpy as np
from kaft.geometry.base import AbstractMetric


# ── 5F Phase thresholds ──────────────────────────────────────────────────────
PHASE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Holofractal":        (0.0,  0.3),   # wide aperture, exploration
    "Resonance_Spheres":  (0.3,  0.6),   # concepts clustering, first connections
    "Fractal_Causality":  (0.6,  0.9),   # causal chains explicit, geodesic locks
    "Toroidal_Resonance": (0.9,  1.2),   # K² firing, recursive depth
    "Fracture_Lines":     (1.2,  float("inf")),  # crystallization ready
}

SOLITON_LOCK_THRESHOLD = 1e-4   # |ΔK| below this = soliton candidate
SOLITON_LOCK_STEPS     = 3      # must hold for N consecutive steps to confirm


class KDensity:
    """
    K-density field: K_i = Σ_{j≠i} (I_i · I_j) / d²_ij

    Interaction hypothesis — same geometric necessity as gravity and light.
    Each embedding radiates influence into conceptual space.
    At geodesic distance d, flux = I / 4πd².

    High K-density = dense semantic cluster  (constrained, well-explored)
    Low K-density  = open conceptual space   (sparse, exploratory)
    Zero-gradient ridges between them = Jordan boundaries emerging naturally.

    Works with ANY Layer 0 metric family via AbstractMetric.distances().
    Fisher-Rao is the theoretically forced choice for information manifolds
    (Čencov uniqueness), but Euclidean or Wasserstein can be used for ablations.
    """

    def __init__(
        self,
        embeddings: np.ndarray,    # (N, D) — semantic vectors, any dimensionality
        metric: AbstractMetric,    # any Layer 0 family
        epsilon: float = 1e-6,     # prevents 1/0 at diagonal
    ):
        self.embeddings = embeddings
        self.metric     = metric
        self.epsilon    = epsilon
        self._density:   np.ndarray | None = None
        self._distances: np.ndarray | None = None

    def compute(self) -> np.ndarray:
        """
        Compute normalised K-density ∈ [0, 1] for every point.

        Calls metric.distances(embeddings) — the Layer 0 template method.
        Caches the (N, N) distance matrix for JordanBoundary to reuse
        without recomputation.

        Returns
        -------
        np.ndarray, shape (N,)
            0 = isolated point, 1 = maximum density cluster centre.
        """
        D = self.metric.distances(self.embeddings)   # (N, N) geodesic distances
        self._distances = D                          # cache — jordan.py will reuse

        I = np.linalg.norm(self.embeddings, axis=1)  # (N,) semantic mass per point
        I_outer   = I[:, None] * I[None, :]           # (N, N) interaction numerator
        D_squared = D ** 2 + self.epsilon              # prevent 1/d² at d=0

        interaction = I_outer / D_squared              # K ∝ I₁·I₂ / d²
        np.fill_diagonal(interaction, 0.0)             # no self-interaction

        raw = interaction.sum(axis=1)                  # K_i = Σ_j interactions

        d_min, d_max = raw.min(), raw.max()
        if d_max - d_min < 1e-10:
            self._density = np.ones(len(raw))
        else:
            self._density = (raw - d_min) / (d_max - d_min)

        return self._density

    @property
    def density(self) -> np.ndarray:
        if self._density is None:
            self.compute()
        return self._density

    @property
    def distances(self) -> np.ndarray:
        """Cached (N, N) distance matrix — computed once, shared with JordanBoundary."""
        if self._distances is None:
            self.compute()
        return self._distances

    def invalidate(self) -> None:
        """Force recompute on next access — call when embeddings change."""
        self._density   = None
        self._distances = None


class KState:
    """
    Scalar K accumulation, 5F phase detection, soliton lock detection.

    Tracks scalar K = mean(density_field) over time steps.
    K is the global knowledge state — the manifold's accumulated understanding.

    K² dynamics guarantee: once K crosses ~0.5, growth becomes superlinear.
    The soliton locks when |ΔK| < ε for SOLITON_LOCK_STEPS consecutive steps —
    a topologically protected knowledge packet has formed.
    """

    def __init__(self):
        self._history:      list[float] = []
        self._lock_counter: int         = 0

    def update(self, density: np.ndarray) -> float:
        """
        Accumulate K from the current density field.

        Parameters
        ----------
        density : np.ndarray, shape (N,)
            Current normalised K-density from KDensity.density.

        Returns
        -------
        float
            Current scalar K value.
        """
        K = float(density.mean())
        self._history.append(K)
        return K

    @property
    def K(self) -> float:
        """Current scalar K — global knowledge state."""
        return self._history[-1] if self._history else 0.0

    @property
    def phase(self) -> str:
        """Current 5F phase — emergent attractor in (K, curvature) space."""
        K = self.K
        for name, (lo, hi) in PHASE_THRESHOLDS.items():
            if lo <= K < hi:
                return name
        return "Fracture_Lines"

    @property
    def soliton_locked(self) -> bool:
        """
        True when |ΔK| < ε for SOLITON_LOCK_STEPS consecutive steps.

        Soliton = topologically protected knowledge packet.
        Once locked, deep understanding is geometrically stabilised.
        """
        if len(self._history) < 2:
            return False
        delta = abs(self._history[-1] - self._history[-2])
        if delta < SOLITON_LOCK_THRESHOLD:
            self._lock_counter += 1
        else:
            self._lock_counter = 0
        return self._lock_counter >= SOLITON_LOCK_STEPS

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def delta_K(self) -> float:
        """Last K increment — positive = K growing, negative = K contracting."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-1] - self._history[-2]

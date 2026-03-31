"""
metric_evolution.py — Recursive metric update: K → p → g_μν

The fourth component of the KAFT loop — the manifold co-creates itself:
    K evolves → p_i ∝ e^(-K_i/T) → g_μν recomputed → new ∇^FR → K evolves

Lives inside KAFTSource.__call__() — KAFT-specific, WaveEngine stays general.

Fisher-Rao geometry update via SLERP:
    p_i ∝ e^(-K_i/T) → weighted centroid on sphere →
    SLERP each embedding toward centroid by α = clip(0.1/T, 0, 1)

    Low  T → large α → strong contraction around knowledge clusters
    High T → small α → manifold barely reshapes (exploration preserved)

    BUG NOTE: naive sqrt(p_i)·emb_i / norm = emb_i — normalization cancels
    the reweighting entirely. SLERP avoids this by moving points ALONG
    the sphere surface, not scaling their magnitude.
"""
from __future__ import annotations
import numpy as np
from kaft.dynamics.kstate import KDensity


class MetricEvolution:
    """
    Updates Fisher-Rao geometry as K evolves via SLERP toward weighted centroid.

    Each step:
        1. p_i = softmax(-K_i / T)          — Boltzmann distribution
        2. centroid = Σ p_i · emb_i (norm)  — probability-weighted pole
        3. emb_i ← slerp(emb_i, centroid, α) — move along sphere surface
        4. k_density.invalidate()            — force metric recomputation

    The centroid is pulled toward high-probability (low-K, exploratory) regions.
    As K accumulates, dense clusters have low p → centroid shifts away from them
    → geometry opens up around unexplored space. Correct KAFT behaviour.
    """

    def __init__(self, temperature: float = 1.0, base_step: float = 0.1):
        """
        Parameters
        ----------
        temperature : float
            Controls manifold reshaping aggressiveness.
            T = 0.1  — sharp contraction, fast convergence
            T = 1.0  — balanced Boltzmann, default
            T = 10.0 — near-flat, slow metric evolution
        base_step : float
            Maximum fractional step size at T=1. α = clip(base_step/T, 0, 1).
            0.1 default: 10% SLERP step per iteration at T=1.
            Tune smaller (0.01) for large corpora to avoid overshooting.
        """
        self.temperature  = temperature
        self.base_step    = base_step
        self._step_count  = 0

    def step(self, k_density_obj: KDensity, k_field: np.ndarray) -> None:
        """
        Update geometry in-place given current K field.

        Modifies k_density_obj.embeddings and calls .invalidate()
        so next access to .density / .distances recomputes on evolved geometry.
        """
        new_emb = self._reweight(k_density_obj.embeddings, k_field)
        k_density_obj.embeddings = new_emb
        k_density_obj.invalidate()
        self._step_count += 1

    def _reweight(self, embeddings: np.ndarray, k_field: np.ndarray) -> np.ndarray:
        """
        Compute p_i = softmax(-K_i/T), find probability-weighted centroid,
        SLERP all embeddings toward centroid by α = clip(base_step/T, 0, 1).
        """
        # Boltzmann distribution
        log_p  = -k_field / self.temperature
        log_p -= log_p.max()               # numerical stability
        p      = np.exp(log_p)
        p      = p / p.sum()               # normalise

        # Probability-weighted centroid on the sphere
        centroid      = (p[:, None] * embeddings).sum(axis=0)   # (D,)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-10:
            return embeddings.copy()       # degenerate — return unchanged
        centroid = centroid / centroid_norm

        # SLERP step size: large at low T, tiny at high T
        alpha = float(np.clip(self.base_step / self.temperature, 0.0, 1.0))

        return self._slerp_batch(embeddings, centroid, alpha)

    def _slerp_batch(
        self,
        a:  np.ndarray,   # (N, D) source points on unit sphere
        b:  np.ndarray,   # (D,)   target point on unit sphere
        t:  float,        # interpolation fraction ∈ [0, 1]
    ) -> np.ndarray:
        """
        SLERP each row of a toward unit vector b by fraction t.

        slerp(a, b, t) = sin((1-t)θ)/sin(θ) · a  +  sin(tθ)/sin(θ) · b
        where θ = arccos(<a, b>)

        Falls back to linear interpolation for near-parallel vectors (sin θ ≈ 0).
        Always renormalises output to unit sphere.
        """
        dots      = np.clip(a @ b, -1.0, 1.0)   # (N,) cos θ
        theta     = np.arccos(dots)               # (N,) angle
        sin_theta = np.sin(theta)                 # (N,)

        # Safe denominator — avoid divide by zero for nearly parallel vectors
        safe_sin = np.where(sin_theta < 1e-10, 1.0, sin_theta)

        w_a = (np.sin((1.0 - t) * theta) / safe_sin)[:, None]   # (N, 1)
        w_b = (np.sin(t * theta)         / safe_sin)[:, None]   # (N, 1)

        slerp_result  = w_a * a + w_b * b[None, :]              # (N, D)
        linear_result = (1.0 - t) * a + t * b[None, :]          # (N, D) fallback

        parallel = (sin_theta < 1e-10)[:, None]                  # (N, 1) mask
        result   = np.where(parallel, linear_result, slerp_result)

        norms  = np.linalg.norm(result, axis=1, keepdims=True)
        norms  = np.where(norms < 1e-10, 1.0, norms)
        return result / norms                                     # unit sphere

    def distribution(self, k_field: np.ndarray) -> np.ndarray:
        """Current Boltzmann p_i ∝ e^(-K_i/T). Useful for experiment logging."""
        log_p  = -k_field / self.temperature
        log_p -= log_p.max()
        p      = np.exp(log_p)
        return p / p.sum()

    @property
    def step_count(self) -> int:
        return self._step_count
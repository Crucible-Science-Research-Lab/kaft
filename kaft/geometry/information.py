
import numpy as np
from kaft.geometry.base import AbstractMetric



class FisherRaoMetric(AbstractMetric):
    """
    Fisher-Rao information metric on the statistical manifold.

    Treats each vector as an unnormalised probability distribution.
    Distance is the geodesic angle on the unit sphere of sqrt-distributions:
        d(p, q) = 2 * arccos( Σ sqrt(p_i * q_i) )

    The natural metric for ML embeddings, statistical mechanics,
    and any domain where vectors represent distributions over outcomes.
    """

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        shifted = vectors - vectors.min(axis=1, keepdims=True) + 1e-10
        probs = shifted / shifted.sum(axis=1, keepdims=True)
        sq = np.sqrt(probs)
        dot = np.clip(sq @ sq.T, -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def speed_field(self, grid_size: int) -> np.ndarray:
        """Information density slows waves near the centre."""
        cx, cy = grid_size // 2, grid_size // 2
        x, y = np.arange(grid_size), np.arange(grid_size)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2) / (grid_size / 2)
        return 0.3 + 0.5 * r

    def geometry_type(self) -> str:
        return "fisher_rao"

class KLDivergence(AbstractMetric):
    """
    Kullback-Leibler divergence — the asymmetric information divergence.

    KL(p||q) = Σ p_i * log(p_i / q_i)

    Measures the information lost when q is used to approximate p.
    Asymmetric: KL(p||q) ≠ KL(q||p).
    The natural divergence for variational inference and maximum likelihood.

    Note: not a true metric (no symmetry, no triangle inequality),
    but registered here as a divergence measure — valid for dynamics.
    """

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        shifted = vectors - vectors.min(axis=1, keepdims=True) + 1e-10
        probs = shifted / shifted.sum(axis=1, keepdims=True)

        # KL(p||q) for all pairs — uses log(p/q) = log(p) - log(q)
        log_p = np.log(probs)                        # (N, d)
        # KL(i||j) = Σ_k p_i_k * (log_p_i_k - log_p_j_k)
        # = p @ log_p.T - diag(p @ log_p.T) broadcast
        kl = (probs * log_p).sum(axis=1, keepdims=True) - probs @ log_p.T
        return np.maximum(kl, 0.0)   # numerical clip — KL ≥ 0

    def speed_field(self, grid_size: int) -> np.ndarray:
        """
        KL geometry creates strongly anisotropic speed variation.
        The asymmetry of KL produces a non-symmetric field — slower
        in one direction, creating directional wave propagation.
        """
        cx, cy = grid_size // 2, grid_size // 2
        x, y = np.arange(grid_size), np.arange(grid_size)
        xx, yy = np.meshgrid(x, y)
        # Asymmetric gradient — steeper drop in one direction
        dx = (xx - cx) / grid_size
        dy = (yy - cy) / grid_size
        return 0.4 + 0.4 * (0.5 + 0.5 * np.tanh(3 * dx)) * np.exp(-dy**2 / 0.1)

    def geometry_type(self) -> str:
        return "kl_divergence"
    

class JensenShannonMetric(AbstractMetric):
    """
    Jensen-Shannon divergence — the symmetric, bounded information metric.

    JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)  where m = 0.5*(p+q)

    Unlike KL, JS is:
    - Symmetric: JS(p||q) = JS(q||p)
    - Bounded: JS ∈ [0, log(2)] ≈ [0, 0.693]
    - Always finite, even when p and q have non-overlapping supports

    sqrt(JS) is a true metric — the Jensen-Shannon distance.
    Used in generative models, NLP, and biological sequence comparison.
    """

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        shifted = vectors - vectors.min(axis=1, keepdims=True) + 1e-10
        probs = shifted / shifted.sum(axis=1, keepdims=True)   # (N, d)

        js = np.zeros((len(probs), len(probs)))
        log_p = np.log(probs)

        for i in range(len(probs)):
            m = 0.5 * (probs[i] + probs)                       # (N, d)
            log_m = np.log(m + 1e-15)
            kl_pm = (probs[i] * (log_p[i] - log_m)).sum(axis=1)
            kl_qm = (probs * (log_p - log_m)).sum(axis=1)
            js[i] = np.maximum(0.5 * (kl_pm + kl_qm), 0.0)

        return js  # JS divergence — take sqrt for true metric

    def speed_field(self, grid_size: int) -> np.ndarray:
        """
        JS geometry is symmetric and bounded — the speed field reflects this.
        Smooth, radially symmetric slowdown — no directional bias unlike KL.
        The bounded nature means the field never goes to zero, just slows gently.
        """
        cx, cy = grid_size // 2, grid_size // 2
        x, y = np.arange(grid_size), np.arange(grid_size)
        xx, yy = np.meshgrid(x, y)
        r2 = ((xx - cx)**2 + (yy - cy)**2) / (grid_size / 2)**2
        # Symmetric, gentle slowdown — bounded nature shows as smooth variation
        return 0.35 + 0.45 * (1.0 - np.exp(-r2 / 0.4))

    def geometry_type(self) -> str:
        return "jensen_shannon"
    


class AlphaDivergence(AbstractMetric):
    """
    Alpha-divergence — the parametric family that unifies information geometry.

    D_α(p||q) = (4 / (1 - α²)) * (1 - Σ p^((1+α)/2) * q^((1-α)/2))

    Special cases:
    - α → +1  :  KL divergence  KL(p||q)
    - α → -1  :  Reverse KL    KL(q||p)
    - α =  0  :  Hellinger distance (squared)
    - α =  0.5:  Bhattacharyya divergence
    - α = -1/3:  Used in turbulence modelling

    This is Nielsen's key contribution: all f-divergences form a 1-parameter
    family on the statistical manifold, interpolating smoothly between regimes.
    """

    def __init__(self, alpha: float = 0.0):
        if abs(alpha) == 1.0:
            raise ValueError(
                "alpha=±1 is a limiting case — use KLDivergence directly. "
                "Values like 0.99 or -0.99 approximate it numerically."
            )
        self.alpha = alpha
        self._coeff = 4.0 / (1.0 - alpha ** 2)

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        shifted = vectors - vectors.min(axis=1, keepdims=True) + 1e-10
        probs = shifted / shifted.sum(axis=1, keepdims=True)

        exp_p = (1.0 + self.alpha) / 2.0
        exp_q = (1.0 - self.alpha) / 2.0

        p_pow = probs ** exp_p    # (N, d)
        q_pow = probs ** exp_q    # (N, d)

        # cross: inner(p^exp_p, q^exp_q) for all pairs
        cross = p_pow @ q_pow.T   # (N, N)
        return np.maximum(self._coeff * (1.0 - cross), 0.0)

    def speed_field(self, grid_size: int) -> np.ndarray:
        """
        Alpha controls the geometry's 'sharpness'.
        α near +1 (KL-like): sharp asymmetric field
        α = 0 (Hellinger): smooth radially symmetric field
        α near -1 (reverse KL): flipped asymmetry
        """
        cx, cy = grid_size // 2, grid_size // 2
        x, y = np.arange(grid_size), np.arange(grid_size)
        xx, yy = np.meshgrid(x, y)
        dx = (xx - cx) / grid_size
        dy = (yy - cy) / grid_size
        r = np.sqrt(dx**2 + dy**2)

        # alpha interpolates between symmetric (0) and directional (±1)
        symmetric   = 0.35 + 0.45 * (1.0 - np.exp(-r**2 / 0.15))
        directional = 0.4 + 0.4 * (0.5 + 0.5 * np.tanh(3 * dx * np.sign(self.alpha)))
        return symmetric * (1 - abs(self.alpha)) + directional * abs(self.alpha)

    def geometry_type(self) -> str:
        return f"alpha_{self.alpha:.2f}"


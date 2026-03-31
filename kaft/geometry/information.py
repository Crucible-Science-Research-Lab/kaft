
import numpy as np
from kaft.geometry.base import AbstractMetric
from scipy.optimize import linprog


class FisherRaoMetric(AbstractMetric):
    """
    Fisher-Rao information metric on the statistical manifold.

    Treats each vector as an unnormalised probability distribution.
    Distance is the geodesic angle on the unit sphere of sqrt-distributions:
        d(p, q) = 2 * arccos( Σ sqrt(p_i * q_i) )

    The natural metric for ML embeddings, statistical mechanics,
    and any domain where vectors represent distributions over outcomes.
    """

    # def distances(self, vectors: np.ndarray) -> np.ndarray:
    #     shifted = vectors - vectors.min(axis=1, keepdims=True) + 1e-10
    #     probs = shifted / shifted.sum(axis=1, keepdims=True)
    #     sq = np.sqrt(probs)
    #     dot = np.clip(sq @ sq.T, -1.0, 1.0)
    #     return 2.0 * np.arccos(dot)
    
    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        a_s = a - a.min() + 1e-10
        b_s = b - b.min() + 1e-10
        u = np.sqrt(a_s)
        v = np.sqrt(b_s)
        bc = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(2.0 * np.arccos(np.clip(bc, -1.0, 1.0)))

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        shifted = vectors - vectors.min(axis=1, keepdims=True) + 1e-10
        sq = np.sqrt(shifted)
        norms = np.linalg.norm(sq, axis=1, keepdims=True)
        sq_n = sq / norms          # L2-normalize the sqrt-vectors
        cos_sim = sq_n @ sq_n.T    # diagonal is EXACTLY 1.0
        return 2.0 * np.arccos(np.clip(cos_sim, -1.0, 1.0))

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
    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        a_s = a - a.min() + 1e-10
        b_s = b - b.min() + 1e-10
        p = a_s / a_s.sum()
        q = b_s / b_s.sum()
        return float(np.maximum(np.sum(p * (np.log(p) - np.log(q))), 0.0))
    
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
    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        a_s = a - a.min() + 1e-10
        b_s = b - b.min() + 1e-10
        p = a_s / a_s.sum()
        q = b_s / b_s.sum()
        m = 0.5 * (p + q)
        log_m = np.log(m + 1e-15)
        kl_pm = np.sum(p * (np.log(p) - log_m))
        kl_qm = np.sum(q * (np.log(q) - log_m))
        return float(np.maximum(0.5 * (kl_pm + kl_qm), 0.0))
    
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
    
    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """Scalar alpha-divergence between two distributions."""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        # normalize to simplex
        a = np.maximum(a, 1e-12); a = a / a.sum()
        b = np.maximum(b, 1e-12); b = b / b.sum()
        exp_p = (1.0 + self.alpha) / 2.0
        exp_q = (1.0 - self.alpha) / 2.0
        cross = np.sum(a ** exp_p * b ** exp_q)
        return float(max(self._coeff * (1.0 - cross), 0.0))

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



class BregmanDivergence(AbstractMetric):
    """
    D_F(p || q) = F(p) - F(q) - <∇F(q), p - q>

    The Hessian of F is the metric tensor at each point.
    All instances below are special cases via choice of generator F.
    """
    def __init__(self, F, grad_F, hess_F=None, name="bregman"):
        self.F = F
        self.grad_F = grad_F
        self.hess_F = hess_F
        self._name = name

    def compute(self, p: np.ndarray, q: np.ndarray) -> float:
        p, q = np.asarray(p, float), np.asarray(q, float)
        return float(self.F(p) - self.F(q) - np.dot(self.grad_F(q), p - q))

    def speed_field(self, grid: np.ndarray) -> np.ndarray:
        """Wavefront speed = 1/curvature(F). Flat F → fast. Curved F → slow."""
        if self.hess_F is not None:
            return 1.0 / (np.abs(self.hess_F(grid)) + 1e-8)
        eps = 1e-5
        curv = np.abs(self.F(grid+eps) - 2*self.F(grid) + self.F(grid-eps)) / eps**2
        return 1.0 / (curv + 1e-8)

    def geometry_type(self) -> str:
        return f"bregman_{self._name}"


# --- Named instances: each F generates a distinct geometry ---

def bregman_kl() -> BregmanDivergence:
    """KL divergence as Bregman. F = Σ p·log(p). Recovers standard KL exactly."""
    return BregmanDivergence(
        F      = lambda p: np.sum(p * np.log(p + 1e-12)),
        grad_F = lambda p: np.log(p + 1e-12) + 1.0,
        hess_F = lambda p: 1.0 / (p + 1e-12),
        name   = "kl"
    )

def bregman_tsallis(q: float) -> BregmanDivergence:
    """Tsallis divergence. q→1 recovers KL. q=2 → χ²-like. Non-extensive stats."""
    assert q != 1, "Use bregman_kl() for q=1"
    return BregmanDivergence(
        F      = lambda p: np.sum(p**q / (q * (q - 1))),
        grad_F = lambda p: p**(q - 1) / (q - 1),
        hess_F = lambda p: p**(q - 2),
        name   = f"tsallis_q{q}"
    )

def bregman_euclidean() -> BregmanDivergence:
    """Squared Euclidean. F = ||p||². Flat geometry, uniform curvature."""
    return BregmanDivergence(
        F      = lambda p: float(np.dot(p, p)),
        grad_F = lambda p: 2 * p,
        hess_F = lambda p: 2 * np.ones_like(p),
        name   = "euclidean"
    )

def bregman_itakura_saito() -> BregmanDivergence:
    """Itakura-Saito. F = -log(p). Scale-invariant. Used in audio/NMF."""
    return BregmanDivergence(
        F      = lambda p: -np.sum(np.log(p + 1e-12)),
        grad_F = lambda p: -1.0 / (p + 1e-12),
        hess_F = lambda p:  1.0 / (p + 1e-12)**2,
        name   = "itakura_saito"
    )






class WassersteinMetric(AbstractMetric):
    """
    Wasserstein-p distance (Earth Mover's Distance).
    
    W_p(μ,ν) = (inf_{γ∈Γ(μ,ν)} ∫ d(x,y)^p dγ)^(1/p)
    
    Geometric character:
      - Symmetric:  W(p,q) = W(q,p)         ← unlike KL
      - True metric: triangle inequality holds ← unlike KL  
      - Support-aware: cost ∝ distance mass travels
      - No absolute continuity required       ← unlike KL (no log(0) crisis)
    """

    def __init__(self, p: int = 2):
        self.p = p

    def _cost_matrix(self, xs, ys):
        xs = np.asarray(xs, float).reshape(-1, 1)
        ys = np.asarray(ys, float).reshape(-1, 1)
        diff = xs[:, np.newaxis, :] - ys[np.newaxis, :, :]
        return (np.sum(diff ** 2, axis=-1) ** 0.5) ** self.p

    def compute(self, a: np.ndarray, b: np.ndarray,
                x_support=None, y_support=None) -> float:
        """Scalar Wasserstein-p distance between two distributions."""
        a = np.asarray(a, float); a /= a.sum()
        b = np.asarray(b, float); b /= b.sum()
        n, m = len(a), len(b)

        xs = np.arange(n, dtype=float) if x_support is None else np.asarray(x_support, float)
        ys = np.arange(m, dtype=float) if y_support is None else np.asarray(y_support, float)

        if xs.ndim == 1 and ys.ndim == 1 and n == m and np.allclose(xs, ys):
            return self._exact_1d(a, b, xs)

        return self._lp_emd(a, b, xs, ys)


    def _exact_1d(self, p, q, support):
        """
        W₁ = ∫|CDF_p - CDF_q| dx   (quantile/CDF formula, exact, O(n))
        W₂ = (∫|CDF_p - CDF_q|² dx)^(1/2)
        """
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        spacings = np.diff(support)               # length n-1, right-side widths
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        diff  = np.abs(cdf_p - cdf_q)[:-1]       # drop last (always 0)
        if self.p == 1:
            return float(np.dot(diff, spacings))
        else:
            return float(np.dot(diff ** self.p, spacings) ** (1 / self.p))

    def _lp_emd(self, p, q, xs, ys):
        """General case: LP over transport plan γ[i,j]"""
        C = self._cost_matrix(xs, ys)
        n, m = len(p), len(q)
        c = C.flatten()

        # Row sums = p, col sums = q
        A_eq = np.zeros((n + m, n * m))
        for i in range(n):
            A_eq[i, i * m:(i + 1) * m] = 1.0
        for j in range(m):
            A_eq[n + j, j::m] = 1.0

        result = linprog(c, A_eq=A_eq, b_eq=np.concatenate([p, q]),
                         bounds=[(0, None)] * (n * m), method='highs')

        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")
        raw = float(result.fun)
        return raw ** (1 / self.p) if self.p > 1 else raw

    def geometry_type(self):
        return f"wasserstein_p{self.p}"
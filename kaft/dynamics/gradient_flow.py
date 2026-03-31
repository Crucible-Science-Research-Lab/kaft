"""
gradient_flow.py — GradientFlow: general divergence descent engine

    ∂K/∂t = -∇D(K ‖ q)

Works for any divergence family from Layer 0.
This is the experiment engine for the arXiv smoking gun:

    KL divergence gradient flow   → softmax proxy (flat-space baseline)
    Fisher-Rao gradient flow      → KAFT (curved-space, Cencov-forced)

    Prediction from genesis crystal:
        KAFT variance DECREASES with N  (universal low-dimensional structure)
        KL/softmax variance INCREASES with N (topology exhaustion)

GradientFlow is deliberately simpler than WaveEngine:
    - Euler only (gradient descent does not benefit from RK4)
    - No after_step protocol (pure descent, no metric feedback)
    - Loss plateau detection instead of variance convergence window
    - FlowResult carries loss_history + variance_history for experiment comparison
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable


PLATEAU_WINDOW  = 10    # steps loss must be flat to declare convergence
PLATEAU_EPSILON = 1e-6  # |loss_t - loss_{t-1}| threshold


@dataclass
class FlowResult:
    K_final:          np.ndarray   # (N,)
    K_history:        np.ndarray   # (n_steps+1, N)
    loss_history:     np.ndarray   # (n_steps+1,)  D(K‖q) over time
    variance_history: np.ndarray   # (n_steps+1,)  var(K) over time
    converged:        bool
    convergence_step: int | None = None


class GradientFlow:
    """
    General divergence gradient descent: ∂K/∂t = -∇D(K ‖ q)

    Parameters
    ----------
    divergence_fn : callable (K: ndarray(N,), q: ndarray(N,)) → float
        Scalar divergence D(K ‖ q). Any Layer 0 family works here.
        Gradient is computed via central finite differences — no autograd needed.
    lr : float
        Learning rate (step size). 0.01 default stable across most families.
    grad_eps : float
        Finite difference epsilon. 1e-5 gives stable gradients without noise.
    """

    def __init__(
        self,
        divergence_fn: Callable[[np.ndarray, np.ndarray], float],
        lr:            float = 0.01,
        grad_eps:      float = 1e-5,
    ):
        self.divergence_fn = divergence_fn
        self.lr            = lr
        self.grad_eps      = grad_eps

    def _gradient(self, K: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Central finite-difference gradient of D(K ‖ q) w.r.t. K.

        ∂D/∂K_i ≈ [D(K + ε·eᵢ ‖ q) - D(K - ε·eᵢ ‖ q)] / 2ε

        O(ε²) accuracy. Works with any divergence_fn — no analytical form needed.
        """
        N, eps = len(K), self.grad_eps
        grad   = np.zeros(N)
        for i in range(N):
            Kp = K.copy(); Kp[i] += eps
            Km = K.copy(); Km[i] -= eps
            grad[i] = (self.divergence_fn(Kp, q) - self.divergence_fn(Km, q)) / (2 * eps)
        return grad

    def step(self, K: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Single gradient descent step: K ← K - lr · ∇D(K ‖ q)"""
        return K - self.lr * self._gradient(K, q)

    def run(
        self,
        K0:             np.ndarray,
        target:         np.ndarray,
        n_steps:        int,
        early_stopping: bool = True,
    ) -> FlowResult:
        """
        Descend D(K ‖ target) from K0 for n_steps steps.

        Returns FlowResult with loss_history and variance_history
        — the two signals compared in the softmax vs Fisher-Rao experiment.
        """
        N                   = len(K0)
        K_history           = np.zeros((n_steps + 1, N))
        loss_history        = np.zeros(n_steps + 1)
        variance_history    = np.zeros(n_steps + 1)

        K                   = K0.copy()
        K_history[0]        = K
        loss_history[0]     = self.divergence_fn(K, target)
        variance_history[0] = np.var(K)

        converged        = False
        convergence_step = None
        plateau_count    = 0

        for s in range(n_steps):
            K                       = self.step(K, target)
            loss                    = self.divergence_fn(K, target)
            K_history[s + 1]        = K
            loss_history[s + 1]     = loss
            variance_history[s + 1] = np.var(K)

            if abs(loss_history[s + 1] - loss_history[s]) < PLATEAU_EPSILON:
                plateau_count += 1
                if plateau_count >= PLATEAU_WINDOW and early_stopping:
                    converged        = True
                    convergence_step = s + 1
                    K_history        = K_history[:convergence_step + 1]
                    loss_history     = loss_history[:convergence_step + 1]
                    variance_history = variance_history[:convergence_step + 1]
                    break
            else:
                plateau_count = 0

        return FlowResult(
            K_final=K, K_history=K_history,
            loss_history=loss_history, variance_history=variance_history,
            converged=converged, convergence_step=convergence_step,
        )


# ── Built-in divergence functions ─────────────────────────────────────────────
# Standalone — no Layer 0 import needed.
# Layer 0 families (FisherRao, KLDivergence) can be wrapped identically.

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    D_KL(p ‖ q) = Σ p_i · log(p_i / q_i)

    Softmax proxy for flat-space attention gradient flow.
    Normalises both inputs to valid distributions before computation.
    """
    pn = np.abs(p) + eps;  pn /= pn.sum()
    qn = np.abs(q) + eps;  qn /= qn.sum()
    return float(np.sum(pn * np.log(pn / qn)))


def euclidean_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Squared Euclidean distance ‖p - q‖²

    Flat-space baseline. Gradient: 2(p - q).
    Constant metric tensor g_μν = δ_μν — no curvature.
    """
    return float(np.sum((p - q) ** 2))


def fisher_rao_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Fisher-Rao geodesic distance (Bhattacharyya angle):
        D_FR(p, q) = arccos(Σ sqrt(p_i · q_i))

    The information-geometrically correct divergence.
    Forced by Cencov uniqueness theorem for probability manifolds.
    Gradient flow under this = KAFT Term 2 in the continuous limit.
    """
    pn = np.abs(p) + eps;  pn /= pn.sum()
    qn = np.abs(q) + eps;  qn /= qn.sum()
    bc = np.sum(np.sqrt(pn * qn))
    return float(np.arccos(np.clip(bc, -1.0 + eps, 1.0 - eps)))

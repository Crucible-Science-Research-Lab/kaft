"""
wave.py — WaveEngine: general first-order field integrator

    ∂K/∂t = source_fn(K, t)

Works for any source_fn — KAFT, Maxwell, Schwarzschild, custom.
The metric shapes the geometry. source_fn defines the physics.
WaveEngine knows neither — it only integrates.

Schemes:
    euler — fast, stable for dt < 0.1
    rk4   — 4th-order accurate, use for publication runs

after_step protocol:
    If source_fn implements after_step(K_new, t_new), WaveEngine calls it
    once per full step — NOT per RK4 sub-step.
    KAFTSource uses this for metric evolution.
    Generic sources (Maxwell, etc.) simply don't implement it.

# TODO wave_v2:
#   ∂²K/∂t² - v²_cog·Δ_g K = 2c·v²_cog·K·∂K/∂t
#   Left side = covariant d'Alembertian □_g K
#   Recovers full Levi-Civita parallel transport structure
#   Requires: leapfrog/Störmer-Verlet scheme, second K_dot state variable
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


CONVERGENCE_WINDOW  = 5      # consecutive steps variance must be stable
CONVERGENCE_EPSILON = 1e-5   # |var(K_t) - var(K_{t-1})| threshold


@dataclass
class WaveResult:
    K_final:          np.ndarray
    K_history:        np.ndarray
    t_history:        np.ndarray
    converged:        bool
    convergence_step: int | None = None


class WaveEngine:
    """
    General first-order field integrator: ∂K/∂t = source_fn(K, t)

    Parameters
    ----------
    source_fn : callable (K: ndarray(N,), t: float) → ndarray(N,)
        Returns dK/dt at every point. Must be stateless w.r.t. time.
        Stateful updates (metric evolution, logging) go in after_step().
    dt : float
        Timestep. Euler stable for dt < 0.1. RK4 tolerates up to ~0.5.
    scheme : str
        "euler" — prototyping and ablations
        "rk4"   — publication runs
    """

    def __init__(self, source_fn: callable, dt: float = 0.01, scheme: str = "euler"):
        self.source_fn = source_fn
        self.dt        = dt
        if scheme not in ("euler", "rk4"):
            raise ValueError(f"scheme must be 'euler' or 'rk4', got {scheme!r}")
        self.scheme = scheme

    def step(self, K: np.ndarray, t: float) -> np.ndarray:
        """Advance K by one timestep dt."""
        K_new = K + self.dt * self.source_fn(K, t) if self.scheme == "euler" \
                else self._rk4_step(K, t)
        if hasattr(self.source_fn, "after_step"):
            self.source_fn.after_step(K_new, t + self.dt)
        return K_new

    def _rk4_step(self, K: np.ndarray, t: float) -> np.ndarray:
        dt = self.dt
        k1 = self.source_fn(K,                t         )
        k2 = self.source_fn(K + dt/2 * k1,    t + dt/2  )
        k3 = self.source_fn(K + dt/2 * k2,    t + dt/2  )
        k4 = self.source_fn(K + dt    * k3,   t + dt    )
        return K + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def run(self, K0: np.ndarray, n_steps: int, early_stopping: bool = True) -> WaveResult:
        """
        Integrate ∂K/∂t from K0 for n_steps timesteps.

        Convergence criterion (from paper):
            |var(K_t) - var(K_{t-1})| < CONVERGENCE_EPSILON
            for CONVERGENCE_WINDOW consecutive steps.
        """
        N            = len(K0)
        K_history    = np.zeros((n_steps + 1, N))
        t_history    = np.zeros(n_steps + 1)
        K_history[0] = K0.copy()

        K                = K0.copy()
        t                = 0.0
        converged        = False
        convergence_step = None
        stable_count     = 0
        prev_var         = np.var(K)

        for s in range(n_steps):
            K   = self.step(K, t)
            t  += self.dt
            K_history[s + 1] = K
            t_history[s + 1] = t

            curr_var = np.var(K)
            if abs(curr_var - prev_var) < CONVERGENCE_EPSILON:
                stable_count += 1
                if stable_count >= CONVERGENCE_WINDOW and early_stopping:
                    converged        = True
                    convergence_step = s + 1
                    K_history        = K_history[:convergence_step + 1]
                    t_history        = t_history[:convergence_step + 1]
                    break
            else:
                stable_count = 0
            prev_var = curr_var

        return WaveResult(
            K_final=K, K_history=K_history, t_history=t_history,
            converged=converged, convergence_step=convergence_step,
        )

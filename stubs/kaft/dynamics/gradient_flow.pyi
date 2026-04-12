import numpy as np
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Callable

PLATEAU_WINDOW: int
PLATEAU_EPSILON: float

@dataclass
class FlowResult:
    K_final: np.ndarray
    K_history: np.ndarray
    loss_history: np.ndarray
    variance_history: np.ndarray
    converged: bool
    convergence_step: int | None = ...

class GradientFlow:
    divergence_fn: Incomplete
    lr: Incomplete
    grad_eps: Incomplete
    def __init__(self, divergence_fn: Callable[[np.ndarray, np.ndarray], float], lr: float = 0.01, grad_eps: float = 1e-05) -> None: ...
    def step(self, K: np.ndarray, q: np.ndarray) -> np.ndarray: ...
    def run(self, K0: np.ndarray, target: np.ndarray, n_steps: int, early_stopping: bool = True) -> FlowResult: ...

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float: ...
def euclidean_divergence(p: np.ndarray, q: np.ndarray) -> float: ...
def fisher_rao_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float: ...

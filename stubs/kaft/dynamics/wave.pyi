import numpy as np
from _typeshed import Incomplete
from dataclasses import dataclass

CONVERGENCE_WINDOW: int
CONVERGENCE_EPSILON: float

@dataclass
class WaveResult:
    K_final: np.ndarray
    K_history: np.ndarray
    t_history: np.ndarray
    converged: bool
    convergence_step: int | None = ...

class WaveEngine:
    source_fn: Incomplete
    dt: Incomplete
    scheme: Incomplete
    def __init__(self, source_fn: callable, dt: float = 0.01, scheme: str = 'euler') -> None: ...
    def step(self, K: np.ndarray, t: float) -> np.ndarray: ...
    def run(self, K0: np.ndarray, n_steps: int, early_stopping: bool = True) -> WaveResult: ...

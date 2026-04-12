import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

class SoftmaxResult(NamedTuple):
    K: np.ndarray
    attn: np.ndarray
    K_mean: float
    K_std: float
    K_var: float

@dataclass
class SoftmaxDynamics:
    temperature: float = ...
    def run(self, embeddings: np.ndarray) -> SoftmaxResult: ...
    def distribution_divergence(self, K_kaft: np.ndarray, K_softmax: np.ndarray) -> float: ...

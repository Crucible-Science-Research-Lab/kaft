import numpy as np
from _typeshed import Incomplete
from kaft.dynamics.kstate import KDensity as KDensity

class ResonanceField:
    k_density: Incomplete
    v_cog: Incomplete
    def __init__(self, k_density: KDensity, v_cog: float = 1.0) -> None: ...
    def compute(self) -> np.ndarray: ...
    def peak_indices(self, top_k: int = 5) -> np.ndarray: ...

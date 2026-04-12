import numpy as np
from _typeshed import Incomplete
from kaft.dynamics.jordan import JordanNoise as JordanNoise
from kaft.dynamics.kstate import KDensity as KDensity, KState as KState
from kaft.dynamics.metric_evolution import MetricEvolution as MetricEvolution
from kaft.dynamics.resonance import ResonanceField as ResonanceField
from kaft.dynamics.transport import KAFTTerm2 as KAFTTerm2

class KAFTSource:
    k_density: Incomplete
    resonance: Incomplete
    term2: Incomplete
    jordan_noise: Incomplete
    metric_evolver: Incomplete
    c: Incomplete
    kstate: Incomplete
    def __init__(self, k_density: KDensity, resonance: ResonanceField, term2: KAFTTerm2, jordan_noise: JordanNoise, metric_evolver: MetricEvolution, c: float = 1.0) -> None: ...
    def __call__(self, K_field: np.ndarray, t: float) -> np.ndarray: ...
    def after_step(self, K_new: np.ndarray, t_new: float) -> None: ...
    @property
    def phase(self) -> str: ...
    @property
    def soliton_locked(self) -> bool: ...

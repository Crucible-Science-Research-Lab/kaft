import numpy as np
from dataclasses import dataclass, field as field
from kaft.dynamics.jordan import JordanBoundary as JordanBoundary, JordanNoise as JordanNoise
from kaft.dynamics.kstate import KDensity as KDensity, KState as KState
from kaft.dynamics.metric_evolution import MetricEvolution as MetricEvolution
from kaft.dynamics.resonance import ResonanceField as ResonanceField
from kaft.dynamics.softmax_dynamics import SoftmaxDynamics as SoftmaxDynamics
from kaft.dynamics.transport import KAFTTerm2 as KAFTTerm2
from kaft.navigate.seeder import DomainSeeder as DomainSeeder
from kaft.simulate.crucible import SimulationCrucible as SimulationCrucible

@dataclass
class Frame:
    step: int
    n_papers: int
    K_field: np.ndarray
    K_softmax: np.ndarray
    K_mean: float
    K_var: float
    softmax_K_mean: float
    softmax_K_var: float
    divergence_l1: float
    phase: str
    soliton_locked: bool
    delta_K: float
    boltzmann_dist: np.ndarray
    boundary_count: int
    boundary_indices: list

class SimulationRunner:
    log: list[Frame]
    def __init__(self, embeddings: np.ndarray, records: list[dict], k_density: KDensity, k_state: KState, jordan: JordanBoundary, crucible: SimulationCrucible, metric_evolver: MetricEvolution, softmax: SoftmaxDynamics, noise: JordanNoise) -> None: ...
    @classmethod
    def from_crucible(cls, crucible: SimulationCrucible, seeds_root: str | None = None) -> SimulationRunner: ...
    def step(self, new_embedding: np.ndarray | None = None) -> Frame: ...
    def run(self, new_embeddings: list[np.ndarray]) -> list[Frame]: ...
    def run_static(self, n_steps: int) -> list[Frame]: ...
    @property
    def embeddings(self) -> np.ndarray: ...
    @property
    def records(self) -> list[dict]: ...
    @property
    def current_phase(self) -> str: ...
    @property
    def K_history(self) -> list[float]: ...

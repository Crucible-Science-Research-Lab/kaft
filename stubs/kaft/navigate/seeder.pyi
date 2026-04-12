import numpy as np
from kaft.dynamics.jordan import JordanBoundary as JordanBoundary
from kaft.dynamics.kstate import KDensity as KDensity, KState as KState
from kaft.geometry.base import AbstractMetric as AbstractMetric

class DomainSeeder:
    @staticmethod
    def save(path: str, embeddings: np.ndarray, records: list[dict], metric: AbstractMetric, k_density: KDensity, k_state: KState, jordan: JordanBoundary, umap_2d: np.ndarray | None = None) -> None: ...
    @staticmethod
    def load(path: str, metric: AbstractMetric) -> dict: ...

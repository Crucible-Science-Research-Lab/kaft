import numpy as np
from _typeshed import Incomplete
from dataclasses import dataclass
from kaft.dynamics.kstate import KDensity as KDensity
from kaft.geometry.base import AbstractMetric as AbstractMetric

@dataclass
class GeodesicPoint:
    t: float
    embedding: np.ndarray
    k_density: float
    nearest_idx: int
    nearest_text: str
    metric_distance_to_nearest: float

@dataclass
class GeodesicPath:
    source_idx: int
    target_idx: int
    source_text: str
    target_text: str
    waypoints: list[GeodesicPoint]
    total_length: float
    k_profile: list[float]
    boundary_crossings: list[tuple[float, str]]

class GeodesicNavigator:
    embeddings: Incomplete
    records: Incomplete
    metric: Incomplete
    n_steps: Incomplete
    def __init__(self, embeddings: np.ndarray, records: list[dict], metric: AbstractMetric, n_steps: int = 20) -> None: ...
    def trace(self, source_idx: int, target_idx: int) -> GeodesicPath: ...

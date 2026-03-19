"""GeodesicPath — shortest path through Fisher-Rao knowledge space."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from geomstats.geometry.hypersphere import Hypersphere

from kaft.core.manifold import Manifold
from kaft.core.metric import FisherRaoMetric


@dataclass
class GeodesicPoint:
    t: float
    embedding: np.ndarray
    k_density: float
    nearest_idx: int
    nearest_text: str
    fr_distance_to_nearest: float


@dataclass
class GeodesicPath:
    source_idx: int
    target_idx: int
    source_text: str
    target_text: str
    waypoints: List[GeodesicPoint]
    total_fr_length: float
    k_profile: List[float]
    boundary_crossings: List[Tuple[float, str]]


class GeodesicNavigator:
    def __init__(self, state: Manifold, metric: FisherRaoMetric, n_steps: int = 20):
        self.state = state
        self.metric = metric
        self.n_steps = n_steps
        self.sphere = Hypersphere(dim=state.embeddings.shape[1] - 1)

    def _arccos_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))

    def _k_density_at(self, point: np.ndarray, epsilon: float = 0.05) -> float:
        """Inverse square law K-density. Skip corpus points closer than epsilon
        to avoid singularity when waypoint coincides with a corpus embedding."""
        densities = []
        for emb in self.state.embeddings:
            d = self._arccos_dist(point, emb)
            if d < epsilon:
                continue  # skip near-coincident points — not self-interaction
            densities.append(1.0 / (d ** 2))
        return float(np.mean(densities)) if densities else 0.0

    def _nearest(self, point: np.ndarray) -> Tuple[int, str, float]:
        distances = [self._arccos_dist(point, emb) for emb in self.state.embeddings]
        idx = int(np.argmin(distances))
        text = self.state.records[idx]["text"]
        return idx, text, distances[idx]

    def _detect_boundary_crossing(
        self, k_profile: List[float], t_values: List[float]
    ) -> List[Tuple[float, str]]:
        crossings = []
        gradient = np.gradient(k_profile)
        for i in range(1, len(gradient)):
            if gradient[i - 1] > 0 and gradient[i] < 0:
                crossings.append((t_values[i], "local_maximum — potential Jordan ridge"))
            elif gradient[i - 1] < 0 and gradient[i] > 0:
                crossings.append((t_values[i], "local_minimum — inter-domain valley"))
        return crossings

    def trace(self, source_idx: int, target_idx: int) -> GeodesicPath:
        source_emb = self.state.embeddings[source_idx]
        target_emb = self.state.embeddings[target_idx]
        source_text = self.state.records[source_idx]["text"]
        target_text = self.state.records[target_idx]["text"]
        total_length = float(self.metric.geodesic_distance(source_idx, target_idx))

        log_vec = self.sphere.metric.log(
            np.array([target_emb]), np.array([source_emb])
        )[0]

        t_values = np.linspace(0.0, 1.0, self.n_steps)
        waypoints = []
        k_profile = []

        for t in t_values:
            point = self.sphere.metric.exp(
                np.array([t * log_vec]), np.array([source_emb])
            )[0]
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm

            k = self._k_density_at(point)
            nearest_idx, nearest_text, fr_dist = self._nearest(point)

            waypoints.append(GeodesicPoint(
                t=float(t),
                embedding=point,
                k_density=k,
                nearest_idx=nearest_idx,
                nearest_text=nearest_text,
                fr_distance_to_nearest=fr_dist
            ))
            k_profile.append(k)

        boundary_crossings = self._detect_boundary_crossing(k_profile, list(t_values))

        return GeodesicPath(
            source_idx=source_idx,
            target_idx=target_idx,
            source_text=source_text,
            target_text=target_text,
            waypoints=waypoints,
            total_fr_length=total_length,
            k_profile=k_profile,
            boundary_crossings=boundary_crossings
        )

# kaft/navigate/geodesic.py
"""GeodesicNavigator — shortest path through knowledge space via any Layer 0 metric."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from geomstats.geometry.hypersphere import Hypersphere

from kaft.geometry.base import AbstractMetric
from kaft.dynamics.kstate import KDensity


@dataclass
class GeodesicPoint:
    t:                          float
    embedding:                  np.ndarray
    k_density:                  float
    nearest_idx:                int
    nearest_text:               str
    metric_distance_to_nearest: float


@dataclass
class GeodesicPath:
    source_idx:         int
    target_idx:         int
    source_text:        str
    target_text:        str
    waypoints:          List[GeodesicPoint]
    total_length:       float
    k_profile:          List[float]
    boundary_crossings: List[Tuple[float, str]]


class GeodesicNavigator:
    def __init__(
        self,
        embeddings:  np.ndarray,    # (N, D) raw embeddings
        records:     list[dict],    # RouterRecord list — parallel to embeddings
        metric:      AbstractMetric,
        n_steps:     int = 20,
    ):
        self.embeddings = embeddings
        self.records    = records
        self.metric     = metric
        self.n_steps    = n_steps

        # Build KDensity once — warms the (N, N) distance cache
        self._kd = KDensity(embeddings, metric)
        self._kd.compute()

        self._corpus_norms = np.linalg.norm(embeddings, axis=1)   # (N,) semantic mass
        self._sphere       = Hypersphere(dim=embeddings.shape[1] - 1)

    def _k_density_at(self, point: np.ndarray, epsilon: float = 1e-6) -> float:
        """
        Interaction hypothesis at any interpolated waypoint:
        K = Σ_j (I_point · I_j) / d²(point, corpus_j)
        Uses metric.compute() — geometrically honest, metric-agnostic.
        """
        I_point = float(np.linalg.norm(point))
        total   = 0.0
        for j, emb_j in enumerate(self.embeddings):
            d = self.metric.compute(point, emb_j)
            if d < epsilon:
                continue
            total += (I_point * self._corpus_norms[j]) / (d ** 2)
        return total / len(self.embeddings)

    def _nearest(self, point: np.ndarray) -> Tuple[int, str, float]:
        distances = [self.metric.compute(point, emb) for emb in self.embeddings]
        idx       = int(np.argmin(distances))
        text      = self.records[idx].get("text", self.records[idx].get("title", ""))
        return idx, text, distances[idx]

    def _detect_boundary_crossings(
        self, k_profile: List[float], t_values: List[float]
    ) -> List[Tuple[float, str]]:
        crossings = []
        gradient  = np.gradient(k_profile)
        for i in range(1, len(gradient)):
            if gradient[i - 1] > 0 and gradient[i] < 0:
                crossings.append((t_values[i], "local_maximum — Jordan ridge"))
            elif gradient[i - 1] < 0 and gradient[i] > 0:
                crossings.append((t_values[i], "local_minimum — inter-domain valley"))
        return crossings

    def trace(self, source_idx: int, target_idx: int) -> GeodesicPath:
        src_emb  = self.embeddings[source_idx]
        tgt_emb  = self.embeddings[target_idx]
        src_text = self.records[source_idx].get("text", self.records[source_idx].get("title", ""))
        tgt_text = self.records[target_idx].get("text", self.records[target_idx].get("title", ""))

        total_length = self.metric.compute(src_emb, tgt_emb)

        # Intrinsic SLERP — log map at source, exp map along tangent
        log_vec  = self._sphere.metric.log(
            np.array([tgt_emb]), np.array([src_emb])
        )[0]

        t_values  = np.linspace(0.0, 1.0, self.n_steps)
        waypoints = []
        k_profile = []

        for t in t_values:
            point = self._sphere.metric.exp(
                np.array([t * log_vec]), np.array([src_emb])
            )[0]
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm

            k               = self._k_density_at(point)
            idx, text, dist = self._nearest(point)

            waypoints.append(GeodesicPoint(
                t=float(t),
                embedding=point,
                k_density=k,
                nearest_idx=idx,
                nearest_text=text,
                metric_distance_to_nearest=dist,
            ))
            k_profile.append(k)

        return GeodesicPath(
            source_idx=source_idx,
            target_idx=target_idx,
            source_text=src_text,
            target_text=tgt_text,
            waypoints=waypoints,
            total_length=total_length,
            k_profile=k_profile,
            boundary_crossings=self._detect_boundary_crossings(k_profile, list(t_values)),
        )



# """GeodesicPath — shortest path through Fisher-Rao knowledge space."""
# from __future__ import annotations
# import numpy as np
# from dataclasses import dataclass
# from typing import List, Tuple
# from geomstats.geometry.hypersphere import Hypersphere

# from kaft.core.manifold import Manifold
# from kaft.core.metric import FisherRaoMetric


# @dataclass
# class GeodesicPoint:
#     t: float
#     embedding: np.ndarray
#     k_density: float
#     nearest_idx: int
#     nearest_text: str
#     fr_distance_to_nearest: float


# @dataclass
# class GeodesicPath:
#     source_idx: int
#     target_idx: int
#     source_text: str
#     target_text: str
#     waypoints: List[GeodesicPoint]
#     total_fr_length: float
#     k_profile: List[float]
#     boundary_crossings: List[Tuple[float, str]]


# class GeodesicNavigator:
#     def __init__(self, state: Manifold, metric: FisherRaoMetric, n_steps: int = 20):
#         self.state = state
#         self.metric = metric
#         self.n_steps = n_steps
#         self.sphere = Hypersphere(dim=state.embeddings.shape[1] - 1)

#     def _arccos_dist(self, a: np.ndarray, b: np.ndarray) -> float:
#         return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))

#     def _k_density_at(self, point: np.ndarray, epsilon: float = 0.05) -> float:
#         """Inverse square law K-density. Skip corpus points closer than epsilon
#         to avoid singularity when waypoint coincides with a corpus embedding."""
#         densities = []
#         for emb in self.state.embeddings:
#             d = self._arccos_dist(point, emb)
#             if d < epsilon:
#                 continue  # skip near-coincident points — not self-interaction
#             densities.append(1.0 / (d ** 2))
#         return float(np.mean(densities)) if densities else 0.0

#     def _nearest(self, point: np.ndarray) -> Tuple[int, str, float]:
#         distances = [self._arccos_dist(point, emb) for emb in self.state.embeddings]
#         idx = int(np.argmin(distances))
#         text = self.state.records[idx]["text"]
#         return idx, text, distances[idx]

#     def _detect_boundary_crossing(
#         self, k_profile: List[float], t_values: List[float]
#     ) -> List[Tuple[float, str]]:
#         crossings = []
#         gradient = np.gradient(k_profile)
#         for i in range(1, len(gradient)):
#             if gradient[i - 1] > 0 and gradient[i] < 0:
#                 crossings.append((t_values[i], "local_maximum — potential Jordan ridge"))
#             elif gradient[i - 1] < 0 and gradient[i] > 0:
#                 crossings.append((t_values[i], "local_minimum — inter-domain valley"))
#         return crossings

#     def trace(self, source_idx: int, target_idx: int) -> GeodesicPath:
#         source_emb = self.state.embeddings[source_idx]
#         target_emb = self.state.embeddings[target_idx]
#         source_text = self.state.records[source_idx]["text"]
#         target_text = self.state.records[target_idx]["text"]
#         total_length = float(self.metric.geodesic_distance(source_idx, target_idx))

#         log_vec = self.sphere.metric.log(
#             np.array([target_emb]), np.array([source_emb])
#         )[0]

#         t_values = np.linspace(0.0, 1.0, self.n_steps)
#         waypoints = []
#         k_profile = []

#         for t in t_values:
#             point = self.sphere.metric.exp(
#                 np.array([t * log_vec]), np.array([source_emb])
#             )[0]
#             norm = np.linalg.norm(point)
#             if norm > 0:
#                 point = point / norm

#             k = self._k_density_at(point)
#             nearest_idx, nearest_text, fr_dist = self._nearest(point)

#             waypoints.append(GeodesicPoint(
#                 t=float(t),
#                 embedding=point,
#                 k_density=k,
#                 nearest_idx=nearest_idx,
#                 nearest_text=nearest_text,
#                 fr_distance_to_nearest=fr_dist
#             ))
#             k_profile.append(k)

#         boundary_crossings = self._detect_boundary_crossing(k_profile, list(t_values))

#         return GeodesicPath(
#             source_idx=source_idx,
#             target_idx=target_idx,
#             source_text=source_text,
#             target_text=target_text,
#             waypoints=waypoints,
#             total_fr_length=total_length,
#             k_profile=k_profile,
#             boundary_crossings=boundary_crossings
#         )

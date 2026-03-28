"""
kaft.geometry.divergences

Layer 0: Abstract metric interface and divergence registry.
Any distance/divergence measure can be registered here —
Fisher-Rao, KL, Hellinger, Euclidean, Minkowski, or custom.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type
import numpy as np


class AbstractMetric(ABC):
    """Base class for all distance/divergence measures."""

    @abstractmethod
    def distances(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.

        Parameters
        ----------
        vectors : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n, n)
        """
        ...

    @abstractmethod
    def geometry_type(self) -> str:
        """Human-readable name: 'euclidean', 'fisher_rao', etc."""
        ...


class EuclideanMetric(AbstractMetric):
    """Flat Euclidean distance — the baseline geometry."""

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        diff = vectors[:, None, :] - vectors[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def geometry_type(self) -> str:
        return "euclidean"


class GaussianCurvedMetric(AbstractMetric):
    """
    A manifold with a Gaussian curvature bump at the centre.
    Wave speed is locally reduced where curvature is highest —
    causing wavefronts to bend, slow, and distort visibly.
    """

    def __init__(self, curvature_strength: float = 0.85, width: float = 0.25):
        self.curvature_strength = curvature_strength
        self.width = width

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        # Flat Euclidean baseline — curvature lives in the wave speed field
        diff = vectors[:, None, :] - vectors[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def speed_field(self, grid_size: int) -> np.ndarray:
        """
        Returns a 2D array of local wave speeds.
        1.0 = flat, <1.0 = curved (slower propagation).
        """
        cx, cy = grid_size // 2, grid_size // 2
        x = np.arange(grid_size)
        y = np.arange(grid_size)
        xx, yy = np.meshgrid(x, y)
        r2 = ((xx - cx) / (self.width * grid_size))**2 + \
             ((yy - cy) / (self.width * grid_size))**2
        bump = np.exp(-r2)
        return 1.0 - self.curvature_strength * bump

    def geometry_type(self) -> str:
        return "curved"


class MinkowskiMetric(AbstractMetric):
    """
    Spacetime metric with a light cone speed limit.
    Wavefronts cannot propagate beyond c * t from the source.
    """

    def __init__(self, c: float = 0.6):
        self.c = c   # speed of light in grid units

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        diff = vectors[:, None, :] - vectors[None, :, :]
        spatial = np.sqrt((diff ** 2).sum(axis=-1))
        return spatial

    def speed_field(self, grid_size: int) -> np.ndarray:
        """Uniform speed — the cone comes from the wave equation structure."""
        return np.full((grid_size, grid_size), self.c)

    def geometry_type(self) -> str:
        return "minkowski"


class FisherRaoMetric(AbstractMetric):
    """
    Fisher-Rao information metric on the statistical manifold.

    Treats each vector as an unnormalised probability distribution.
    Distance is the geodesic angle on the unit sphere of sqrt-distributions:
        d(p, q) = 2 * arccos( Σ sqrt(p_i * q_i) )

    The natural metric for ML embeddings, statistical mechanics,
    and any domain where vectors represent distributions over outcomes.
    """

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        shifted = vectors - vectors.min(axis=1, keepdims=True) + 1e-10
        probs = shifted / shifted.sum(axis=1, keepdims=True)
        sq = np.sqrt(probs)
        dot = np.clip(sq @ sq.T, -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def speed_field(self, grid_size: int) -> np.ndarray:
        """Information density slows waves near the centre."""
        cx, cy = grid_size // 2, grid_size // 2
        x, y = np.arange(grid_size), np.arange(grid_size)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2) / (grid_size / 2)
        return 0.3 + 0.5 * r

    def geometry_type(self) -> str:
        return "fisher_rao"


class DivergenceRegistry:
    """
    Registry of named metric/divergence implementations.

    Usage
    -----
    registry = DivergenceRegistry()
    registry.register("euclidean", EuclideanMetric)
    metric = registry.get("euclidean")
    distances = metric.distances(vectors)
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[AbstractMetric]] = {}
        self.register("euclidean", EuclideanMetric)
        self.register("gaussian_curved", GaussianCurvedMetric)
        self.register("minkowski", MinkowskiMetric)
        self.register("fisher_rao", FisherRaoMetric)

    def register(self, name: str, metric_cls: Type[AbstractMetric]) -> None:
        """Register a metric class under a string key."""
        if not issubclass(metric_cls, AbstractMetric):
            raise TypeError(f"{metric_cls} must subclass AbstractMetric")
        self._registry[name] = metric_cls

    def get(self, name: str) -> AbstractMetric:
        """Instantiate and return a registered metric by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Metric '{name}' not found. Available: {available}")
        return self._registry[name]()

    def available(self) -> list:
        """List all registered metric names."""
        return list(self._registry.keys())
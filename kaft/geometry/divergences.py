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
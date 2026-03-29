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
    def speed_field(self, grid_size: int) -> np.ndarray:
        """Default: uniform flat field — Euclidean-like geometry."""
        return np.ones((grid_size, grid_size)) * 0.7




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
        from kaft.geometry.classical import EuclideanMetric, GaussianCurvedMetric, MinkowskiMetric
        from kaft.geometry.information import KLDivergence, JensenShannonMetric, AlphaDivergence, FisherRaoMetric
        self._registry: Dict[str, Type[AbstractMetric]] = {}
        self.register("euclidean", EuclideanMetric)
        self.register("gaussian_curved", GaussianCurvedMetric)
        self.register("minkowski", MinkowskiMetric)
        self.register("fisher_rao", FisherRaoMetric)
        self.register("kl_divergence", KLDivergence)
        self.register("jensen_shannon", JensenShannonMetric)
        self._registry["alpha_hellinger"]    = AlphaDivergence(alpha=0.0)
        self._registry["alpha_bhattacharyya"] = AlphaDivergence(alpha=0.5)
        self._registry["alpha_reverse"]      = AlphaDivergence(alpha=-0.5) # reverse-KL direction


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
        obj = self._registry[name]
        return obj() if isinstance(obj, type) else obj

    def available(self) -> list:
        """List all registered metric names."""
        return list(self._registry.keys())
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type
import numpy as np


class AbstractMetric(ABC):
    """
    Base class for all geometric structures in kaft.
    
    The universal contract is compute(a, b) → float.
    distances() is derived from compute() by default — classical metrics
    override it with a fast vectorized implementation.
    """

    @abstractmethod
    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Scalar distance or divergence between two points.

        Parameters
        ----------
        a, b : np.ndarray
            Two points in the appropriate domain for this metric.
            Classical: vectors in ℝⁿ
            Information: probability distributions on the simplex

        Returns
        -------
        float
        """
        ...

    @abstractmethod
    def geometry_type(self) -> str:
        """Human-readable label: 'euclidean', 'fisher_rao', 'kl_divergence', etc."""
        ...

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        """
        Pairwise distance/divergence matrix via compute().

        Default implementation loops over pairs — O(n²) but universally correct.
        Classical metrics override this with a fast vectorized version.

        Parameters
        ----------
        vectors : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n, n)
        """
        n = len(vectors)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.compute(vectors[i], vectors[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def speed_field(self, grid_size: int) -> np.ndarray:
        """Default: uniform flat field — Euclidean-like geometry."""
        return np.ones((grid_size, grid_size)) * 0.7


class DivergenceRegistry:
    """Registry of named metric/divergence instances."""

    def __init__(self):
        self._registry: Dict[str, AbstractMetric] = {}

    def register(self, name: str, metric: AbstractMetric) -> None:
        """Register a metric instance under a string key."""
        if not isinstance(metric, AbstractMetric):
            raise TypeError(f"{metric} must be an instance of AbstractMetric")
        self._registry[name] = metric

    def get(self, name: str) -> AbstractMetric:
        """Return a registered metric instance by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Metric '{name}' not found. Available: {available}")
        return self._registry[name]

    def available(self) -> list:
        """List all registered metric names."""
        return list(self._registry.keys())









# from __future__ import annotations
# from abc import ABC, abstractmethod
# from typing import Dict, Type
# import numpy as np


# class AbstractMetric(ABC):
#     """Base class for all distance/divergence measures."""

#     @abstractmethod
#     def distances(self, vectors: np.ndarray) -> np.ndarray:
#         """
#         Compute pairwise distance matrix.

#         Parameters
#         ----------
#         vectors : np.ndarray of shape (n, d)

#         Returns
#         -------
#         np.ndarray of shape (n, n)
#         """
#         ...
    
#     @abstractmethod
#     def geometry_type(self) -> str:
#         """Human-readable name: 'euclidean', 'fisher_rao', etc."""
#         ...
#     def speed_field(self, grid_size: int) -> np.ndarray:
#         """Default: uniform flat field — Euclidean-like geometry."""
#         return np.ones((grid_size, grid_size)) * 0.7




# class DivergenceRegistry:
#     """
#     Registry of named metric/divergence implementations.

#     Usage
#     -----
#     registry = DivergenceRegistry()
#     registry.register("euclidean", EuclideanMetric)
#     metric = registry.get("euclidean")
#     distances = metric.distances(vectors)
#     """

#     def __init__(self):
#         self._registry = {} 

#     def register(self, name: str, metric: Type[AbstractMetric]) -> None:
#         """Register a metric class under a string key."""
#         if not isinstance(metric, AbstractMetric):
#             raise TypeError(f"{metric} must be an instance of AbstractMetric")
#         self._registry[name] = metric

#     def get(self, name: str) -> AbstractMetric:
#         """Instantiate and return a registered metric by name."""
#         if name not in self._registry:
#             available = list(self._registry.keys())
#             raise KeyError(f"Metric '{name}' not found. Available: {available}")
#         obj = self._registry[name]
#         return obj() if isinstance(obj, type) else obj

#     def available(self) -> list:
#         """List all registered metric names."""
#         return list(self._registry.keys())
    


    
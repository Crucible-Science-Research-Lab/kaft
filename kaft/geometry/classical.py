import numpy as np
from kaft.geometry.base import AbstractMetric



class EuclideanMetric(AbstractMetric):
    """Flat Euclidean distance — the baseline geometry."""

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))
    
    def distances(self, vectors: np.ndarray) -> np.ndarray:
        diff = vectors[:, None, :] - vectors[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def geometry_type(self) -> str:
        return "euclidean"

class MinkowskiMetric(AbstractMetric):
    """
    Spacetime metric with a light cone speed limit.
    Wavefronts cannot propagate beyond c * t from the source.
    """

    def __init__(self, c: float = 0.6):
        self.c = c

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        diff = vectors[:, None, :] - vectors[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def speed_field(self, grid_size: int) -> np.ndarray:
        return np.full((grid_size, grid_size), self.c)

    def geometry_type(self) -> str:
        return "minkowski"

class GaussianCurvedMetric(AbstractMetric):
    """
    A manifold with a Gaussian curvature bump at the centre.
    Curvature lives in speed_field() — distances() stays flat.
    """

    def __init__(self, curvature_strength: float = 0.85, width: float = 0.25):
        self.curvature_strength = curvature_strength
        self.width = width

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        diff = vectors[:, None, :] - vectors[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def speed_field(self, grid_size: int) -> np.ndarray:
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


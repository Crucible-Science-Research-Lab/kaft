import numpy as np
from kaft.geometry.base import AbstractMetric



class EuclideanMetric(AbstractMetric):
    """Flat Euclidean distance — the baseline geometry."""

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


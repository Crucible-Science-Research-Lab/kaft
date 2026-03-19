"""
AbstractSimulator — the interface every simulation model implements.

Adding a new mathematical system (Maxwell, Schrödinger, Navier-Stokes)
= write one class that inherits from AbstractSimulator.
kaft.simulate() stays clean. The library never changes.
"""

from abc import ABC, abstractmethod
import numpy as np


class AbstractSimulator(ABC):

    @abstractmethod
    def run(self, manifold: dict, dt: float = 1.0) -> dict:
        """
        Evolve the manifold state by one timestep dt.

        Parameters
        ----------
        manifold : dict
            Current geometric state (embeddings, k_density, metric).
        dt : float
            Timestep for evolution.

        Returns
        -------
        dict
            Updated geometric state after evolution.
        """
        ...

    @abstractmethod
    def geometry_type(self) -> str:
        """
        Returns 'curved' or 'flat'.
        Used by compare() to label topology divergence.
        """
        ...

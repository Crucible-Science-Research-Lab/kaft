from __future__ import annotations
import numpy as np
from kaft.core.topology import KDensity


class ResonanceField:
    """
    R ≡ K² · v²_cog

    Attention density — regions where knowledge self-amplifies most strongly.
    NOT designed. Falls directly out of K² recursion.

    High R = powerful attractor. Knowledge flows toward these regions.
    Low R  = exploratory space. Jordan noise dominates here.

    This is the first term in the master evolution equation.
    """

    def __init__(self, k_density: KDensity, v_cog: float = 1.0):
        self.k_density = k_density
        self.v_cog = v_cog

    def compute(self) -> np.ndarray:
        """
        Returns R = K² · v²_cog for every point.
        Shape (N,) — same shape as K-density.
        """
        K = self.k_density.density          # (N,)
        return (K ** 2) * (self.v_cog ** 2) # emerges — not imposed

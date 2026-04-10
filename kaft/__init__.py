"""
kaft — Information geometry engine.

Dense knowledge curves information space;
curved space tells knowledge where to navigate.
"""

from kaft.dynamics.kstate import KDensity
from kaft.geometry import AbstractMetric, DivergenceRegistry
from kaft.simulate.base import AbstractManifoldDynamics


__version__ = "0.3.0"
__author__  = "Crucible Science"

__all__ = [
    # Core geometry
    "KDensity",
    # Simulation
    "KAFTEvolution",
    "AbstractManifoldDynamics",
    "AbstractMetric", "DivergenceRegistry", 
    "FisherRaoMetric", 
    "KLDivergence" , "JensenShannonMetric", "AlphaDivergence"
]

"""
kaft — Information geometry engine.

Dense knowledge curves information space;
curved space tells knowledge where to navigate.
"""

from kaft.core.manifold import build_manifold, Manifold
from kaft.core.metric import FisherRaoMetric
from kaft.core.topology import KDensity, JordanBoundary
from kaft.core.resonance import ResonanceField
from kaft.simulate.compare import compare
from kaft.simulate.kaft_sim import KAFTEvolution
from kaft.simulate.softmax_sim import EuclideanBaseline
from kaft.simulate.base import AbstractManifoldDynamics
from kaft.ingest.router import ArxivRouter
from kaft.geometry.divergences import AbstractMetric, DivergenceRegistry, FisherRaoMetric


__version__ = "0.2.0"
__author__  = "Crucible Science"

__all__ = [
    # Core geometry
    "build_manifold", "Manifold",
    "FisherRaoMetric",
    "KDensity", "JordanBoundary",
    "ResonanceField",
    # Simulation
    "KAFTEvolution", "EuclideanBaseline",
    "AbstractManifoldDynamics",
    "compare",
    "AbstractMetric", "DivergenceRegistry", "FisherRaoMetric"
]

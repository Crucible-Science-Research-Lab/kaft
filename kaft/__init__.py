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
from kaft.simulate.kaft import KAFTSimulator
from kaft.simulate.softmax import SoftmaxSimulator
from kaft.simulate.base import AbstractSimulator
from kaft.ingest.router import ArxivRouter

__version__ = "0.1.0"
__author__  = "Crucible Science"

__all__ = [
    # Core geometry
    "build_manifold", "Manifold",
    "FisherRaoMetric",
    "KDensity", "JordanBoundary",
    "ResonanceField",
    # Simulation
    "KAFTSimulator", "SoftmaxSimulator",
    "AbstractSimulator",
    "compare",
]

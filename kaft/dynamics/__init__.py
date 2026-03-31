"""
kaft.dynamics — Layer 1: Knowledge dynamics engine

The KAFT master evolution equation:
    ∂K/∂t = c·v²_cog·K²  +  P(V)·∇^FR(K)  +  ξ_J(t)
              Term 1           Term 2           Term 3
              resonance        transport        jordan noise

And the derived wave equation:
    ∂²K/∂t² - v²_cog·Δ_g K = 2c·v²_cog·K·∂K/∂t

Layer 0 (kaft.geometry) provides the metric.
Layer 1 (kaft.dynamics) runs the physics ON that metric.

Public API:
    KDensity        — K-density field (interaction hypothesis)
    KState          — scalar K, 5F phase, soliton detection
    ResonanceField  — Term 1: R = K²·v²_cog
    JordanBoundary  — emergent semantic boundaries
    JordanNoise     — Term 3: ξ_J(t) stochastic exploration
    WaveEngine      — full ∂²K/∂t² integrator (wave.py)
    GradientFlow    — ∂K/∂t = -∇D for divergence families (gradient_flow.py)
    ParallelTransport — Term 2: P(V)·∇^FR(K) (transport.py)
"""
from kaft.dynamics.kstate    import KDensity, KState
from kaft.dynamics.resonance import ResonanceField
from kaft.dynamics.jordan    import JordanBoundary, JordanNoise
from kaft.dynamics.transport import ParallelTransport
from kaft.dynamics.wave      import WaveEngine
from kaft.dynamics.metric_evolution import MetricEvolution
from kaft.dynamics.gradient_flow import GradientFlow, FlowResult
from kaft.dynamics.gradient_flow import kl_divergence, euclidean_divergence, fisher_rao_divergence



__all__ = [
    "KDensity",
    "KState",
    "ResonanceField",
    "JordanBoundary",
    "JordanNoise",
    "ParallelTransport",
    "WaveEngine",
    "MetricEvolution",
    "GradientFlow", "FlowResult",
    "kl_divergence","euclidean_divergence","fisher_rao_divergence"
]

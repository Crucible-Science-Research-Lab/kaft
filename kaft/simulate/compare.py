"""
Runs two simulators on the SAME manifold.
Returns their evolved topologies and the divergence between them.

compare(KAFTEvolution(), EuclideanBaseline(), manifold=state)

"""
from __future__ import annotations
import numpy as np
from kaft.simulate.base import AbstractManifoldDynamics
from kaft.core.manifold import Manifold
from kaft.core.metric import FisherRaoMetric
from kaft.core.topology import KDensity
#from kaft.simulate.kaft_sim import KAFTEvolution


def compare(
    model_a: AbstractManifoldDynamics,
    model_b: AbstractManifoldDynamics,
    manifold: Manifold,
    steps: int = 20,
    dt: float = 0.1,
) -> dict:
    """
    Runs two simulators on the SAME manifold.
    Returns their evolved topologies and the divergence between them.
    """
    history_a, history_b = [], []
    divergence_curve = []

    result_a = model_a.run(manifold, dt)
    result_b = model_b.run(manifold, dt)

    for step in range(steps):
        # Rebuild manifold state with evolved K (same embeddings, updated density)
        manifold_a = _inject_k(manifold, result_a["k_density"])
        manifold_b = _inject_k(manifold, result_b["k_density"])

        result_a = model_a.run(manifold_a, dt)
        result_b = model_b.run(manifold_b, dt)

        history_a.append(result_a["k_density"].copy())
        history_b.append(result_b["k_density"].copy())

        # Divergence: how different are the two K-density fields?
        div = _kl_divergence(result_a["k_density"], result_b["k_density"])
        divergence_curve.append(div)

    final_a = history_a[-1]
    final_b = history_b[-1]

    return {
        "topology_a"      : final_a,
        "topology_b"      : final_b,
        "geometry_type_a" : model_a.geometry_type(),
        "geometry_type_b" : model_b.geometry_type(),
        "divergence"      : float(divergence_curve[-1]),
        "divergence_curve": divergence_curve,
        "history_a"       : history_a,
        "history_b"       : history_b,
    }


def _inject_k(manifold: Manifold, k_new: np.ndarray) -> Manifold:
    """Return a new Manifold with updated k_density — embeddings unchanged."""
    from kaft.core.manifold import Manifold as M
    new = M(
        embeddings=manifold.embeddings,
        records=manifold.records,
        domain_type=manifold.domain_type,
        k_density=k_new,
    )
    return new


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL divergence D(p||q) — measures topology divergence between two models."""
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

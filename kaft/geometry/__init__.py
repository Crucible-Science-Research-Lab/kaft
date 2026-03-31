# kaft/geometry/__init__.py

from .base import AbstractMetric, DivergenceRegistry

_registry = DivergenceRegistry()


def get(name: str) -> AbstractMetric:
    return _registry.get(name)


def list_metrics() -> list:
    return _registry.available()

def register(name: str, metric: AbstractMetric):
    """Public API — lets users register their own custom metrics."""
    _registry.register(name, metric)


def _bootstrap():
    """Called once at import time. Single source of truth for all built-in metrics."""
    import numpy as np
    from .information import (
        KLDivergence, FisherRaoMetric, JensenShannonMetric, 
        AlphaDivergence, BregmanDivergence,
        WassersteinMetric, bregman_euclidean, bregman_itakura_saito, bregman_kl, bregman_tsallis
        
    )
    from .classical import (
        EuclideanMetric, GaussianCurvedMetric, MinkowskiMetric
    )
    # Add your spatial metrics imports too
    # from .spatial import EuclideanMetric, MinkowskiMetric, GaussianCurvedMetric

    _registry.register("euclidean",              EuclideanMetric())
    _registry.register("gaussian_curved",        GaussianCurvedMetric())
    _registry.register("minkowski",              MinkowskiMetric())
    _registry.register("fisher_rao",             FisherRaoMetric())
    _registry.register("kl_divergence",          KLDivergence())
    _registry.register("jensen_shannon",         JensenShannonMetric())
    _registry.register("alpha_hellinger",        AlphaDivergence(alpha=0.0))
    _registry.register("alpha_bhattacharyya",    AlphaDivergence(alpha=0.5))
    _registry.register("alpha_reverse",          AlphaDivergence(alpha=2.0))
    _registry.register("bregman_kl",             BregmanDivergence(
        F      = lambda p: np.sum(p * np.log(p + 1e-12)),
        grad_F = lambda p: np.log(p + 1e-12) + 1.0,
        hess_F = lambda p: 1.0 / (p + 1e-12),
        name   = "kl"
    ))
    _registry.register("bregman_euclidean",      bregman_euclidean())
    _registry.register("bregman_itakura_saito",  bregman_itakura_saito())
    _registry.register("bregman_tsallis_2",      bregman_tsallis(q=2.0))
    _registry.register("bregman_tsallis_0.5",    bregman_tsallis(q=0.5))
    _registry.register("wasserstein_1",          WassersteinMetric(p=1))
    _registry.register("wasserstein_2",          WassersteinMetric(p=2))


_bootstrap()  # runs once when kaft.geometry is imported
"""
kaft geometry test suite.

Note: test_jordan_boundaries_emerge_not_imposed() is not just a unit test.
It is a philosophical assertion baked into the CI pipeline.
"""
import numpy as np
import pytest

from kaft.core.manifold import build_manifold, Manifold
from kaft.core.metric import FisherRaoMetric
from kaft.core.topology import KDensity, JordanBoundary


# Minimal corpus — two clearly separated semantic clusters
MATH_RECORDS = [
    {"text": "Riemannian geometry curvature tensor manifold"},
    {"text": "Fisher information metric statistical manifold"},
    {"text": "geodesic distance information geometry hypersphere"},
]
BIO_RECORDS = [
    {"text": "CRISPR Cas9 gene editing DNA repair mechanism"},
    {"text": "mRNA vaccine immune response spike protein"},
    {"text": "drug target binding affinity molecular dynamics"},
]
TWO_CLUSTER_CORPUS = MATH_RECORDS + BIO_RECORDS


def test_build_manifold_end_to_end():
    """build_manifold(records) → Manifold with correct shape embeddings."""
    state = build_manifold(TWO_CLUSTER_CORPUS)

    assert isinstance(state, Manifold)
    assert state.embeddings.shape == (6, 384)
    assert len(state.records) == 6
    assert state.records[0]["text"] == TWO_CLUSTER_CORPUS[0]["text"]


def test_fisher_rao_returns_correct_shape():
    """Fisher-Rao metric tensor must be (N, N) for N corpus points."""
    state = build_manifold(TWO_CLUSTER_CORPUS)
    metric = FisherRaoMetric(state)
    D = metric.compute()

    assert D.shape == (6, 6)
    # Diagonal must be zero — zero distance from point to itself
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-3)
    # Symmetric
    np.testing.assert_allclose(D, D.T, atol=1e-6)
    # All distances in [0, π]
    assert D.min() >= 0.0
    assert D.max() <= np.pi + 1e-6


def test_k_density_normalised_0_to_1():
    """K-density values must live in [0, 1] — normalised interaction field."""
    state = build_manifold(TWO_CLUSTER_CORPUS)
    metric = FisherRaoMetric(state)
    metric.compute()
    kdensity = KDensity(state, metric)
    K = kdensity.compute()

    assert K.shape == (6,)
    assert K.min() >= 0.0 - 1e-6
    assert K.max() <= 1.0 + 1e-6


def test_jordan_boundaries_emerge_not_imposed():
    """
    Jordan boundaries must be detected as level sets of K-density gradients.
    They are NOT configurable parameters. They crystallise from the geometry.
    Two clearly separated semantic clusters must produce at least one boundary.
    """
    state = build_manifold(TWO_CLUSTER_CORPUS)
    metric = FisherRaoMetric(state)
    metric.compute()
    kdensity = KDensity(state, metric)
    kdensity.compute()

    boundaries = JordanBoundary(state, kdensity).detect()

    # Geometry must find at least one boundary between math and bio clusters
    assert len(boundaries) >= 1

    # Each boundary must have the required fields
    for b in boundaries:
        assert "boundary_point" in b
        assert "separates" in b
        assert "energy_barrier" in b
        assert "k_density_at_boundary" in b
        # Boundary points must be low K-density — inter-domain valleys
        assert b["k_density_at_boundary"] < 0.5

import kaft.geometry as geo
import numpy as np

def test_registry_populated_at_import():
    metrics = geo.list_metrics()
    assert len(metrics) >= 10


def test_fisher_rao_registered():
    from kaft.geometry.base import AbstractMetric
    m = geo.get("fisher_rao")
    assert isinstance(m, AbstractMetric)

def test_metric_compute_returns_non_negative_float(synthetic_embeddings, fr_metric):
    a, b = synthetic_embeddings[0], synthetic_embeddings[1]
    d = fr_metric.compute(a, b)
    assert isinstance(d, float)
    assert d >= 0.0


def test_distances_matrix_shape_and_diagonal(synthetic_embeddings, fr_metric):
    D = fr_metric.distances(synthetic_embeddings)
    N = len(synthetic_embeddings)
    assert D.shape == (N, N)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)


def test_distances_matrix_symmetric(synthetic_embeddings, fr_metric):
    D = fr_metric.distances(synthetic_embeddings)
    assert np.allclose(D, D.T, atol=1e-8)


def test_jordan_boundaries_emerge_not_imposed(warmed_kd):
    from kaft.dynamics.jordan import JordanBoundary
    jb = JordanBoundary(warmed_kd)
    jb.detect()
    assert isinstance(jb.boundaries, list)
    for b in jb.boundaries:
        assert "boundary_point" in b
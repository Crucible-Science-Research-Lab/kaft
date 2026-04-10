import numpy as np
from kaft.navigate.geodesic import GeodesicNavigator, GeodesicPath, GeodesicPoint
from kaft.navigate.seeder import DomainSeeder
from kaft.dynamics.kstate import KState
from kaft.dynamics.jordan import JordanBoundary

def test_geodesic_returns_path(synthetic_embeddings, synthetic_records, fr_metric):
    nav  = GeodesicNavigator(synthetic_embeddings, synthetic_records, fr_metric, n_steps=10)
    path = nav.trace(0, 9)
    assert isinstance(path, GeodesicPath)
    assert path.total_length > 0.0

def test_geodesic_waypoint_count(synthetic_embeddings, synthetic_records, fr_metric):
    nav  = GeodesicNavigator(synthetic_embeddings, synthetic_records, fr_metric, n_steps=10)
    path = nav.trace(0, 9)
    assert len(path.waypoints) == 10

def test_geodesic_waypoints_type(synthetic_embeddings, synthetic_records, fr_metric):
    nav  = GeodesicNavigator(synthetic_embeddings, synthetic_records, fr_metric, n_steps=5)
    path = nav.trace(0, 11)
    for wp in path.waypoints:
        assert isinstance(wp, GeodesicPoint)
        assert 0.0 <= wp.t <= 1.0




def test_seeder_round_trip(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd, tmp_path):
    ks = KState()
    ks.update(warmed_kd.density)
    jb = JordanBoundary(warmed_kd)
    jb.detect()
    path = str(tmp_path / "seed.json")
    DomainSeeder.save(path, synthetic_embeddings, synthetic_records, fr_metric, warmed_kd, ks, jb)
    loaded = DomainSeeder.load(path, fr_metric)
    assert abs(loaded["k_value"] - ks.K) < 1e-4

def test_seeder_cache_preloaded(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd, tmp_path):
    ks = KState(); ks.update(warmed_kd.density)
    jb = JordanBoundary(warmed_kd); jb.detect()
    path = str(tmp_path / "seed2.json")
    DomainSeeder.save(path, synthetic_embeddings, synthetic_records, fr_metric, warmed_kd, ks, jb)
    loaded = DomainSeeder.load(path, fr_metric)
    assert loaded["k_density"]._density  is not None
    assert loaded["k_density"]._distances is not None

def test_domain_labels_not_in_kdensity(synthetic_embeddings, fr_metric):
    from kaft.dynamics.kstate import KDensity
    kd = KDensity(synthetic_embeddings, fr_metric)
    kd.compute()
    assert not hasattr(kd, "labels")
    assert not hasattr(kd, "domain_label")
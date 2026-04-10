import pytest
import numpy as np
from kaft.geometry import get as get_metric
from kaft.dynamics.kstate import KDensity, KState

@pytest.fixture(scope="session")
def synthetic_embeddings():
    rng = np.random.default_rng(42)
    clusters = [
        rng.normal([1,0,0,0], 0.1, (3,4)),
        rng.normal([0,1,0,0], 0.1, (3,4)),
        rng.normal([0,0,1,0], 0.1, (3,4)),
        rng.normal([0,0,0,1], 0.1, (3,4)),
    ]
    emb = np.vstack(clusters)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

@pytest.fixture(scope="session")
def synthetic_records():
    domains = ["math"]*3 + ["biology"]*3 + ["fusion"]*3 + ["ai"]*3
    return [{"title": f"{d}_paper_{i}", "domain_label": d}
            for i, d in enumerate(domains)]

@pytest.fixture(scope="session")
def fr_metric():
    return get_metric("fisher_rao")

@pytest.fixture(scope="session")
def warmed_kd(synthetic_embeddings, fr_metric):
    kd = KDensity(synthetic_embeddings, fr_metric)
    kd.compute()
    return kd
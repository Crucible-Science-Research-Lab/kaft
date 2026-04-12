import numpy as np

def test_kdensity_shape_and_range(warmed_kd, synthetic_embeddings):
    d = warmed_kd.density
    assert d.shape == (len(synthetic_embeddings),)
    assert d.min() >= 0.0
    assert d.max() <= 1.0


def test_kdensity_distances_cached(warmed_kd, synthetic_embeddings):
    N = len(synthetic_embeddings)
    assert warmed_kd.distances.shape == (N, N)


def test_kstate_phase_valid(warmed_kd):
    from kaft.dynamics.kstate import KState
    ks = KState()
    ks.update(warmed_kd.density)
    valid = {"diffuse","clustering","locked",
             "recursive","saturated"}
    assert ks.phase in valid

def test_kstate_delta_k_zero_on_first(warmed_kd):
    from kaft.dynamics.kstate import KState
    ks = KState()
    ks.update(warmed_kd.density)
    assert ks.delta_K == 0.0

def test_kvar_kaft_greater_than_zero(synthetic_embeddings, fr_metric):
    from kaft.dynamics.kstate import KDensity
    kd = KDensity(synthetic_embeddings, fr_metric)
    K_field = kd.compute()
    assert float(K_field.var()) > 0.0


def test_kvar_softmax_near_zero(synthetic_embeddings):
    from kaft.dynamics.softmax_dynamics import SoftmaxDynamics
    sm = SoftmaxDynamics(temperature=1.0)
    result = sm.run(synthetic_embeddings)
    assert float(result.K_var) < 1e-4

def test_soliton_called_once_per_step():
    from kaft.dynamics.kstate import KState, SOLITON_LOCK_STEPS
    ks = KState()
    density = np.full(10, 0.5)
    for _ in range(SOLITON_LOCK_STEPS + 2):
        ks.update(density)
    locked = ks.soliton_locked  # call once, store
    assert isinstance(locked, bool)

def test_gradient_flow_loss_decreases():
    from kaft.dynamics.gradient_flow import GradientFlow, kl_divergence
    rng = np.random.default_rng(0)
    K0 = np.abs(rng.normal(size=8)) + 0.01
    q  = np.abs(rng.normal(size=8)) + 0.01
    result = GradientFlow(kl_divergence).run(K0, q, n_steps=200)
    assert result.loss_history[-1] < result.loss_history[0]
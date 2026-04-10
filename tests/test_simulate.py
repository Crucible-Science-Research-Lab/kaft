import numpy as np
import pytest
from kaft.simulate.crucible import SimulationCrucible
from kaft.simulate.runner import SimulationRunner, Frame
from kaft.dynamics.kstate import KState
from kaft.dynamics.jordan import JordanBoundary, JordanNoise
from kaft.dynamics.metric_evolution import MetricEvolution
from kaft.dynamics.softmax_dynamics import SoftmaxDynamics

def test_crucible_invalid_vcog():
    with pytest.raises(ValueError, match="v_cog"):
        SimulationCrucible("d", "fisher_rao", "p", v_cog=-1.0)

def test_crucible_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        SimulationCrucible("d", "fisher_rao", "p", temperature=0.0)

def test_crucible_invalid_boundary_threshold():
    with pytest.raises(ValueError, match="boundary_threshold"):
        SimulationCrucible("d", "fisher_rao", "p", boundary_threshold=1.5)




def _make_runner(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd):
    ks = KState(); ks.update(warmed_kd.density)
    jb = JordanBoundary(warmed_kd); jb.detect()
    crucible = SimulationCrucible("test", "fisher_rao", "unused")
    return SimulationRunner(
        embeddings     = synthetic_embeddings,
        records        = synthetic_records,
        k_density      = warmed_kd,
        k_state        = ks,
        jordan         = jb,
        crucible       = crucible,
        metric_evolver = MetricEvolution(temperature=1.0, base_step=0.1),
        softmax        = SoftmaxDynamics(temperature=1.0),
        noise          = JordanNoise(sigma=0.05, rng=np.random.default_rng(42)),
    )

def test_runner_step_returns_frame(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd):
    runner = _make_runner(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd)
    frame  = runner.step()
    assert isinstance(frame, Frame)

def test_frame_fields_populated(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd):
    runner = _make_runner(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd)
    frame  = runner.step()
    assert frame.K_field.shape == (len(synthetic_embeddings),)
    assert 0.0 <= frame.K_mean <= 1.0
    assert isinstance(frame.soliton_locked, bool)
    assert isinstance(frame.boundary_indices, list)

def test_runner_log_grows(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd):
    runner = _make_runner(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd)
    runner.run_static(n_steps=4)
    assert len(runner.log) == 4


def test_runner_ingest_expands_corpus(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd):
    runner   = _make_runner(synthetic_embeddings, synthetic_records, fr_metric, warmed_kd)
    N_before = len(runner.embeddings)
    runner.step(new_embedding=synthetic_embeddings[0].copy())
    assert len(runner.embeddings) == N_before + 1
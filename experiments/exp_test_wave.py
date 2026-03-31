"""
test_wave.py — Tests for wave.py and kaft_source.py
Run from project root: python test_wave.py
"""
import numpy as np


def test_euler_decay():
    from kaft.dynamics.wave import WaveEngine
    K0     = np.ones(10) * 2.0
    result = WaveEngine(lambda K, t: -K, dt=0.01, scheme="euler").run(K0, 200, early_stopping=False)
    error  = np.abs(result.K_final - K0 * np.exp(-result.t_history[-1])).max()
    assert error < 0.02
    assert result.K_history.shape == (201, 10)
    print(f"  PASS euler_decay    — t={result.t_history[-1]:.2f}, max_error={error:.6f}")


def test_rk4_more_accurate():
    from kaft.dynamics.wave import WaveEngine
    K0 = np.ones(10) * 2.0
    re = WaveEngine(lambda K, t: -K, dt=0.05, scheme="euler").run(K0, 100, early_stopping=False)
    rr = WaveEngine(lambda K, t: -K, dt=0.05, scheme="rk4"  ).run(K0, 100, early_stopping=False)
    err_e = np.abs(re.K_final - K0 * np.exp(-re.t_history[-1])).max()
    err_r = np.abs(rr.K_final - K0 * np.exp(-rr.t_history[-1])).max()
    assert err_r < err_e, f"RK4 ({err_r:.6f}) should beat Euler ({err_e:.6f})"
    print(f"  PASS rk4_accuracy   — euler={err_e:.6f}  rk4={err_r:.6f}")


def test_early_stopping():
    from kaft.dynamics.wave import WaveEngine
    result = WaveEngine(lambda K, t: np.zeros_like(K), dt=0.01).run(np.ones(10)*0.5, 500)
    assert result.converged
    assert result.convergence_step < 50
    assert len(result.K_history) < 502
    print(f"  PASS early_stopping — converged at step {result.convergence_step}")


def test_after_step_once_per_step():
    """Critical: after_step must fire n_steps times, not 4*n_steps under RK4."""
    from kaft.dynamics.wave import WaveEngine
    log = {"src": 0, "after": 0}

    class Tracked:
        def __call__(self, K, t):
            log["src"] += 1
            return -K
        def after_step(self, K_new, t_new):
            log["after"] += 1

    WaveEngine(Tracked(), dt=0.01, scheme="rk4").run(np.ones(5), n_steps=10, early_stopping=False)
    assert log["after"] == 10, f"after_step {log['after']}x — expected 10"
    assert log["src"]   == 40, f"source {log['src']}x — expected 40 (rk4×4)"
    print(f"  PASS after_step     — source={log['src']}x (rk4×4)  after={log['after']}x (once/step)")


def test_kaft_source_assembles():
    import sys, types
    for mod in ["kaft","kaft.dynamics","kaft.dynamics.kstate",
                "kaft.dynamics.resonance","kaft.dynamics.transport",
                "kaft.dynamics.jordan","kaft.dynamics.metric_evolution"]:
        sys.modules.setdefault(mod, types.ModuleType(mod))

    N, ones = 12, np.ones(12)

    class _KD:
        embeddings = np.random.randn(12, 4)
        @property
        def density(self): return ones * 0.5

    class _R:
        def compute(self): return ones * 0.10

    class _T2:
        def compute(self, kd): return ones * 0.20

    class _JN:
        def sample(self, K): return ones * 0.05

    class _ME:
        steps = 0
        def step(self, kd, K): self.steps += 1

    class _KS:
        def update(self, K): pass
        phase = "Holofractal"
        soliton_locked = False

    from kaft.dynamics import kaft_source as ksmod
    src = ksmod.KAFTSource.__new__(ksmod.KAFTSource)
    src.k_density = _KD(); src.resonance = _R(); src.term2 = _T2()
    src.jordan_noise = _JN(); src.metric_evolver = _ME(); src.c = 1.0; src.kstate = _KS()

    dK = src(ones * 0.5, 0.0)
    assert dK.shape == (N,)
    assert np.allclose(dK, 0.35), f"Expected 0.35, got {dK[0]:.4f}"
    src.after_step(ones * 0.6, 0.01)
    assert src.metric_evolver.steps == 1
    print(f"  PASS kaft_source    — dK={dK[0]:.4f}  metric_steps={src.metric_evolver.steps}")


if __name__ == "__main__":
    print("\n── wave.py: WaveEngine ──────────────────────────────")
    test_euler_decay()
    test_rk4_more_accurate()
    test_early_stopping()
    test_after_step_once_per_step()
    print("\n── kaft_source.py: KAFTSource ───────────────────────")
    test_kaft_source_assembles()
    print("\n✓ All tests passed\n")

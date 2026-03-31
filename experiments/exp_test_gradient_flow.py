"""
test_gradient_flow.py — Tests for gradient_flow.py
Run from project root: python test_gradient_flow.py
"""
import numpy as np


def test_euclidean_converges():
    """Euclidean gradient descent on ‖K - q‖² should reach q."""
    from kaft.dynamics.gradient_flow import GradientFlow, euclidean_divergence

    rng    = np.random.default_rng(0)
    N      = 10
    K0     = rng.random(N)
    q      = rng.random(N)
    flow   = GradientFlow(divergence_fn=euclidean_divergence, lr=0.1)
    result = flow.run(K0, target=q, n_steps=500)

    assert result.converged,                         "Should converge"
    assert result.loss_history[-1] < 1e-4,          f"Final loss too high: {result.loss_history[-1]:.6f}"
    assert result.loss_history[0] > result.loss_history[-1], "Loss should decrease"
    print(f"  PASS euclidean_converges  — steps={result.convergence_step}, "
          f"loss: {result.loss_history[0]:.4f} → {result.loss_history[-1]:.6f}")


def test_kl_decreases():
    """KL divergence should strictly decrease toward target distribution."""
    from kaft.dynamics.gradient_flow import GradientFlow, kl_divergence

    rng    = np.random.default_rng(1)
    N      = 8
    K0     = np.abs(rng.random(N)) + 0.1
    q      = np.abs(rng.random(N)) + 0.1
    flow   = GradientFlow(divergence_fn=kl_divergence, lr=0.01)
    result = flow.run(K0, target=q, n_steps=300, early_stopping=False)

    assert result.loss_history[-1] < result.loss_history[0], "KL should decrease"
    assert np.all(np.isfinite(result.loss_history)),         "No NaN/Inf in loss"
    print(f"  PASS kl_decreases         — loss: {result.loss_history[0]:.4f} → {result.loss_history[-1]:.4f}")


def test_fisher_rao_decreases():
    """Fisher-Rao divergence should decrease toward target."""
    from kaft.dynamics.gradient_flow import GradientFlow, fisher_rao_divergence

    rng    = np.random.default_rng(2)
    N      = 8
    K0     = np.abs(rng.random(N)) + 0.1
    q      = np.abs(rng.random(N)) + 0.1
    flow   = GradientFlow(divergence_fn=fisher_rao_divergence, lr=0.01)
    result = flow.run(K0, target=q, n_steps=300, early_stopping=False)

    assert result.loss_history[-1] < result.loss_history[0], "FR divergence should decrease"
    assert np.all(np.isfinite(result.loss_history)),         "No NaN/Inf in loss"
    print(f"  PASS fisher_rao_decreases — loss: {result.loss_history[0]:.4f} → {result.loss_history[-1]:.4f}")


def test_variance_history_tracked():
    """FlowResult must carry variance_history for experiment comparison."""
    from kaft.dynamics.gradient_flow import GradientFlow, euclidean_divergence

    rng    = np.random.default_rng(3)
    N      = 20
    K0     = rng.random(N)
    q      = np.ones(N) * 0.5       # uniform target — variance should decrease
    flow   = GradientFlow(divergence_fn=euclidean_divergence, lr=0.1)
    result = flow.run(K0, target=q, n_steps=200)

    assert result.variance_history.shape[0] == result.loss_history.shape[0]
    assert result.variance_history[0] > result.variance_history[-1],         "Descending toward uniform should reduce variance"
    print(f"  PASS variance_tracked     — var: {result.variance_history[0]:.4f} → {result.variance_history[-1]:.6f}")


def test_kl_vs_fisher_rao_variance_signal():
    """
    The smoking gun preview — small N sanity check.

    Both flows start from same K0, same target q.
    We verify both produce valid trajectories with tracked variance.
    Full N-scaling experiment lives in experiments/exp_arxiv_softmax_vs_fr.py
    """
    from kaft.dynamics.gradient_flow import (
        GradientFlow, kl_divergence, fisher_rao_divergence
    )

    rng = np.random.default_rng(42)
    N   = 15
    K0  = np.abs(rng.random(N)) + 0.1
    q   = np.abs(rng.random(N)) + 0.1

    r_kl = GradientFlow(kl_divergence,         lr=0.01).run(K0, q, 200, early_stopping=False)
    r_fr = GradientFlow(fisher_rao_divergence,  lr=0.01).run(K0, q, 200, early_stopping=False)

    assert r_kl.K_history.shape == r_fr.K_history.shape
    assert np.all(np.isfinite(r_kl.variance_history))
    assert np.all(np.isfinite(r_fr.variance_history))

    final_var_kl = r_kl.variance_history[-1]
    final_var_fr = r_fr.variance_history[-1]
    print(f"  PASS kl_vs_fr_preview     — "
          f"final var(KL)={final_var_kl:.4f}  var(FR)={final_var_fr:.4f}")
    print(f"                              (N-scaling signal lives in exp_arxiv_softmax_vs_fr.py)")


if __name__ == "__main__":
    print("\n── gradient_flow.py ─────────────────────────────────")
    test_euclidean_converges()
    test_kl_decreases()
    test_fisher_rao_decreases()
    test_variance_history_tracked()
    test_kl_vs_fisher_rao_variance_signal()
    print("\n✓ All tests passed")
    print("\n── dynamics/ Layer 1 complete ───────────────────────")
    print("   kstate.py  resonance.py  jordan.py  transport.py")
    print("   metric_evolution.py  wave.py  kaft_source.py  gradient_flow.py\n")

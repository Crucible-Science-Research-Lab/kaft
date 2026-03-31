"""
Quick smoke test for transport.py and metric_evolution.py.
Run from project root: python test_transport.py
Does NOT require the full kaft library — uses mock data and
a lightweight stub for KDensity to isolate the transport logic.
"""
import numpy as np

# ── Minimal KDensity stub (no kaft.geometry needed) ──────────────────────────
class _MockKDensity:
    def __init__(self, N=20, D=8, seed=42):
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal((N, D))
        self.embeddings = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self._density = rng.random(N)          # fake normalised K in [0,1]
        # compute cosine distances as stand-in for Fisher-Rao
        dots = self.embeddings @ self.embeddings.T
        self._distances = np.arccos(np.clip(dots, -1.0, 1.0))
        np.fill_diagonal(self._distances, 0.0)

    @property
    def density(self):   return self._density
    @property
    def distances(self): return self._distances
    def invalidate(self): pass

# ── Test 1: FisherRaoGradient ─────────────────────────────────────────────────
def test_gradient():
    from kaft.dynamics.transport import FisherRaoGradient
    mock   = _MockKDensity(N=20, D=8)
    grad   = FisherRaoGradient()              # auto k_neighbors
    field  = grad.compute(mock.embeddings, mock.density, mock.distances)
    mags   = grad.magnitude(field)

    assert field.shape == (20, 8),  f"Expected (20,8), got {field.shape}"
    assert mags.shape  == (20,),    f"Expected (20,),  got {mags.shape}"
    assert np.all(mags >= 0),       "Magnitudes must be non-negative"
    # tangent vectors must be orthogonal to their base point (sphere tangent space)
    dots = np.einsum("nd,nd->n", field, mock.embeddings)
    assert np.allclose(dots, 0, atol=1e-5), "Gradient not in tangent space"
    print(f"  PASS gradient — k={grad._resolve_k(20)}, max_mag={mags.max():.4f}")

# ── Test 2: ParallelTransport ─────────────────────────────────────────────────
def test_transport():
    from kaft.dynamics.transport import FisherRaoGradient, ParallelTransport
    mock   = _MockKDensity(N=15, D=4)
    field  = FisherRaoGradient().compute(mock.embeddings, mock.density, mock.distances)
    pt     = ParallelTransport()
    trans, att_idx = pt.to_attractor(mock.embeddings, field, mock.density)

    assert trans.shape == (15, 4), f"Expected (15,4), got {trans.shape}"
    assert isinstance(att_idx, int)
    assert att_idx == int(np.argmax(mock.density))
    print(f"  PASS transport — attractor_idx={att_idx}, "
          f"density_at_attractor={mock.density[att_idx]:.4f}")

# ── Test 3: KAFTTerm2 ─────────────────────────────────────────────────────────
def test_term2():
    from kaft.dynamics.transport import KAFTTerm2
    mock   = _MockKDensity(N=25, D=8)
    term2  = KAFTTerm2()
    result = term2.compute(mock)

    assert result.shape == (25,), f"Expected (25,), got {result.shape}"
    assert np.all(np.isfinite(result)), "Term2 contains NaN or Inf"
    print(f"  PASS term2 — mean={result.mean():.4f}, "
          f"range=[{result.min():.4f}, {result.max():.4f}]")

# ── Test 4: MetricEvolution ───────────────────────────────────────────────────
def test_metric_evolution():
    # No kaft import needed — MetricEvolution only uses numpy
    import sys, importlib, types
    # Provide a stub for kaft.dynamics.kstate so import works standalone
    stub_mod = types.ModuleType("kaft.dynamics.kstate")
    stub_mod.KDensity = object
    sys.modules.setdefault("kaft", types.ModuleType("kaft"))
    sys.modules.setdefault("kaft.dynamics", types.ModuleType("kaft.dynamics"))
    sys.modules["kaft.dynamics.kstate"] = stub_mod

    from kaft.dynamics.metric_evolution import MetricEvolution
    mock = _MockKDensity(N=20, D=8)

    # Temperature check: higher T → less movement
    me_lo = MetricEvolution(temperature=0.1)
    me_hi = MetricEvolution(temperature=10.0)

    emb_before = mock.embeddings.copy()
    k_field    = mock.density

    new_lo = me_lo._reweight(emb_before, k_field)
    new_hi = me_hi._reweight(emb_before, k_field)

    diff_lo = np.linalg.norm(new_lo - emb_before)
    diff_hi = np.linalg.norm(new_hi - emb_before)

    assert diff_lo > diff_hi, "Low T should reshape geometry more than high T"
    # All reweighted embeddings must remain on unit sphere
    norms_lo = np.linalg.norm(new_lo, axis=1)
    assert np.allclose(norms_lo, 1.0, atol=1e-6), "Reweighted embeddings not on unit sphere"
    print(f"  PASS metric_evolution — diff(T=0.1)={diff_lo:.4f} > diff(T=10)={diff_hi:.4f}")

    # step() integration
    me_step = MetricEvolution(temperature=1.0)
    me_step.step(mock, k_field)
    assert me_step.step_count == 1
    print(f"  PASS metric_evolution.step() — step_count={me_step.step_count}")


if __name__ == "__main__":
    print("\n── transport.py ─────────────────────────────")
    test_gradient()
    test_transport()
    test_term2()
    print("\n── metric_evolution.py ──────────────────────")
    test_metric_evolution()
    print("\n✓ All tests passed\n")

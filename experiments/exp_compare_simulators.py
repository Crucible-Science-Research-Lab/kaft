import sys
import os
from kaft.navigate.seeder import DomainSeeder
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kaft.core import build_manifold
from kaft.simulate import compare, KAFTSimulator, SoftmaxSimulator

# corpus = [
#     # Biology cluster
#     {"text": "CRISPR-Cas9 gene editing mechanism"},
#     {"text": "protein folding molecular dynamics"},
#     {"text": "drug resistance mutation pathway"},
#     {"text": "mRNA vaccine immunological response"},
#     {"text": "enzyme catalysis biochemical kinetics"},
#     # Math/Geometry cluster
#     {"text": "Fisher-Rao information geometry manifold"},
#     {"text": "Riemannian curvature tensor geodesic"},
#     {"text": "probability space covariant derivative"},
#     {"text": "topological invariant homology group"},
#     {"text": "differential geometry Lie bracket"},
#     # Boundary / cross-domain
#     {"text": "statistical mechanics entropy thermodynamics"},
#     {"text": "neural network gradient descent optimization"},
# ]

# state  = build_manifold(corpus)
BASE = os.path.dirname(os.path.abspath(__file__))
state = DomainSeeder.load(os.path.join(BASE, "geometric_state_arxiv.json"))
print(f"[Corpus] {len(state.records)} real arXiv papers loaded")


# result = compare(KAFTSimulator(), SoftmaxSimulator(), manifold=state, steps=20, dt=0.1)
result = compare(KAFTSimulator(c=2.0, v_cog=1.5),
                 SoftmaxSimulator(),
                 manifold=state,
                 steps=50,
                 dt=0.2)


print("── KAFT (curved) vs Softmax (flat) ──\n")
print(f"Final divergence D(kaft||softmax) = {result['divergence']:.4f}")
print(f"  curved geometry : {result['geometry_type_a']}")
print(f"  flat geometry   : {result['geometry_type_b']}")

print("\n── Final K-density: KAFT (curved) ──")
for i, (r, k) in enumerate(zip(state.records, result['topology_a'])):
    bar = '█' * int(k * 20)
    print(f"  [{i}] K={k:.3f} {bar:<20}  {r['text'][:40]}")

print("\n── Final K-density: Softmax (flat) ──")
for i, (r, k) in enumerate(zip(state.records, result['topology_b'])):
    bar = '█' * int(k * 20)
    print(f"  [{i}] K={k:.3f} {bar:<20}  {r['text'][:40]}")

print("\n── Divergence curve (how fast they separated) ──")
for step, div in enumerate(result['divergence_curve']):
    bar = '▓' * int(div * 30)
    print(f"  step {step+1:02d}  {bar}  {div:.4f}")

print("\n Compare complete ✓")

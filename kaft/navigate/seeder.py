# kaft/navigate/seeder.py
"""DomainSeeder — persist and restore a fully-warmed corpus topology.

Saves: embeddings, records, k_density, distance matrix, jordan boundaries
       (enriched with record text), metric_name, K scalar, phase, and
       optionally umap_2d if the caller passes it in.

On load: reconstructs KDensity with the cached distance matrix already
         injected — no recomputation needed. JordanBoundary is
         reconstructed from the stored boundary list directly.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from kaft.geometry.base import AbstractMetric
from kaft.dynamics.kstate import KDensity, KState
from kaft.dynamics.jordan import JordanBoundary


class DomainSeeder:

    @staticmethod
    def save(
        path:          str,
        embeddings:    np.ndarray,          # (N, D) raw
        records:       list[dict],          # RouterRecord list
        metric:        AbstractMetric,      # Layer 0 instance
        k_density:     KDensity,            # must already be computed
        k_state:       KState,
        jordan:        JordanBoundary,      # must already be detected
        umap_2d:       np.ndarray | None = None,  # (N, 2) — caller computes
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Enrich jordan boundaries with record text at save time
        enriched_boundaries = []
        for b in jordan.boundaries:
            bp = b["boundary_point"]
            rec = records[bp]
            enriched_boundaries.append({
                **b,
                "text": rec.get("text", rec.get("title", ""))[:120],
            })

        data = {
            "metric_name":   metric.__class__.__name__,
            "embeddings":    embeddings.tolist(),
            "records":       records,
            "k_density":     k_density.density.tolist(),
            "distances":     k_density.distances.tolist(),
            "k_value":       k_state.K,
            "phase":         k_state.phase,
            "boundaries":    enriched_boundaries,
            "umap_2d":       umap_2d.tolist() if umap_2d is not None else None,
        }

        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[DomainSeeder] Saved → {path}")
        print(f"  Points: {len(records)} | Dim: {embeddings.shape[1]}"
              f" | K={k_state.K:.3f} | Phase={k_state.phase}"
              f" | Boundaries: {len(enriched_boundaries)}")

    @staticmethod
    def load(
        path:   str,
        metric: AbstractMetric,             # caller passes the right metric
    ) -> dict:
        """
        Returns a dict with fully-reconstructed kaft objects — no recomputation.

        Keys:
            embeddings   : np.ndarray (N, D)
            records      : list[dict]
            k_density    : KDensity   — density + distance cache pre-loaded
            k_state      : KState     — K history seeded with saved K value
            jordan       : JordanBoundary — boundaries pre-loaded
            k_value      : float
            phase        : str
            boundaries   : list[dict] — enriched with text
            umap_2d      : np.ndarray | None
        """
        data = json.loads(Path(path).read_text())

        embeddings = np.array(data["embeddings"])
        records    = data["records"]

        # Reconstruct KDensity with cached state — no recompute
        kd = KDensity(embeddings, metric)
        kd._density   = np.array(data["k_density"])
        kd._distances = np.array(data["distances"])

        # Reconstruct KState seeded from saved K value
        ks = KState()
        ks.update(kd.density)   # seeds history from stored density

        # Reconstruct JordanBoundary with pre-loaded boundaries
        jb = JordanBoundary(kd)
        jb._boundaries = data["boundaries"]

        umap_2d = np.array(data["umap_2d"]) if data.get("umap_2d") else None

        print(f"[DomainSeeder] Loaded ← {path}")
        print(f"  Points: {len(records)} | Dim: {embeddings.shape[1]}"
              f" | K={data['k_value']:.3f} | Phase={data['phase']}"
              f" | Boundaries: {len(data['boundaries'])}")

        return {
            "embeddings":  embeddings,
            "records":     records,
            "k_density":   kd,
            "k_state":     ks,
            "jordan":      jb,
            "k_value":     data["k_value"],
            "phase":       data["phase"],
            "boundaries":  data["boundaries"],
            "umap_2d":     umap_2d,
        }















# """DomainSeeder — loads a geometric_state.json as initial manifold topology."""
# from __future__ import annotations
# import json
# import numpy as np
# from pathlib import Path

# from kaft.core.manifold import Manifold


# class DomainSeeder:

#     @staticmethod
#     def save(state: Manifold, path: str) -> None:
#         # Create parent directory if it doesn't exist
#         Path(path).parent.mkdir(parents=True, exist_ok=True)
#         data = {
#             "embeddings": state.embeddings.tolist(),
#             "records": state.records,
#             "domain_type": getattr(state, 'domain_type', 'auto'),
#             "k_density": state.k_density.tolist() if state.k_density is not None else []
#         }
#         Path(path).write_text(json.dumps(data, indent=2))
#         print(f"[DomainSeeder] Manifold saved → {path}")
#         print(f"  Points: {len(state.records)} | Embedding dim: {state.embeddings.shape[1]}")

#     @staticmethod
#     def load(path: str) -> Manifold:
#         data = json.loads(Path(path).read_text())
#         state = Manifold(
#             embeddings=np.array(data["embeddings"]),
#             records=data.get("records", []),
#             domain_type=data.get("domain_type", "auto")
#         )
#         if data.get("k_density"):
#             state.k_density = np.array(data["k_density"])
#         print(f"[DomainSeeder] Manifold loaded ← {path}")
#         print(f"  Points: {len(state.records)} | Shape: {state.embeddings.shape}")
#         return state

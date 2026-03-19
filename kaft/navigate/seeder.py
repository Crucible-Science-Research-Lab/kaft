"""DomainSeeder — loads a geometric_state.json as initial manifold topology."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from kaft.core.manifold import Manifold


class DomainSeeder:

    @staticmethod
    def save(state: Manifold, path: str) -> None:
        # Create parent directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embeddings": state.embeddings.tolist(),
            "records": state.records,
            "domain_type": getattr(state, 'domain_type', 'auto'),
            "k_density": state.k_density.tolist() if state.k_density is not None else []
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[DomainSeeder] Manifold saved → {path}")
        print(f"  Points: {len(state.records)} | Embedding dim: {state.embeddings.shape[1]}")

    @staticmethod
    def load(path: str) -> Manifold:
        data = json.loads(Path(path).read_text())
        state = Manifold(
            embeddings=np.array(data["embeddings"]),
            records=data.get("records", []),
            domain_type=data.get("domain_type", "auto")
        )
        if data.get("k_density"):
            state.k_density = np.array(data["k_density"])
        print(f"[DomainSeeder] Manifold loaded ← {path}")
        print(f"  Points: {len(state.records)} | Shape: {state.embeddings.shape}")
        return state

"""
Manifold — the base geometric object.

A Manifold wraps a corpus of records into an embedded point cloud
with a computed Fisher-Rao metric and K-density field.
Everything else in kaft navigates or simulates on top of this.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Manifold:
    """
    Core geometric state object.
    Everything in kaft navigates or simulates on top of this.
    """
    embeddings: np.ndarray        # shape (N, 384) — unified metric space
    records: list                 # original corpus with "text" key
    domain_type: str = "auto"
    metric_tensor: np.ndarray | None = None
    k_density: np.ndarray | None = None
    jordan_boundaries: list | None = None


def build_manifold(records: list[dict], domain_type: str = "auto") -> Manifold:
    """
    Entry point: corpus → Manifold.
    Embeds all records into one unified Fisher-Rao coordinate space.

    Parameters
    ----------
    records : list[dict]
        Each record must have a "text" key.
    domain_type : str
        Routing hint — 'research', 'molecular', 'patent', 'auto'

    Returns
    -------
    Manifold
        Geometric state with embeddings ready for metric computation.

    Example
    -------
    >>> state = build_manifold([
    ...     {"text": "CRISPR-Cas9 gene editing mechanism"},
    ...     {"text": "drug resistance mutation pathway"},
    ... ])
    >>> state.embeddings.shape
    (2, 384)
    """
    from sentence_transformers import SentenceTransformer

    if not records:
        raise ValueError("records cannot be empty — no manifold to build")

    texts = [r["text"] for r in records]

    # all-MiniLM-L6-v2: 384 dimensions, fast, strong semantic geometry
    # This is the ONE embedding model used everywhere in kaft
    # It fixes the two-stream divergence from geometric_bridge vs geometric_prompting
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)

    return Manifold(
        embeddings=np.array(embeddings),
        records=records,
        domain_type=domain_type,
    )

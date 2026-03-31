from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Embedder:
    """Thin SentenceTransformer wrapper.

    Returns raw (N, D) float32 — NO normalisation.
    Raw norm  ||e_i||  =  I_i  (semantic mass) fed directly to KDensity.
    FisherRaoMetric.distances() does its own shift→sqrt→L2-norm→arccos
    internally, so pre-normalising here would destroy the mass signal.

    Args:
        model_name: any sentence-transformers model name.
                    Default: all-MiniLM-L6-v2 (384-dim, fast, good quality).
    """

    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed — pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], *, batch_size: int = 64) -> np.ndarray:
        """Encode texts → raw (N, D) float32.

        No normalisation.  Batch-size exposed for large corpora.
        """
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # ← critical: keeps L2 norm as I_i
            show_progress_bar=False,
        )
        return embs.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

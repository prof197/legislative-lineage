"""Embed windows with a pinned ONNX model, on the CPU, offline.

The model is loaded from the directory `lineage model fetch` filled (lineage.models), so
fastembed never downloads anything. Vectors come back unit length, so a dot product is the
cosine similarity. See DESIGN.md#semantic-stage.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

from lineage.models import MODELS, MODELS_DIR, require_model

# fastembed is given a local directory and has no reason to reach the Hugging Face hub; this
# makes sure nothing it imports tries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_model(
    name: str, models_dir: Path = MODELS_DIR, *, threads: int | None = None
) -> TextEmbedding:
    path = require_model(name, models_dir)
    return TextEmbedding(
        model_name=MODELS[name].fastembed_name, specific_model_path=str(path), threads=threads
    )


def embed(
    model: TextEmbedding, texts: Sequence[str], *, batch_size: int = 64
) -> NDArray[np.float32]:
    """One unit-length float32 row per text."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    rows = np.stack(
        [np.asarray(v, dtype=np.float32) for v in model.embed(texts, batch_size=batch_size)]
    )
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.where(norms == 0, 1, norms)).astype(np.float32, copy=False)

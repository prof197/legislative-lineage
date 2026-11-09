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

# Texts per batch. Sorted by length, a batch's texts pad little whatever its size; the rates in
# reports/embed_throughput.json are at this size.
BATCH_SIZE = 32


def load_model(
    name: str, models_dir: Path = MODELS_DIR, *, threads: int | None = None
) -> TextEmbedding:
    path = require_model(name, models_dir)
    return TextEmbedding(
        model_name=MODELS[name].fastembed_name, specific_model_path=str(path), threads=threads
    )


def embed(
    model: TextEmbedding, texts: Sequence[str], *, batch_size: int = BATCH_SIZE
) -> NDArray[np.float32]:
    """One unit-length float32 row per text, in the order given.

    A batch is padded to its longest text, so texts are embedded in order of token count: a
    batch of like lengths pads little, where a batch drawn at random pads most texts to the
    longest window in it. Masked padding leaves each vector as it was, to within float error.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    order = np.argsort([model.token_count(t) for t in texts], kind="stable")
    positions: list[int] = order.tolist()
    rows = np.stack(
        [
            np.asarray(v, dtype=np.float32)
            for v in model.embed([texts[i] for i in positions], batch_size=batch_size)
        ]
    )
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    out = np.empty_like(rows)
    out[order] = rows / np.where(norms == 0, 1, norms)
    return out

"""`lineage eval cost`: what each stage costs to build and to query (step 26).

The build's stages are timed by their own commands, so their seconds, peak memory and stored
bytes are read from the committed reports, each cited by file and field, never measured again
here. A query is timed here, stage by stage (lineage.query): each development act's text is
queried once from a cold start, when the semantic stage's vectors are not yet in the page cache,
and again warm. No stage makes a model call, so retrieval costs no dollars; the model's labels
and explanations are costed where they are made. See DESIGN.md#cost-and-runtime.
"""

import json
from pathlib import Path
from typing import Any

from lineage.acts import enacted_text
from lineage.query import query_text
from lineage.semantic import K, SemanticSearch

REPORT = Path("reports/cost.json")
COMMAND = "uv run lineage eval cost"
_NEIGHBOR_BYTES = 4 + 2  # an int32 id and a float16 cosine per kept slot


def _read(reports: Path, name: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((reports / name).read_text())
    return doc


def build_costs(reports: Path) -> dict[str, dict[str, Any]]:
    """Each build stage's seconds, peak memory and stored bytes, from its committed report."""
    index = _read(reports, "index.json")
    embedding = _read(reports, "embedding.json")
    neighbors = _read(reports, "neighbors.json")
    pairs = _read(reports, "semantic_pairs.json")
    workers = max(p for p, _ in embedding["processes_threads"])
    peaks = embedding["peak_memory_bytes"]
    return {
        "near_duplicate_index": {
            "source": "reports/index.json (build, signature_bytes, lsh.index_bytes, pairs)",
            "seconds": index["build"]["seconds"],
            "workers": index["build"]["workers"],
            "peak_memory_bytes": index["build"]["peak_memory_bytes"],
            "stored_bytes": index["signature_bytes"]
            + index["lsh"]["index_bytes"]
            + index["pairs"]["stored_bytes"],
            "dollars": 0,
        },
        "embedding": {
            "source": "reports/embedding.json (seconds, processes_threads, peak_memory_bytes, "
            "vector_bytes, texts_embedded_in_runs)",
            "seconds": embedding["seconds"],
            # The runs' seconds leave out the shards embedded before runs were logged.
            "texts_timed": embedding["texts_embedded_in_runs"],
            "texts": embedding["distinct_texts"],
            "workers": workers,
            # Every process's own peak, summed: a bound, since the peaks need not coincide.
            "peak_memory_bytes": workers * peaks["worker_peak_bytes"] + peaks["main_peak_bytes"],
            "stored_bytes": embedding["vector_bytes"],
            "dollars": 0,
        },
        "neighbors": {
            "source": "reports/neighbors.json (seconds, peak_memory_bytes, texts, k)",
            "seconds": neighbors["seconds"],
            "workers": None,
            "peak_memory_bytes": neighbors["peak_memory_bytes"],
            "stored_bytes": neighbors["texts"] * neighbors["k"] * _NEIGHBOR_BYTES,
            "dollars": 0,
        },
        "semantic_pairs": {
            "source": "reports/semantic_pairs.json (seconds, peak_memory_bytes, bytes)",
            "seconds": pairs["seconds"],
            "workers": None,
            "peak_memory_bytes": pairs["peak_memory_bytes"],
            "stored_bytes": pairs["bytes"],
            "dollars": 0,
        },
    }


def query_costs(
    tables_dir: Path, acts_dir: Path, semantic: SemanticSearch | None
) -> dict[str, dict[str, Any]]:
    """Each act text's query seconds by stage, first from cold and then warm."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(acts_dir.glob("*.txt")):
        text = enacted_text(path.read_text(encoding="utf-8-sig"))
        first = query_text(text, tables_dir, semantic=semantic)
        warm = query_text(text, tables_dir, semantic=semantic)
        out[path.stem] = {
            "sections_searched": first.searched,
            "first": {k: round(v, 3) for k, v in first.seconds.items()},
            "warm": {k: round(v, 3) for k, v in warm.seconds.items()},
        }
    return out


def cost_report(
    tables_dir: Path, acts_dir: Path, reports: Path, semantic: SemanticSearch | None
) -> dict[str, Any]:
    return {
        "command": COMMAND,
        "k": K,
        "build": build_costs(reports),
        "query": query_costs(tables_dir, acts_dir, semantic),
    }


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")

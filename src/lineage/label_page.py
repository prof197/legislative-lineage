"""`lineage label-page` and `lineage label-import`: label a sample on a phone, not a terminal.

The page is the labeling session of lineage.labels in another form. `label-page` reads the
sample's pairs from the tables exactly as `lineage label` does and writes the page and its data
to a local directory (gitignored, since it holds section text): each pending pair's two display
texts, the keys a label names them by, and the aligned spans, so the page marks copied,
paraphrased and unmatched text as the terminal does. Like the session it is blind: the data
carries no stage and no model label. The labeler's GitHub handle is checked when the data is
written, and every label the page saves names it.

The page is published as a private claude.ai artifact and keeps each label in that artifact's
own store as it is given, one document per pair, so labeling stops and resumes on any device.
`label-import` takes those documents back, one JSON object per line, and appends the label rows
to the labels file after checking each against the sample: the same two sections, by bill id
and text hash, a label on the scale, and the handle the page was written for. A row already in
the file is not appended twice.
"""

import datetime
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lineage.align import Embedder, align
from lineage.labels import PERSON, SCALE, Item, LabelError, person
from lineage.pair_data import LABELS

TEMPLATE = Path(__file__).parent / "pages" / "label.html"
OUT = Path("data/label_page")
DATA = "pairs.json"
PAGE = "index.html"
_ROW_KEYS = ("query", "match", "label", "labeler", "at")


def page_pair(item: Item, *, embed: Embedder | None) -> dict[str, Any]:
    """One pair as the page shows it: both sections and their aligned spans, no stages."""
    spans = align(item.query.text, item.match.text, embed=embed).spans
    return {
        "pair": item.pair.pair,
        "query": {**item.query.ref.as_dict(), "text": item.query.text},
        "match": {**item.match.ref.as_dict(), "text": item.match.text},
        "spans": [
            {
                "kind": s.kind,
                "a": None if s.a is None else list(s.a),
                "b": None if s.b is None else list(s.b),
            }
            for s in spans
        ],
    }


def page_data(
    items: Sequence[Item], *, handle: str, sample: Path, embed: Embedder | None
) -> dict[str, Any]:
    """Everything the page reads: who labels, the scale, and each pending pair."""
    return {
        "labeler": person(handle),
        "sample": sample.name,
        "paraphrase_marked": embed is not None,
        "scale": [
            {"label": i, "name": name.replace("_", " "), "gloss": gloss}
            for i, (name, gloss) in enumerate(zip(LABELS, SCALE, strict=True))
        ],
        "pairs": [page_pair(i, embed=embed) for i in items],
    }


def write_page(data: Mapping[str, Any], out: Path = OUT) -> Path:
    """The page and its data in `out`; returns the page's path."""
    out.mkdir(parents=True, exist_ok=True)
    (out / DATA).write_text(json.dumps(data, separators=(",", ":")) + "\n")
    page = out / PAGE
    shutil.copyfile(TEMPLATE, page)
    return page


def _date(value: object) -> str:
    try:
        return datetime.date.fromisoformat(str(value)).isoformat()
    except ValueError as e:
        raise LabelError(f"{value!r} is not a date") from e


def imported_rows(
    saved: Sequence[Mapping[str, Any]],
    items: Sequence[Item],
    existing: Sequence[Mapping[str, Any]],
    *,
    handle: str,
) -> list[dict[str, Any]]:
    """The label rows of the page's saved documents that are not in the labels file yet, each
    checked against the sample; any document that fails a check refuses the whole import."""
    labeler = person(handle)
    by_pair = {i.pair.pair: i for i in items}
    have = {json.dumps({k: r[k] for k in _ROW_KEYS}, sort_keys=True) for r in existing}
    order = {p: n for n, p in enumerate(by_pair)}
    rows: list[tuple[int, dict[str, Any]]] = []
    for doc in saved:
        pair = str(doc.get("pair"))
        item = by_pair.get(pair)
        if item is None:
            raise LabelError(f"saved label for {pair!r}, which is not a pair of the sample")
        if (
            doc.get("query") != item.query.ref.as_dict()
            or doc.get("match") != item.match.ref.as_dict()
        ):
            raise LabelError(f"{pair}: the saved sections are not the sample's current sections")
        label = doc.get("label")
        if not isinstance(label, int) or isinstance(label, bool) or label not in range(len(LABELS)):
            raise LabelError(
                f"{pair}: {label!r} is not a label on the 0 to {len(LABELS) - 1} scale"
            )
        if doc.get("labeler") != labeler or not labeler.startswith(PERSON):
            raise LabelError(f"{pair}: labeled by {doc.get('labeler')!r}, not {labeler!r}")
        row = {
            "query": item.query.ref.as_dict(),
            "match": item.match.ref.as_dict(),
            "label": label,
            "labeler": labeler,
            "at": _date(doc.get("at")),
        }
        key = json.dumps(row, sort_keys=True)
        if key not in have:
            have.add(key)
            rows.append((order[pair], row))
    # Sample order, so the file reads like a terminal session's.
    return [row for _, row in sorted(rows, key=lambda r: r[0])]


def load_saved(path: Path) -> list[dict[str, Any]]:
    """The page's saved documents, one JSON object per line."""
    out: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as e:
            raise LabelError(f"{path}:{n}: not JSON ({e})") from e
        if not isinstance(doc, dict):
            raise LabelError(f"{path}:{n}: not a saved label")
        out.append(dict(doc))  # pyright: ignore[reportUnknownArgumentType]
    return out


def append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + "\n")

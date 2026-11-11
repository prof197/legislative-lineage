"""`lineage label`: a person labels sampled matches, one pair at a time, resumable.

The precision sample (step 25) is a file of section pairs, one JSON object per line: the pair's
id, the stages that found it, and each side's bill id and section ordinal in the build it was
drawn from. It holds no text; the sections are read from the tables when the pair is shown,
aligned (lineage.align) so copied, paraphrased and unmatched spans stand out.

Each pair gets a label on the labeled pair dataset's scale, unrelated (0) to identical (4)
(lineage.pair_data). A match counts as correct from CORRECT_FROM, related, up: the bar the
semantic stage's threshold was set to (DESIGN.md#precision-sample).

A label is keyed by what survives a re-split of the corpus: each side's Open States bill id and
the SHA-256 of its section's normalized text. The heading and ordinal are kept for reading
only, since a re-split renumbers sections. A label whose text no longer matches a section of
its bill is dropped when the labels are loaded, and counted. Labelers are named by GitHub
handle, or for a provisional model label by model id and prompt version, never by name; only a
person's label takes a pair off the list. A session is blind to the stages that found each
pair. Each label is appended as it is given, so a session stops and resumes at will. Nothing
here calls a model; provisional model labels come from lineage.model_labels.
"""

import hashlib
import json
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from lineage import queries
from lineage.align import Embedder, Kind, align
from lineage.pair_data import LABELS

CORRECT_FROM = 2  # related; the pair evaluation's positive bar (lineage.pair_eval.POSITIVE_FROM)
# What each label means, shown to a person at the start of a session and to a model in its
# prompt, so both label on the same scale. The names are the pair dataset's.
SCALE = (
    "different subjects",
    "a shared subject, but different provisions",
    "the same provision or purpose, worded or detailed differently",
    "the same provision in mostly the same words, with small edits",
    "the same text, apart from formatting, numbering or place names",
)
LABELS_PATH = Path("evals/precision_labels.jsonl")
PERSON = "github:"
MODEL = "model:"
SKIP = "s"
QUIT = "q"
# GitHub's rule: up to 39 letters, digits and single hyphens, neither first nor last.
_HANDLE = re.compile(r"^(?!.*--)[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_MARKS: dict[Kind, tuple[str, str]] = {
    "copied": ("[=", "=]"),
    "paraphrased": ("[~", "~]"),
    "unmatched": ("[+", "+]"),
}
_COLORS: dict[Kind, dict[str, Any]] = {
    "copied": {"fg": typer.colors.GREEN},
    "paraphrased": {"fg": typer.colors.YELLOW},
    "unmatched": {"dim": True},
}


class LabelError(Exception):
    pass


def person(handle: str) -> str:
    """A person's labeler name: their GitHub handle, which a real name cannot pass for."""
    if not _HANDLE.fullmatch(handle):
        raise LabelError(f"{handle!r} is not a GitHub handle; label under your handle")
    return PERSON + handle


def model_labeler(model_id: str, prompt_version: str) -> str:
    return f"{MODEL}{model_id}@{prompt_version}"


def text_sha256(normalized: str) -> str:
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SectionKey:
    """Where a sample section sits in the build it was drawn from."""

    bill_id: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class SectionRef:
    """How a label names a section, so the label outlives a re-split."""

    bill_id: str  # Open States bill id
    heading: str | None  # for reading only
    ordinal: int  # for reading only
    text_sha256: str  # of the section's normalized text

    def as_dict(self) -> dict[str, Any]:
        return {
            "bill_id": self.bill_id,
            "heading": self.heading,
            "ordinal": self.ordinal,
            "text_sha256": self.text_sha256,
        }

    @property
    def key(self) -> tuple[str, str]:
        return (self.bill_id, self.text_sha256)


@dataclass(frozen=True, slots=True)
class Section:
    ref: SectionRef
    text: str  # display text


@dataclass(frozen=True, slots=True)
class Pair:
    pair: str
    stages: tuple[str, ...]
    query: SectionKey
    match: SectionKey


@dataclass(frozen=True, slots=True)
class Item:
    """A sample pair with both its sections read from the tables."""

    pair: Pair
    query: Section
    match: Section
    # The aligned passages (query side, match side) a relative is judged on; none for a pair.
    passages: tuple[tuple[str, str], ...] = ()


def load_sample(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            pair = Pair(
                str(row["pair"]),
                tuple(str(s) for s in row["stages"]),
                SectionKey(str(row["query"]["bill_id"]), int(row["query"]["ordinal"])),
                SectionKey(str(row["match"]["bill_id"]), int(row["match"]["ordinal"])),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise LabelError(f"{path}:{n}: not a sample pair ({e})") from e
        if pair.pair in seen:
            raise LabelError(f"{path}: pair {pair.pair} appears twice")
        seen.add(pair.pair)
        pairs.append(pair)
    return pairs


def load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _side(row: dict[str, Any], side: str) -> tuple[str, str]:
    return (row[side]["bill_id"], row[side]["text_sha256"])


def current_labels(
    labels: Sequence[dict[str, Any]], sections: Collection[tuple[str, str]]
) -> tuple[list[dict[str, Any]], int]:
    """The labels whose two sections still exist, by bill id and text hash, among `sections`;
    and how many were dropped because a section's text changed or is gone."""
    kept = [r for r in labels if _side(r, "query") in sections and _side(r, "match") in sections]
    return kept, len(labels) - len(kept)


def pending(items: Sequence[Item], labels: Sequence[dict[str, Any]]) -> list[Item]:
    """The pairs no person has labeled yet, in sample order."""
    done = {
        (_side(r, "query"), _side(r, "match"))
        for r in labels
        if str(r["labeler"]).startswith(PERSON)
    }
    return [i for i in items if (i.query.ref.key, i.match.ref.key) not in done]


def read_items(tables_dir: Path, pairs: Sequence[Pair]) -> list[Item]:
    """Every pair's two sections, read in one pass; a missing section is refused."""
    keys = sorted({(k.bill_id, k.ordinal) for p in pairs for k in (p.query, p.match)})
    with queries.connect_quiet() as con:
        found = queries.sections_at(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet"), keys=keys
        )
    missing = [k for k in keys if k not in found]
    if missing:
        bill, ordinal = missing[0]
        raise LabelError(
            f"{len(missing)} sample sections are not in {tables_dir / 'sections'}, "
            f"{bill} section {ordinal} first; was the sample drawn from another build?"
        )

    def section(key: SectionKey) -> Section:
        heading, text, normalized = found[(key.bill_id, key.ordinal)]
        return Section(SectionRef(key.bill_id, heading, key.ordinal, text_sha256(normalized)), text)

    return [Item(p, section(p.query), section(p.match)) for p in pairs]


def section_hashes(tables_dir: Path, labels: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    """(bill id, text hash) of every current section of the bills the labels name."""
    bills = sorted({r[side]["bill_id"] for r in labels for side in ("query", "match")})
    if not bills:
        return set()
    with queries.connect_quiet() as con:
        rows = queries.normalized_texts_of(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet"), bill_ids=bills
        )
    return {(bill, text_sha256(normalized)) for bill, normalized in rows}


def marked(a: str, b: str, *, embed: Embedder | None = None, color: bool = True) -> tuple[str, str]:
    """Both texts with their aligned spans marked: in color for a terminal, else in brackets.
    Without `embed`, nothing is marked as paraphrase."""
    spans = align(a, b, embed=embed).spans
    out: list[str] = []
    for side, text in (("a", a), ("b", b)):
        parts: list[str] = []
        for span in spans:
            at: tuple[int, int] | None = span.a if side == "a" else span.b
            if at is None:
                continue
            piece = text[at[0] : at[1]]
            if color:
                parts.append(typer.style(piece, **_COLORS[span.kind]))
            else:
                start, end = _MARKS[span.kind]
                parts.append(f"{start}{piece}{end}")
        out.append(" ".join(parts))
    return out[0], out[1]


def scale_lines() -> list[str]:
    return [
        f"{i} {name.replace('_', ' ')}: {gloss}"
        for i, (name, gloss) in enumerate(zip(LABELS, SCALE, strict=True))
    ]


def _prompt() -> str:
    scale = ", ".join(f"{i} {name.replace('_', ' ')}" for i, name in enumerate(LABELS))
    return f"label ({scale}; {SKIP} skip, {QUIT} quit): "


def session(
    items: Sequence[Item],
    *,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    labels_path: Path,
    labeler: str,
    today: str,
    embed: Embedder | None = None,
    color: bool = False,
) -> int:
    """Ask for a label for each pair in turn; append each as it is given. Returns how many
    were labeled before the pairs ran out or the labeler quit."""
    labeled = 0
    show("\n".join(scale_lines()))
    for n, item in enumerate(items, start=1):
        pair = item.pair
        first, second = marked(item.query.text, item.match.text, embed=embed, color=color)
        # The stages that found a pair are not shown, so they cannot sway its label.
        show(f"\n--- {n} of {len(items)}: {pair.pair}")
        show(f"query  {pair.query.bill_id} section {pair.query.ordinal}\n{first}")
        show(f"match  {pair.match.bill_id} section {pair.match.ordinal}\n{second}")
        while True:
            answer = ask(_prompt()).strip().lower()
            if answer == QUIT:
                return labeled
            if answer == SKIP:
                break
            if answer.isdigit() and int(answer) in range(len(LABELS)):
                row = {
                    "query": item.query.ref.as_dict(),
                    "match": item.match.ref.as_dict(),
                    "label": int(answer),
                    "labeler": labeler,
                    "at": today,
                }
                with labels_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                labeled += 1
                break
            show(f"not a label: {answer!r}")
    return labeled

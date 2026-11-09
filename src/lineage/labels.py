"""`lineage label`: a person labels sampled matches, one pair at a time, resumable.

The precision sample (step 25) is a file of section pairs, one JSON object per line: the pair's
id, the stages that found it, and each side's bill id and section ordinal. It holds no text;
the texts are read from the sections table when the pair is shown, aligned (lineage.align) so
copied, paraphrased and unmatched spans stand out.

Each pair gets a label on the labeled pair dataset's scale, unrelated (0) to identical (4)
(lineage.pair_data), so the sample's precision counts what the semantic stage's threshold was
set to find: related or closer. A label is appended to the labels file with its labeler and the
date as soon as it is given, so a session can stop at any pair and resume at the next one
still unlabeled. Provisional labels from a model are written with their own labeler and do not
take a pair off a person's list. Nothing here calls a model.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from lineage import queries
from lineage.align import Kind, align
from lineage.pair_data import LABELS

HUMAN = "human"
SKIP = "s"
QUIT = "q"
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


@dataclass(frozen=True, slots=True)
class SectionKey:
    bill_id: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class Pair:
    pair: str
    stages: tuple[str, ...]
    query: SectionKey
    match: SectionKey


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


def pending(sample: Sequence[Pair], labels: Sequence[dict[str, Any]]) -> list[Pair]:
    """The pairs no person has labeled yet, in sample order."""
    done = {r["pair"] for r in labels if r["labeler"] == HUMAN}
    return [p for p in sample if p.pair not in done]


def marked(a: str, b: str, *, meaning_match: bool, color: bool = True) -> tuple[str, str]:
    """Both texts with their aligned spans marked: in color for a terminal, else in brackets."""
    spans = align(a, b, meaning_match=meaning_match).spans
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


def _prompt() -> str:
    scale = ", ".join(f"{i} {name.replace('_', ' ')}" for i, name in enumerate(LABELS))
    return f"label ({scale}; {SKIP} skip, {QUIT} quit): "


def session(
    pairs: Sequence[Pair],
    *,
    texts: Callable[[SectionKey], str],
    ask: Callable[[str], str],
    show: Callable[[str], None],
    labels_path: Path,
    today: str,
    color: bool = False,
) -> int:
    """Ask for a label for each pair in turn; append each as it is given. Returns how many
    were labeled before the pairs ran out or the person quit."""
    labeled = 0
    for n, pair in enumerate(pairs, start=1):
        first, second = marked(
            texts(pair.query),
            texts(pair.match),
            meaning_match="semantic" in pair.stages,
            color=color,
        )
        show(f"\n--- {n} of {len(pairs)}: {pair.pair} (found by {', '.join(pair.stages)})")
        show(f"query  {pair.query.bill_id} section {pair.query.ordinal}\n{first}")
        show(f"match  {pair.match.bill_id} section {pair.match.ordinal}\n{second}")
        while True:
            answer = ask(_prompt()).strip().lower()
            if answer == QUIT:
                return labeled
            if answer == SKIP:
                break
            if answer.isdigit() and int(answer) in range(len(LABELS)):
                row = {"pair": pair.pair, "label": int(answer), "labeler": HUMAN, "at": today}
                with labels_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                labeled += 1
                break
            show(f"not a label: {answer!r}")
    return labeled


LABELS_PATH = Path("evals/precision_labels.jsonl")


def section_texts(tables_dir: Path, pairs: Sequence[Pair]) -> dict[SectionKey, str]:
    """Every pair's two section texts, read in one pass; a missing section is refused."""
    keys = sorted({(k.bill_id, k.ordinal) for p in pairs for k in (p.query, p.match)})
    with queries.connect_quiet() as con:
        found = queries.section_texts_at(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet"), keys=keys
        )
    missing = [k for k in keys if k not in found]
    if missing:
        bill, ordinal = missing[0]
        raise LabelError(
            f"{len(missing)} sample sections are not in {tables_dir / 'sections'}, "
            f"{bill} section {ordinal} first; was the sample drawn from another build?"
        )
    return {SectionKey(b, o): t for (b, o), t in found.items()}

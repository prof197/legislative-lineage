"""Uniform act texts as query documents for the commission recall eval.

An act as the commission publishes it is mostly not law: a title page and committee list, a
table of contents, a prefatory note, and an official comment or legislative note after many
sections. States enact the sections. Scoring is from the query's side, the share of its
sections a bill matches, so every comment left in would lower every enacted bill's score.
`enacted_text` keeps the numbered sections and drops the rest.

Act texts are copyrighted and never committed. Each user downloads them from the commission
into `data/acts/<act>.txt` as plain text, and the held-out acts into `data/acts/held-out/`.
See DESIGN.md#commission-recall.
"""

import re
from pathlib import Path

ACTS = (
    "public-expression-protection",
    "special-deposits",
    "electronic-estate-planning-documents",
)
# Acts never examined while tuning, evaluated once at the end of M5. Chosen by the commission's
# count of enactments in the corpus years, not by anything the pipeline found.
HELD_OUT_ACTS = (
    "assignment-for-benefit-of-creditors",
    "easement-relocation",
    "mortgage-modification",
    "directed-trust",
)
ACT_NAMES = {
    "public-expression-protection": "Public Expression Protection",
    "special-deposits": "Special Deposits",
    "electronic-estate-planning-documents": "Electronic Estate Planning Documents",
    "assignment-for-benefit-of-creditors": "Assignment for Benefit of Creditors",
    "easement-relocation": "Easement Relocation",
    "mortgage-modification": "Mortgage Modification",
    "directed-trust": "Directed Trust",
}
ACTS_DIR = Path("data/acts")
HELD_OUT_DIR = ACTS_DIR / "held-out"

# "Section 1. Short Title", or "[Section 3. Registration" for an optional section. Acts
# divided into articles number sections by article ("Section 101.") under headings such as
# "[Article] 1", or "[[Article] 3" for an optional article.
_HEADING = re.compile(
    r"^\s*(?:\[?Section\s+\d+[A-Za-z]?\.|\[{0,2}Article\]?\s+\d+\s*$)", re.IGNORECASE
)
# A table-of-contents entry ends in its page number, after a tab, dot leaders or a wide gap.
_CONTENTS_ENTRY = re.compile(r"(?:\t|\.{3,}|\s{2,})\s*\d+\s*$")
# A comment, or a legislative note, runs from its heading to the next section or article:
# in the commission's final acts both close a section, and either can run several paragraphs.
_COMMENT = re.compile(r"^\s*(?:official\s+)?(?:comments?|reporter'?s\s+notes?)\s*$", re.IGNORECASE)
_NOTE = re.compile(r"^\s*(?:legislative|drafting)\s+notes?\b", re.IGNORECASE)


class ActTextError(Exception):
    pass


def _is_heading(line: str) -> bool:
    return _HEADING.match(line) is not None and _CONTENTS_ENTRY.search(line) is None


def enacted_text(raw: str) -> str:
    """The act's numbered sections, without front matter, comments or legislative notes."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if _is_heading(line)), None)
    if start is None:
        raise ActTextError("no section or article heading: is this the act's final text?")
    kept: list[str] = []
    dropping = False
    for line in lines[start:]:
        if _is_heading(line):
            dropping = False
        elif _COMMENT.match(line) or _NOTE.match(line):
            dropping = True
        if not dropping:
            kept.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip() + "\n"


def load_act(act: str, acts_dir: Path = ACTS_DIR) -> str:
    """One act's enacted text, from `<acts_dir>/<act>.txt`."""
    path = acts_dir / f"{act}.txt"
    if not path.exists():
        raise ActTextError(
            f"{path} not found: download the final text of {act} from uniformlaws.org and "
            "save it there as plain text"
        )
    return enacted_text(path.read_text(encoding="utf-8-sig"))

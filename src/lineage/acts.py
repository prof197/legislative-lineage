"""Uniform act texts as query documents for the commission recall eval.

An act as the commission publishes it is mostly not law: a title page and committee list, a
table of contents, a prefatory note, and an official comment or legislative note after many
sections. States enact the sections. Scoring is from the query's side, the share of its
sections a bill matches, so every comment left in would lower every enacted bill's score.
`enacted_text` keeps the numbered sections and drops the rest.

Act texts are copyrighted and never committed. Each user downloads them from the commission
into `data/acts/<act>.txt` as plain text. See DESIGN.md#commission-recall.
"""

import re
from pathlib import Path

ACTS = (
    "public-expression-protection",
    "special-deposits",
    "electronic-estate-planning-documents",
)
ACTS_DIR = Path("data/acts")

# "Section 1. Short Title", or "[Section 3. Registration" for an optional section.
_HEADING = re.compile(r"^\s*\[?Section\s+\d+[A-Za-z]?\.", re.IGNORECASE)
_FIRST_HEADING = re.compile(r"^\s*\[?Section\s+1\.", re.IGNORECASE)
# A table-of-contents entry ends in its page number, after a tab, dot leaders or a wide gap.
_CONTENTS_ENTRY = re.compile(r"(?:\t|\.{3,}|\s{2,})\s*\d+\s*$")
# A comment runs from its heading to the next section.
_COMMENT = re.compile(r"^\s*(?:official\s+)?(?:comments?|reporter'?s\s+notes?)\s*$", re.IGNORECASE)
# A legislative note runs to the next blank line, heading or comment.
_NOTE = re.compile(r"^\s*(?:legislative|drafting)\s+notes?\b", re.IGNORECASE)


class ActTextError(Exception):
    pass


def _is_heading(line: str, pattern: re.Pattern[str] = _HEADING) -> bool:
    return pattern.match(line) is not None and _CONTENTS_ENTRY.search(line) is None


def enacted_text(raw: str) -> str:
    """The act's numbered sections, without front matter, comments or legislative notes."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if _is_heading(line, _FIRST_HEADING)), None)
    if start is None:
        raise ActTextError("no Section 1 heading: is this the act's final text?")
    kept: list[str] = []
    in_comment = in_note = False
    for line in lines[start:]:
        if _is_heading(line):
            in_comment = in_note = False
        elif _COMMENT.match(line):
            in_comment, in_note = True, False
        elif _NOTE.match(line):
            in_note = True
        elif in_note and not line.strip():
            in_note = False
        if not (in_comment or in_note):
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

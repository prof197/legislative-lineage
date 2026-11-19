"""Model texts as matching reads them: the law a state would enact, without the drafter's asides.

A published model act carries more than its sections: front matter, commentary, drafting and
legislative notes that tell a state how to adapt a section, and bracketed instructions to the
drafter such as "[cite to the state's commercial code]" or "[specify the court]". No bill
carries those words, so leaving them in would lower the share of a model's words any bill can
match. The commission's acts already have a reader for their layout,
`lineage.evals.acts.enacted_text`, which keeps the numbered sections and drops front matter,
comments and notes; it is reused here, so the eval's query and the collected document drop the same
commentary.

Brackets are dropped only where they hold an instruction. The commission brackets much that a
state enacts as written: "[act]", "[this state]", alternatives such as "[shall][may]", periods
such as "[60]" days, and optional sections. Those are kept as written, as the eval's query keeps
them (DESIGN.md#commission-recall, Placeholders): "[act]" enacted as "act" costs nothing. An
instruction such as "[insert citation]" is never enacted, so it goes. See
DESIGN.md#model-documents.
"""

import io
import re
import zipfile
from xml.etree import ElementTree

from lineage.evals.acts import enacted_text

# A bracket whose first word tells the drafter what to put there, or a bracketed gap. An
# instruction can hold an enacted bracket ("[cite to the law ... of this [act]]"), so one level
# of nesting is inside it; "[not less than [insert a number]" keeps its outer, enacted, bracket.
_INSTRUCTION = re.compile(
    r"[ \t]*\[\s*(?:(?:cite|citation|insert|specify|identify|describe|enter|appropriate)\b"
    r"|name\s+of\b)"
    r"[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"
    r"|[ \t]*\[\s*(?:…|\.(?:\s*\.)+|_+)?\s*\]",
    re.IGNORECASE,
)
# A drafting note or comment set inline in brackets, as some model texts give them.
_INLINE_NOTE = re.compile(
    r"[ \t]*\[\s*(?:drafting\s+note|legislative\s+note|note|comment)s?\s*:[^\[\]]*\]", re.IGNORECASE
)

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class ModelTextError(Exception):
    pass


def drop_instructions(text: str) -> str:
    """`text` without bracketed instructions to the drafter or inline notes; enacted brackets
    stay as written."""
    out = _INLINE_NOTE.sub("", text)
    while True:
        dropped = _INSTRUCTION.sub("", out)
        if dropped == out:
            break
        out = dropped
    return "\n".join(line.rstrip() for line in out.split("\n"))


def clean_act(raw: str) -> str:
    """A commission act's text as matching reads it: its numbered sections
    (lineage.evals.acts), with no comment, note or instruction to the drafter."""
    # A note that stood on its own line leaves an empty one behind.
    return re.sub(r"\n{3,}", "\n\n", drop_instructions(enacted_text(raw))).strip() + "\n"


def docx_text(data: bytes) -> str:
    """A Word document's body text, one line per paragraph, tables' included. Footnotes, headers
    and footers are not body text."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        raise ModelTextError(f"unreadable Word document: {type(e).__name__}: {e}") from e
    body = root.find(f"{_W}body")
    if body is None:
        raise ModelTextError("a Word document without a body")
    return "\n".join(_paragraph_text(p) for p in body.iter(f"{_W}p"))


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """A paragraph's own text. A paragraph nested in it, as in a text box, is a paragraph of its
    own, so its text is not repeated here. Deleted tracked changes hold w:delText, never read,
    and a struck-through run is wording an amendment deletes, so it is not read either."""
    parts: list[str] = []
    stack = list(reversed(list(paragraph)))
    while stack:
        node = stack.pop()
        if node.tag == f"{_W}p" or (node.tag == f"{_W}r" and _struck(node)):
            continue
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in {f"{_W}br", f"{_W}cr"}:
            parts.append("\n")
        stack.extend(reversed(list(node)))
    return "".join(parts)


def _struck(run: ElementTree.Element) -> bool:
    """Whether a run is set in single or double strikethrough; `w:val` of 0, false or off unsets
    it, as a style's strike can be turned off for one run."""
    props = run.find(f"{_W}rPr")
    if props is None:
        return False
    for tag in ("strike", "dstrike"):
        mark = props.find(f"{_W}{tag}")
        if mark is not None and mark.get(f"{_W}val", "true").lower() not in {"0", "false", "off"}:
            return True
    return False

"""Cleaning model texts (DESIGN.md#model-documents): sections kept, commentary and instructions
to the drafter dropped, enacted brackets kept as written.

The real-sample tests read the commission's acts from the local data directory (LINEAGE_DATA)
and are skipped without them, so CI passes; they assert counts only, never print act text."""

import io
import re
import zipfile
from pathlib import Path

import pytest

from lineage.evals.acts import (
    ACTS,
    ACTS_DIR,
    HELD_OUT_ACTS,
    HELD_OUT_DIR,
    ORIGIN_HELD_OUT_ACTS,
    enacted_text,
)
from lineage.modeltext import ModelTextError, clean_act, docx_text, drop_instructions
from tests.recorded_site import SYNTHETIC_ACT, docx_bytes

CLEANED = (
    "Section 1. Short Title\n"
    "This [act] may be cited as the Synthetic Lantern Registry Act.\n"
    "\n"
    "Section 2. Definitions\n"
    "In this [act]:\n"
    '(1) "Lantern" means a portable lamp kept under.\n'
    '(2) "Registry" means the list kept by [the secretary of state][the registrar].\n'
    "\n"
    "Section 3. Registration\n"
    "A keeper shall register each lantern within [60] days, before the or the registrar.\n"
    "[Section 4. Optional Fees\n"
    "The registrar may charge a fee not less than [ dollars.]\n"
    "\n"
    "Section 5. Effective Date\n"
    "This [act] takes effect [the effective date of this [act]].\n"
)


def test_an_act_keeps_its_sections_and_loses_commentary_and_instructions() -> None:
    assert clean_act(SYNTHETIC_ACT.read_text()) == CLEANED


@pytest.mark.parametrize(
    ("given", "cleaned"),
    [
        ("under [cite to the state's commercial code], a", "under, a"),
        ("before the [specify the court in the enacting state].", "before the."),
        ("filed with [insert office]", "filed with"),
        ("the [Name of agency] shall", "the shall"),
        ("[Identify the fund] and [describe the account]", " and"),
        ("as [ . . . ] and […] and [___] and [ ]", "as and and and"),
        ("[Drafting Note: rename it.] The registry", " The registry"),
        ("under [citation to the insurance code], a", "under, a"),
        ("in [appropriate section of the state code].", "in."),
        ("a fee [Comment: optional.] of", "a fee of"),
    ],
)
def test_instructions_to_the_drafter_go(given: str, cleaned: str) -> None:
    assert drop_instructions(given) == cleaned


@pytest.mark.parametrize(
    "kept",
    [
        "This [act] applies in [this state].",
        "The court [shall][may] dismiss within [60] days.",
        "[Section 4. Optional Fees",
        "[the effective date of this [act]]",
        "as amended [, and regulations issued thereunder]",
        "the [secretary of state] keeps it",  # an office named in brackets is enacted wording
    ],
)
def test_brackets_a_state_enacts_stay_as_written(kept: str) -> None:
    assert drop_instructions(kept) == kept


def test_an_instruction_inside_an_enacted_bracket_goes_and_the_bracket_stays() -> None:
    given = "not less than [[insert a number not less than 90] days]"
    assert drop_instructions(given) == "not less than [ days]"


def test_an_instruction_holding_an_enacted_bracket_goes_whole() -> None:
    given = "under [cite to the law on lanterns, as used in this [act]]."
    assert drop_instructions(given) == "under."


def test_a_word_document_reads_as_its_text_does() -> None:
    text = SYNTHETIC_ACT.read_text()
    assert clean_act(docx_text(docx_bytes(text))) == clean_act(text)
    with pytest.raises(ModelTextError, match="unreadable Word document"):
        docx_text(b"not a zip")


def test_struck_wording_in_a_word_file_is_not_read() -> None:
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    runs = (
        '<w:r><w:t xml:space="preserve">The fee is </w:t></w:r>'
        '<w:r><w:rPr><w:strike/></w:rPr><w:t xml:space="preserve">ten</w:t></w:r>'
        '<w:r><w:rPr><w:dstrike w:val="1"/></w:rPr><w:t xml:space="preserve">nine</w:t></w:r>'
        '<w:r><w:rPr><w:strike w:val="false"/></w:rPr><w:t xml:space="preserve">twenty</w:t></w:r>'
        '<w:r><w:t xml:space="preserve"> dollars.</w:t></w:r>'
    )
    document = f'<w:document xmlns:w="{w}"><w:body><w:p>{runs}</w:p></w:body></w:document>'
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("word/document.xml", document)
    assert docx_text(out.getvalue()) == "The fee is twenty dollars."


# ---- real samples, local only ---------------------------------------------------------------

_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|insert|specify)\b", re.IGNORECASE)
_NOTE_LINE = re.compile(r"^\s*(?:legislative|drafting)\s+notes?\b", re.IGNORECASE | re.MULTILINE)
_HEADING = re.compile(r"^\s*\[{0,2}(?:Section\s+\d+[A-Za-z]?\.|Article\]?\s+\d+)", re.I | re.M)


def _local_acts() -> list[Path]:
    paths = [ACTS_DIR / f"{a}.txt" for a in ACTS] + [
        HELD_OUT_DIR / f"{a}.txt" for a in HELD_OUT_ACTS + ORIGIN_HELD_OUT_ACTS
    ]
    return [p for p in paths if p.exists()]


@pytest.mark.skipif(not _local_acts(), reason="no act texts under the data directory's acts/")
@pytest.mark.parametrize("path", _local_acts(), ids=lambda p: p.stem)
def test_a_real_act_keeps_every_section_and_no_instruction(path: Path) -> None:
    raw = path.read_text(encoding="utf-8-sig")
    sections = enacted_text(raw)
    cleaned = clean_act(raw)

    # Counts only, so a failure never prints the act's text.
    headings, headings_before = len(_HEADING.findall(cleaned)), len(_HEADING.findall(sections))
    instructions_left = len(_INSTRUCTION_LEFT.findall(cleaned))
    note_lines = len(_NOTE_LINE.findall(cleaned))
    kept, before = len(cleaned.split()), len(sections.split())
    settled = clean_act(cleaned) == cleaned

    assert headings == headings_before > 0
    assert instructions_left == 0
    assert note_lines == 0
    # Instructions are a few words per section; losing more would be cleaning law away.
    assert 0.9 * before <= kept <= before
    assert settled


def _local_word_files() -> list[Path]:
    return sorted(p for d in (ACTS_DIR, HELD_OUT_DIR) if d.exists() for p in d.glob("*.docx"))


@pytest.mark.skipif(
    not _local_word_files(), reason="no Word files under the data directory's acts/"
)
@pytest.mark.parametrize("path", _local_word_files(), ids=lambda p: p.stem[:24])
def test_a_real_word_file_reads_as_its_saved_text(path: Path) -> None:
    from_word = clean_act(docx_text(path.read_bytes())).split()
    # The plain-text copy saved beside it is the one sharing the most distinct words.
    best = max(
        (
            clean_act(p.read_text(encoding="utf-8-sig")).split()
            for p in _local_acts()
            if p.parent == path.parent
        ),
        key=lambda words: len(set(words) & set(from_word)) / len(set(words) | set(from_word)),
    )
    words_apart = abs(len(best) - len(from_word))
    distinct_apart = len(set(best) ^ set(from_word))
    # A list bullet the plain-text conversion wrote is the only difference seen.
    assert words_apart <= 0.01 * len(best)
    assert distinct_apart <= 2

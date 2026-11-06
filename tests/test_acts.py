"""Act texts as query documents: only the enacted provisions are kept.

The act below is invented, laid out the way the commission's final acts are: a title page and
committee list, a table of contents, a prefatory note, then sections, each followed by an
official comment or a legislative note. No real act text is committed (DESIGN.md#commission-recall).
"""

from pathlib import Path

import pytest

from lineage.acts import ACTS, ActTextError, enacted_text, load_act

ACT = """\
Uniform Widget Registration Act

drafted by the

NATIONAL CONFERENCE OF COMMISSIONERS ON UNIFORM STATE LAWS

and by it

APPROVED AND RECOMMENDED FOR ENACTMENT IN ALL THE STATES

The Committee appointed by and representing the commission consists of the following:

A Commissioner, Chair

Copyright © 2099 by the commission

Table of Contents

Section 1. Short Title\t1
Section 2. Definitions\t1
[Section 3. Registration]\t3
Section 4. Effective Date\t4

Uniform Widget Registration Act

Prefatory Note

Section 1 of this act names it. Widgets are registered so that their owners can be found.

Section 1. Short Title

This [act] may be cited as the Uniform Widget Registration Act.

Comment

The short title follows the commission's usual form. Section 2 defines its terms.

Section 2. Definitions

In this [act]:

(1) "Widget" means a device registered under Section 3.

(2) "Owner" means a person that holds title to a widget.

Legislative Note: A state that already defines "owner" should cross-reference its definition.

Comment

1. The definition of widget is deliberately broad.

2. Section 3 sets out the registration duty.

[Section 3. Registration

(a) An owner shall register a widget with the [secretary of state] not later than [30] days
after acquiring it.

(b) A registration expires after [five] years.]

Legislative Note: Bracketed text may be adapted to the state's own offices and periods.

Section 4. Effective Date

This [act] takes effect [effective date].
"""


def test_keeps_the_sections_and_drops_everything_else() -> None:
    text = enacted_text(ACT)

    assert text.startswith("Section 1. Short Title")
    assert text.rstrip().endswith("This [act] takes effect [effective date].")
    for kept in ("may be cited as", '"Widget" means', "(b) A registration expires", "[30] days"):
        assert kept in text
    for dropped in (
        "Commissioner, Chair",
        "Table of Contents",
        "Prefatory Note",
        "Widgets are registered so that",
        "Comment",
        "deliberately broad",
        "Legislative Note",
        "cross-reference its definition",
        "Bracketed text may be adapted",
    ):
        assert dropped not in text


def test_a_bracketed_optional_section_is_kept_as_a_section() -> None:
    text = enacted_text(ACT)

    assert "[Section 3. Registration" in text
    assert "Section 3 sets out" not in text  # from a comment, not the act


def test_an_act_with_no_numbered_sections_is_refused() -> None:
    with pytest.raises(ActTextError, match="no Section 1"):
        enacted_text("Uniform Widget Registration Act\n\nPrefatory Note\n\nWidgets matter.\n")


def test_loads_a_named_act_from_the_local_directory(tmp_path: Path) -> None:
    (tmp_path / f"{ACTS[0]}.txt").write_text(ACT, encoding="utf-8")

    assert load_act(ACTS[0], tmp_path).startswith("Section 1. Short Title")
    with pytest.raises(ActTextError, match=ACTS[1]):
        load_act(ACTS[1], tmp_path)

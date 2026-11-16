"""Model documents (DESIGN.md#model-documents): a source says what kind of organization it is and
whether it is leaked, and a document's texts are kept per version, outside git."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from lineage.modeldocs import (
    ModelDoc,
    ModelDocError,
    Source,
    load_text,
    read_docs,
    read_versions,
    sha256,
    store,
    text_path,
    withdraw,
)

SOURCE = Source(
    key="example-library",
    organization="Example Council",
    descriptor="nonprofit membership organization, 501(c)(3)",
    library_url="https://example.org/models",
    terms_note="viewing and copying with attribution",
    terms_read_on=date(2026, 11, 15),
)


def _doc(text: str, on: date, raw: bytes = b"%PDF") -> ModelDoc:
    return ModelDoc(
        id="example-library/open-records-act",
        source=SOURCE.key,
        title="Model Open Records Act",
        url="https://example.org/models/open-records.pdf",
        collected_on=on,
        robots="allowed",
        leaked=False,
        raw_sha256=sha256(raw),
        media_type="application/pdf",
        text_sha256=sha256(text.encode()),
    )


def test_a_members_line_comes_with_its_source() -> None:
    line = "Commissioners appointed by each state."
    with pytest.raises(ValidationError, match="members line"):
        Source.model_validate({**SOURCE.model_dump(), "members": line})
    with pytest.raises(ValidationError):
        Source.model_validate(
            {**SOURCE.model_dump(), "members": line, "members_source": "http://example.org"}
        )
    sourced = {**SOURCE.model_dump(), "members": line, "members_source": "https://example.org/a"}
    assert Source.model_validate(sourced).members == line


def test_a_leaked_source_says_how_it_became_public() -> None:
    with pytest.raises(ValidationError, match="leak_note"):
        Source.model_validate({**SOURCE.model_dump(), "leaked": True})
    leaked = Source.model_validate(
        {**SOURCE.model_dump(), "leaked": True, "leak_note": "published in 2011"}
    )
    with pytest.raises(ModelDocError, match="leaked"):
        store(leaked, _doc("text", date(2026, 11, 15)), raw=b"%PDF", text="text", root=Path())


def test_a_joint_document_names_its_co_publishers_and_older_records_have_none() -> None:
    doc = _doc("t", date(2026, 11, 15))
    assert doc.co_publishers == ()  # a record stored before the field reads as it did
    joint = ModelDoc.model_validate({**doc.model_dump(), "co_publishers": ["Second Institute"]})
    assert joint.co_publishers == ("Second Institute",)
    for names in (["  "], ["Second Institute", "Second Institute"]):
        with pytest.raises(ValidationError, match="co-publisher"):
            ModelDoc.model_validate({**doc.model_dump(), "co_publishers": names})


def test_a_document_belongs_to_the_source_its_id_names() -> None:
    with pytest.raises(ValidationError, match="not in source"):
        ModelDoc.model_validate(
            {**_doc("t", date(2026, 11, 15)).model_dump(), "id": "other/open-records-act"}
        )


def test_collecting_again_changes_nothing_and_a_new_text_keeps_the_old(tmp_path: Path) -> None:
    first, second = date(2026, 11, 15), date(2026, 12, 1)
    old = _doc("Section 1. Records are open.", first)

    assert store(SOURCE, old, raw=b"%PDF", text="Section 1. Records are open.", root=tmp_path)
    again = _doc("Section 1. Records are open.", second)
    assert not store(SOURCE, again, raw=b"%PDF", text="Section 1. Records are open.", root=tmp_path)
    assert read_docs(SOURCE, tmp_path) == [old]  # the re-run left even its date alone

    new = _doc("Section 1. Records are open within ten days.", second, raw=b"%PDF-2")
    assert store(
        SOURCE,
        new,
        raw=b"%PDF-2",
        text="Section 1. Records are open within ten days.",
        root=tmp_path,
    )

    assert read_docs(SOURCE, tmp_path) == [new]
    versions = read_versions(SOURCE, tmp_path)
    assert [(v.first_seen, v.replaced_on) for v in versions] == [(first, second), (second, None)]
    assert text_path(old, tmp_path).exists()  # replaced, not erased
    assert load_text(new, tmp_path) == "Section 1. Records are open within ten days."


def test_a_withdrawn_document_leaves_the_current_list_and_keeps_its_text(tmp_path: Path) -> None:
    doc = _doc("Section 1. Records are open.", date(2026, 11, 1))
    store(SOURCE, doc, raw=b"%PDF", text="Section 1. Records are open.", root=tmp_path)

    assert withdraw(SOURCE, doc.id, on=date(2026, 11, 15), root=tmp_path)
    assert read_docs(SOURCE, tmp_path) == []
    assert [v.replaced_on for v in read_versions(SOURCE, tmp_path)] == [date(2026, 11, 15)]
    assert text_path(doc, tmp_path).exists()
    assert not withdraw(SOURCE, doc.id, on=date(2026, 11, 16), root=tmp_path)


def test_hashes_must_match_what_is_stored(tmp_path: Path) -> None:
    doc = _doc("the text", date(2026, 11, 15))
    with pytest.raises(ModelDocError, match="hashes"):
        store(SOURCE, doc, raw=b"%PDF", text="another text", root=tmp_path)
    store(SOURCE, doc, raw=b"%PDF", text="the text", root=tmp_path)
    text_path(doc, tmp_path).write_text("edited by hand")
    with pytest.raises(ModelDocError, match="does not match"):
        load_text(doc, tmp_path)

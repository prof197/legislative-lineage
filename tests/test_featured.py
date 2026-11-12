"""The featured scope: an anchor's listed section pairs with its relatives, from the anchor's side,
and the relatives introduced after it (DESIGN.md#explanations)."""

import numpy as np
import pytest

from lineage.featured import (
    GEORGIA_SECTIONS,
    STORIES,
    FeaturedError,
    Relatives,
    in_scope,
    introduced_after,
    pairs_with,
)
from lineage.listing import ListedPairs
from lineage.pairs import SectionPairs
from lineage.semantic_pairs import SemanticPairs

# Sections 0-1 are the anchor's, 2-3 relative r's, 4 another bill's.
BILLS = ["anchor", "anchor", "r", "r", "x"]
ORDINALS = np.array([1, 2, 1, 5, 1], dtype=np.int64)


def _u32(*v: int) -> np.ndarray:
    return np.array(v, dtype=np.uint32)


LISTED = ListedPairs(
    SectionPairs(_u32(0, 1), _u32(2, 4), np.array([9, 9], dtype=np.uint8)),
    SemanticPairs(
        _u32(0, 1, 3),
        _u32(2, 3, 4),
        np.array([0.95, 0.96, 0.99], dtype=np.float32),
        _u32(0, 0, 0),
        _u32(0, 0, 0),
    ),
    {"near_duplicate": 0, "semantic": 0},
)


def test_every_listed_pair_with_a_relative_once_from_the_anchors_side() -> None:
    got = pairs_with(LISTED, BILLS, ORDINALS, "anchor", ["r"])

    # (0, 2) is found by both stages and counted once; (1, 4) is with a bill not asked for.
    assert got == [(1, "r", 1), (2, "r", 5)]
    assert pairs_with(LISTED, BILLS, ORDINALS, "r", ["anchor"]) == [
        (1, "anchor", 1),
        (5, "anchor", 2),
    ]


def test_only_relatives_introduced_after_the_anchor_are_kept() -> None:
    first = {"a": "2023-01-09", "b": "2023-01-10", "c": None}

    assert introduced_after(["a", "b", "c"], first, "2023-01-09") == ["b"]
    with pytest.raises(FeaturedError, match="no dated action"):
        introduced_after(["a"], first, None)


def test_a_scoped_story_keeps_the_pairs_whose_anchor_side_lies_in_its_sections() -> None:
    headings = {1: "SECTION 2-2.", 2: "SECTION 3-2.", 5: None}
    pairs = [(1, "r", 1), (2, "r", 5), (5, "s", 2)]

    assert in_scope(pairs, headings, ("SECTION 2-2.",)) == [(1, "r", 1)]
    assert in_scope(pairs, headings, ()) == pairs
    # A resectioned anchor that no longer has a named heading is an error, never a silent drop.
    with pytest.raises(FeaturedError, match=r"no section headed 'SECTION 9-9\.'"):
        in_scope(pairs, headings, ("SECTION 2-2.", "SECTION 9-9."))


def test_the_stories_are_the_ones_the_review_ruled() -> None:
    assert [(s.name, s.state, s.identifier, s.relatives, s.sections) for s in STORIES] == [
        ("anti-slapp", "UT", "SB 18", Relatives.ENACTMENTS, ()),
        ("deepfakes", "WA", "SB 5152", Relatives.LISTED_LATER, ()),
        ("georgia-social-media", "GA", "SB 351", Relatives.LISTED, GEORGIA_SECTIONS),
    ]
    # Georgia's story is its social-media provisions: instruction on social media and school
    # policies on it (Part II), and the platform chapter (Part III's Chapter 6); not school
    # internet filtering (SECTION 2-3.) or age verification for harmful material (SECTION 3-2.).
    assert "SECTION 2-3." not in GEORGIA_SECTIONS
    assert "SECTION 3-2." not in GEORGIA_SECTIONS
    assert all(s.scope_note for s in STORIES if s.sections)

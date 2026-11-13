"""The static site's data contract: every file `lineage export` writes, as Pydantic models.

The same classes validate what the export writes and emit the JSON Schema the site's
TypeScript types are generated from (`schema/export.schema.json`, committed; a test fails when
it drifts from these models), so the pipeline and the site share one definition of every
field (DESIGN.md#dr-1).

The export holds facts, not display wording: a bill's status as the pipeline derives it,
whether its stored text is a notice or an amendment alone, whether a relative's matched text
names an interstate compact. How the site words or groups them is the site's decision.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Status = Literal["passed", "died", "introduced", "passed_legislature"]
Stage = Literal["near_duplicate", "semantic"]
SpanKind = Literal["copied", "paraphrased", "unmatched"]
ClaimKind = Literal["added", "removed", "changed"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BillRef(_Model):
    """A bill as any page names it."""

    slug: str  # the file name its data is written under
    state: str
    session: str
    identifier: str
    title: str
    status: Status | None
    status_reason: str | None  # the rule that decided the status (DESIGN.md#status)
    status_date: str | None  # the deciding action's day, ISO 8601
    first_action: str | None  # the day of its first recorded action, ISO 8601


class TextInfo(_Model):
    """Which text of a bill the matching read."""

    # Open States' export, the legislature's site, or California's bulk download
    # (DESIGN.md#text-layer)
    source: Literal["openstates", "legislature", "ca_bulk"]
    version: str | None  # the version's label, when the record names one
    url: str | None  # where the text was taken from
    notice_only: bool  # a Georgia local bill's notice of intention, with no bill text
    amendment_only: bool  # opens like an amendment and does not replace the whole bill


class SectionInfo(_Model):
    ordinal: int
    heading: str | None
    words: int
    boilerplate: str | None  # the boilerplate kind that kept it from matching, if any
    with_relatives: int  # how many listed relatives match this section


class SectionPair(_Model):
    """One listed match between a section of this bill and one of the relative's."""

    ordinal: int  # this bill's section
    other_ordinal: int  # the relative's
    stages: list[Stage]
    estimate: float | None  # the near-duplicate stage's Jaccard estimate
    cosine: float | None  # the semantic stage's best cosine between their windows


class Relative(_Model):
    """A related family in another state, shown through its best bill."""

    bill: BillRef
    family_bills: int
    share: float  # of this bill's searched words, the share in sections the family matches
    best_estimate: float  # the highest near-duplicate estimate of any matched pair, or 0
    best_cosine: float | None
    found_by: Literal["wording", "meaning"]  # which list a page shows it in
    pairs: list[SectionPair]  # with the shown bill
    names_compact: bool  # a matched section of either bill names an interstate compact
    comparison: str  # the comparison file's slug


class BillPage(_Model):
    """`bills/<slug>.json`: a hosted bill's page."""

    bill: BillRef
    text: TextInfo | None  # None when the bill has no text
    sections: list[SectionInfo]
    words_searched: int
    families: int  # every related family, however many are listed
    relatives: list[Relative]  # the best ones, best first, up to the display cap


class Closest(_Model):
    slug: str
    state: str
    identifier: str
    share: float


class BillListEntry(_Model):
    bill: BillRef
    families: int
    closest: Closest | None


class BillList(_Model):
    """`bills.json`: every hosted bill, for the browse page's filter."""

    bills: list[BillListEntry]


class Span(_Model):
    id: str
    kind: SpanKind
    a: tuple[int, int] | None  # character offsets in the first bill's section text
    b: tuple[int, int] | None


class Claim(_Model):
    kind: ClaimKind
    summary: str
    origin_span_ids: list[str]
    descendant_span_ids: list[str]


class Explanation(_Model):
    """A model's account of how the second text differs from the first, grounded in the
    alignment's spans; only for featured pairs (DESIGN.md#explanations)."""

    model: str
    withheld: bool  # no grounded account after one retry
    claims: list[Claim]  # every claim grounding kept, in the reply's order


class AlignedPair(_Model):
    ordinal_a: int
    ordinal_b: int
    stages: list[Stage]
    spans: list[Span]
    copied_share_a: float
    copied_share_b: float
    explanation: Explanation | None


class Comparison(_Model):
    """`compare/<slug>.json`: two bills' matched sections, aligned. The first bill is the
    hosted bill, or a featured story's anchor, whose text the comparison starts from."""

    a: BillRef
    b: BillRef
    pairs: list[AlignedPair]


class SectionText(_Model):
    ordinal: int
    heading: str | None
    text: str  # the display text span offsets index


class BillTexts(_Model):
    """`texts/<slug>.json`: a bill's sections that some comparison shows, each once."""

    bill: str
    sections: list[SectionText]


class StoryRelative(_Model):
    bill: BillRef
    section_pairs: int  # featured pairs with the anchor
    comparison: str | None  # None when the bill has no featured pair with the anchor


class Story(_Model):
    """`stories/<slug>.json`: a featured story."""

    slug: str
    anchor: BillRef
    relatives_rule: str
    scope_note: str | None
    relatives: list[StoryRelative]


class ArchiveStamp(_Model):
    session: str
    generated_at: str  # the archive's Generated At, the as-of date of its records


class Recall(_Model):
    enactments: int
    found: int
    recall: float


class Enactment(_Model):
    act: str
    state: str
    year: int
    bill_cell: str
    stratum: str
    found: bool
    miss_reason: str | None


class Share(_Model):
    labeled: int
    correct: int
    share: float | None
    interval_95: tuple[float, float]


class Validation(_Model):
    """The validation page's figures, from the committed results (reports/results.json)."""

    development_recall: Recall
    development_recall_by_act: dict[str, Recall]
    development_enactments: list[Enactment]
    held_out_recall: Recall
    relatives_precision: Share  # model-judged
    relatives_precision_labeler: str
    faithfulness: Share  # model-judged
    faithfulness_judge: str


class Meta(_Model):
    """`meta.json`: the data stamp every page carries."""

    schema_version: int = Field(default=SCHEMA_VERSION)
    state: str
    archives: list[ArchiveStamp]  # the hosted state's
    states_indexed: int
    hosted_bills: int
    stories: list[str]
    validation: Validation


TOP_LEVEL: tuple[type[BaseModel], ...] = (
    Meta,
    BillList,
    BillPage,
    Comparison,
    BillTexts,
    Story,
)

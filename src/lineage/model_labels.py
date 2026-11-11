"""Provisional precision labels from a model (step 25), recorded apart from people's.

A model labels the precision sample on the same scale, with the same definitions, a person
sees (lineage.labels.SCALE), so precision has a provisional estimate before volunteers have
labeled, and where both have labeled, their agreement says how far the provisional one can be
trusted. The model sees the two section texts and nothing else: not the stages that found the
pair, not the alignment. Its reply must fit ModelLabel, one label and no prose.

Each label is appended to the labels file under `model:<model id>@<prompt version>`, the id of
the model that served the reply, so a fallback is on record. The file is the cache: a pair
already labeled under the current prompt version is never asked again, and a new version asks
every pair afresh. A model's label never takes a pair off a person's list
(lineage.labels.pending), and the estimate never pools the two (lineage.precision). A declined
or malformed reply is counted and left for the next run. Nothing is asked without --live.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from lineage.labels import MODEL, Item, model_labeler, scale_lines

MODEL_ID = "claude-opus-5-5"
PROMPT_VERSION = "v1"
# A 0-4 label needs little thought: at low effort, 1,024 tokens holds the thinking and the reply,
# and keeps the run inside its cost cap. A reply cut off at the limit is unusable, not guessed.
MAX_TOKENS = 1024
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT: Effort = "low"
# Standard prices in dollars per million tokens, input and output, from the platform's model
# list as of November 2026; the Batches API charges half. A model not listed is not projected.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}
# The first batch's prompts ran 2.77 characters a token (8.67 million characters, 3,134,052
# input tokens; reports/label_spend.json): legal text tokenizes finer than prose. Fewer than
# that, so the projection errs high.
CHARS_PER_TOKEN = 2.5
MAX_COST = 10.0  # dollars a run may be projected to cost before it is refused
# Declines are rerun on the fallback model the API picks for the refusal's category; the label
# records which model served it.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_SCALE = "\n".join(scale_lines())
SYSTEM = f"""\
You are given two sections of state legislation, "first" and "second", as JSON. Say how \
closely the second's text follows the first's, on this scale:

{_SCALE}

Judge the text alone: what the provisions do and how they are worded. Do not guess who wrote \
either section. Reply with the one label that fits best."""


RELATIVES_SYSTEM = f"""\
You are given two sections of state legislation, "first" and "second", and the passages an \
alignment matched between them, as JSON: each passage is a stretch of the first and the \
stretch of the second it was matched to. A page lists the second section's bill as a relative \
of the first's because of these passages. Say how closely the second follows the first where \
they were matched, reading each passage in its section, on this scale:

{_SCALE}

Judge the text alone: what the matched provisions do and how they are worded. When no passage \
is listed, judge the two sections whole. Do not guess who wrote either section. Reply with the \
one label that fits best."""


class ModelLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: Literal[0, 1, 2, 3, 4]


class UnusableReplyError(Exception):
    """The model declined, or its reply did not fit the schema."""


Ask = Callable[[str, str], tuple[int, str]]  # (system, user) -> (label, served by)


def user_message(item: Item) -> str:
    return json.dumps(
        {"first": item.query.text, "second": item.match.text}, ensure_ascii=False, indent=1
    )


def relative_message(item: Item) -> str:
    """The two sections and the matched passages, not marked as copied or paraphrased."""
    return json.dumps(
        {
            "first": item.query.text,
            "second": item.match.text,
            "passages": [{"first": a, "second": b} for a, b in item.passages],
        },
        ensure_ascii=False,
        indent=1,
    )


@dataclass(frozen=True, slots=True)
class Prompt:
    """What a model is asked and how hard it thinks. The version names the labeler, so a
    change to any of these is another labeler, and its labels are asked afresh."""

    version: str
    system: str
    message: Callable[[Item], str]
    effort: Effort
    max_tokens: int
    passages: bool = False  # the items must carry their aligned passages


# At high effort, thinking runs longer than 1,024 tokens holds.
HIGH_EFFORT_MAX_TOKENS = 4096
SECTIONS = Prompt(PROMPT_VERSION, SYSTEM, user_message, EFFORT, MAX_TOKENS)
SECTIONS_HIGH = Prompt(
    f"{PROMPT_VERSION}-high", SYSTEM, user_message, "high", HIGH_EFFORT_MAX_TOKENS
)
RELATIVES = Prompt("relatives-v1", RELATIVES_SYSTEM, relative_message, EFFORT, MAX_TOKENS, True)
RELATIVES_HIGH = Prompt(
    "relatives-v1-high", RELATIVES_SYSTEM, relative_message, "high", HIGH_EFFORT_MAX_TOKENS, True
)
PROMPTS = {p.version: p for p in (SECTIONS, SECTIONS_HIGH, RELATIVES, RELATIVES_HIGH)}


def projected_cost(
    items: Sequence[Item],
    *,
    model: str = MODEL_ID,
    max_tokens: int | None = None,
    batch: bool = False,
    prompt: Prompt = SECTIONS,
) -> float:
    """An upper estimate, in dollars, of asking every pair: each prompt's tokens from its
    characters, and each reply, thinking included, at its token limit."""
    if model not in PRICES:
        raise ValueError(f"no price for {model}; add it to PRICES before asking it")
    input_price, output_price = PRICES[model]
    limit = prompt.max_tokens if max_tokens is None else max_tokens
    chars = sum(len(prompt.system) + len(prompt.message(i)) for i in items)
    cost = (chars / CHARS_PER_TOKEN * input_price + len(items) * limit * output_price) / 1e6
    return cost / 2 if batch else cost


def model_pending(
    items: Sequence[Item], labels: Sequence[Mapping[str, Any]], prompt: Prompt = SECTIONS
) -> list[Item]:
    """The pairs no model has labeled under the prompt's version."""
    suffix = f"@{prompt.version}"
    done = {
        (
            (r["query"]["bill_id"], r["query"]["text_sha256"]),
            (r["match"]["bill_id"], r["match"]["text_sha256"]),
        )
        for r in labels
        if str(r["labeler"]).startswith(MODEL) and str(r["labeler"]).endswith(suffix)
    }
    return [i for i in items if (i.query.ref.key, i.match.ref.key) not in done]


def label_with_model(
    items: Sequence[Item], *, ask: Ask, labels_path: Path, today: str, prompt: Prompt = SECTIONS
) -> tuple[int, int]:
    """Ask for each pair's label in turn and append it as it comes. Returns how many were
    labeled and how many replies were unusable."""
    labeled = unusable = 0
    for item in items:
        try:
            label, served_by = ask(prompt.system, prompt.message(item))
        except UnusableReplyError:
            unusable += 1
            continue
        row = {
            "query": item.query.ref.as_dict(),
            "match": item.match.ref.as_dict(),
            "label": label,
            "labeler": model_labeler(served_by, prompt.version),
            "at": today,
        }
        with labels_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        labeled += 1
    return labeled, unusable


def anthropic_ask(model: str = MODEL_ID, prompt: Prompt = SECTIONS) -> Ask:
    """Ask through the Anthropic SDK with structured output from ModelLabel. The key comes from
    the environment, as every secret here does."""
    import anthropic

    client = anthropic.Anthropic()

    def ask(system: str, user: str) -> tuple[int, str]:
        try:
            response = client.beta.messages.parse(
                model=model,
                max_tokens=prompt.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=ModelLabel,
                output_config={"effort": prompt.effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except ValidationError as e:
            raise UnusableReplyError(f"the reply does not fit the schema: {e}") from e
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise UnusableReplyError(f"no usable reply (stop reason {response.stop_reason})")
        return response.parsed_output.label, response.model

    return ask


@dataclass(frozen=True, slots=True)
class BatchReply:
    custom_id: str  # the pair's id
    label: int | None  # None when the reply is unusable
    served_by: str | None
    why: str | None = None  # why it is unusable
    input_tokens: int = 0
    output_tokens: int = 0  # thinking included


class Batches(Protocol):
    """The Batches API as the labeling needs it: half price, results within a day."""

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str: ...
    def ended(self, batch_id: str) -> bool: ...
    def results(self, batch_id: str) -> Iterable[BatchReply]: ...


def request_params(item: Item, model: str, prompt: Prompt = SECTIONS) -> dict[str, Any]:
    """One pair's request: the scale, the texts, and a reply that must fit ModelLabel."""
    return {
        "model": model,
        "max_tokens": prompt.max_tokens,
        "system": prompt.system,
        "messages": [{"role": "user", "content": prompt.message(item)}],
        "output_config": {
            "effort": prompt.effort,
            "format": {"type": "json_schema", "schema": ModelLabel.model_json_schema()},
        },
    }


def label_in_batch(
    items: Sequence[Item],
    *,
    batches: Batches,
    model: str,
    labels_path: Path,
    today: str,
    wait: Callable[[], None],
    batch_id: str | None = None,
    submitted: Callable[[str], None] = lambda _: None,
    prompt: Prompt = SECTIONS,
) -> tuple[int, int, str]:
    """Send every pair in one batch, or collect `batch_id` if it was sent already; wait until it
    ends, then append each usable label. Returns how many were labeled, how many replies were
    unusable, and the batch id. A batch is not resumed part way: its results are read whole."""
    if batch_id is None:
        batch_id = batches.submit([(i.pair.pair, request_params(i, model, prompt)) for i in items])
        submitted(batch_id)
    while not batches.ended(batch_id):
        wait()
    by_id = {i.pair.pair: i for i in items}
    labeled = unusable = 0
    with labels_path.open("a") as f:
        for reply in batches.results(batch_id):
            item = by_id.get(reply.custom_id)
            if item is None:
                raise UnusableReplyError(f"batch {batch_id} answered {reply.custom_id}, not asked")
            if reply.label is None or reply.served_by is None:
                unusable += 1
                continue
            row = {
                "query": item.query.ref.as_dict(),
                "match": item.match.ref.as_dict(),
                "label": reply.label,
                "labeler": model_labeler(reply.served_by, prompt.version),
                "at": today,
            }
            f.write(json.dumps(row) + "\n")
            labeled += 1
    return labeled, unusable, batch_id


def anthropic_batches() -> Batches:
    """The Batches API through the Anthropic SDK; the key comes from the environment. Batches
    take no server-side fallback, so a declined pair is unusable, counted and left."""
    import anthropic

    client = anthropic.Anthropic()

    class _Batches:
        def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str:
            batch = client.messages.batches.create(
                requests=cast(Any, [{"custom_id": c, "params": p} for c, p in requests])
            )
            return batch.id

        def ended(self, batch_id: str) -> bool:
            return client.messages.batches.retrieve(batch_id).processing_status == "ended"

        def results(self, batch_id: str) -> Iterable[BatchReply]:
            for entry in client.messages.batches.results(batch_id):
                result = entry.result
                if result.type != "succeeded":
                    yield BatchReply(entry.custom_id, None, None, result.type)
                    continue
                message = result.message
                # A reply is billed whether or not it is usable.
                tokens = (message.usage.input_tokens, message.usage.output_tokens)
                if message.stop_reason != "end_turn":
                    why = f"stop {message.stop_reason}"
                    yield BatchReply(entry.custom_id, None, None, why, *tokens)
                    continue
                text = next((b.text for b in message.content if b.type == "text"), "")
                try:
                    label = ModelLabel.model_validate_json(text).label
                except ValidationError:
                    yield BatchReply(entry.custom_id, None, None, "not the schema", *tokens)
                    continue
                yield BatchReply(entry.custom_id, label, message.model, None, *tokens)

    return _Batches()


SPEND = Path("reports/label_spend.json")
# Dollars every batch of precision labels may cost together, spent and projected: the cap the
# review set for step 25's labels. A run whose projection would pass it is refused.
BUDGET = 30.0


def batch_dollars(model: str, input_tokens: int, output_tokens: int) -> float:
    """What a batch's tokens cost, at half the standard price."""
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1e6 / 2


def batch_usage(batches: Batches, batch_id: str) -> tuple[int, int, int]:
    """(replies, input tokens, output tokens) of an ended batch, from its results."""
    replies = input_tokens = output_tokens = 0
    for reply in batches.results(batch_id):
        replies += 1
        input_tokens += reply.input_tokens
        output_tokens += reply.output_tokens
    return replies, input_tokens, output_tokens


def load_spend(path: Path = SPEND) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text())["batches"])


def spent(path: Path = SPEND) -> float:
    return sum(float(b["dollars"]) for b in load_spend(path))


def record_spend(
    path: Path,
    *,
    batch_id: str,
    model: str,
    prompt: str,
    usage: tuple[int, int, int],
    today: str,
) -> dict[str, Any]:
    """Record a batch's tokens and cost, once per batch id, and return the ledger."""
    replies, input_tokens, output_tokens = usage
    entry = {
        "batch_id": batch_id,
        "model": model,
        "prompt": prompt,
        "replies": replies,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "dollars": round(batch_dollars(model, input_tokens, output_tokens), 4),
        "at": today,
    }
    batches = [b for b in load_spend(path) if b["batch_id"] != batch_id] + [entry]
    doc = {
        "command": "uv run lineage label-model --batch",
        "budget": BUDGET,
        "spent": round(sum(float(b["dollars"]) for b in batches), 4),
        "batches": batches,
    }
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc

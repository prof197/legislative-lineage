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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from lineage.labels import MODEL, Item, model_labeler, scale_lines

MODEL_ID = "claude-opus-5-5"
PROMPT_VERSION = "v1"
_MAX_TOKENS = 4000
# Standard prices in dollars per million tokens, input and output, from the platform's model
# list as of November 2026; the Batches API charges half. A model not listed is not projected.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}
# Fewer characters a token than English prose usually takes, so the projection errs high.
CHARS_PER_TOKEN = 3.5
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


def projected_cost(
    items: Sequence[Item],
    *,
    model: str = MODEL_ID,
    max_tokens: int = _MAX_TOKENS,
    batch: bool = False,
) -> float:
    """An upper estimate, in dollars, of asking every pair: each prompt's tokens from its
    characters, and each reply, thinking included, at its token limit."""
    if model not in PRICES:
        raise ValueError(f"no price for {model}; add it to PRICES before asking it")
    input_price, output_price = PRICES[model]
    chars = sum(len(SYSTEM) + len(user_message(i)) for i in items)
    cost = (chars / CHARS_PER_TOKEN * input_price + len(items) * max_tokens * output_price) / 1e6
    return cost / 2 if batch else cost


def model_pending(items: Sequence[Item], labels: Sequence[Mapping[str, Any]]) -> list[Item]:
    """The pairs no model has labeled under the current prompt version."""
    suffix = f"@{PROMPT_VERSION}"
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
    items: Sequence[Item], *, ask: Ask, labels_path: Path, today: str
) -> tuple[int, int]:
    """Ask for each pair's label in turn and append it as it comes. Returns how many were
    labeled and how many replies were unusable."""
    labeled = unusable = 0
    for item in items:
        try:
            label, served_by = ask(SYSTEM, user_message(item))
        except UnusableReplyError:
            unusable += 1
            continue
        row = {
            "query": item.query.ref.as_dict(),
            "match": item.match.ref.as_dict(),
            "label": label,
            "labeler": model_labeler(served_by, PROMPT_VERSION),
            "at": today,
        }
        with labels_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        labeled += 1
    return labeled, unusable


def anthropic_ask(model: str = MODEL_ID) -> Ask:
    """Ask through the Anthropic SDK with structured output from ModelLabel. The key comes from
    the environment, as every secret here does."""
    import anthropic

    client = anthropic.Anthropic()

    def ask(system: str, user: str) -> tuple[int, str]:
        try:
            response = client.beta.messages.parse(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=ModelLabel,
                output_config={"effort": "medium"},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except ValidationError as e:
            raise UnusableReplyError(f"the reply does not fit the schema: {e}") from e
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise UnusableReplyError(f"no usable reply (stop reason {response.stop_reason})")
        return response.parsed_output.label, response.model

    return ask

"""One structured-output request to the model, every reply metered, usable or not.

The reply is asked for as JSON fitting a Pydantic class's schema (`output_config.format`) and
validated against the class in code. The schema sent leaves out the length bounds constrained
decoding does not enforce (maxLength, maxItems and the like); the class still checks them, so a
reply over a bound is unusable, with its reason, rather than an error that loses the call's
usage. The SDK's `parse` helper raised on such a reply before returning it, so its tokens went
unrecorded: 85 of the first explanation run's 173 calls (DESIGN.md#explanations). Declines are
rerun server-side on the fallback model the API picks for the refusal's category, and the reply
records which model served it.
"""

import json
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ValidationError

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Bounds the API's schema subset does not take; the Pydantic class checks them on the reply.
_UNSENT = frozenset({"maxLength", "minLength", "maxItems", "minItems", "title"})


class UnusableReplyError(Exception):
    """The model declined, or its reply did not fit the schema; with what the call cost."""

    def __init__(
        self,
        why: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        served_by: str | None = None,
    ) -> None:
        super().__init__(why)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.served_by = served_by


def sent_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The class's JSON schema without the bounds the API does not take, closed to extra keys."""

    def strip(node: object) -> object:
        if isinstance(node, dict):
            items = cast(dict[str, object], node)
            out: dict[str, object] = {k: strip(v) for k, v in items.items() if k not in _UNSENT}
            if out.get("type") == "object":
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [strip(v) for v in cast(list[object], node)]
        return node

    schema = cast(dict[str, Any], strip(model.model_json_schema()))
    return schema


@dataclass(frozen=True, slots=True)
class Structured[T: BaseModel]:
    value: T
    served_by: str
    input_tokens: int
    output_tokens: int


def reply_text(content: list[Any]) -> str | None:
    """The reply's JSON: its last text block, after any fallback block."""
    texts = [b.text for b in content if getattr(b, "type", None) == "text"]
    return texts[-1] if texts else None


def validated[T: BaseModel](
    cls: type[T], text: str | None, stop_reason: str | None, tokens: tuple[int, int], model: str
) -> Structured[T]:
    """The reply as `cls`, or UnusableReplyError saying why, with the call's usage."""
    if stop_reason == "refusal":
        raise UnusableReplyError("declined", *tokens, model)
    if stop_reason == "max_tokens":
        raise UnusableReplyError("cut off at the token limit", *tokens, model)
    if text is None:
        raise UnusableReplyError(
            f"no text in the reply (stop reason {stop_reason})", *tokens, model
        )
    try:
        value = cls.model_validate(json.loads(text))
    except json.JSONDecodeError as e:
        raise UnusableReplyError(f"not JSON: {e.msg}", *tokens, model) from e
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise UnusableReplyError(f"off the schema: {problems}", *tokens, model) from e
    return Structured(value, model, *tokens)


def ask_structured[T: BaseModel](
    client: Any,
    cls: type[T],
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    effort: str,
) -> Structured[T]:
    """One request through the Anthropic SDK, with the server-side fallback for a decline."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": sent_schema(cls)},
        },
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    tokens = (int(response.usage.input_tokens), int(response.usage.output_tokens))
    return validated(
        cls, reply_text(list(response.content)), response.stop_reason, tokens, str(response.model)
    )

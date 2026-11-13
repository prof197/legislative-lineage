"""What M6's model calls cost, recorded as they are made, against one cap.

Explanations and the faithfulness judge are asked one request at a time rather than in batches:
the API's server-side fallback, which reruns a declined request on another model, is not
offered on the Batches API. Each call's tokens are added to the run's row in the ledger as soon
as it returns, so a run stopped part way still records what it spent. A run is projected before
anything is sent, every prompt's tokens from its characters and every reply at its token limit,
each asked as often as it may be retried, and refused when what is spent and what is projected
together pass BUDGET. Before each call the same bound is checked again for that call alone, so
the ledger cannot pass the cap however the run goes. The ledger keeps bounds and actual spend
apart: each run's row records the bound it was projected at beside the dollars its recorded
usage comes to, and the totals give what usage records (metered) apart from the bound counted
for calls whose usage was lost (bounded); the cap counts both. See DESIGN.md#explanations.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lineage.model_labels import CHARS_PER_TOKEN, PRICES

SPEND = Path("reports/explanation_spend.json")
# Dollars every M6 model call may cost together, explanations and judge, spent and projected,
# for the featured scope only: the M5 review set $30, the M6 review raised it to $40 to ask
# again for the withheld explanations and the changed claims the grounding rules now drop, and
# the faithfulness ruling raised it to $55 so the last re-ask and judge run fit their bounds.
BUDGET = 55.0


class BudgetError(Exception):
    pass


def _prices(model: str) -> tuple[float, float]:
    if model not in PRICES:
        raise BudgetError(f"no price for {model}; add it to lineage.model_labels.PRICES first")
    return PRICES[model]


def dollars(model: str, input_tokens: int, output_tokens: int) -> float:
    """What a call's tokens cost at the standard price."""
    input_price, output_price = _prices(model)
    return (input_tokens * input_price + output_tokens * output_price) / 1e6


def upper_bound(model: str, prompt_chars: Sequence[int], max_tokens: int, attempts: int) -> float:
    """At most what asking each prompt costs: its tokens from its characters, its reply at the
    token limit, asked `attempts` times."""
    input_price, output_price = _prices(model)
    one = sum(c / CHARS_PER_TOKEN * input_price + max_tokens * output_price for c in prompt_chars)
    return one * attempts / 1e6


def load_runs(path: Path = SPEND) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text())["runs"])


def spent(path: Path = SPEND) -> float:
    return sum(float(r["dollars"]) for r in load_runs(path))


def check_projection(projected: float, path: Path = SPEND, budget: float = BUDGET) -> None:
    """Refuse a run, or a call, whose projection with what is spent passes the budget."""
    so_far = spent(path)
    if so_far + projected > budget:
        raise BudgetError(
            f"projected ${projected:.2f} with ${so_far:.2f} spent passes the ${budget:.2f} "
            "budget for M6's model calls"
        )


def _write(path: Path, runs: list[dict[str, Any]]) -> None:
    metered = sum(float(r["dollars"]) for r in runs if r.get("metered", True))
    bounded = sum(float(r["dollars"]) for r in runs if not r.get("metered", True))
    doc = {
        "budget": BUDGET,
        "metered": round(metered, 4),
        "bounded": round(bounded, 4),
        "spent": round(metered + bounded, 4),
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")


@dataclass
class Meter:
    """One run's row in the ledger, updated after every call."""

    path: Path
    purpose: str  # "explain" or "judge"
    model: str
    prompt: str
    today: str
    projected: float | None = None  # the run's bound before anything was sent
    run: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    served_by: dict[str, int] = field(default_factory=dict[str, int])

    def __post_init__(self) -> None:
        if not self.run:
            self.run = f"{self.purpose}-{self.today}-{len(load_runs(self.path)) + 1}"

    def before_call(self, prompt_chars: int, max_tokens: int) -> None:
        check_projection(upper_bound(self.model, [prompt_chars], max_tokens, 1), self.path)

    def add(self, input_tokens: int, output_tokens: int, served_by: str | None) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if served_by is not None:
            self.served_by[served_by] = self.served_by.get(served_by, 0) + 1
        row = {
            "run": self.run,
            "purpose": self.purpose,
            "model": self.model,
            "prompt": self.prompt,
            "projected": None if self.projected is None else round(self.projected, 4),
            "metered": True,
            "calls": self.calls,
            "served_by": dict(sorted(self.served_by.items())),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "dollars": round(dollars(self.model, self.input_tokens, self.output_tokens), 4),
            "at": self.today,
        }
        runs = [r for r in load_runs(self.path) if r["run"] != self.run] + [row]
        _write(self.path, runs)

"""The text layer's rules, read from rules/text_layer.toml.

What counts as usable text, how each state's bill versions are fetched (link format, PDF
reading, a site's content element), which states are left out and why, and where fetched text
replaces Open States' text. The file holds the decisions; this module applies them, and checks
them against the quality test's report. See DESIGN.md#dr-6.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lineage.sections import clean

RULES = Path("rules/text_layer.toml")
FORMATS = frozenset({"html", "pdf"})
READINGS = frozenset({"plain", "layout"})
SOURCES = frozenset({"legislature", "bulk"})

_TOKEN = re.compile(r"\S+")
# A real word: letters only, with a vowel, two or more of them. Text a PDF's font encoding
# garbles, or a page of numbers and codes, falls below the share.
_REAL_WORD = re.compile(r"[A-Za-z]*[AEIOUYaeiouy][A-Za-z]*")
_PUNCTUATION = ".,;:()[]\"'!?\u201c\u201d\u2018\u2019"


@dataclass(frozen=True, slots=True)
class StateRule:
    fetch: bool
    source: str  # "legislature" (each record's linked version) or "bulk" (the state's download)
    formats: tuple[str, ...]  # link formats in order of preference; a record lacking the first
    # is fetched in the next. Empty when not fetched from the legislature's links.
    reading: str  # a PDF's reading: "plain" (text order) or "layout" (word positions)
    reason: str | None  # why the state is not fetched

    @property
    def format(self) -> str | None:
        return self.formats[0] if self.formats else None


@dataclass(frozen=True, slots=True)
class TextRules:
    min_words: int
    min_real_word_share: float
    min_usable_rate: float
    html_margin: float
    replace_margin: float
    replace_min_compared: int
    states: dict[str, StateRule]
    containers: dict[str, str]  # host -> its content element, "tag#id" or "tag.class"
    retired: dict[str, str]  # host -> why its links no longer lead to bill text


def load_rules(path: Path = RULES) -> TextRules:
    raw = tomllib.loads(path.read_text())
    states = {state: _state_rule(state, rule) for state, rule in raw.get("states", {}).items()}
    hosts: dict[str, dict[str, str]] = raw.get("hosts", {})
    containers = {host: rule["container"] for host, rule in hosts.items() if "container" in rule}
    retired = {host: rule["retired"] for host, rule in hosts.items() if "retired" in rule}
    return TextRules(
        min_words=raw["usable"]["min_words"],
        min_real_word_share=raw["usable"]["min_real_word_share"],
        min_usable_rate=raw["fetch"]["min_usable_rate"],
        html_margin=raw["fetch"]["html_margin"],
        replace_margin=raw["replace"]["margin"],
        replace_min_compared=raw["replace"]["min_compared"],
        states=states,
        containers=containers,
        retired=retired,
    )


def _state_rule(state: str, rule: dict[str, Any]) -> StateRule:
    fetch = rule.get("fetch", True)
    source = rule.get("source", "legislature")
    formats = tuple(rule.get("formats", [rule["format"]] if "format" in rule else []))
    reading = rule.get("reading", "plain")
    reason = rule.get("reason")
    if source not in SOURCES or reading not in READINGS or not set(formats) <= FORMATS:
        raise ValueError(f"{state}: unknown source, format or reading in {rule}")
    if fetch and source == "legislature" and not formats:
        raise ValueError(f"{state}: a fetched state names its format, html or pdf")
    if not fetch and not reason:
        raise ValueError(f"{state}: a state left out says why")
    return StateRule(fetch, source, formats, reading, reason)


def usable(text: str, rules: TextRules) -> bool:
    """Non-trivial length, mostly real words: the bar fetched and stored text must clear."""
    tokens = _TOKEN.findall(clean(text))
    if len(tokens) < rules.min_words:
        return False
    real = 0
    for token in tokens:
        word = token.strip(_PUNCTUATION)
        real += len(word) >= 2 and _REAL_WORD.fullmatch(word) is not None
    return real >= rules.min_real_word_share * len(tokens)


def replaced_states(report: dict[str, Any], rules: TextRules) -> list[str]:
    """States whose fetched text replaces Open States' stored text.

    On the sampled records that have both, our clean rate (with the state's format and reading)
    must exceed theirs by the margin, over enough records to tell.
    """
    chosen: list[str] = []
    for state, rule in sorted(rules.states.items()):
        cell = report["states"].get(state, {}).get(rule.format or "")
        if not rule.fetch or not cell:
            continue
        ours = cell["readings"].get(rule.reading, {}).get("clean_rate", {}).get("with_text")
        theirs = cell["theirs"]["clean_rate"]
        if ours is None or theirs is None or cell["compared"] < rules.replace_min_compared:
            continue
        # Rates are stored to three decimals; compare at that precision, not in binary float.
        if round(ours - theirs, 3) >= rules.replace_margin:
            chosen.append(state)
    return chosen

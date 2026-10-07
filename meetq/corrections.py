"""Term corrections: fix words the STT model consistently mishears.

Homophones are the typical case — 仕様 and 使用 are both しよう, 移行 and 以降 are both
いこう, and the model tends to pick the everyday word. A rule list like

    使用 => 仕様
    以降 => 移行
    使用料 => 使用料

is applied to the raw STT text. The longest match wins, so a rule whose two sides are
equal protects a term from a shorter rule (「使用料」 stays as it is). Segments keep
that raw text, so changing or removing a rule and re-applying restores the original.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SEPARATORS = ("=>", "→", "->")
EXAMPLES_PER_RULE = 3
CONTEXT_CHARS = 14


@dataclass
class Rule:
    wrong: str
    right: str


@dataclass
class RulePreview:
    wrong: str
    right: str
    count: int = 0
    examples: list[dict] = field(default_factory=list)


def parse_rules(text: str) -> tuple[list[Rule], list[str]]:
    """One rule per line, `wrong => right`. Blank lines and `#` comments are ignored."""
    rules: list[Rule] = []
    errors: list[str] = []
    seen: set[str] = set()
    for number, line in enumerate((text or "").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        separator = next((sep for sep in SEPARATORS if sep in stripped), None)
        if separator is None:
            errors.append(f"{number} 行目: 「誤 => 正」の形で書いてください: {stripped}")
            continue
        wrong, _, right = stripped.partition(separator)
        wrong, right = wrong.strip(), right.strip()
        if not wrong or not right:
            errors.append(f"{number} 行目: 左右どちらかが空です: {stripped}")
            continue
        if wrong in seen:
            errors.append(f"{number} 行目: 「{wrong}」のルールが重複しています")
            continue
        seen.add(wrong)
        rules.append(Rule(wrong=wrong, right=right))
    return rules, errors


def _pattern(rules: list[Rule]) -> re.Pattern | None:
    if not rules:
        return None
    # longest first so 「使用策定」 wins over 「使用」 when both are rules
    ordered = sorted((r.wrong for r in rules), key=len, reverse=True)
    return re.compile("|".join(re.escape(w) for w in ordered))


def apply_rules(text: str, rules: list[Rule]) -> tuple[str, int]:
    """Replace every rule in a single pass (no cascades). Returns (text, replacements)."""
    pattern = _pattern(rules)
    if pattern is None or not text:
        return text, 0
    mapping = {r.wrong: r.right for r in rules}
    count = 0

    def substitute(match: re.Match) -> str:
        nonlocal count
        replacement = mapping[match.group(0)]
        if replacement != match.group(0):  # protecting rules change nothing
            count += 1
        return replacement

    return pattern.sub(substitute, text), count


def apply_text(text: str, rules_text: str) -> str:
    rules, _ = parse_rules(rules_text)
    return apply_rules(text, rules)[0]


def targets(rules_text: str) -> list[str]:
    """The corrected terms, fed to STT as keywords so they are heard right next time."""
    rules, _ = parse_rules(rules_text)
    out: list[str] = []
    for rule in rules:
        if rule.right not in out:
            out.append(rule.right)
    return out


def preview(segments: list[dict], rules: list[Rule]) -> list[RulePreview]:
    """How many times each rule fires on the raw text, with a little context."""
    pattern = _pattern(rules)
    previews = {r.wrong: RulePreview(wrong=r.wrong, right=r.right) for r in rules}
    if pattern is None:
        return list(previews.values())
    for seg in segments:
        raw = seg.get("raw_text") or seg.get("text") or ""
        for match in pattern.finditer(raw):
            item = previews[match.group(0)]
            item.count += 1
            if len(item.examples) < EXAMPLES_PER_RULE:
                start = max(0, match.start() - CONTEXT_CHARS)
                end = min(len(raw), match.end() + CONTEXT_CHARS)
                item.examples.append(
                    {
                        "segment_id": seg.get("id"),
                        "start_sec": seg.get("start_sec", 0.0),
                        "before": raw[start : match.start()],
                        "match": match.group(0),
                        "after": raw[match.end() : end],
                    }
                )
    return [previews[r.wrong] for r in rules]

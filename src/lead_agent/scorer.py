"""Hybrid ICP scoring: deterministic hard-filter rules plus LLM-judged soft signals.

Hard filters gate (a None field value fails — lenient extraction couldn't confirm it).
Under the 'gate' policy a hard-filter failure short-circuits before any LLM call, so
disqualified firms cost nothing. Soft signals are rated 1-10 in a single batched call
per firm, normalized to [0,1], and combined by weight. The pipeline persists score and
score_breakdown; this module only computes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .llm import LLMSettings

if TYPE_CHECKING:
    from .config import HardFilter, ICPConfig, SoftSignal
    from .llm import CallStats, LLMClient


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    score: float
    qualified: bool
    passed_hard_filters: bool
    signal_ratings: dict[str, int]
    breakdown: dict[str, Any]
    stats: list[CallStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hard filters (pure)
# ---------------------------------------------------------------------------

def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None  # don't treat True/False as 1/0 for numeric comparisons
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _apply_operator(operator: str, field_value: object, filter_value: object) -> bool:
    """Evaluate one hard-filter operator. A missing (None) field value always fails."""
    if field_value is None:
        return False

    if operator in ("gte", "lte"):
        a, b = _as_number(field_value), _as_number(filter_value)
        if a is None or b is None:
            return False
        return a >= b if operator == "gte" else a <= b

    if operator == "between":
        if not isinstance(filter_value, list) or len(filter_value) != 2:
            return False
        a = _as_number(field_value)
        lo, hi = _as_number(filter_value[0]), _as_number(filter_value[1])
        if a is None or lo is None or hi is None:
            return False
        return lo <= a <= hi

    if operator == "eq":
        if isinstance(field_value, str) and isinstance(filter_value, str):
            return field_value.strip().lower() == filter_value.strip().lower()
        return field_value == filter_value

    if operator == "in":
        if not isinstance(filter_value, list):
            return False
        needle = field_value.strip().lower() if isinstance(field_value, str) else field_value
        haystack = [v.strip().lower() if isinstance(v, str) else v for v in filter_value]
        return needle in haystack

    if operator == "contains":
        needle = str(filter_value).strip().lower()
        if isinstance(field_value, list):
            return any(needle in str(item).lower() for item in field_value)
        if isinstance(field_value, str):
            return needle in field_value.lower()
        return False

    return False


def evaluate_hard_filters(
    profile: dict[str, Any] | None, filters: list[HardFilter]
) -> tuple[bool, list[dict[str, Any]]]:
    """Return (all_passed, per-filter detail). A None field value fails its filter."""
    details: list[dict[str, Any]] = []
    all_passed = True
    for hf in filters:
        field_value = profile.get(hf.field) if profile else None
        passed = _apply_operator(hf.operator, field_value, hf.value)
        details.append(
            {
                "field": hf.field,
                "operator": hf.operator,
                "value": hf.value,
                "field_value": field_value,
                "passed": passed,
            }
        )
        if not passed:
            all_passed = False
    return all_passed, details


# ---------------------------------------------------------------------------
# Soft signals (LLM, batched)
# ---------------------------------------------------------------------------

class SignalRating(BaseModel):
    name: str
    rating: int


class SignalRatings(BaseModel):
    ratings: list[SignalRating]


_SCORING_SYSTEM = (
    "You evaluate a firm against specific scoring criteria using only its website text. "
    "For each criterion, return an integer rating from 1 to 10 following its instruction. "
    "Return a rating for every criterion, keyed by its exact name."
)


def _clamp_rating(rating: int) -> int:
    return max(1, min(10, int(rating)))


def _build_scoring_prompt(icp: ICPConfig, combined_text: str, *, max_chars: int) -> str:
    lines = [
        "Rate the firm below against each criterion. Return an integer 1-10 for every "
        "criterion, keyed by its exact name.",
        "",
        "Criteria:",
    ]
    for signal in icp.soft_signals:
        lines.append(f"- name: {signal.name}\n  instruction: {signal.prompt.strip()}")
    lines.extend(["", "Firm website text:", combined_text[:max_chars]])
    return "\n".join(lines)


async def rate_soft_signals(
    combined_text: str,
    icp: ICPConfig,
    client: LLMClient,
    *,
    max_chars: int | None = None,
) -> tuple[dict[str, int], list[CallStats]]:
    """One batched LLM call rating all soft signals. Missing ratings default to 1; clamped 1-10."""
    if max_chars is None:
        max_chars = LLMSettings().llm_input_max_chars
    prompt = _build_scoring_prompt(icp, combined_text, max_chars=max_chars)
    response = await client.extract(prompt, SignalRatings, system=_SCORING_SYSTEM)
    returned = {r.name: _clamp_rating(r.rating) for r in response.content.ratings}
    ratings = {signal.name: returned.get(signal.name, 1) for signal in icp.soft_signals}
    return ratings, [response.stats]


# ---------------------------------------------------------------------------
# Combine (pure)
# ---------------------------------------------------------------------------

def combine_score(
    ratings: dict[str, int], soft_signals: list[SoftSignal], normalization: str
) -> tuple[float, dict[str, Any]]:
    """Combine 1-10 ratings into a [0,1] score. Each rating normalizes as rating/10."""
    detail: dict[str, Any] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for signal in soft_signals:
        raw = ratings.get(signal.name, 1)
        normalized = raw / 10.0
        contribution = signal.weight * normalized
        weighted_sum += contribution
        weight_total += signal.weight
        detail[signal.name] = {
            "rating": raw,
            "normalized": normalized,
            "weight": signal.weight,
            "contribution": contribution,
        }
    if normalization == "weighted_average":
        score = weighted_sum / weight_total if weight_total > 0 else 0.0
    else:  # "sum"
        score = weighted_sum
    return max(0.0, min(1.0, score)), detail


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def score_firm(
    profile: dict[str, Any] | None,
    combined_text: str,
    icp: ICPConfig,
    client: LLMClient,
) -> ScoreResult:
    """Score one firm: hard-filter gate, then batched soft-signal rating and weighted combine."""
    passed, filter_detail = evaluate_hard_filters(profile, icp.hard_filters)
    policy = icp.scoring.hard_filter_policy
    breakdown: dict[str, Any] = {"policy": policy, "hard_filters": filter_detail}

    if policy == "gate" and not passed:
        breakdown["soft_signals"] = {}
        breakdown["reason"] = "failed hard filters"
        breakdown["score"] = 0.0
        return ScoreResult(
            score=0.0,
            qualified=False,
            passed_hard_filters=False,
            signal_ratings={},
            breakdown=breakdown,
        )

    ratings, stats = await rate_soft_signals(combined_text, icp, client)
    score, soft_detail = combine_score(
        ratings, icp.soft_signals, icp.scoring.soft_signal_normalization
    )
    breakdown["soft_signals"] = soft_detail
    breakdown["score"] = score

    qualified = score >= icp.scoring.min_qualify_score
    if policy == "gate":
        qualified = qualified and passed
    return ScoreResult(
        score=score,
        qualified=qualified,
        passed_hard_filters=passed,
        signal_ratings=ratings,
        breakdown=breakdown,
        stats=stats,
    )

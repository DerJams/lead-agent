"""Tests for hybrid ICP scoring: hard-filter operators, soft-signal rating, and combine."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from lead_agent.config import ICPConfig
from lead_agent.llm import CallStats, LLMResponse
from lead_agent.scorer import (
    ScoreResult,
    SignalRating,
    SignalRatings,
    _apply_operator,
    combine_score,
    evaluate_hard_filters,
    rate_soft_signals,
    score_firm,
)

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

_BASE_ICP: dict = {
    "name": "Test ICP",
    "description": "A minimal test ICP.",
    "search_queries": {
        "templates": ["{city} test"],
        "geo_focus": ["Dallas"],
        "negative_keywords": [],
    },
    "extraction_schema": [
        {"name": "firm_name", "type": "string", "description": "Name"},
        {"name": "attorney_count", "type": "integer", "description": "Count"},
        {"name": "practice_areas", "type": "list", "description": "Areas"},
    ],
    "hard_filters": [{"field": "attorney_count", "operator": "between", "value": [3, 15]}],
    "soft_signals": [
        {"name": "sig_a", "description": "A", "weight": 0.5, "prompt": "Rate A 1-10."},
        {"name": "sig_b", "description": "B", "weight": 0.5, "prompt": "Rate B 1-10."},
    ],
    "scoring": {
        "hard_filter_policy": "gate",
        "soft_signal_normalization": "weighted_average",
        "min_qualify_score": 0.55,
    },
    "output_fields": ["score", "sig_a", "sig_b"],
}


def make_icp(
    *,
    hard_filters: list[dict] | None = None,
    soft_signals: list[dict] | None = None,
    scoring: dict | None = None,
) -> ICPConfig:
    d = copy.deepcopy(_BASE_ICP)
    if hard_filters is not None:
        d["hard_filters"] = hard_filters
    if soft_signals is not None:
        d["soft_signals"] = soft_signals
        d["output_fields"] = ["score"] + [s["name"] for s in soft_signals]
    if scoring is not None:
        d["scoring"].update(scoring)
    return ICPConfig.model_validate(d)


class FakeLLM:
    """extract() returns responder(prompt, response_model); records calls."""

    def __init__(self, responder: Callable[[str, type[BaseModel]], BaseModel]) -> None:
        self._responder = responder
        self.calls: list[tuple[str, type[BaseModel]]] = []

    async def extract(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system: str = "",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> LLMResponse[BaseModel]:
        self.calls.append((prompt, response_model))
        content = self._responder(prompt, response_model)
        stats = CallStats(
            model="fake", prompt_tokens=9, completion_tokens=3, cost_usd=0.0, duration_ms=1
        )
        return LLMResponse(content=content, stats=stats)


def ratings_responder(values: dict[str, int]) -> Callable[[str, type[BaseModel]], BaseModel]:
    def responder(_: str, __: type[BaseModel]) -> BaseModel:
        return SignalRatings(ratings=[SignalRating(name=k, rating=v) for k, v in values.items()])

    return responder


# ---------------------------------------------------------------------------
# _apply_operator
# ---------------------------------------------------------------------------

class TestApplyOperator:
    def test_none_field_always_fails(self) -> None:
        assert _apply_operator("gte", None, 3) is False
        assert _apply_operator("contains", None, "x") is False

    def test_gte(self) -> None:
        assert _apply_operator("gte", 5, 3) is True
        assert _apply_operator("gte", 3, 3) is True
        assert _apply_operator("gte", 2, 3) is False

    def test_lte(self) -> None:
        assert _apply_operator("lte", 3, 5) is True
        assert _apply_operator("lte", 6, 5) is False

    def test_between_inclusive(self) -> None:
        assert _apply_operator("between", 3, [3, 15]) is True
        assert _apply_operator("between", 15, [3, 15]) is True
        assert _apply_operator("between", 16, [3, 15]) is False
        assert _apply_operator("between", 2, [3, 15]) is False

    def test_between_malformed_value_fails(self) -> None:
        assert _apply_operator("between", 5, [3]) is False

    def test_numeric_coercion_from_string(self) -> None:
        assert _apply_operator("between", "7", [3, 15]) is True
        assert _apply_operator("gte", "not a number", 3) is False

    def test_eq_string_case_insensitive(self) -> None:
        assert _apply_operator("eq", "Texas", "texas") is True
        assert _apply_operator("eq", "Texas", "Florida") is False

    def test_in_membership(self) -> None:
        assert _apply_operator("in", "TX", ["tx", "ca"]) is True
        assert _apply_operator("in", "NY", ["tx", "ca"]) is False

    def test_contains_in_list_case_insensitive(self) -> None:
        areas = ["Commercial Real Estate Transactions", "Leasing"]
        assert _apply_operator("contains", areas, "commercial real estate") is True
        assert _apply_operator("contains", areas, "family law") is False

    def test_contains_in_string(self) -> None:
        assert _apply_operator("contains", "We handle CRE deals", "cre") is True
        assert _apply_operator("contains", "We handle CRE deals", "tax") is False

    def test_boolean_not_treated_as_number(self) -> None:
        assert _apply_operator("gte", True, 1) is False


# ---------------------------------------------------------------------------
# evaluate_hard_filters
# ---------------------------------------------------------------------------

class TestEvaluateHardFilters:
    def test_all_pass(self) -> None:
        icp = make_icp()
        profile = {"attorney_count": 7, "practice_areas": ["CRE"]}
        passed, detail = evaluate_hard_filters(profile, icp.hard_filters)
        assert passed is True
        assert detail[0]["passed"] is True

    def test_one_failure_fails_overall(self) -> None:
        icp = make_icp(
            hard_filters=[
                {"field": "attorney_count", "operator": "between", "value": [3, 15]},
                {"field": "practice_areas", "operator": "contains", "value": "cre"},
            ]
        )
        profile = {"attorney_count": 7, "practice_areas": ["Family Law"]}
        passed, detail = evaluate_hard_filters(profile, icp.hard_filters)
        assert passed is False
        assert [d["passed"] for d in detail] == [True, False]

    def test_missing_field_fails(self) -> None:
        icp = make_icp()
        passed, _ = evaluate_hard_filters({"practice_areas": ["CRE"]}, icp.hard_filters)
        assert passed is False

    def test_none_profile_fails(self) -> None:
        icp = make_icp()
        passed, _ = evaluate_hard_filters(None, icp.hard_filters)
        assert passed is False

    def test_detail_records_field_value(self) -> None:
        icp = make_icp()
        _, detail = evaluate_hard_filters({"attorney_count": 7}, icp.hard_filters)
        assert detail[0]["field"] == "attorney_count"
        assert detail[0]["field_value"] == 7


# ---------------------------------------------------------------------------
# combine_score
# ---------------------------------------------------------------------------

class TestCombineScore:
    def test_weighted_average(self) -> None:
        icp = make_icp()
        score, _ = combine_score({"sig_a": 8, "sig_b": 6}, icp.soft_signals, "weighted_average")
        # 0.5*0.8 + 0.5*0.6 = 0.7
        assert score == pytest.approx(0.7)

    def test_sum(self) -> None:
        icp = make_icp()
        score, _ = combine_score({"sig_a": 10, "sig_b": 10}, icp.soft_signals, "sum")
        # 0.5*1.0 + 0.5*1.0 = 1.0
        assert score == pytest.approx(1.0)

    def test_missing_rating_defaults_to_one(self) -> None:
        icp = make_icp()
        score, detail = combine_score({"sig_a": 10}, icp.soft_signals, "weighted_average")
        # sig_b missing -> rating 1 -> 0.1 ; 0.5*1.0 + 0.5*0.1 = 0.55
        assert detail["sig_b"]["rating"] == 1
        assert score == pytest.approx(0.55)

    def test_detail_structure(self) -> None:
        icp = make_icp()
        _, detail = combine_score({"sig_a": 8, "sig_b": 4}, icp.soft_signals, "weighted_average")
        assert detail["sig_a"] == {
            "rating": 8,
            "normalized": pytest.approx(0.8),
            "weight": 0.5,
            "contribution": pytest.approx(0.4),
        }


# ---------------------------------------------------------------------------
# rate_soft_signals
# ---------------------------------------------------------------------------

class TestRateSoftSignals:
    async def test_returns_ratings_keyed_by_name(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 7, "sig_b": 9}))
        ratings, stats = await rate_soft_signals("text", icp, llm)
        assert ratings == {"sig_a": 7, "sig_b": 9}
        assert len(stats) == 1

    async def test_clamps_out_of_range(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 0, "sig_b": 15}))
        ratings, _ = await rate_soft_signals("text", icp, llm)
        assert ratings == {"sig_a": 1, "sig_b": 10}

    async def test_missing_signal_defaults_to_one(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 8}))  # sig_b omitted by model
        ratings, _ = await rate_soft_signals("text", icp, llm)
        assert ratings == {"sig_a": 8, "sig_b": 1}

    async def test_prompt_contains_signal_instructions_and_text(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 5, "sig_b": 5}))
        await rate_soft_signals("UNIQUE_TEXT_MARKER", icp, llm)
        prompt = llm.calls[0][0]
        assert "sig_a" in prompt and "Rate A 1-10." in prompt
        assert "UNIQUE_TEXT_MARKER" in prompt

    async def test_combined_text_truncated_to_max_chars(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 5, "sig_b": 5}))
        head = "HEAD_MARKER" + "A" * 90
        tail = "B" * 200 + "TAIL_MARKER"
        await rate_soft_signals(head + tail, icp, llm, max_chars=len(head))
        prompt = llm.calls[0][0]
        assert "HEAD_MARKER" in prompt
        assert "TAIL_MARKER" not in prompt


# ---------------------------------------------------------------------------
# score_firm
# ---------------------------------------------------------------------------

class TestScoreFirm:
    async def test_gate_failure_short_circuits_no_llm_call(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 10, "sig_b": 10}))
        result = await score_firm({"attorney_count": 50}, "text", icp, llm)
        assert result.qualified is False
        assert result.passed_hard_filters is False
        assert result.score == 0.0
        assert result.signal_ratings == {}
        assert result.stats == []
        assert llm.calls == []  # disqualified firms cost nothing

    async def test_gate_pass_high_ratings_qualifies(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 8, "sig_b": 8}))
        result = await score_firm({"attorney_count": 7}, "text", icp, llm)
        assert result.passed_hard_filters is True
        assert result.score == pytest.approx(0.8)
        assert result.qualified is True
        assert len(llm.calls) == 1

    async def test_gate_pass_low_ratings_does_not_qualify(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 3, "sig_b": 3}))
        result = await score_firm({"attorney_count": 7}, "text", icp, llm)
        assert result.passed_hard_filters is True
        assert result.score == pytest.approx(0.3)
        assert result.qualified is False  # below min_qualify_score 0.55

    async def test_weighted_policy_does_not_gate(self) -> None:
        icp = make_icp(scoring={"hard_filter_policy": "weighted"})
        llm = FakeLLM(ratings_responder({"sig_a": 8, "sig_b": 8}))
        result = await score_firm({"attorney_count": 50}, "text", icp, llm)
        # hard filter fails, but weighted policy ignores the gate
        assert result.passed_hard_filters is False
        assert result.qualified is True
        assert len(llm.calls) == 1

    async def test_breakdown_structure(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 8, "sig_b": 8}))
        result = await score_firm({"attorney_count": 7}, "text", icp, llm)
        assert result.breakdown["policy"] == "gate"
        assert result.breakdown["hard_filters"][0]["passed"] is True
        assert set(result.breakdown["soft_signals"].keys()) == {"sig_a", "sig_b"}
        assert result.breakdown["score"] == pytest.approx(0.8)

    async def test_gate_failure_breakdown_records_reason(self) -> None:
        icp = make_icp()
        llm = FakeLLM(ratings_responder({"sig_a": 10, "sig_b": 10}))
        result = await score_firm({"attorney_count": 50}, "text", icp, llm)
        assert result.breakdown["reason"] == "failed hard filters"
        assert result.breakdown["soft_signals"] == {}


def test_score_result_is_dataclass() -> None:
    r = ScoreResult(
        score=0.7, qualified=True, passed_hard_filters=True,
        signal_ratings={"sig_a": 7}, breakdown={},
    )
    assert r.stats == []  # default factory

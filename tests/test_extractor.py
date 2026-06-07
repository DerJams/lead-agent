"""Tests for structured extraction: dynamic model building and extract_profile orchestration."""

from __future__ import annotations

import copy
from collections.abc import Callable

from pydantic import BaseModel

from lead_agent.config import ExtractionField, ICPConfig
from lead_agent.extractor import ExtractionResult, build_extraction_model, extract_profile
from lead_agent.llm import CallStats, LLMResponse

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

_VALID_ICP: dict = {
    "name": "Test ICP",
    "description": "A minimal test ICP.",
    "search_queries": {
        "templates": ["{city} test"],
        "geo_focus": ["Dallas"],
        "negative_keywords": [],
    },
    "extraction_schema": [
        {"name": "firm_name", "type": "string", "description": "Legal name", "required": True},
        {
            "name": "attorney_count",
            "type": "integer",
            "description": "Number of attorneys",
            "required": True,
        },
        {
            "name": "practice_areas",
            "type": "list",
            "description": "Practice areas",
            "required": True,
        },
        {"name": "contact_info", "type": "string", "description": "Contact", "required": False},
    ],
    "hard_filters": [{"field": "attorney_count", "operator": "between", "value": [3, 15]}],
    "soft_signals": [
        {"name": "spec", "description": "spec", "weight": 1.0, "prompt": "Rate 1-10."}
    ],
    "scoring": {
        "hard_filter_policy": "gate",
        "soft_signal_normalization": "weighted_average",
        "min_qualify_score": 0.55,
    },
    "output_fields": ["score", "spec"],
}


def make_icp(extraction_schema: list[dict] | None = None) -> ICPConfig:
    d = copy.deepcopy(_VALID_ICP)
    if extraction_schema is not None:
        d["extraction_schema"] = extraction_schema
    return ICPConfig.model_validate(d)


class FakeLLM:
    """Stands in for LLMClient; extract() returns responder(prompt, response_model)."""

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
            model="fake", prompt_tokens=11, completion_tokens=7, cost_usd=0.0, duration_ms=1
        )
        return LLMResponse(content=content, stats=stats)


# ---------------------------------------------------------------------------
# build_extraction_model
# ---------------------------------------------------------------------------

_ALL_TYPES = [
    ExtractionField(name="s", type="string", description="a string"),
    ExtractionField(name="i", type="integer", description="an integer"),
    ExtractionField(name="lst", type="list", description="a list"),
    ExtractionField(name="b", type="boolean", description="a boolean"),
]


class TestBuildExtractionModel:
    def test_field_names_match_schema(self) -> None:
        model = build_extraction_model(_ALL_TYPES)
        assert set(model.model_fields.keys()) == {"s", "i", "lst", "b"}

    def test_all_fields_nullable_empty_instance_valid(self) -> None:
        model = build_extraction_model(_ALL_TYPES)
        instance = model()  # no args -> must validate
        assert instance.model_dump() == {"s": None, "i": None, "lst": None, "b": None}

    def test_integer_coerced_from_string(self) -> None:
        model = build_extraction_model(_ALL_TYPES)
        instance = model.model_validate({"i": "7"})
        assert instance.i == 7

    def test_list_field_accepts_list_of_strings(self) -> None:
        model = build_extraction_model(_ALL_TYPES)
        instance = model.model_validate({"lst": ["a", "b"]})
        assert instance.lst == ["a", "b"]

    def test_boolean_field(self) -> None:
        model = build_extraction_model(_ALL_TYPES)
        assert model.model_validate({"b": True}).b is True

    def test_descriptions_attached_to_fields(self) -> None:
        model = build_extraction_model(_ALL_TYPES)
        assert model.model_fields["s"].description == "a string"
        assert model.model_fields["lst"].description == "a list"

    def test_required_flag_does_not_make_field_mandatory(self) -> None:
        # 'required: True' fields are still nullable under the lenient design
        schema = [ExtractionField(name="firm_name", type="string", description="x", required=True)]
        model = build_extraction_model(schema)
        assert model().firm_name is None


# ---------------------------------------------------------------------------
# extract_profile
# ---------------------------------------------------------------------------

class TestExtractProfile:
    async def test_returns_profile_dict_and_stats(self) -> None:
        icp = make_icp()

        def responder(_: str, model: type[BaseModel]) -> BaseModel:
            return model.model_validate(
                {"firm_name": "Acme Law", "attorney_count": 7, "practice_areas": ["CRE"]}
            )

        result = await extract_profile("scraped text", icp, FakeLLM(responder))
        assert result.ok
        assert result.profile["firm_name"] == "Acme Law"
        assert result.profile["attorney_count"] == 7
        assert result.profile["practice_areas"] == ["CRE"]
        assert result.profile["contact_info"] is None  # absent optional -> null
        assert result.stats.prompt_tokens == 11

    async def test_combined_text_included_in_prompt(self) -> None:
        icp = make_icp()
        llm = FakeLLM(lambda p, m: m())
        await extract_profile("UNIQUE_SCRAPE_MARKER body", icp, llm)
        assert "UNIQUE_SCRAPE_MARKER body" in llm.calls[0][0]

    async def test_source_url_included_when_given(self) -> None:
        icp = make_icp()
        llm = FakeLLM(lambda p, m: m())
        await extract_profile("text", icp, llm, source_url="https://acme.com/")
        assert "https://acme.com/" in llm.calls[0][0]

    async def test_required_fields_listed_in_prompt(self) -> None:
        icp = make_icp()
        llm = FakeLLM(lambda p, m: m())
        await extract_profile("text", icp, llm)
        prompt = llm.calls[0][0]
        # firm_name, attorney_count, practice_areas are required; contact_info is not
        assert "firm_name" in prompt
        assert "attorney_count" in prompt
        assert "contact_info" not in prompt

    async def test_extract_called_with_dynamic_model(self) -> None:
        icp = make_icp()
        llm = FakeLLM(lambda p, m: m())
        await extract_profile("text", icp, llm)
        _, response_model = llm.calls[0]
        assert issubclass(response_model, BaseModel)
        assert set(response_model.model_fields.keys()) == {
            "firm_name", "attorney_count", "practice_areas", "contact_info"
        }

    async def test_exception_returns_error_result(self) -> None:
        icp = make_icp()

        def boom(_: str, __: type[BaseModel]) -> BaseModel:
            raise ValueError("model exhausted retries")

        result = await extract_profile("text", icp, FakeLLM(boom))
        assert not result.ok
        assert result.profile is None
        assert "model exhausted retries" in result.error
        assert result.stats is None

    async def test_combined_text_truncated_to_max_chars(self) -> None:
        icp = make_icp()
        llm = FakeLLM(lambda p, m: m())
        head = "HEAD_MARKER" + "A" * 90
        tail = "B" * 200 + "TAIL_MARKER"
        long_text = head + tail
        await extract_profile(long_text, icp, llm, max_chars=len(head))
        prompt = llm.calls[0][0]
        assert "HEAD_MARKER" in prompt
        assert "TAIL_MARKER" not in prompt


class TestExtractionResult:
    def test_ok_true_when_profile_present_no_error(self) -> None:
        assert ExtractionResult(profile={"firm_name": "X"}).ok

    def test_ok_false_when_error(self) -> None:
        assert not ExtractionResult(profile=None, error="boom").ok

    def test_ok_false_when_profile_none(self) -> None:
        assert not ExtractionResult(profile=None).ok

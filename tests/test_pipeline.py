"""Tests for pipeline orchestration: fresh runs, error isolation, resume, stats, limit."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import BaseModel

from lead_agent.config import ICPConfig
from lead_agent.llm import CallStats, LLMResponse
from lead_agent.pipeline import RunResult, resume_run, run_pipeline
from lead_agent.scorer import SignalRating, SignalRatings
from lead_agent.scraper import ScrapedPage, ScrapeResult
from lead_agent.search import FilterBatchResult, FilterDecision, SearchResult
from lead_agent.storage import Storage

# ---------------------------------------------------------------------------
# ICP fixture
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


def make_icp() -> ICPConfig:
    return ICPConfig.model_validate(copy.deepcopy(_BASE_ICP))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = {"firm_name": "Firm", "attorney_count": 7, "practice_areas": ["cre"]}
_DEFAULT_RATINGS = {"sig_a": 8, "sig_b": 8}


class FakeLLM:
    """Dispatches extract() on response_model: filter / scoring / dynamic extraction model."""

    def __init__(
        self,
        *,
        profiles: dict[str, dict] | None = None,
        ratings: dict[str, dict] | None = None,
    ) -> None:
        self.profiles = profiles or {}
        self.ratings = ratings or {}
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
        content = self._respond(prompt, response_model)
        stats = CallStats(
            model="fake", prompt_tokens=10, completion_tokens=5, cost_usd=0.0, duration_ms=1
        )
        return LLMResponse(content=content, stats=stats)

    def _respond(self, prompt: str, model: type[BaseModel]) -> BaseModel:
        if model is FilterBatchResult:
            return FilterBatchResult(
                decisions=[FilterDecision(index=i, is_firm=True) for i in range(50)]
            )
        if model is SignalRatings:
            vals = self._match(prompt, self.ratings) or _DEFAULT_RATINGS
            return SignalRatings(ratings=[SignalRating(name=k, rating=v) for k, v in vals.items()])
        vals = self._match(prompt, self.profiles) or _DEFAULT_PROFILE
        return model.model_validate(vals)

    @staticmethod
    def _match(prompt: str, mapping: dict[str, dict]) -> dict | None:
        for key, val in mapping.items():
            if key in prompt:
                return val
        return None

    def extraction_calls_for(self, marker: str) -> int:
        return sum(
            1
            for prompt, model in self.calls
            if model not in (FilterBatchResult, SignalRatings) and marker in prompt
        )


class FakeProvider:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        return [
            SearchResult(url=u, title="t", snippet="s", score=0.5, query=query)
            for u in self.urls
        ]


class FakeScraper:
    def __init__(self, fail_markers: tuple[str, ...] = ()) -> None:
        self.fail_markers = fail_markers
        self.scraped: list[str] = []

    async def scrape_firm(self, url: str) -> ScrapeResult:
        self.scraped.append(url)
        if any(m in url for m in self.fail_markers):
            return ScrapeResult(url=url, error="scrape failed")
        return ScrapeResult(
            url=url,
            pages=[ScrapedPage(url=url, page_type="home", text="x")],
            combined_text=f"WEBSITE {url} content",
            bytes_fetched=100,
        )


async def _run(icp: ICPConfig, db: Storage, llm: FakeLLM, provider: FakeProvider,
               scraper: FakeScraper, **kwargs: object) -> RunResult:
    return await run_pipeline(
        icp, storage=db, client=llm, provider=provider, scraper=scraper,
        augment_queries=False, **kwargs,
    )


# ---------------------------------------------------------------------------
# Fresh run
# ---------------------------------------------------------------------------

class TestFreshRun:
    async def test_end_to_end_all_qualified(self, tmp_path: Path) -> None:
        icp = make_icp()
        provider = FakeProvider(["https://firm1.com/", "https://firm2.com/"])
        scraper = FakeScraper()
        async with Storage(tmp_path / "t.db") as db:
            result = await _run(icp, db, FakeLLM(), provider, scraper)
            run = await db.get_run(result.run_id)
        assert result.stage_counts == {"completed": 2}
        assert len(result.qualified) == 2
        assert all(f["score_breakdown"]["qualified"] for f in result.qualified)
        assert run["status"] == "completed"

    async def test_scored_sorted_descending(self, tmp_path: Path) -> None:
        icp = make_icp()
        provider = FakeProvider(["https://firm1.com/", "https://firm2.com/"])
        # firm2 rated lower than firm1
        llm = FakeLLM(ratings={"firm2.com": {"sig_a": 6, "sig_b": 6}})
        async with Storage(tmp_path / "t.db") as db:
            result = await _run(icp, db, llm, provider, FakeScraper())
        scores = [f["score"] for f in result.scored]
        assert scores == sorted(scores, reverse=True)
        assert result.scored[0]["url"] == "https://firm1.com/"

    async def test_firms_persisted_with_score_and_breakdown(self, tmp_path: Path) -> None:
        icp = make_icp()
        provider = FakeProvider(["https://firm1.com/"])
        async with Storage(tmp_path / "t.db") as db:
            result = await _run(icp, db, FakeLLM(), provider, FakeScraper())
            completed = await db.get_firms_by_stage(result.run_id, "completed")
        assert len(completed) == 1
        assert completed[0]["score"] == pytest.approx(0.8)
        assert "sig_a" in completed[0]["score_breakdown"]["soft_signals"]


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------

class TestErrorIsolation:
    async def test_one_firm_fails_others_complete(self, tmp_path: Path) -> None:
        icp = make_icp()
        provider = FakeProvider(["https://firm1.com/", "https://firm2.com/"])
        scraper = FakeScraper(fail_markers=("firm2.com",))
        async with Storage(tmp_path / "t.db") as db:
            result = await _run(icp, db, FakeLLM(), provider, scraper)
            failed = await db.get_firms_by_stage(result.run_id, "failed")
            run = await db.get_run(result.run_id)
        assert result.stage_counts == {"completed": 1, "failed": 1}
        assert len(result.qualified) == 1
        assert failed[0]["url"] == "https://firm2.com/"
        assert failed[0]["error"] == "scrape failed"
        # per-firm failure does not fail the whole run
        assert run["status"] == "completed"


# ---------------------------------------------------------------------------
# Gate disqualification
# ---------------------------------------------------------------------------

class TestGateDisqualification:
    async def test_disqualified_firm_completed_but_not_qualified(self, tmp_path: Path) -> None:
        icp = make_icp()
        provider = FakeProvider(["https://firm1.com/", "https://firm2.com/"])
        # firm2 has too many attorneys -> fails hard filter -> gate short-circuit
        llm = FakeLLM(profiles={"firm2.com": {"firm_name": "F2", "attorney_count": 50}})
        async with Storage(tmp_path / "t.db") as db:
            result = await _run(icp, db, llm, provider, FakeScraper())
            firms = {f["url"]: f for f in await db.get_firms_by_stage(result.run_id, "completed")}
        assert result.stage_counts == {"completed": 2}
        assert len(result.qualified) == 1
        assert result.qualified[0]["url"] == "https://firm1.com/"
        assert firms["https://firm2.com/"]["score"] == 0.0
        assert firms["https://firm2.com/"]["score_breakdown"]["qualified"] is False
        # gate short-circuit: 1 filter + firm1 (extract+score) + firm2 (extract only) = 4
        assert result.stats.llm_calls == 4


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------

class TestLimit:
    async def test_limit_caps_processed_firms(self, tmp_path: Path) -> None:
        icp = make_icp()
        provider = FakeProvider(
            ["https://firm1.com/", "https://firm2.com/", "https://firm3.com/"]
        )
        async with Storage(tmp_path / "t.db") as db:
            result = await _run(icp, db, FakeLLM(), provider, FakeScraper(), limit=2)
        assert sum(result.stage_counts.values()) == 2


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------

class TestStatsAggregation:
    async def test_aggregates_calls_tokens_and_bytes(self, tmp_path: Path) -> None:
        icp = make_icp()
        provider = FakeProvider(["https://firm1.com/", "https://firm2.com/"])
        async with Storage(tmp_path / "t.db") as db:
            result = await _run(icp, db, FakeLLM(), provider, FakeScraper())
        # 1 discovery filter call + 2 firms x (extract + score) = 5 calls
        assert result.stats.llm_calls == 5
        assert result.stats.prompt_tokens == 50
        assert result.stats.completion_tokens == 25
        assert result.stats.total_tokens == 75
        assert result.stats.bytes_scraped == 200


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume:
    async def test_resume_processes_only_non_terminal_and_reuses_profile(
        self, tmp_path: Path
    ) -> None:
        icp = make_icp()
        async with Storage(tmp_path / "t.db") as db:
            run_id = await db.create_run(icp.name)
            a = await db.add_firm(run_id, "https://firma.com/")  # already completed
            b = await db.add_firm(run_id, "https://firmb.com/")  # already failed
            c = await db.add_firm(run_id, "https://firmc.com/")  # extracted, resume reuses
            await db.add_firm(run_id, "https://firmd.com/")  # pending
            await db.update_firm_stage(
                a, "completed", score=0.9, score_breakdown={"qualified": True}
            )
            await db.update_firm_stage(b, "failed", error="prior error")
            await db.update_firm_stage(
                c, "extracted",
                extracted_profile={
                    "firm_name": "C", "attorney_count": 7, "practice_areas": ["cre"]
                },
            )

            llm = FakeLLM()
            scraper = FakeScraper()
            result = await resume_run(run_id, icp, storage=db, client=llm, scraper=scraper)

        # A (completed) and B (failed) untouched; C and D advanced to completed
        assert result.stage_counts == {"completed": 3, "failed": 1}
        # only non-terminal firms were scraped
        assert set(scraper.scraped) == {"https://firmc.com/", "https://firmd.com/"}
        # C reused its persisted profile (no extraction call); D was extracted fresh
        assert llm.extraction_calls_for("firmc.com") == 0
        assert llm.extraction_calls_for("firmd.com") == 1
        # resume stats: C (score only) + D (extract + score) = 3 calls, no discovery
        assert result.stats.llm_calls == 3

    async def test_resume_marks_run_completed(self, tmp_path: Path) -> None:
        icp = make_icp()
        async with Storage(tmp_path / "t.db") as db:
            run_id = await db.create_run(icp.name)
            await db.add_firm(run_id, "https://firm1.com/")
            result = await resume_run(
                run_id, icp, storage=db, client=FakeLLM(), scraper=FakeScraper()
            )
            run = await db.get_run(result.run_id)
        assert run["status"] == "completed"
        assert result.stage_counts.get("completed") == 1

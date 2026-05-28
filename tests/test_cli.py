"""Tests for the CLI: CSV row building, CSV writing, the run core, and arg/error handling."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from pydantic import BaseModel
from typer.testing import CliRunner

from lead_agent.cli import _async_run, app, build_rows, write_csv
from lead_agent.config import ICPConfig
from lead_agent.llm import CallStats, LLMResponse
from lead_agent.pipeline import RunResult, RunStats
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
    "output_fields": ["firm_name", "practice_areas", "score", "sig_a", "sig_b"],
}


def make_icp() -> ICPConfig:
    return ICPConfig.model_validate(copy.deepcopy(_BASE_ICP))


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "icp.yaml"
    path.write_text(yaml.safe_dump(_BASE_ICP), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fakes (offline pipeline dependencies)
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = {"firm_name": "Firm", "attorney_count": 7, "practice_areas": ["cre"]}
_DEFAULT_RATINGS = {"sig_a": 8, "sig_b": 8}


class FakeLLM:
    def __init__(self, *, profiles: dict[str, dict] | None = None) -> None:
        self.profiles = profiles or {}

    async def extract(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system: str = "",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> LLMResponse[BaseModel]:
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
            return SignalRatings(
                ratings=[SignalRating(name=k, rating=v) for k, v in _DEFAULT_RATINGS.items()]
            )
        for marker, profile in self.profiles.items():
            if marker in prompt:
                return model.model_validate(profile)
        return model.model_validate(_DEFAULT_PROFILE)


class FakeProvider:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(url=u, title="t", snippet="s", score=0.5, query=query)
            for u in self.urls
        ]


class FakeScraper:
    async def scrape_firm(self, url: str) -> ScrapeResult:
        return ScrapeResult(
            url=url,
            pages=[ScrapedPage(url=url, page_type="home", text="x")],
            combined_text=f"WEBSITE {url} content",
            bytes_fetched=100,
        )


# ---------------------------------------------------------------------------
# build_rows
# ---------------------------------------------------------------------------

def _firm(url: str, score: float, *, qualified: bool, profile: dict, ratings: dict) -> dict:
    return {
        "url": url,
        "score": score,
        "extracted_profile": profile,
        "score_breakdown": {
            "soft_signals": {k: {"rating": v} for k, v in ratings.items()},
            "qualified": qualified,
        },
    }


class TestBuildRows:
    def test_maps_output_fields(self) -> None:
        icp = make_icp()
        firm = _firm(
            "https://acme.com/", 0.8, qualified=True,
            profile={"firm_name": "Acme", "practice_areas": ["CRE", "Leasing"]},
            ratings={"sig_a": 8, "sig_b": 7},
        )
        result = RunResult("r", {"completed": 1}, qualified=[firm], scored=[firm], stats=RunStats())
        rows = build_rows(result, icp)
        assert rows == [
            {
                "firm_name": "Acme",
                "practice_areas": "CRE; Leasing",
                "score": "0.800",
                "sig_a": "8",
                "sig_b": "7",
            }
        ]

    def test_missing_values_render_empty(self) -> None:
        icp = make_icp()
        firm = {"url": "https://x.com/", "score": None, "extracted_profile": None,
                "score_breakdown": None}
        result = RunResult("r", {}, qualified=[firm], scored=[firm], stats=RunStats())
        rows = build_rows(result, icp)
        assert rows[0]["firm_name"] == ""
        assert rows[0]["score"] == ""
        assert rows[0]["sig_a"] == ""

    def test_include_all_includes_disqualified(self) -> None:
        icp = make_icp()
        good = _firm("https://a.com/", 0.8, qualified=True, profile=_DEFAULT_PROFILE,
                     ratings=_DEFAULT_RATINGS)
        bad = _firm("https://b.com/", 0.0, qualified=False, profile=_DEFAULT_PROFILE,
                    ratings=_DEFAULT_RATINGS)
        result = RunResult("r", {}, qualified=[good], scored=[good, bad], stats=RunStats())
        assert len(build_rows(result, icp)) == 1
        assert len(build_rows(result, icp, include_all=True)) == 2


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "out.csv"
        rows = [{"firm_name": "Acme", "score": "0.800"}]
        write_csv(rows, ["firm_name", "score"], path)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == "firm_name,score"
        assert lines[1] == "Acme,0.800"

    def test_empty_rows_writes_header_only(self, tmp_path: Path) -> None:
        path = tmp_path / "out.csv"
        write_csv([], ["firm_name", "score"], path)
        assert path.read_text(encoding="utf-8").splitlines() == ["firm_name,score"]


# ---------------------------------------------------------------------------
# _async_run (offline, injected deps)
# ---------------------------------------------------------------------------

class TestAsyncRun:
    async def test_end_to_end_writes_ranked_csv(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        out = tmp_path / "out.csv"
        result, out_path = await _async_run(
            config,
            db_path=tmp_path / "t.db",
            limit=50,
            output=out,
            resume=None,
            augment=False,
            include_all=False,
            client=FakeLLM(),
            provider=FakeProvider(["https://firm1.com/"]),
            scraper=FakeScraper(),
        )
        assert out_path == out
        assert len(result.qualified) == 1
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "firm_name,practice_areas,score,sig_a,sig_b"
        assert lines[1] == "Firm,cre,0.800,8,8"

    async def test_include_all_writes_disqualified_rows(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        out = tmp_path / "out.csv"
        llm = FakeLLM(profiles={"firm2.com": {"firm_name": "F2", "attorney_count": 50}})
        result, _ = await _async_run(
            config,
            db_path=tmp_path / "t.db",
            limit=50,
            output=out,
            resume=None,
            augment=False,
            include_all=True,
            client=llm,
            provider=FakeProvider(["https://firm1.com/", "https://firm2.com/"]),
            scraper=FakeScraper(),
        )
        assert len(result.qualified) == 1
        data_lines = out.read_text(encoding="utf-8").splitlines()[1:]
        assert len(data_lines) == 2  # both firms present with --all

    async def test_default_output_path_used_when_omitted(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        _, out_path = await _async_run(
            config,
            db_path=tmp_path / "t.db",
            limit=50,
            output=None,
            resume=None,
            augment=False,
            include_all=False,
            client=FakeLLM(),
            provider=FakeProvider(["https://firm1.com/"]),
            scraper=FakeScraper(),
        )
        assert out_path.suffix == ".csv"
        assert out_path.name.startswith("icp_")
        assert out_path.exists()

    async def test_resume_path(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        db_path = tmp_path / "t.db"
        async with Storage(db_path) as db:
            run_id = await db.create_run("Test ICP")
            await db.add_firm(run_id, "https://firm1.com/")

        out = tmp_path / "out.csv"
        result, _ = await _async_run(
            config,
            db_path=db_path,
            limit=None,
            output=out,
            resume=run_id,
            augment=False,
            include_all=False,
            client=FakeLLM(),
            scraper=FakeScraper(),
        )
        assert result.run_id == run_id
        assert result.stage_counts.get("completed") == 1
        assert out.exists()


# ---------------------------------------------------------------------------
# Command-line arg / error handling
# ---------------------------------------------------------------------------

class TestCommandLine:
    def test_missing_config_exits_nonzero(self) -> None:
        result = CliRunner().invoke(app, ["run", "--config", "does_not_exist.yaml"])
        assert result.exit_code == 1

    def test_run_requires_config_option(self) -> None:
        result = CliRunner().invoke(app, ["run"])
        assert result.exit_code != 0

    def test_eval_is_placeholder(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        result = CliRunner().invoke(app, ["eval", "--config", str(config)])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output

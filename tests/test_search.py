"""Tests for search: query generation, Tavily mapping, prefilter, dedup, filtering, discovery."""

from __future__ import annotations

import copy
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from lead_agent.config import ICPConfig
from lead_agent.llm import CallStats, LLMResponse
from lead_agent.search import (
    DiscoveryResult,
    FilterBatchResult,
    FilterDecision,
    GeneratedQueries,
    SearchResult,
    TavilySearch,
    dedupe_by_domain,
    discover_candidates,
    domain_of,
    expand_templates,
    filter_candidates,
    generate_queries,
    normalize_url,
    prefilter,
)
from lead_agent.storage import Storage

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

_VALID_ICP: dict = {
    "name": "Test ICP",
    "description": "A minimal test ICP.",
    "search_queries": {
        "templates": ["law firm {city}", "boutique attorneys"],
        "geo_focus": ["Dallas", "Austin"],
        "negative_keywords": ["directory", "ranking"],
    },
    "extraction_schema": [
        {"name": "firm_name", "type": "string", "description": "Firm name"},
        {"name": "attorney_count", "type": "integer", "description": "Attorney count"},
        {"name": "practice_areas", "type": "list", "description": "Practice areas"},
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
    "output_fields": ["firm_name", "attorney_count", "practice_areas", "score", "spec"],
}


def make_icp(**search_queries_overrides: object) -> ICPConfig:
    d = copy.deepcopy(_VALID_ICP)
    d["search_queries"].update(search_queries_overrides)
    return ICPConfig.model_validate(d)


class FakeLLM:
    """Stands in for LLMClient; only implements extract() via a responder callable."""

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
            model="fake", prompt_tokens=1, completion_tokens=1, cost_usd=0.0, duration_ms=1
        )
        return LLMResponse(content=content, stats=stats)


class FakeProvider:
    def __init__(self, mapping: dict[str, list[SearchResult]]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls.append(query)
        return self._mapping.get(query, [])


class FakeRawTavily:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.calls: list[tuple[str, Any]] = []

    async def search(
        self, query: str, max_results: int | None = None, **_: object
    ) -> dict[str, Any]:
        self.calls.append((query, max_results))
        return self._response


def _keep_all_firms(_: str, model: type[BaseModel]) -> BaseModel:
    """Responder that keeps every candidate (used for discover_candidates integration)."""
    if model is GeneratedQueries:
        return GeneratedQueries(queries=[])
    # FilterBatchResult: mark a generous range of indices as firms
    return FilterBatchResult(
        decisions=[FilterDecision(index=i, is_firm=True) for i in range(50)]
    )


def _keep_index_0(_: str, __: type[BaseModel]) -> FilterBatchResult:
    """Responder that keeps only index 0 of each batch."""
    return FilterBatchResult(decisions=[FilterDecision(index=0, is_firm=True)])


# ---------------------------------------------------------------------------
# expand_templates
# ---------------------------------------------------------------------------

class TestExpandTemplates:
    def test_city_placeholder_expands_over_geo(self) -> None:
        icp = make_icp(templates=["law firm {city}"], geo_focus=["Dallas", "Austin"])
        assert expand_templates(icp) == ["law firm Dallas", "law firm Austin"]

    def test_template_without_placeholder_emitted_once(self) -> None:
        icp = make_icp(templates=["boutique attorneys"], geo_focus=["Dallas", "Austin"])
        assert expand_templates(icp) == ["boutique attorneys"]

    def test_mixed_templates(self) -> None:
        icp = make_icp(templates=["law firm {city}", "boutique attorneys"], geo_focus=["Dallas"])
        assert expand_templates(icp) == ["law firm Dallas", "boutique attorneys"]

    def test_duplicates_removed_preserving_order(self) -> None:
        icp = make_icp(templates=["firm {city}", "firm {city}"], geo_focus=["Dallas"])
        assert expand_templates(icp) == ["firm Dallas"]


# ---------------------------------------------------------------------------
# normalize_url / domain_of
# ---------------------------------------------------------------------------

class TestUrlUtils:
    def test_normalize_strips_www_trailing_slash_query_fragment(self) -> None:
        assert normalize_url("https://www.Example.com/About/?x=1#frag") == "https://example.com/About"

    def test_normalize_adds_scheme_when_missing(self) -> None:
        assert normalize_url("example.com/x") == "https://example.com/x"

    def test_normalize_root_has_no_trailing_slash(self) -> None:
        assert normalize_url("https://example.com/") == "https://example.com"

    def test_domain_of_strips_www_and_lowercases(self) -> None:
        assert domain_of("https://WWW.Example.COM/about") == "example.com"

    def test_domain_of_keeps_subdomain(self) -> None:
        assert domain_of("https://news.example.com") == "news.example.com"

    def test_domain_of_handles_schemeless(self) -> None:
        assert domain_of("example.com/path") == "example.com"


# ---------------------------------------------------------------------------
# prefilter
# ---------------------------------------------------------------------------

def _r(url: str, title: str = "", snippet: str = "", score: float = 0.0) -> SearchResult:
    return SearchResult(url=url, title=title, snippet=snippet, score=score)


class TestPrefilter:
    def test_blocks_known_directory_domain(self) -> None:
        icp = make_icp()
        out = prefilter([_r("https://www.avvo.com/firm")], icp)
        assert out == []

    def test_blocks_subdomain_of_blocked_domain(self) -> None:
        icp = make_icp()
        out = prefilter([_r("https://profiles.martindale.com/x")], icp)
        assert out == []

    def test_blocks_aggregator_leak_domain(self) -> None:
        icp = make_icp()
        out = prefilter([_r("https://www.primerus.com/houston")], icp)
        assert out == []

    def test_drops_negative_keyword_in_title(self) -> None:
        icp = make_icp(negative_keywords=["ranking"])
        out = prefilter([_r("https://goodfirm.com", title="Top 10 Ranking of Firms")], icp)
        assert out == []

    def test_keeps_legit_firm(self) -> None:
        icp = make_icp(negative_keywords=["directory"])
        results = [_r("https://smithlaw.com", title="Smith Law - CRE Attorneys")]
        assert prefilter(results, icp) == results

    def test_negative_keyword_in_host(self) -> None:
        icp = make_icp(negative_keywords=["news"])
        out = prefilter([_r("https://news.somesite.com", title="Firm")], icp)
        assert out == []


# ---------------------------------------------------------------------------
# dedupe_by_domain
# ---------------------------------------------------------------------------

class TestDedupeByDomain:
    def test_collapses_pages_of_same_domain_to_one(self) -> None:
        results = [
            _r("https://smithlaw.com/about"),
            _r("https://smithlaw.com/attorneys"),
            _r("https://www.smithlaw.com/"),
        ]
        out = dedupe_by_domain(results)
        assert len(out) == 1
        assert domain_of(out[0].url) == "smithlaw.com"

    def test_representative_url_rewritten_to_origin_root(self) -> None:
        out = dedupe_by_domain([_r("https://smithlaw.com/about/team")])
        assert out[0].url == "https://smithlaw.com/"

    def test_prefers_shortest_path(self) -> None:
        results = [
            _r("https://smithlaw.com/about/team/deep", score=0.9),
            _r("https://smithlaw.com/x", score=0.1),
        ]
        out = dedupe_by_domain(results)
        # shortest path wins regardless of score; rewritten to root
        assert out[0].url == "https://smithlaw.com/"
        assert len(out) == 1

    def test_tie_break_by_score_when_same_path_length(self) -> None:
        results = [
            _r("https://smithlaw.com/aaaa", score=0.2, title="low"),
            _r("https://smithlaw.com/bbbb", score=0.8, title="high"),
        ]
        out = dedupe_by_domain(results)
        assert out[0].title == "high"

    def test_distinct_domains_kept_separately(self) -> None:
        out = dedupe_by_domain([_r("https://a.com"), _r("https://b.com")])
        assert {domain_of(r.url) for r in out} == {"a.com", "b.com"}


# ---------------------------------------------------------------------------
# TavilySearch
# ---------------------------------------------------------------------------

class TestTavilySearch:
    async def test_maps_response_to_search_results(self) -> None:
        raw = FakeRawTavily(
            {
                "results": [
                    {"url": "https://a.com", "title": "A", "content": "snip a", "score": 0.7},
                    {"url": "https://b.com", "title": "B", "content": "snip b", "score": 0.5},
                ]
            }
        )
        provider = TavilySearch(raw)
        out = await provider.search("query", max_results=5)
        assert [r.url for r in out] == ["https://a.com", "https://b.com"]
        assert out[0].title == "A"
        assert out[0].snippet == "snip a"
        assert out[0].score == pytest.approx(0.7)
        assert out[0].query == "query"

    async def test_skips_items_without_url(self) -> None:
        raw = FakeRawTavily({"results": [{"title": "no url"}, {"url": "https://a.com"}]})
        out = await TavilySearch(raw).search("q", max_results=5)
        assert [r.url for r in out] == ["https://a.com"]

    async def test_passes_max_results(self) -> None:
        raw = FakeRawTavily({"results": []})
        await TavilySearch(raw).search("q", max_results=3)
        assert raw.calls == [("q", 3)]

    async def test_empty_results_key(self) -> None:
        out = await TavilySearch(FakeRawTavily({})).search("q", max_results=5)
        assert out == []


# ---------------------------------------------------------------------------
# generate_queries
# ---------------------------------------------------------------------------

class TestGenerateQueries:
    async def test_augment_false_returns_templates_only_no_stats(self) -> None:
        icp = make_icp(templates=["law firm {city}"], geo_focus=["Dallas"])
        llm = FakeLLM(lambda p, m: GeneratedQueries(queries=["should not appear"]))
        queries, stats = await generate_queries(icp, llm, augment=False)
        assert queries == ["law firm Dallas"]
        assert stats is None
        assert llm.calls == []  # no LLM call when augment disabled

    async def test_augment_merges_and_dedupes(self) -> None:
        icp = make_icp(templates=["law firm Dallas"], geo_focus=["Dallas"])
        llm = FakeLLM(
            lambda p, m: GeneratedQueries(queries=["law firm Dallas", "CRE counsel Dallas"])
        )
        queries, stats = await generate_queries(icp, llm, augment=True)
        assert queries == ["law firm Dallas", "CRE counsel Dallas"]  # duplicate collapsed
        assert stats is not None

    async def test_augment_calls_llm_with_generated_queries_model(self) -> None:
        icp = make_icp()
        llm = FakeLLM(lambda p, m: GeneratedQueries(queries=[]))
        await generate_queries(icp, llm, augment=True)
        assert len(llm.calls) == 1
        assert llm.calls[0][1] is GeneratedQueries


# ---------------------------------------------------------------------------
# filter_candidates
# ---------------------------------------------------------------------------

class TestFilterCandidates:
    async def test_keeps_only_is_firm_true(self) -> None:
        icp = make_icp()
        results = [_r("https://a.com"), _r("https://b.com"), _r("https://c.com")]

        def responder(_: str, model: type[BaseModel]) -> BaseModel:
            return FilterBatchResult(
                decisions=[
                    FilterDecision(index=0, is_firm=True),
                    FilterDecision(index=1, is_firm=False),
                    FilterDecision(index=2, is_firm=True),
                ]
            )

        kept, stats = await filter_candidates(results, icp, FakeLLM(responder), batch_size=10)
        assert [r.url for r in kept] == ["https://a.com", "https://c.com"]
        assert len(stats) == 1

    async def test_empty_input_short_circuits(self) -> None:
        kept, stats = await filter_candidates([], make_icp(), FakeLLM(lambda p, m: None))
        assert kept == []
        assert stats == []

    async def test_missing_decision_drops_result(self) -> None:
        icp = make_icp()
        results = [_r("https://a.com"), _r("https://b.com")]
        # Only index 0 returned; index 1 has no decision -> dropped
        kept, _ = await filter_candidates(results, icp, FakeLLM(_keep_index_0), batch_size=10)
        assert [r.url for r in kept] == ["https://a.com"]

    async def test_invalid_index_ignored(self) -> None:
        icp = make_icp()
        results = [_r("https://a.com")]

        def responder(_: str, __: type[BaseModel]) -> FilterBatchResult:
            return FilterBatchResult(
                decisions=[
                    FilterDecision(index=0, is_firm=True),
                    FilterDecision(index=99, is_firm=True),
                ]
            )

        kept, _ = await filter_candidates(results, icp, FakeLLM(responder), batch_size=10)
        assert [r.url for r in kept] == ["https://a.com"]

    async def test_batches_produce_one_stats_per_batch(self) -> None:
        icp = make_icp()
        results = [_r(f"https://firm{i}.com") for i in range(5)]
        # keep index 0 of each batch
        kept, stats = await filter_candidates(results, icp, FakeLLM(_keep_index_0), batch_size=2)
        assert len(stats) == 3  # ceil(5/2) batches
        assert len(kept) == 3  # index 0 kept from each batch


# ---------------------------------------------------------------------------
# discover_candidates (integration with fakes + real Storage)
# ---------------------------------------------------------------------------

class TestDiscoverCandidates:
    async def test_end_to_end_returns_firm_urls(self, tmp_path: Path) -> None:
        icp = make_icp(
            templates=["law firm {city}"],
            geo_focus=["Dallas"],
            negative_keywords=["directory"],
        )
        provider = FakeProvider(
            {
                "law firm Dallas": [
                    _r("https://smithlaw.com/about", score=0.6),
                    _r("https://www.avvo.com/listing"),  # blocked
                    _r("https://joneslaw.com/", score=0.4),
                ]
            }
        )
        async with Storage(tmp_path / "t.db") as db:
            result = await discover_candidates(
                icp, FakeLLM(_keep_all_firms), provider, storage=db, augment_queries=False
            )
        assert isinstance(result, DiscoveryResult)
        assert set(result.urls) == {"https://smithlaw.com/", "https://joneslaw.com/"}

    async def test_search_cache_avoids_second_provider_call(self, tmp_path: Path) -> None:
        icp = make_icp(templates=["law firm {city}"], geo_focus=["Dallas"], negative_keywords=[])
        provider = FakeProvider({"law firm Dallas": [_r("https://smithlaw.com/", score=0.6)]})
        async with Storage(tmp_path / "t.db") as db:
            await discover_candidates(
                icp, FakeLLM(_keep_all_firms), provider, storage=db, augment_queries=False
            )
            assert provider.calls == ["law firm Dallas"]
            # second run: served from search_cache, provider not called again
            result2 = await discover_candidates(
                icp, FakeLLM(_keep_all_firms), provider, storage=db, augment_queries=False
            )
        assert provider.calls == ["law firm Dallas"]  # unchanged
        assert result2.urls == ["https://smithlaw.com/"]

    async def test_aggregates_llm_stats(self, tmp_path: Path) -> None:
        icp = make_icp(templates=["law firm {city}"], geo_focus=["Dallas"], negative_keywords=[])
        provider = FakeProvider({"law firm Dallas": [_r("https://smithlaw.com/")]})
        # augment_queries=True -> 1 querygen call; 1 filter batch -> 2 calls total
        result = await discover_candidates(
            icp, FakeLLM(_keep_all_firms), provider, augment_queries=True
        )
        assert len(result.llm_calls) == 2
        assert result.total_tokens == 4  # 2 calls x (1 prompt + 1 completion)

    async def test_works_without_storage(self) -> None:
        icp = make_icp(templates=["law firm {city}"], geo_focus=["Dallas"], negative_keywords=[])
        provider = FakeProvider({"law firm Dallas": [_r("https://smithlaw.com/")]})
        result = await discover_candidates(
            icp, FakeLLM(_keep_all_firms), provider, augment_queries=False
        )
        assert result.urls == ["https://smithlaw.com/"]


# ---------------------------------------------------------------------------
# filter_candidates: persistence + caching
# ---------------------------------------------------------------------------

def _decisions_with_reasons(_: str, __: type[BaseModel]) -> FilterBatchResult:
    """Responder that returns two decisions with non-empty reasons (keep + reject)."""
    return FilterBatchResult(
        decisions=[
            FilterDecision(index=0, is_firm=True, reason="clear firm site"),
            FilterDecision(index=1, is_firm=False, reason="directory aggregator page"),
        ]
    )


def _read_filter_decisions(db_path: Path, run_id: str) -> list[dict[str, Any]]:
    """Plain sqlite3 read after Storage has been closed — avoids touching internals."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT url, is_firm, reason FROM filter_decisions "
            "WHERE run_id = ? ORDER BY url",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


class TestFilterPersistenceAndCache:
    async def test_persists_kept_and_rejected_with_reasons(self, tmp_path: Path) -> None:
        from lead_agent.storage import Storage

        icp = make_icp()
        results = [
            _r("https://a.com", title="Firm A"),
            _r("https://b.com", title="Directory B"),
        ]
        db_path = tmp_path / "t.db"
        async with Storage(db_path) as db:
            run_id = await db.create_run(icp.name)
            await filter_candidates(
                results, icp, FakeLLM(_decisions_with_reasons),
                storage=db, run_id=run_id,
            )

        rows = _read_filter_decisions(db_path, run_id)
        assert rows == [
            {"url": "https://a.com", "is_firm": 1, "reason": "clear firm site"},
            {"url": "https://b.com", "is_firm": 0, "reason": "directory aggregator page"},
        ]

    async def test_missing_decision_audited_as_no_decision(self, tmp_path: Path) -> None:
        from lead_agent.storage import Storage

        icp = make_icp()
        results = [_r("https://a.com"), _r("https://b.com")]
        db_path = tmp_path / "t.db"
        async with Storage(db_path) as db:
            run_id = await db.create_run(icp.name)
            # Responder returns no decision for index 1
            await filter_candidates(
                results, icp, FakeLLM(_keep_index_0), storage=db, run_id=run_id,
            )

        rows = _read_filter_decisions(db_path, run_id)
        assert rows[0]["reason"] == ""  # index 0 kept, default empty reason
        assert rows[1]["reason"] == "[no decision returned by LLM]"
        assert rows[1]["is_firm"] == 0

    async def test_cache_hit_skips_llm_on_second_call(self, tmp_path: Path) -> None:
        from lead_agent.storage import Storage

        icp = make_icp()
        results = [_r("https://a.com", title="A"), _r("https://b.com", title="B")]
        llm = FakeLLM(_decisions_with_reasons)

        async with Storage(tmp_path / "t.db") as db:
            kept1, _ = await filter_candidates(results, icp, llm, storage=db)
            assert len(llm.calls) == 1  # cache miss on first call
            kept2, stats2 = await filter_candidates(results, icp, llm, storage=db)

        assert len(llm.calls) == 1  # cache hit on second; LLM not called again
        assert [r.url for r in kept2] == [r.url for r in kept1]
        assert stats2[0].model == "cache"
        assert stats2[0].prompt_tokens == 0

    async def test_different_icp_name_misses_cache(self, tmp_path: Path) -> None:
        from lead_agent.storage import Storage

        results = [_r("https://a.com", title="A")]
        llm = FakeLLM(_decisions_with_reasons)
        icp_a = make_icp()  # name = "Test ICP"
        other = copy.deepcopy(_VALID_ICP)
        other["name"] = "Different ICP Name"
        icp_b = ICPConfig.model_validate(other)

        async with Storage(tmp_path / "t.db") as db:
            await filter_candidates(results, icp_a, llm, storage=db)
            await filter_candidates(results, icp_b, llm, storage=db)

        assert len(llm.calls) == 2  # different icp.name -> different cache key

    async def test_different_batch_contents_miss_cache(self, tmp_path: Path) -> None:
        from lead_agent.storage import Storage

        icp = make_icp()
        llm = FakeLLM(_decisions_with_reasons)

        async with Storage(tmp_path / "t.db") as db:
            await filter_candidates([_r("https://a.com", title="A")], icp, llm, storage=db)
            await filter_candidates(
                [_r("https://a.com", title="DIFFERENT TITLE")], icp, llm, storage=db,
            )

        assert len(llm.calls) == 2  # different batch_hash -> cache miss

    async def test_prompt_version_change_misses_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lead_agent import search as search_mod
        from lead_agent.storage import Storage

        icp = make_icp()
        results = [_r("https://a.com", title="A")]
        llm = FakeLLM(_decisions_with_reasons)

        async with Storage(tmp_path / "t.db") as db:
            monkeypatch.setattr(search_mod, "_FILTER_PROMPT_VERSION", "v1")
            await filter_candidates(results, icp, llm, storage=db)
            assert len(llm.calls) == 1  # first call: cache miss
            # Same batch, same ICP — but prompt version changed
            monkeypatch.setattr(search_mod, "_FILTER_PROMPT_VERSION", "v2")
            await filter_candidates(results, icp, llm, storage=db)

        assert len(llm.calls) == 2  # cache miss on version change

    async def test_no_run_id_skips_logging_but_caching_still_works(
        self, tmp_path: Path
    ) -> None:
        from lead_agent.storage import Storage

        icp = make_icp()
        results = [_r("https://a.com", title="A")]
        llm = FakeLLM(_decisions_with_reasons)
        db_path = tmp_path / "t.db"

        async with Storage(db_path) as db:
            # No run_id supplied
            await filter_candidates(results, icp, llm, storage=db)
            # Second call should still hit cache
            await filter_candidates(results, icp, llm, storage=db)

        assert len(llm.calls) == 1  # caching active
        con = sqlite3.connect(str(db_path))
        try:
            count = con.execute("SELECT COUNT(*) FROM filter_decisions").fetchone()[0]
        finally:
            con.close()
        assert count == 0  # nothing logged without run_id

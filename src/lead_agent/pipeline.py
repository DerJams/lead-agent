"""Pipeline orchestration: search -> scrape -> extract -> score, with resumable run state.

Discovery is a batch step; each candidate URL becomes a 'pending' firm. Firms are
then processed concurrently and in isolation — a failure marks only that firm
'failed' and never aborts the batch. State is persisted per stage so a run can be
resumed; an interrupted firm already at 'extracted' reuses its stored profile and
only re-scores. CSV output lives in the CLI (step 10); this module persists to
SQLite and returns a RunResult.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .extractor import extract_profile
from .llm import get_client
from .scorer import score_firm
from .scraper import ScrapeSettings, get_scraper
from .search import discover_candidates, get_search_provider

if TYPE_CHECKING:
    from .config import ICPConfig
    from .llm import CallStats, LLMClient
    from .scraper import Scraper
    from .search import SearchProvider
    from .storage import Storage


_NON_TERMINAL_STAGES = ("pending", "scraped", "extracted")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    bytes_scraped: int = 0

    def add_call(self, call: CallStats) -> None:
        self.llm_calls += 1
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens
        self.cost_usd += call.cost_usd

    def add_calls(self, calls: list[CallStats]) -> None:
        for call in calls:
            self.add_call(call)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class FirmOutcome:
    firm_id: str
    stage: str
    calls: list[CallStats] = field(default_factory=list)
    bytes_scraped: int = 0


@dataclass
class RunResult:
    run_id: str
    stage_counts: dict[str, int]
    qualified: list[dict[str, Any]]
    scored: list[dict[str, Any]]
    stats: RunStats


# ---------------------------------------------------------------------------
# Per-firm processing
# ---------------------------------------------------------------------------

async def _process_firm(
    firm: dict[str, Any],
    icp: ICPConfig,
    storage: Storage,
    scraper: Scraper,
    client: LLMClient,
) -> FirmOutcome:
    """Scrape -> extract -> score one firm. Isolates failures to this firm."""
    firm_id = firm["firm_id"]
    url = firm["url"]
    stage = firm["stage"]
    calls: list[CallStats] = []
    try:
        scrape = await scraper.scrape_firm(url)
        if not scrape.ok:
            await storage.update_firm_stage(
                firm_id, "failed", error=scrape.error or "scrape produced no content"
            )
            return FirmOutcome(firm_id, "failed", calls)
        await storage.update_firm_stage(firm_id, "scraped", scraped_at=_now())
        combined_text = scrape.combined_text
        bytes_scraped = scrape.bytes_fetched

        profile = firm.get("extracted_profile")
        if stage == "extracted" and profile is not None:
            pass  # resume: reuse persisted profile, skip the extraction LLM call
        else:
            extraction = await extract_profile(combined_text, icp, client, source_url=url)
            if extraction.stats is not None:
                calls.append(extraction.stats)
            if not extraction.ok:
                await storage.update_firm_stage(
                    firm_id, "failed", error=extraction.error or "extraction failed"
                )
                return FirmOutcome(firm_id, "failed", calls, bytes_scraped)
            profile = extraction.profile
            await storage.update_firm_stage(firm_id, "extracted", extracted_profile=profile)

        result = await score_firm(profile, combined_text, icp, client)
        calls.extend(result.stats)
        breakdown = {**result.breakdown, "qualified": result.qualified}
        await storage.update_firm_stage(
            firm_id, "completed", score=result.score, score_breakdown=breakdown
        )
        return FirmOutcome(firm_id, "completed", calls, bytes_scraped)
    except Exception as exc:  # isolate any per-firm failure
        await storage.update_firm_stage(firm_id, "failed", error=str(exc))
        return FirmOutcome(firm_id, "failed", calls)


async def _process_firms(
    firms: list[dict[str, Any]],
    icp: ICPConfig,
    storage: Storage,
    scraper: Scraper,
    client: LLMClient,
    concurrency: int,
) -> list[FirmOutcome]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(firm: dict[str, Any]) -> FirmOutcome:
        async with semaphore:
            return await _process_firm(firm, icp, storage, scraper, client)

    return await asyncio.gather(*(run_one(firm) for firm in firms))


async def _run_firms(
    firms: list[dict[str, Any]],
    icp: ICPConfig,
    storage: Storage,
    client: LLMClient,
    scraper: Scraper | None,
    concurrency: int,
) -> list[FirmOutcome]:
    """Process firms, opening a default scraper if one was not injected."""
    if scraper is not None:
        return await _process_firms(firms, icp, storage, scraper, client, concurrency)
    async with get_scraper(storage) as owned:
        return await _process_firms(firms, icp, storage, owned, client, concurrency)


def _accumulate(stats: RunStats, outcomes: list[FirmOutcome]) -> None:
    for outcome in outcomes:
        stats.add_calls(outcome.calls)
        stats.bytes_scraped += outcome.bytes_scraped


async def _build_result(
    run_id: str, storage: Storage, stats: RunStats
) -> RunResult:
    stage_counts = await storage.count_firms_by_stage(run_id)
    completed = await storage.get_firms_by_stage(run_id, "completed")
    scored = sorted(completed, key=lambda f: f.get("score") or 0.0, reverse=True)
    qualified = [f for f in scored if (f.get("score_breakdown") or {}).get("qualified")]
    return RunResult(
        run_id=run_id,
        stage_counts=stage_counts,
        qualified=qualified,
        scored=scored,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_pipeline(
    icp: ICPConfig,
    *,
    storage: Storage,
    client: LLMClient | None = None,
    provider: SearchProvider | None = None,
    scraper: Scraper | None = None,
    limit: int | None = None,
    augment_queries: bool = True,
    concurrency: int | None = None,
) -> RunResult:
    """Run a fresh pipeline: discover candidates, then scrape/extract/score each firm."""
    client = client or get_client()
    provider = provider or get_search_provider()
    concurrency = concurrency or ScrapeSettings().scrape_concurrency
    stats = RunStats()

    run_id = await storage.create_run(icp.name)
    try:
        discovery = await discover_candidates(
            icp,
            client,
            provider,
            storage=storage,
            run_id=run_id,
            augment_queries=augment_queries,
        )
        stats.add_calls(discovery.llm_calls)
        urls = discovery.urls[:limit] if limit is not None else discovery.urls
        for url in urls:
            await storage.add_firm(run_id, url)

        firms = await storage.get_firms_by_stage(run_id, "pending")
        outcomes = await _run_firms(firms, icp, storage, client, scraper, concurrency)
        _accumulate(stats, outcomes)
    except Exception:
        await storage.complete_run(run_id, status="failed")
        raise

    await storage.complete_run(run_id, status="completed")
    return await _build_result(run_id, storage, stats)


async def resume_run(
    run_id: str,
    icp: ICPConfig,
    *,
    storage: Storage,
    client: LLMClient | None = None,
    scraper: Scraper | None = None,
    limit: int | None = None,
    concurrency: int | None = None,
) -> RunResult:
    """Resume a partial run: reprocess only non-terminal firms (pending/scraped/extracted)."""
    client = client or get_client()
    concurrency = concurrency or ScrapeSettings().scrape_concurrency
    stats = RunStats()

    firms: list[dict[str, Any]] = []
    for stage in _NON_TERMINAL_STAGES:
        firms.extend(await storage.get_firms_by_stage(run_id, stage))
    if limit is not None:
        firms = firms[:limit]

    try:
        outcomes = await _run_firms(firms, icp, storage, client, scraper, concurrency)
        _accumulate(stats, outcomes)
    except Exception:
        await storage.complete_run(run_id, status="failed")
        raise

    await storage.complete_run(run_id, status="completed")
    return await _build_result(run_id, storage, stats)

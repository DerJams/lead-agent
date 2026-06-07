"""Query generation, search execution, candidate filtering, and dedup.

Discovery only: produces candidate firm URLs from an ICP config. The pipeline
persists them as `pending` firms via storage.add_firm(); this module never
writes the firms or scrape_cache tables. It optionally uses storage.search_cache
to avoid re-hitting the search API on re-runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .llm import CallStats  # runtime: synthesised for cache-hit stats

if TYPE_CHECKING:
    from .config import ICPConfig
    from .llm import LLMClient
    from .storage import Storage


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SearchSettings(BaseSettings):
    search_provider: Literal["tavily", "serper"] = "tavily"
    tavily_api_key: str = ""
    serper_api_key: str = ""
    max_results_per_query: int = 8
    search_concurrency: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    query: str = ""


class GeneratedQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


class FilterDecision(BaseModel):
    index: int
    is_firm: bool
    reason: str = ""


class FilterBatchResult(BaseModel):
    decisions: list[FilterDecision] = Field(default_factory=list)


@dataclass
class DiscoveryResult:
    urls: list[str]
    results: list[SearchResult]
    llm_calls: list[CallStats] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.llm_calls)


# ---------------------------------------------------------------------------
# Search provider
# ---------------------------------------------------------------------------

class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class TavilySearch:
    """Tavily-backed search provider. Client is injected for testability."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        response = await self._client.search(query, max_results=max_results)
        results: list[SearchResult] = []
        for item in response.get("results", []):
            url = item.get("url")
            if not url:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title") or "",
                    snippet=item.get("content") or "",
                    score=float(item.get("score") or 0.0),
                    query=query,
                )
            )
        return results


def get_search_provider() -> SearchProvider:
    """Read env/settings and return a configured SearchProvider."""
    settings = SearchSettings()
    if settings.search_provider == "tavily":
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when SEARCH_PROVIDER=tavily")
        from tavily import AsyncTavilyClient

        return TavilySearch(AsyncTavilyClient(api_key=settings.tavily_api_key))
    raise ValueError(
        f"SEARCH_PROVIDER {settings.search_provider!r} is not implemented in v1 (use 'tavily')"
    )


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

_QUERYGEN_SYSTEM = (
    "You generate web search queries to discover official websites of individual firms "
    "or businesses matching a target market, for a lead-generation pipeline.\n\n"
    "Effective queries pair a geographic anchor (city + state) with vocabulary firms "
    "actually use on their own websites: specific practice areas, deal types, or "
    "services they advertise. Avoid:\n"
    "- Adjectives like 'boutique', 'small', 'owner-operated', 'mid-size' — these "
    "rarely appear on firms' own pages and don't constrain ranking.\n"
    "- Words that pull aggregator/directory results: 'directory', 'best', 'top', "
    "'ranking', 'rated', 'services', 'advice', 'search'.\n"
    "- Paraphrastic restatements of the same noun phrase (e.g. 'law firm', 'legal "
    "services', 'law offices', 'attorney services' for the same vertical).\n"
    "- Boolean operators, quotes, and site: filters — the search backend ignores these.\n\n"
    "Prefer queries that vary on practice subvertical (specific deal types, "
    "transaction types, or sub-specialties the target firms list as their services) "
    "rather than synonym swaps on the same broad term."
)


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def expand_templates(icp: ICPConfig) -> list[str]:
    """Deterministic cartesian expansion of templates x geo_focus on the {city} slot."""
    sq = icp.search_queries
    queries: list[str] = []
    for template in sq.templates:
        if "{city}" in template:
            for city in sq.geo_focus:
                queries.append(template.replace("{city}", city))
        else:
            queries.append(template)
    return _dedupe_preserving_order(queries)


async def generate_queries(
    icp: ICPConfig,
    client: LLMClient,
    *,
    augment: bool = True,
    extra_count: int = 8,
) -> tuple[list[str], CallStats | None]:
    """Return (queries, stats). Template expansion plus optional LLM-augmented variations."""
    base = expand_templates(icp)
    if not augment:
        return base, None

    negatives = icp.search_queries.negative_keywords
    avoid_line = (
        f"Words to AVOID in queries (these signal noise in this domain): {', '.join(negatives)}\n"
        if negatives
        else ""
    )
    prompt = (
        f"Target market: {icp.name}\n"
        f"Description: {icp.description}\n"
        f"Regions of focus: {', '.join(icp.search_queries.geo_focus)}\n"
        + avoid_line
        + "Existing queries:\n" + "\n".join(f"- {q}" for q in base) + "\n\n"
        f"Propose {extra_count} additional, distinct web search queries that would surface "
        "official websites of firms matching this market. Vary phrasing and angle; avoid "
        "duplicating the existing queries."
    )
    response = await client.extract(prompt, GeneratedQueries, system=_QUERYGEN_SYSTEM)
    merged = _dedupe_preserving_order(base + response.content.queries)
    return merged, response.stats


# ---------------------------------------------------------------------------
# URL utilities, prefilter, dedup
# ---------------------------------------------------------------------------

_DEFAULT_BLOCKED_DOMAINS: frozenset[str] = frozenset(
    {
        "martindale.com", "avvo.com", "findlaw.com", "justia.com", "lawyers.com",
        "superlawyers.com", "nolo.com", "hg.org", "lawyer.com", "expertise.com",
        "yelp.com", "yellowpages.com", "bbb.org", "wikipedia.org", "glassdoor.com",
        "indeed.com", "mapquest.com", "crunchbase.com", "bloomberg.com", "reuters.com",
        "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
        "youtube.com",
        "primerus.com", "axiomlaw.com", "bcgsearch.com", "lexinter.net",
        "findarealestateattorney.com", "fortworthchamber.com", "bestlawyers.com",
        "lawfirmsquare.com", "jurisolutions.com",
        "lawinfo.com", "contractscounsel.com", "texaslandcan.org",
        "ipx1031.com", "apiexchange.com", "legal1031.com", "firstexchange.com",
        "rattikinexchange.com", "excel1031exchange.org",
        "ctic.com", "ltic.com",
    }
)


def _ensure_scheme(url: str) -> str:
    if "//" not in url:
        return "https://" + url
    return url


def domain_of(url: str) -> str:
    """Lowercased hostname with a leading 'www.' stripped; '' if unparseable."""
    host = urlsplit(_ensure_scheme(url)).hostname or ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def normalize_url(url: str) -> str:
    """Canonical dedup key: lowercased host (no www), path without trailing slash, no query."""
    parts = urlsplit(_ensure_scheme(url))
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    scheme = parts.scheme.lower() or "https"
    return urlunsplit((scheme, host, path, "", ""))


def _origin_root(url: str) -> str:
    """Site root (scheme://netloc/) for the given URL, preserving scheme and host."""
    parts = urlsplit(_ensure_scheme(url))
    scheme = parts.scheme.lower() or "https"
    return urlunsplit((scheme, parts.netloc, "/", "", ""))


def _is_blocked_domain(domain: str) -> bool:
    return any(domain == b or domain.endswith("." + b) for b in _DEFAULT_BLOCKED_DOMAINS)


def prefilter(results: list[SearchResult], icp: ICPConfig) -> list[SearchResult]:
    """Drop directory/social/news noise via a built-in blocklist and ICP negative keywords."""
    negatives = [k.lower() for k in icp.search_queries.negative_keywords]
    kept: list[SearchResult] = []
    for r in results:
        domain = domain_of(r.url)
        if not domain or _is_blocked_domain(domain):
            continue
        haystack = f"{domain} {r.title}".lower()
        if any(neg in haystack for neg in negatives):
            continue
        kept.append(r)
    return kept


def dedupe_by_domain(results: list[SearchResult]) -> list[SearchResult]:
    """Collapse to one result per registrable domain.

    Representative = shortest path (closest to homepage), tie-broken by highest score.
    The kept result's URL is rewritten to the site root for downstream scraping.
    """
    best: dict[str, SearchResult] = {}
    for r in results:
        domain = domain_of(r.url)
        if not domain:
            continue
        current = best.get(domain)
        if current is None or _is_better_representative(r, current):
            best[domain] = r
    return [r.model_copy(update={"url": _origin_root(r.url)}) for r in best.values()]


def _is_better_representative(candidate: SearchResult, current: SearchResult) -> bool:
    cand_path_len = len(urlsplit(_ensure_scheme(candidate.url)).path.rstrip("/"))
    cur_path_len = len(urlsplit(_ensure_scheme(current.url)).path.rstrip("/"))
    if cand_path_len != cur_path_len:
        return cand_path_len < cur_path_len
    return candidate.score > current.score


# ---------------------------------------------------------------------------
# LLM candidate filtering
# ---------------------------------------------------------------------------

_FILTER_PROMPT_VERSION = "v2"

_FILTER_SYSTEM = (
    "You are a classifier deciding whether each search result is the official website of a "
    "single firm or business — as opposed to a directory, aggregator, ranking list, news "
    "article, social-media page, or other non-firm page.\n\n"
    "Decide based on page type only. Do NOT reject a result based on:\n"
    "- The firm's size (number of attorneys, employees, partners, offices)\n"
    "- Whether the firm specializes in exactly the target practice area\n"
    "- The firm's specific city, as long as it operates in the broad region\n\n"
    "Those nuances are evaluated downstream from the full website content. Your only job "
    'is "is this a single firm\'s website, yes or no?"\n\n'
    "When uncertain about page type, prefer is_firm=true and let downstream scoring filter "
    "it out. Reject only when clearly a directory, aggregator, news article, social media "
    "page, or other non-firm content."
)


def _chunk(items: list[SearchResult], size: int) -> list[list[SearchResult]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_filter_prompt(icp: ICPConfig, batch: list[SearchResult]) -> str:
    # Note: icp.description is intentionally NOT injected — its size/geo/specialization
    # details bias the classifier toward over-rejection. Only icp.name passes through, to
    # supply the vertical (e.g. "Law Boutique" tells the LLM not to keep CPA firms).
    lines = [
        f"We are sourcing leads for firms in this broad domain: {icp.name}",
        "",
        "Classify each numbered search result below. For each, decide whether it is the "
        "official website of a single firm in this broad domain (is_firm=true) or not "
        "(is_firm=false). Return one decision per index with a brief reason.",
        "",
    ]
    for i, r in enumerate(batch):
        snippet = r.snippet[:300]
        lines.append(f"[{i}] url: {r.url}\n    title: {r.title}\n    snippet: {snippet}")
    return "\n".join(lines)


def _batch_hash(batch: list[SearchResult]) -> str:
    """Stable SHA-256 over (url, title, snippet[:300]) tuples in batch order.

    Matches the 300-char snippet truncation used in `_build_filter_prompt`, so the cache
    key reflects exactly what the LLM saw. Order-sensitive because decision indices are
    positional within the batch.
    """
    payload = json.dumps(
        [(r.url, r.title, r.snippet[:300]) for r in batch],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decisions_for_audit(
    batch: list[SearchResult], decisions: list[FilterDecision]
) -> list[dict[str, object]]:
    """Pair every batch item with its decision; mark missing ones explicitly."""
    by_index = {d.index: d for d in decisions}
    rows: list[dict[str, object]] = []
    for i, r in enumerate(batch):
        d = by_index.get(i)
        rows.append(
            {
                "url": r.url,
                "title": r.title,
                "snippet": r.snippet[:300],
                "is_firm": bool(d.is_firm) if d is not None else False,
                "reason": d.reason if d is not None else "[no decision returned by LLM]",
            }
        )
    return rows


async def filter_candidates(
    results: list[SearchResult],
    icp: ICPConfig,
    client: LLMClient,
    *,
    batch_size: int = 10,
    storage: Storage | None = None,
    run_id: str | None = None,
) -> tuple[list[SearchResult], list[CallStats]]:
    """Keep results the LLM classifies as firm websites. Batched; missing decisions drop.

    Decisions (kept and rejected, with their `reason` strings) are appended to the
    `filter_decisions` audit table when `storage` and `run_id` are both supplied, and
    cached by (icp.name, batch_hash) when `storage` is supplied. On a cache hit the LLM
    is not called; the returned `CallStats` has model='cache' and zero tokens/cost.
    """
    if not results:
        return [], []

    batches = _chunk(results, batch_size)

    async def classify(batch: list[SearchResult]) -> tuple[list[SearchResult], CallStats]:
        batch_h = _batch_hash(batch)
        cached = (
            await storage.get_cached_filter(icp.name, _FILTER_PROMPT_VERSION, batch_h)
            if storage is not None
            else None
        )
        if cached is not None:
            decisions = [FilterDecision(**d) for d in cached]
            stats = CallStats(
                model="cache",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                duration_ms=0,
            )
        else:
            prompt = _build_filter_prompt(icp, batch)
            response = await client.extract(prompt, FilterBatchResult, system=_FILTER_SYSTEM)
            decisions = response.content.decisions
            stats = response.stats
            if storage is not None:
                await storage.cache_filter(
                    icp.name,
                    _FILTER_PROMPT_VERSION,
                    batch_h,
                    [d.model_dump() for d in decisions],
                )

        if storage is not None and run_id is not None:
            await storage.log_filter_decisions(
                run_id, icp.name, _decisions_for_audit(batch, decisions)
            )

        keep_idx = {d.index for d in decisions if d.is_firm}
        kept = [batch[i] for i in sorted(keep_idx) if 0 <= i < len(batch)]
        return kept, stats

    outcomes = await asyncio.gather(*(classify(b) for b in batches))
    kept: list[SearchResult] = []
    stats: list[CallStats] = []
    for batch_kept, batch_stats in outcomes:
        kept.extend(batch_kept)
        stats.append(batch_stats)
    return kept, stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def _search_one(
    provider: SearchProvider,
    query: str,
    max_results: int,
    storage: Storage | None,
    semaphore: asyncio.Semaphore,
) -> list[SearchResult]:
    if storage is not None:
        cached = await storage.get_cached_search(query)
        if cached is not None:
            return [SearchResult(**item) for item in cached]
    async with semaphore:
        try:
            results = await provider.search(query, max_results)
        except Exception:
            return []
    if storage is not None and results:
        await storage.cache_search(query, [r.model_dump() for r in results])
    return results


async def discover_candidates(
    icp: ICPConfig,
    client: LLMClient,
    provider: SearchProvider,
    *,
    storage: Storage | None = None,
    run_id: str | None = None,
    augment_queries: bool = True,
    max_results_per_query: int = 8,
    search_concurrency: int = 5,
) -> DiscoveryResult:
    """Full discovery: generate queries, search, prefilter, dedupe, LLM-filter to firm URLs.

    When `storage` is supplied the LLM filter consults `filter_cache` and writes new
    decisions to it; when `run_id` is also supplied, every decision (kept and rejected)
    is appended to `filter_decisions` for audit.
    """
    queries, qstats = await generate_queries(icp, client, augment=augment_queries)

    semaphore = asyncio.Semaphore(search_concurrency)
    per_query = await asyncio.gather(
        *(
            _search_one(provider, q, max_results_per_query, storage, semaphore)
            for q in queries
        )
    )
    all_results = [r for batch in per_query for r in batch]

    prefiltered = prefilter(all_results, icp)
    deduped = dedupe_by_domain(prefiltered)
    kept, fstats = await filter_candidates(
        deduped, icp, client, storage=storage, run_id=run_id
    )

    llm_calls: list[CallStats] = ([qstats] if qstats else []) + fstats
    return DiscoveryResult(
        urls=[r.url for r in kept],
        results=kept,
        llm_calls=llm_calls,
    )

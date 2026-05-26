"""Async polite site scraper.

Fetches a firm's homepage and a handful of relevant internal pages (about,
attorneys/team, practice areas, contact), extracts readable text, and combines
them into a single labeled document for the extractor.

Polite by construction: per-domain rate limiting, robots.txt compliance, an
identifiable User-Agent, request timeouts, and a response-size cap. Touches only
storage.scrape_cache (raw HTML, re-cleaned on read); the pipeline owns firm-stage
transitions.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from .search import domain_of, normalize_url

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .storage import Storage


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class ScrapeSettings(BaseSettings):
    user_agent: str = "lead-agent/0.1 (+https://github.com/DerJams/lead-agent)"
    scrape_concurrency: int = 5
    scrape_delay_seconds: float = 1.0
    request_timeout_seconds: float = 15.0
    max_pages_per_firm: int = 5
    max_response_bytes: int = 2_000_000
    scrape_cache_ttl_hours: float = 24.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ScrapedPage(BaseModel):
    url: str
    page_type: str
    title: str = ""
    text: str = ""


@dataclass
class ScrapeResult:
    url: str
    pages: list[ScrapedPage] = field(default_factory=list)
    combined_text: str = ""
    error: str | None = None
    bytes_fetched: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.combined_text)

    @property
    def pages_fetched(self) -> int:
        return len(self.pages)


# ---------------------------------------------------------------------------
# Page discovery and text extraction (pure)
# ---------------------------------------------------------------------------

_PAGE_PRIORITY: tuple[str, ...] = ("about", "team", "practice", "contact")

_PAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "about": ("about", "who-we-are", "the-firm", "our-firm", "firm-overview"),
    "team": (
        "attorney", "attorneys", "lawyer", "lawyers", "people", "team",
        "professionals", "our-team", "staff",
    ),
    "practice": ("practice", "services", "expertise", "what-we-do", "areas"),
    "contact": ("contact", "locations"),
}

_ASSET_EXTS: tuple[str, ...] = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".mp4", ".css", ".js",
)

_STRIP_TAGS: tuple[str, ...] = (
    "script", "style", "nav", "header", "footer", "noscript", "aside", "form",
)

_MAX_CHARS_PER_PAGE = 6000
_MAX_TOTAL_CHARS = 24000


def _match_page_type(haystack: str) -> str | None:
    for page_type in _PAGE_PRIORITY:
        if any(kw in haystack for kw in _PAGE_KEYWORDS[page_type]):
            return page_type
    return None


def select_relevant_links(
    homepage_html: str, base_url: str, max_pages: int
) -> list[tuple[str, str]]:
    """Return up to (max_pages - 1) same-domain (url, page_type) pairs, one per page type.

    Homepage is fetched separately, so it is excluded here. Page types are returned
    in priority order: about, team, practice, contact.
    """
    soup = BeautifulSoup(homepage_html, "html.parser")
    base_domain = domain_of(base_url)
    home_norm = normalize_url(base_url)
    found: dict[str, str] = {}

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        resolved = urljoin(base_url, href)
        parts = urlsplit(resolved)
        if parts.scheme not in ("http", "https"):
            continue
        if domain_of(resolved) != base_domain:
            continue
        if parts.path.lower().endswith(_ASSET_EXTS):
            continue
        norm = normalize_url(resolved)
        if norm == home_norm:
            continue
        haystack = f"{parts.path.lower()} {anchor.get_text(' ', strip=True).lower()}"
        page_type = _match_page_type(haystack)
        if page_type is None or page_type in found:
            continue
        found[page_type] = resolved

    ordered = [(found[pt], pt) for pt in _PAGE_PRIORITY if pt in found]
    return ordered[: max(0, max_pages - 1)]


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, readable_text) from raw HTML, stripping boilerplate tags."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
    return title, text


def combine_pages(
    pages: list[ScrapedPage],
    *,
    max_chars_per_page: int = _MAX_CHARS_PER_PAGE,
    max_total_chars: int = _MAX_TOTAL_CHARS,
) -> str:
    """Join pages into one labeled document, truncating per-page and overall."""
    sections: list[str] = []
    total = 0
    for page in pages:
        body = page.text[:max_chars_per_page]
        section = f"## {page.page_type.upper()} — {page.url}\n{page.title}\n{body}".strip()
        sections.append(section)
        total += len(section)
        if total >= max_total_chars:
            break
    return "\n\n".join(sections)[:max_total_chars]


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class Scraper:
    """Async context manager owning the HTTP client and per-domain politeness state.

        async with Scraper(ScrapeSettings(), storage=db) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
    """

    def __init__(
        self,
        settings: ScrapeSettings,
        *,
        storage: Storage | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_fetch: dict[str, float] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}

    async def __aenter__(self) -> Scraper:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.request_timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": self._settings.user_agent},
            )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- politeness ----------------------------------------------------------

    def _domain_lock(self, domain: str) -> asyncio.Lock:
        lock = self._domain_locks.get(domain)
        if lock is None:
            lock = asyncio.Lock()
            self._domain_locks[domain] = lock
        return lock

    async def _wait_for_delay(self, domain: str) -> None:
        last = self._last_fetch.get(domain)
        if last is not None:
            wait = self._settings.scrape_delay_seconds - (time.monotonic() - last)
            if wait > 0:
                await self._sleep(wait)

    async def _get_robots(self, domain: str) -> RobotFileParser | None:
        if domain in self._robots:
            return self._robots[domain]
        parser: RobotFileParser | None = None
        try:
            resp = await self._client.get(f"https://{domain}/robots.txt")
            if resp.is_success and resp.text:
                parser = RobotFileParser()
                parser.parse(resp.text.splitlines())
        except httpx.HTTPError:
            parser = None
        self._robots[domain] = parser
        return parser

    async def _robots_allowed(self, url: str) -> bool:
        parser = await self._get_robots(domain_of(url))
        if parser is None:
            return True
        return parser.can_fetch(self._settings.user_agent, url)

    # -- fetching ------------------------------------------------------------

    async def _fetch(self, url: str) -> tuple[str | None, int]:
        """Return (raw_html, network_bytes). Cache hits report 0 bytes; None on any skip."""
        if self._storage is not None:
            cached = await self._storage.get_cached_scrape(
                url, ttl_hours=self._settings.scrape_cache_ttl_hours
            )
            if cached is not None:
                return cached, 0

        if not await self._robots_allowed(url):
            return None, 0

        domain = domain_of(url)
        async with self._domain_lock(domain):
            await self._wait_for_delay(domain)
            try:
                resp = await self._client.get(url)
            finally:
                self._last_fetch[domain] = time.monotonic()

        if not resp.is_success:
            return None, 0
        content_type = resp.headers.get("content-type", "").lower()
        if content_type and "html" not in content_type:
            return None, 0
        nbytes = len(resp.content)
        if nbytes > self._settings.max_response_bytes:
            return None, 0

        html = resp.text
        if self._storage is not None:
            await self._storage.cache_scrape(url, html)
        return html, nbytes

    # -- public API ----------------------------------------------------------

    async def scrape_firm(self, url: str) -> ScrapeResult:
        """Scrape one firm: homepage plus relevant internal pages, combined to text."""
        try:
            home_html, home_bytes = await self._fetch(url)
        except httpx.HTTPError as exc:
            return ScrapeResult(url=url, error=f"homepage fetch error: {exc}")
        if home_html is None:
            return ScrapeResult(url=url, error="homepage unavailable")

        title, text = extract_text(home_html)
        pages = [ScrapedPage(url=url, page_type="home", title=title, text=text)]
        total_bytes = home_bytes

        for link_url, page_type in select_relevant_links(
            home_html, url, self._settings.max_pages_per_firm
        ):
            try:
                html, nbytes = await self._fetch(link_url)
            except httpx.HTTPError:
                continue
            if html is None:
                continue
            page_title, page_text = extract_text(html)
            pages.append(
                ScrapedPage(url=link_url, page_type=page_type, title=page_title, text=page_text)
            )
            total_bytes += nbytes

        return ScrapeResult(
            url=url,
            pages=pages,
            combined_text=combine_pages(pages),
            bytes_fetched=total_bytes,
        )

    async def scrape_firms(self, urls: list[str]) -> list[ScrapeResult]:
        """Scrape many firms concurrently, bounded by scrape_concurrency."""
        semaphore = asyncio.Semaphore(self._settings.scrape_concurrency)

        async def one(url: str) -> ScrapeResult:
            async with semaphore:
                return await self.scrape_firm(url)

        return await asyncio.gather(*(one(url) for url in urls))


def get_scraper(storage: Storage | None = None) -> Scraper:
    """Read env/settings and return a Scraper. Use as an async context manager."""
    return Scraper(ScrapeSettings(), storage=storage)

"""Tests for the polite scraper: link discovery, text extraction, robots, caching, politeness."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from lead_agent.scraper import (
    ScrapedPage,
    Scraper,
    ScrapeResult,
    ScrapeSettings,
    combine_pages,
    extract_text,
    select_relevant_links,
)
from lead_agent.storage import Storage

# ---------------------------------------------------------------------------
# Mock HTTP routing
# ---------------------------------------------------------------------------

class RouteHandler:
    """MockTransport handler routing by full URL. /robots.txt handled specially."""

    def __init__(self, pages: dict[str, dict], robots: str | None = None) -> None:
        self._pages = pages
        self._robots = robots
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith("/robots.txt"):
            if self._robots is None:
                return httpx.Response(404)
            return httpx.Response(200, text=self._robots)
        spec = self._pages.get(url)
        if spec is None:
            return httpx.Response(404)
        if spec.get("raise"):
            raise httpx.ConnectError("boom")
        content_type = spec.get("content_type", "text/html")
        body: str = spec.get("html", "")
        return httpx.Response(
            spec.get("status", 200),
            content=body.encode(),
            headers={"content-type": content_type},
        )


@contextlib.asynccontextmanager
async def make_scraper(
    handler: RouteHandler,
    *,
    settings: ScrapeSettings | None = None,
    storage: Storage | None = None,
    sleep: AsyncMock | None = None,
) -> AsyncIterator[Scraper]:
    settings = settings or ScrapeSettings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        kwargs = {"storage": storage, "client": client}
        if sleep is not None:
            kwargs["sleep"] = sleep
        yield Scraper(settings, **kwargs)


# ---------------------------------------------------------------------------
# select_relevant_links
# ---------------------------------------------------------------------------

class TestSelectRelevantLinks:
    def test_matches_page_types_in_priority_order(self) -> None:
        html = """
        <a href="/contact">Contact Us</a>
        <a href="/practice-areas">Practice Areas</a>
        <a href="/attorneys">Our Attorneys</a>
        <a href="/about">About the Firm</a>
        """
        out = select_relevant_links(html, "https://smithlaw.com/", max_pages=5)
        assert [pt for _, pt in out] == ["about", "team", "practice", "contact"]

    def test_resolves_relative_urls(self) -> None:
        html = '<a href="about-us.html">About</a>'
        out = select_relevant_links(html, "https://smithlaw.com/", max_pages=5)
        assert out == [("https://smithlaw.com/about-us.html", "about")]

    def test_excludes_external_domains(self) -> None:
        html = '<a href="https://other.com/about">About</a>'
        assert select_relevant_links(html, "https://smithlaw.com/", max_pages=5) == []

    def test_excludes_mailto_tel_and_assets(self) -> None:
        html = """
        <a href="mailto:x@smithlaw.com">Email</a>
        <a href="tel:+1">Call</a>
        <a href="/brochure-about.pdf">About PDF</a>
        """
        assert select_relevant_links(html, "https://smithlaw.com/", max_pages=5) == []

    def test_excludes_homepage_self_link(self) -> None:
        html = '<a href="/">About Home</a>'
        assert select_relevant_links(html, "https://smithlaw.com/", max_pages=5) == []

    def test_one_link_per_page_type_first_wins(self) -> None:
        html = """
        <a href="/about">About</a>
        <a href="/about-firm">About the firm</a>
        """
        out = select_relevant_links(html, "https://smithlaw.com/", max_pages=5)
        assert out == [("https://smithlaw.com/about", "about")]

    def test_respects_max_pages_cap(self) -> None:
        html = """
        <a href="/about">About</a>
        <a href="/attorneys">Attorneys</a>
        <a href="/practice">Practice</a>
        <a href="/contact">Contact</a>
        """
        # max_pages=3 -> homepage + 2 internal
        out = select_relevant_links(html, "https://smithlaw.com/", max_pages=3)
        assert len(out) == 2
        assert [pt for _, pt in out] == ["about", "team"]

    def test_unmatched_links_ignored(self) -> None:
        html = '<a href="/blog/post-1">Some blog post</a>'
        assert select_relevant_links(html, "https://smithlaw.com/", max_pages=5) == []


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_returns_title_and_text(self) -> None:
        html = "<html><head><title>Smith Law</title></head><body><p>We do CRE.</p></body></html>"
        title, text = extract_text(html)
        assert title == "Smith Law"
        assert "We do CRE." in text

    def test_strips_script_and_style(self) -> None:
        html = (
            "<body><script>var x=1;</script><style>.a{color:red}</style>"
            "<p>Visible</p></body>"
        )
        _, text = extract_text(html)
        assert "Visible" in text
        assert "var x" not in text
        assert "color:red" not in text

    def test_strips_nav_header_footer(self) -> None:
        html = (
            "<body><nav>MENU</nav><header>TOP</header>"
            "<p>Body content</p><footer>BOTTOM</footer></body>"
        )
        _, text = extract_text(html)
        assert "Body content" in text
        assert "MENU" not in text
        assert "BOTTOM" not in text

    def test_no_title_returns_empty(self) -> None:
        title, _ = extract_text("<body><p>hi</p></body>")
        assert title == ""

    def test_collapses_whitespace(self) -> None:
        _, text = extract_text("<body><p>a\n\n   b\t c</p></body>")
        assert text == "a b c"


# ---------------------------------------------------------------------------
# combine_pages
# ---------------------------------------------------------------------------

class TestCombinePages:
    def test_labels_sections_with_type_and_url(self) -> None:
        pages = [
            ScrapedPage(url="https://x.com/", page_type="home", title="Home", text="body1"),
            ScrapedPage(url="https://x.com/about", page_type="about", title="About", text="body2"),
        ]
        out = combine_pages(pages)
        assert "## HOME — https://x.com/" in out
        assert "## ABOUT — https://x.com/about" in out
        assert "body1" in out and "body2" in out

    def test_truncates_per_page(self) -> None:
        pages = [ScrapedPage(url="https://x.com/", page_type="home", text="A" * 100)]
        out = combine_pages(pages, max_chars_per_page=10)
        assert out.count("A") == 10

    def test_truncates_total(self) -> None:
        pages = [
            ScrapedPage(url=f"https://x.com/{i}", page_type="home", text="A" * 50)
            for i in range(10)
        ]
        out = combine_pages(pages, max_chars_per_page=50, max_total_chars=60)
        assert len(out) <= 60


# ---------------------------------------------------------------------------
# Scraper.scrape_firm
# ---------------------------------------------------------------------------

class TestScrapeFirm:
    async def test_fetches_homepage_and_internal_pages(self) -> None:
        handler = RouteHandler(
            {
                "https://smithlaw.com/": {
                    "html": "<title>Smith Law</title><a href='/about'>About</a>"
                    "<a href='/attorneys'>Attorneys</a>"
                },
                "https://smithlaw.com/about": {"html": "<title>About</title><p>Founded 1990.</p>"},
                "https://smithlaw.com/attorneys": {
                    "html": "<title>Attorneys</title><p>7 attorneys.</p>"
                },
            }
        )
        async with make_scraper(handler) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
        assert result.ok
        assert {p.page_type for p in result.pages} == {"home", "about", "team"}
        assert "Founded 1990." in result.combined_text
        assert "7 attorneys." in result.combined_text

    async def test_homepage_404_returns_error(self) -> None:
        handler = RouteHandler({})  # everything 404
        async with make_scraper(handler) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
        assert not result.ok
        assert result.error == "homepage unavailable"
        assert result.pages == []

    async def test_homepage_connection_error_returns_error(self) -> None:
        handler = RouteHandler({"https://smithlaw.com/": {"raise": True}})
        async with make_scraper(handler) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
        assert not result.ok
        assert result.error is not None
        assert "homepage fetch error" in result.error

    async def test_non_html_homepage_skipped(self) -> None:
        handler = RouteHandler(
            {"https://smithlaw.com/": {"html": "%PDF-1.4", "content_type": "application/pdf"}}
        )
        async with make_scraper(handler) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
        assert result.error == "homepage unavailable"

    async def test_failed_internal_page_skipped_not_fatal(self) -> None:
        handler = RouteHandler(
            {
                "https://smithlaw.com/": {
                    "html": "<title>Home</title><a href='/about'>About</a>"
                    "<a href='/contact'>Contact</a>"
                },
                "https://smithlaw.com/about": {"html": "<p>About body</p>"},
                # /contact intentionally absent -> 404, should be skipped
            }
        )
        async with make_scraper(handler) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
        assert result.ok
        assert {p.page_type for p in result.pages} == {"home", "about"}


class TestRobots:
    async def test_disallowed_page_not_fetched(self) -> None:
        handler = RouteHandler(
            {
                "https://smithlaw.com/": {
                    "html": "<title>Home</title><a href='/about'>About</a>"
                    "<a href='/attorneys'>Attorneys</a>"
                },
                "https://smithlaw.com/about": {"html": "<p>About body</p>"},
                "https://smithlaw.com/attorneys": {"html": "<p>Should not be fetched</p>"},
            },
            robots="User-agent: *\nDisallow: /attorneys",
        )
        async with make_scraper(handler) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
        assert {p.page_type for p in result.pages} == {"home", "about"}
        assert "https://smithlaw.com/attorneys" not in handler.requests

    async def test_missing_robots_allows_all(self) -> None:
        handler = RouteHandler(
            {
                "https://smithlaw.com/": {
                    "html": "<title>Home</title><a href='/about'>About</a>"
                },
                "https://smithlaw.com/about": {"html": "<p>About body</p>"},
            },
            robots=None,  # 404
        )
        async with make_scraper(handler) as scraper:
            result = await scraper.scrape_firm("https://smithlaw.com/")
        assert {p.page_type for p in result.pages} == {"home", "about"}


class TestPoliteness:
    async def test_delay_applied_between_same_domain_fetches(self) -> None:
        sleep = AsyncMock()
        settings = ScrapeSettings(scrape_delay_seconds=10.0)
        handler = RouteHandler(
            {
                "https://smithlaw.com/": {
                    "html": "<title>Home</title><a href='/about'>About</a>"
                },
                "https://smithlaw.com/about": {"html": "<p>About body</p>"},
            }
        )
        async with make_scraper(handler, settings=settings, sleep=sleep) as scraper:
            await scraper.scrape_firm("https://smithlaw.com/")
        # homepage fetch: no prior timestamp -> no wait; /about fetch -> one delay
        assert sleep.await_count == 1
        waited = sleep.await_args.args[0]
        assert 0 < waited <= 10.0


class TestScrapeCaching:
    async def test_second_scrape_served_from_cache_no_network(self, tmp_path: Path) -> None:
        handler = RouteHandler(
            {
                "https://smithlaw.com/": {
                    "html": "<title>Home</title><a href='/about'>About</a>"
                },
                "https://smithlaw.com/about": {"html": "<p>About body</p>"},
            }
        )
        async with Storage(tmp_path / "t.db") as db, make_scraper(handler, storage=db) as scraper:
            first = await scraper.scrape_firm("https://smithlaw.com/")
            requests_after_first = len(handler.requests)
            second = await scraper.scrape_firm("https://smithlaw.com/")
        assert first.ok and second.ok
        # No new network requests on the second run (homepage + about served from cache)
        assert len(handler.requests) == requests_after_first
        assert second.combined_text == first.combined_text

    async def test_raw_html_cached(self, tmp_path: Path) -> None:
        handler = RouteHandler(
            {"https://smithlaw.com/": {"html": "<title>Home</title><p>Raw HTML body</p>"}}
        )
        async with Storage(tmp_path / "t.db") as db, make_scraper(handler, storage=db) as scraper:
            await scraper.scrape_firm("https://smithlaw.com/")
            cached = await db.get_cached_scrape("https://smithlaw.com/")
        assert cached is not None
        assert "<title>Home</title>" in cached  # raw HTML, not cleaned text


class TestScrapeFirms:
    async def test_scrapes_multiple_firms(self) -> None:
        handler = RouteHandler(
            {
                "https://a.com/": {"html": "<title>A</title><p>Firm A</p>"},
                "https://b.com/": {"html": "<title>B</title><p>Firm B</p>"},
            }
        )
        async with make_scraper(handler) as scraper:
            results = await scraper.scrape_firms(["https://a.com/", "https://b.com/"])
        assert len(results) == 2
        assert all(r.ok for r in results)
        assert {r.url for r in results} == {"https://a.com/", "https://b.com/"}

    async def test_empty_list_returns_empty(self) -> None:
        handler = RouteHandler({})
        async with make_scraper(handler) as scraper:
            assert await scraper.scrape_firms([]) == []


def test_scrape_result_ok_property() -> None:
    assert not ScrapeResult(url="x", error="boom").ok
    assert not ScrapeResult(url="x", combined_text="").ok
    assert ScrapeResult(url="x", combined_text="text").ok

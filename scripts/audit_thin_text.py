"""Audit thin combined_text across the last full run. Read-only."""
import asyncio
import json
import sqlite3
from pathlib import Path

from lead_agent.scraper import (
    ScrapedPage,
    combine_pages,
    extract_text,
    select_relevant_links,
)
from lead_agent.storage import Storage

RUN_ID = "8ef4c9c0-33e5-477b-8f27-cf554e4355f7"
THIN_COMBINED = 3000  # combined_text under this is suspiciously thin
THIN_PAGE = 500  # any page < this chars after strip
SUBSTANTIAL_HTML = 10000  # but cached HTML was > this — JS-rendered signal
KEY_PAGE_TYPES = ("team", "practice")  # the ones we care about for JS gap


async def audit_firm(db: Storage, url: str) -> dict:
    home_html = await db.get_cached_scrape(url, ttl_hours=168)
    if home_html is None:
        return {"url": url, "skipped": "no homepage in cache"}
    links = select_relevant_links(home_html, url, max_pages=5)
    home_title, home_text = extract_text(home_html)
    pages = [ScrapedPage(url=url, page_type="home", title=home_title, text=home_text)]
    per_page = [{
        "page_type": "home",
        "url": url,
        "html_bytes": len(home_html),
        "text_chars": len(home_text),
    }]
    for u, pt in links:
        cached = await db.get_cached_scrape(u, ttl_hours=168)
        if cached is None:
            per_page.append({"page_type": pt, "url": u, "html_bytes": 0, "text_chars": 0})
            continue
        t, x = extract_text(cached)
        pages.append(ScrapedPage(url=u, page_type=pt, title=t, text=x))
        per_page.append({
            "page_type": pt,
            "url": u,
            "html_bytes": len(cached),
            "text_chars": len(x),
        })
    combined = combine_pages(pages)
    # JS-rendered signal: any KEY page with substantial HTML but tiny text
    js_pages = [
        p for p in per_page
        if p["page_type"] in KEY_PAGE_TYPES
        and p["html_bytes"] >= SUBSTANTIAL_HTML
        and p["text_chars"] < THIN_PAGE
    ]
    return {
        "url": url,
        "combined_chars": len(combined),
        "per_page": per_page,
        "js_signal_pages": [p["page_type"] for p in js_pages],
        "thin_combined": len(combined) < THIN_COMBINED,
        "js_rendered": len(js_pages) > 0,
    }


async def main():
    conn = sqlite3.connect("data/lead_agent.db")
    conn.row_factory = sqlite3.Row
    firms = conn.execute(
        "SELECT url, stage, score, extracted_profile FROM firms WHERE run_id=? ORDER BY url",
        (RUN_ID,),
    ).fetchall()
    conn.close()

    async with Storage(Path("data/lead_agent.db")) as db:
        results = []
        for f in firms:
            audit = await audit_firm(db, f["url"])
            audit["stage"] = f["stage"]
            audit["score"] = f["score"]
            profile = json.loads(f["extracted_profile"]) if f["extracted_profile"] else {}
            audit["firm_name"] = profile.get("firm_name") or f["url"]
            audit["attorney_count"] = profile.get("attorney_count")
            results.append(audit)

    # Table 1: per firm (completed only)
    completed = [r for r in results if r["stage"] == "completed"]
    print(f"=== Per-firm audit ({len(completed)} completed firms) ===")
    print(f"{'#':>3}  {'combined':>8}  {'score':>6}  {'attn':>4}  {'flags':<15}  firm")
    for i, r in enumerate(sorted(completed, key=lambda x: x["combined_chars"]), 1):
        flags = []
        if r["thin_combined"]: flags.append("THIN")
        if r["js_rendered"]: flags.append("JS:" + ",".join(r["js_signal_pages"]))
        flag_str = " ".join(flags)
        score = f"{r['score']:.3f}" if r["score"] is not None else "  -  "
        attn = str(r["attorney_count"]) if r["attorney_count"] is not None else "-"
        print(f"{i:>3}  {r['combined_chars']:>8}  {score:>6}  {attn:>4}  {flag_str:<15}  {r['firm_name'][:60]}")

    # Summary
    n_thin = sum(1 for r in completed if r["thin_combined"])
    n_js = sum(1 for r in completed if r["js_rendered"])
    n_thin_or_js = sum(1 for r in completed if r["thin_combined"] or r["js_rendered"])
    print()
    print("=== Summary ===")
    print(f"Completed firms: {len(completed)}")
    print(f"Thin combined_text (<{THIN_COMBINED} chars): {n_thin}")
    print(f"JS-rendered signature (key page <{THIN_PAGE} chars from >{SUBSTANTIAL_HTML}B HTML): {n_js}")
    print(f"Thin OR JS-rendered: {n_thin_or_js}")

    # Score correlation
    print()
    print("=== Score correlation ===")
    healthy = [r for r in completed if not r["thin_combined"] and not r["js_rendered"] and r["score"] is not None]
    thin_or_js = [r for r in completed if (r["thin_combined"] or r["js_rendered"]) and r["score"] is not None]
    def stats(rows):
        scores = [r["score"] for r in rows]
        if not scores: return "(none)"
        return f"n={len(scores)}, mean={sum(scores)/len(scores):.3f}, min={min(scores):.3f}, max={max(scores):.3f}"
    print(f"Healthy (full text):    {stats(healthy)}")
    print(f"Thin or JS-rendered:    {stats(thin_or_js)}")

    # Buckets by combined_chars
    buckets = [(0, 1500), (1500, 3000), (3000, 6000), (6000, 12000), (12000, 24001)]
    print()
    print("=== Score by combined_text bucket ===")
    for lo, hi in buckets:
        rows = [r for r in completed if lo <= r["combined_chars"] < hi and r["score"] is not None]
        s = stats(rows)
        qualified = sum(1 for r in rows if r["score"] >= 0.55)
        print(f"  {lo:>5}-{hi:<5} chars: {s}  qualified(>=0.55): {qualified}")

    # Failed firms (just count, no audit since no cache)
    failed = [r for r in results if r["stage"] == "failed"]
    print()
    print(f"Failed firms (no scrape): {len(failed)}")


if __name__ == "__main__":
    asyncio.run(main())

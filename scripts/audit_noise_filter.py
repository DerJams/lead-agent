"""Compare old vs new select_relevant_links on every firm in the last run.

Old: current scraper.py with _looks_like_noise short-circuited to always False.
New: scraper.py as-is.

Reports per firm: URLs dropped by the new filter, and any page-type slot
reassignments (e.g. /press-release lost the practice slot to /practice-areas).
Read-only — no LLM calls, no scraping.
"""
import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import patch

from lead_agent import scraper
from lead_agent.scraper import select_relevant_links
from lead_agent.storage import Storage

RUN_ID = "8ef4c9c0-33e5-477b-8f27-cf554e4355f7"


async def main() -> None:
    conn = sqlite3.connect("data/lead_agent.db")
    conn.row_factory = sqlite3.Row
    firms = conn.execute(
        "SELECT url FROM firms WHERE run_id=? AND stage IN ('completed', 'failed') ORDER BY url",
        (RUN_ID,),
    ).fetchall()
    conn.close()

    async with Storage(Path("data/lead_agent.db")) as db:
        new_results: list[tuple[str, list[tuple[str, str]]]] = []
        old_results: list[tuple[str, list[tuple[str, str]]]] = []

        for f in firms:
            url = f["url"]
            html = await db.get_cached_scrape(url, ttl_hours=168)
            if html is None:
                continue
            # New (filter active)
            new_links = select_relevant_links(html, url, max_pages=5)
            # Old (filter disabled)
            with patch.object(scraper, "_looks_like_noise", return_value=False):
                old_links = select_relevant_links(html, url, max_pages=5)
            new_results.append((url, new_links))
            old_results.append((url, old_links))

    # Diff per firm
    print("=== Per-firm changes ===")
    n_dropped_urls = 0
    n_slot_replaced = 0
    n_slot_lost = 0
    n_firms_changed = 0
    for (url, new_links), (_, old_links) in zip(new_results, old_results, strict=True):
        old_map = dict((pt, u) for u, pt in old_links)
        new_map = dict((pt, u) for u, pt in new_links)
        changes = []
        for pt in ("about", "team", "practice", "contact"):
            old_u, new_u = old_map.get(pt), new_map.get(pt)
            if old_u == new_u:
                continue
            if old_u and new_u:
                changes.append(f"  [{pt}] REPLACED: {old_u}  ->  {new_u}")
                n_slot_replaced += 1
                n_dropped_urls += 1
            elif old_u and not new_u:
                changes.append(f"  [{pt}] LOST (no fallback): {old_u}")
                n_slot_lost += 1
                n_dropped_urls += 1
        if changes:
            n_firms_changed += 1
            print(f"\n--- {url} ---")
            for c in changes:
                print(c)

    print()
    print("=== Summary ===")
    print(f"Firms inspected: {len(new_results)}")
    print(f"Firms with any change: {n_firms_changed}")
    print(f"URLs dropped: {n_dropped_urls}")
    print(f"  Slots REPLACED with cleaner URL: {n_slot_replaced}")
    print(f"  Slots LOST (no fallback in homepage anchors): {n_slot_lost}")


if __name__ == "__main__":
    asyncio.run(main())

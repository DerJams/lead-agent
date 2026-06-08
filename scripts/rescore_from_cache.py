"""Re-score the completed firms from a given run using cached scrapes.

Reuses scrape_cache (no fresh HTTP), reuses each firm's persisted
extracted_profile (no re-extraction LLM call), and runs only the scoring
LLM call against the current ICP config + scorer code. Cheap validation
for scorer / config changes.

Each firm's full result is appended to a JSONL artifact the instant
score_firm returns and the file is flushed before the next call, so a
mid-run crash (e.g. Cerebras daily quota) preserves everything paid
for. The artifact is self-describing: every line carries the ICP
weights it was scored under, so weight/threshold sweeps can run offline
against the file with no further model calls.
"""
import argparse
import asyncio
import datetime as dt
import json
import sqlite3
from pathlib import Path

from lead_agent.config import load_icp
from lead_agent.llm import get_client
from lead_agent.scorer import score_firm
from lead_agent.scraper import ScrapeSettings, Scraper
from lead_agent.storage import Storage

PRIOR_RUN = "e81032ba-ffed-4b23-b086-b08ae750859b"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _default_output_path(prior_run: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = prior_run.split("-")[0]
    return Path(f"data/rescore_{short}_{stamp}.jsonl")


async def main(prior_run: str, config_path: Path, output_path: Path) -> None:
    conn = sqlite3.connect("data/lead_agent.db")
    conn.row_factory = sqlite3.Row
    firms = conn.execute(
        "SELECT url, score, extracted_profile FROM firms "
        "WHERE run_id=? AND stage='completed' ORDER BY score DESC",
        (prior_run,),
    ).fetchall()
    conn.close()
    print(f"Loaded {len(firms)} completed firms from run {prior_run}")

    icp = load_icp(config_path)
    client = get_client()
    weights = {s.name: s.weight for s in icp.soft_signals}
    print(f"ICP: {icp.name}")
    print("Signals + weights:")
    for name, w in weights.items():
        print(f"  {name:<22} {w}")
    print()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing per-firm JSONL to: {output_path}")
    print()

    results: list[dict] = []
    with output_path.open("w", encoding="utf-8") as artifact:
        async with Storage(Path("data/lead_agent.db")) as db:
            async with Scraper(ScrapeSettings(), storage=db) as scraper:
                for i, f in enumerate(firms, 1):
                    url = f["url"]
                    profile = (
                        json.loads(f["extracted_profile"]) if f["extracted_profile"] else None
                    )
                    scrape = await scraper.scrape_firm(url)
                    if not scrape.ok:
                        print(f"[{i:>2}/{len(firms)}] SKIP {url}  scrape failed")
                        continue
                    new = await score_firm(profile, scrape.combined_text, icp, client)
                    old_score = f["score"]
                    attn = (profile or {}).get("attorney_count")
                    firm_name = (profile or {}).get("firm_name") or url
                    record = {
                        "timestamp": _now_iso(),
                        "prior_run_id": prior_run,
                        "icp_name": icp.name,
                        "weights": weights,
                        "url": url,
                        "firm_name": firm_name,
                        "attorney_count": attn,
                        "passed_hard_filters": new.passed_hard_filters,
                        "signal_ratings": new.signal_ratings,
                        "score": new.score,
                        "old_score": old_score,
                    }
                    # Write + flush before moving on so a crash here preserves
                    # everything we've paid LLM tokens for.
                    artifact.write(json.dumps(record, ensure_ascii=False) + "\n")
                    artifact.flush()
                    results.append({
                        "url": url,
                        "firm_name": firm_name,
                        "attorney_count": attn,
                        "old_score": old_score,
                        "new_score": new.score,
                        "signal_ratings": new.signal_ratings,
                    })
                    delta = new.score - (old_score or 0)
                    print(
                        f"[{i:>2}/{len(firms)}] {old_score:.3f} -> {new.score:.3f}  "
                        f"({delta:+.3f}) attn={attn}  {firm_name[:50]}"
                    )

    # Sort by new_score desc
    results.sort(key=lambda r: r["new_score"], reverse=True)

    # Spotlight firms
    SPOT = {
        "https://www.andrewsmyers.com/": "Andrews Myers (68 attorneys)",
        "https://goldensteves.com/": "Golden Steves (17 attorneys)",
        "https://unelllaw.com/": "Unell (1 attorney)",
        "https://wallacetexaslaw.com/": "Wallace Law (1 attorney)",
        "https://sa-law.com/": "R L Wilson Law (null attorney_count)",
    }
    print()
    print("=== Spotlight firms ===")
    for url, label in SPOT.items():
        r = next((x for x in results if x["url"] == url), None)
        if not r:
            print(f"  {label}: NOT FOUND")
            continue
        print(f"  {label}")
        print(f"    old={r['old_score']:.3f}  new={r['new_score']:.3f}  delta={r['new_score'] - r['old_score']:+.3f}  attn={r['attorney_count']}")
        for sig, rating in r["signal_ratings"].items():
            print(f"    {sig:<22} = {rating}")

    # Qualification status changes
    THRESH = 0.55
    changed = []
    for r in results:
        old_q = (r["old_score"] or 0) >= THRESH
        new_q = r["new_score"] >= THRESH
        if old_q != new_q:
            changed.append((r, old_q, new_q))
    print()
    print(f"=== Qualification status changes (threshold {THRESH}) ===")
    if not changed:
        print("  (none)")
    for r, old_q, new_q in changed:
        direction = "LOST" if old_q else "GAINED"
        print(f"  {direction}: {r['firm_name'][:50]}  {r['old_score']:.3f} -> {r['new_score']:.3f}  attn={r['attorney_count']}")

    # Top 20 new ranking
    print()
    print("=== Top 20 by new score ===")
    print(f"  {'old':>5}  {'new':>5}  {'delta':>7}  {'attn':>5}  firm")
    for r in results[:20]:
        d = r["new_score"] - (r["old_score"] or 0)
        attn = str(r["attorney_count"]) if r["attorney_count"] is not None else "-"
        print(f"  {(r['old_score'] or 0):.3f}  {r['new_score']:.3f}  {d:+7.3f}  {attn:>5}  {r['firm_name'][:55]}")

    # Summary
    n_old_q = sum(1 for r in results if (r["old_score"] or 0) >= THRESH)
    n_new_q = sum(1 for r in results if r["new_score"] >= THRESH)
    print()
    print(f"Qualified (>= {THRESH}):  old={n_old_q}  new={n_new_q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=PRIOR_RUN)
    parser.add_argument("--config", default="configs/icp_law_boutique.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL output path; defaults to data/rescore_<run>_<timestamp>.jsonl",
    )
    args = parser.parse_args()
    out = Path(args.output) if args.output else _default_output_path(args.run)
    asyncio.run(main(args.run, Path(args.config), out))

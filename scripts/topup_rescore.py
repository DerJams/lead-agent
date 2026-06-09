"""One-off: top up an existing rescore JSONL to all completed firms from a run.

The existing JSONL was produced at older weights but already carries the
1-10 signal_ratings (which are weight-independent). This script:

1. Recomputes each existing firm's score under the current ICP weights
   via combine_score -- math only, no LLM call.
2. LLM-scores only the firms in the run that are NOT yet in the JSONL,
   using cached scrapes plus one LLM call per firm.

Output is a single combined JSONL where every record carries the same
current weights, ready for a threshold sweep. Crash-safe per-firm
write-and-flush is preserved.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sqlite3
from pathlib import Path

from lead_agent.config import load_icp
from lead_agent.llm import get_client
from lead_agent.scorer import combine_score, score_firm
from lead_agent.scraper import ScrapeSettings, Scraper
from lead_agent.storage import Storage

PRIOR_RUN = "e81032ba-ffed-4b23-b086-b08ae750859b"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _default_output_path(prior_run: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = prior_run.split("-")[0]
    return Path(f"data/rescore_{short}_{stamp}_full.jsonl")


async def main(
    prior_run: str, config_path: Path, existing_path: Path, output_path: Path
) -> None:
    existing: dict[str, dict] = {}
    with existing_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            existing[rec["url"]] = rec
    print(f"Loaded {len(existing)} existing records from {existing_path}")

    icp = load_icp(config_path)
    weights = {s.name: s.weight for s in icp.soft_signals}
    print(f"ICP: {icp.name}")
    print("Current committed weights:")
    for name, w in weights.items():
        print(f"  {name:<22} {w}")
    print()

    conn = sqlite3.connect("data/lead_agent.db")
    conn.row_factory = sqlite3.Row
    firms = conn.execute(
        "SELECT url, score, extracted_profile FROM firms "
        "WHERE run_id=? AND stage='completed' ORDER BY score DESC",
        (prior_run,),
    ).fetchall()
    conn.close()
    print(f"Found {len(firms)} completed firms in run {prior_run}")

    missing = [f for f in firms if f["url"] not in existing]
    print(f"Already scored: {len(existing)}; need LLM calls: {len(missing)}")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing combined JSONL to: {output_path}")
    print()

    with output_path.open("w", encoding="utf-8") as artifact:
        # 1. Recompute scores for the existing firms under current weights (no LLM)
        recomputed = 0
        for url, rec in existing.items():
            score, _ = combine_score(
                rec["signal_ratings"],
                icp.soft_signals,
                icp.scoring.soft_signal_normalization,
            )
            new_rec = {
                "timestamp": rec["timestamp"],
                "recomputed_at": _now_iso(),
                "prior_run_id": prior_run,
                "icp_name": icp.name,
                "weights": weights,
                "url": url,
                "firm_name": rec.get("firm_name"),
                "attorney_count": rec.get("attorney_count"),
                "passed_hard_filters": rec.get("passed_hard_filters"),
                "signal_ratings": rec["signal_ratings"],
                "score": score,
                "old_score": rec.get("old_score"),
                "prior_score_at_old_weights": rec.get("score"),
                "source": "recomputed_from_existing",
            }
            artifact.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            artifact.flush()
            recomputed += 1
        print(f"Recomputed {recomputed} existing records under current weights.")
        print()

        if not missing:
            print("Nothing to top up; all firms already scored.")
            return

        # 2. LLM-score the missing firms
        client = get_client()
        async with Storage(Path("data/lead_agent.db")) as db:
            async with Scraper(ScrapeSettings(), storage=db) as scraper:
                for i, f in enumerate(missing, 1):
                    url = f["url"]
                    profile = (
                        json.loads(f["extracted_profile"]) if f["extracted_profile"] else None
                    )
                    scrape = await scraper.scrape_firm(url)
                    if not scrape.ok:
                        print(f"[{i:>2}/{len(missing)}] SKIP {url}  scrape failed")
                        continue
                    new = await score_firm(profile, scrape.combined_text, icp, client)
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
                        "old_score": f["score"],
                        "source": "fresh_llm_score",
                    }
                    artifact.write(json.dumps(record, ensure_ascii=False) + "\n")
                    artifact.flush()
                    delta = new.score - (f["score"] or 0)
                    print(
                        f"[{i:>2}/{len(missing)}] {(f['score'] or 0):.3f} -> {new.score:.3f}  "
                        f"({delta:+.3f}) attn={attn}  {firm_name[:50]}"
                    )

    print()
    print("Done. Combined JSONL ready for threshold sweep.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=PRIOR_RUN)
    parser.add_argument("--config", default="configs/icp_law_boutique.yaml")
    parser.add_argument(
        "--existing",
        required=True,
        help="Existing rescore JSONL to top up (firms in this file are not re-LLM-scored).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Combined JSONL output path; defaults to data/rescore_<run>_<timestamp>_full.jsonl",
    )
    args = parser.parse_args()
    out = Path(args.output) if args.output else _default_output_path(args.run)
    asyncio.run(main(args.run, Path(args.config), Path(args.existing), out))

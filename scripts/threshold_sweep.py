"""Offline qualify-threshold sweep against a rescore JSONL artifact.

Reads each firm's final score from a JSONL produced by
rescore_from_cache.py / topup_rescore.py and reports, for three
candidate qualify cutoffs (0.50, 0.55, 0.60):

  - qualified count
  - ranked qualified list with attorney_count
  - the borderline band: every firm within 0.03 of the cutoff (above
    or below), to make calibration grounded in real firms rather than
    a target count
  - explicit BoyarMiller (15 attn) and Rogers & Whitley (3 attn)
    placement vs. each cutoff

No LLM calls. All firms in the JSONL must already carry the weight
vector being evaluated; the script does not reweight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CUTOFFS = [0.50, 0.55, 0.60]
BORDERLINE_HALF_WIDTH = 0.03

FLAGGED = {
    "https://www.boyarmiller.com/": "BoyarMiller (15 attn)",
    "https://www.rogerswhitleyllp.com/": "Rogers & Whitley (3 attn)",
}


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def fmt_attn(v: object) -> str:
    return str(v) if v is not None else "-"


def report(path: Path) -> None:
    records = load_records(path)
    print(f"Loaded {len(records)} firms from {path}")
    sample_weights = records[0].get("weights") if records else None
    if sample_weights:
        print("Weights in file:")
        for name, w in sample_weights.items():
            print(f"  {name:<22} {w}")
    print()

    ranked = sorted(records, key=lambda r: r["score"], reverse=True)

    for cutoff in CUTOFFS:
        qualified = [r for r in ranked if r["score"] >= cutoff]
        band_lo, band_hi = cutoff - BORDERLINE_HALF_WIDTH, cutoff + BORDERLINE_HALF_WIDTH
        borderline = [r for r in ranked if band_lo <= r["score"] <= band_hi]

        print(f"========== cutoff = {cutoff:.2f} ==========")
        print(f"Qualified: {len(qualified)} / {len(ranked)}")
        print()
        print("  Qualified list (score, attn, firm)")
        for r in qualified:
            attn = fmt_attn(r.get("attorney_count"))
            print(f"    {r['score']:.3f}  attn={attn:>4}  {r.get('firm_name', r['url'])[:55]}")
        print()
        print(f"  Borderline band [{band_lo:.2f}, {band_hi:.2f}]")
        for r in borderline:
            attn = fmt_attn(r.get("attorney_count"))
            side = "ABOVE" if r["score"] >= cutoff else "below"
            margin = r["score"] - cutoff
            print(
                f"    {r['score']:.3f}  ({margin:+.3f}, {side})  attn={attn:>4}  "
                f"{r.get('firm_name', r['url'])[:55]}"
            )
        print()
        print("  Flagged firms (per the prompt)")
        for url, label in FLAGGED.items():
            r = next((x for x in ranked if x["url"] == url), None)
            if r is None:
                print(f"    {label}: NOT in JSONL")
                continue
            margin = r["score"] - cutoff
            side = "qualified" if r["score"] >= cutoff else "DISQUALIFIED"
            print(f"    {label}: score={r['score']:.3f}  ({margin:+.3f} vs cutoff)  {side}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="rescore JSONL with current weights")
    args = parser.parse_args()
    report(args.jsonl)

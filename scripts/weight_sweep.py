"""Offline boutique_size weight sweep against a rescore JSONL artifact.

Reads each firm's per-signal 1-10 ratings from the JSONL emitted by
rescore_from_cache.py and recomputes the combined score under three
boutique_size weights (0.20, 0.25, 0.30), holding cre_specialization at
0.35 and deal_activity at 0.25 fixed, and letting boutique_size take
its extra weight from texas_market_depth and owner_operated
proportionally to their current shares (0.15 and 0.05 -> 3:1).

No LLM calls. Pure arithmetic on the ratings already in the file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Current committed weights (must sum to 1.0).
BASE = {
    "cre_specialization": 0.35,
    "deal_activity": 0.25,
    "boutique_size": 0.20,
    "texas_market_depth": 0.15,
    "owner_operated": 0.05,
}
SWEEP = [0.20, 0.25, 0.30]
THRESH = 0.55

SPOTLIGHT = {
    "https://www.andrewsmyers.com/": "Andrews Myers",
    "https://goldensteves.com/": "Golden Steves",
    "https://unelllaw.com/": "Unell",
    "https://wallacetexaslaw.com/": "Wallace Law",
    "https://sa-law.com/": "R L Wilson",
}
BOYAR_URL_HINTS = ("boyarmiller", "boyar miller")


def weights_for(boutique: float) -> dict[str, float]:
    """Return a full weight vector with boutique_size set to `boutique`.

    cre_specialization and deal_activity are pinned. The extra weight
    pulled into boutique_size is removed from texas_market_depth and
    owner_operated in proportion to their current shares (3:1).
    """
    extra = boutique - BASE["boutique_size"]
    tx_share = BASE["texas_market_depth"] / (
        BASE["texas_market_depth"] + BASE["owner_operated"]
    )
    ow_share = 1.0 - tx_share
    w = {
        "cre_specialization": BASE["cre_specialization"],
        "deal_activity": BASE["deal_activity"],
        "boutique_size": boutique,
        "texas_market_depth": BASE["texas_market_depth"] - extra * tx_share,
        "owner_operated": BASE["owner_operated"] - extra * ow_share,
    }
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-9, f"weights don't sum to 1: {total}"
    return w


def score_under(ratings: dict[str, int], weights: dict[str, float]) -> float:
    """Replicates scorer.combine_score with weighted_average normalization."""
    weighted = 0.0
    wt = 0.0
    for name, w in weights.items():
        r = ratings.get(name, 1)
        weighted += w * (r / 10.0)
        wt += w
    return max(0.0, min(1.0, weighted / wt if wt > 0 else 0.0))


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
    print()

    # Precompute per-weight score for every firm.
    runs: list[tuple[float, dict[str, float], list[dict]]] = []
    for w in SWEEP:
        wv = weights_for(w)
        rescored = []
        for rec in records:
            s = score_under(rec["signal_ratings"], wv)
            rescored.append({
                "url": rec["url"],
                "firm_name": rec.get("firm_name") or rec["url"],
                "attorney_count": rec.get("attorney_count"),
                "ratings": rec["signal_ratings"],
                "score": s,
            })
        rescored.sort(key=lambda r: r["score"], reverse=True)
        runs.append((w, wv, rescored))

    # === Weight vectors ===
    print("=== Weight vectors ===")
    header = f"{'signal':<22} " + " ".join(f"{w:>8.2f}" for w in SWEEP)
    print(header)
    for name in BASE:
        row = f"{name:<22} " + " ".join(f"{wv[name]:>8.4f}" for _, wv, _ in runs)
        print(row)
    print()

    # === Andrews Myers ===
    print("=== Andrews Myers (the firm we want excluded) ===")
    am_url = "https://www.andrewsmyers.com/"
    am_rec = next((r for r in records if r["url"] == am_url), None)
    if am_rec is None:
        print("  NOT FOUND")
    else:
        print(f"  attorney_count: {fmt_attn(am_rec.get('attorney_count'))}")
        print(f"  ratings: {am_rec['signal_ratings']}")
        print()
        print(f"  {'weight':>8}  {'score':>7}  {'vs 0.55':>9}")
        for w, _, rescored in runs:
            am = next(r for r in rescored if r["url"] == am_url)
            margin = am["score"] - THRESH
            print(f"  {w:>8.2f}  {am['score']:>7.3f}  {margin:>+9.3f}")
    print()

    # === Spotlight firms ===
    print("=== Spotlight firms ===")
    print(f"  {'firm':<28} {'attn':>5}  " + "  ".join(f"w={w:.2f}" for w in SWEEP))
    for url, label in SPOTLIGHT.items():
        attn_str = "-"
        cells = []
        for _, _, rescored in runs:
            r = next((x for x in rescored if x["url"] == url), None)
            if r is None:
                cells.append("  -   ")
                continue
            attn_str = fmt_attn(r["attorney_count"])
            cells.append(f"{r['score']:>6.3f}")
        print(f"  {label:<28} {attn_str:>5}  " + "  ".join(cells))
    print()

    # === BoyarMiller (15 attorneys check) ===
    print("=== BoyarMiller check (15 attorneys; prompt says decline steeply above 12) ===")
    boyar = None
    for rec in records:
        url_l = rec["url"].lower()
        name_l = (rec.get("firm_name") or "").lower()
        if any(h in url_l or h in name_l for h in BOYAR_URL_HINTS):
            boyar = rec
            break
    if boyar is None:
        print("  NOT FOUND in JSONL")
    else:
        print(f"  url: {boyar['url']}")
        print(f"  firm_name: {boyar.get('firm_name')}")
        print(f"  attorney_count: {fmt_attn(boyar.get('attorney_count'))}")
        print(f"  ratings: {boyar['signal_ratings']}")
        bs_rating = boyar["signal_ratings"].get("boutique_size")
        print(f"  boutique_size rating: {bs_rating}  (prompt-expected: roughly 3-5 for 15 attorneys)")
        print()
        print(f"  {'weight':>8}  {'score':>7}  {'qualified':>10}")
        for w, _, rescored in runs:
            b = next(r for r in rescored if r["url"] == boyar["url"])
            q = "YES" if b["score"] >= THRESH else "no"
            print(f"  {w:>8.2f}  {b['score']:>7.3f}  {q:>10}")
    print()

    # === Qualified counts and the full qualified set per weight ===
    print(f"=== Qualified counts (threshold {THRESH}) ===")
    for w, _, rescored in runs:
        n_q = sum(1 for r in rescored if r["score"] >= THRESH)
        print(f"  w={w:.2f}: {n_q}/{len(rescored)} qualified")
    print()

    print("=== Qualified firms per weight (with attn) ===")
    # Build a per-firm grid showing qualification across the sweep.
    union_q_urls = set()
    for _, _, rescored in runs:
        for r in rescored:
            if r["score"] >= THRESH:
                union_q_urls.add(r["url"])
    # Order by score under the smallest weight (0.20) for stable presentation.
    base_order = {r["url"]: r["score"] for r in runs[0][2]}
    ordered = sorted(union_q_urls, key=lambda u: base_order.get(u, 0), reverse=True)
    header = f"  {'firm':<40} {'attn':>5}  " + "  ".join(f"w={w:.2f}" for w in SWEEP)
    print(header)
    for url in ordered:
        rows = []
        firm_name = url
        attn = "-"
        for _, _, rescored in runs:
            r = next(x for x in rescored if x["url"] == url)
            firm_name = r["firm_name"]
            attn = fmt_attn(r["attorney_count"])
            mark = "*" if r["score"] >= THRESH else " "
            rows.append(f"{r['score']:>5.3f}{mark}")
        print(f"  {firm_name[:40]:<40} {attn:>5}  " + "  ".join(rows))
    print()
    print("  (* = qualified at threshold)")
    print()

    # === Firms whose qualification flipped across the sweep ===
    print("=== Firms whose qualification flips across the sweep ===")
    flipped = []
    for url in union_q_urls.union(
        {r["url"] for _, _, rs in runs for r in rs if r["score"] >= THRESH}
    ):
        per_w = []
        firm_name = url
        attn = "-"
        for _, _, rescored in runs:
            r = next(x for x in rescored if x["url"] == url)
            firm_name = r["firm_name"]
            attn = fmt_attn(r["attorney_count"])
            per_w.append(r["score"] >= THRESH)
        if len(set(per_w)) > 1:
            flipped.append((firm_name, attn, per_w))
    if not flipped:
        print("  (none — all firms either stay qualified or stay unqualified across 0.20/0.25/0.30)")
    else:
        for firm_name, attn, per_w in flipped:
            marks = "  ".join("Y" if q else "n" for q in per_w)
            print(f"  {firm_name[:48]:<48} attn={attn:>4}  {marks}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="rescore JSONL produced by rescore_from_cache.py")
    args = parser.parse_args()
    report(args.jsonl)

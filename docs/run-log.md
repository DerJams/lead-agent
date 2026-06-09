# Run Log

Brief, factual entries for notable full pipeline runs. Newest first.

---

## 2026-06-08 — Size-enforcement calibration (cached re-score of run `e81032ba`)

Not a new pipeline run. All decisions validated offline against the 43
completed firms from run `e81032ba`, re-scored against the current
config using cached scrapes (no fresh search, no fresh scraping).

**Changes shipped:**

- **Structured profile in scoring prompt** (commit `cc18a4e`):
  vertical-agnostic scorer change. The extracted profile is rendered
  as a `Known facts:` block at the top of every soft-signal prompt,
  so a signal can read the model's already-extracted value (e.g.
  `attorney_count`) instead of re-inferring it from page text.
- **`boutique_size` soft signal added** (commit `0bad26d`): calibrated
  from the case-study client's actual ICP. Sweet spot 4-8 attorneys;
  solos rated as "real but limited fit" so they can be saved by other
  signals; steep decline above 12; 25+ rated 1-2. Reads `attorney_count`
  from the structured-facts block.
- **`boutique_size` weight set to 0.25** (commit `47cfb82`): chosen via
  an offline weight sweep over 0.20/0.25/0.30 on the 29 firms available
  at the time, then confirmed against the full 43 in the threshold
  sweep — same weights validated on consistent samples for both
  decisions. Final vector cre 0.35 / deal 0.25 / boutique 0.25 /
  tx 0.1125 / owner 0.0375. At 0.25 Andrews Myers (68 attorneys) lands
  at 0.495, ~0.055 below the 0.55 line, while every 4-8 sweet-spot firm
  in the file qualifies and strong-fit solos like Saunders survive on
  their CRE and deal signals.
- **`min_qualify_score` kept at 0.55**: validated by a 43-firm offline
  threshold sweep at 0.50 / 0.55 / 0.60. 0.50 leaves Andrews Myers
  (0.495) within scoring jitter of the line, defeating the size
  calibration. 0.60 cuts genuine 4-8 sweet-spot firms (Clausewitz Reyes
  4, Cutler Smith 7, Biggers 5) plus the strong-CRE solo Saunders. 0.55
  keeps the sweet-spot cluster qualified and leaves Andrews Myers a
  clean 0.055 margin.

**Qualified at ≥0.55 under current weights (11 firms):**

| # | Firm | Score | Attorneys |
|---|---|---|---|
| 1 | Golden Steves & Gordon, LLP | 0.760 | 17 |
| 2 | Johnson Petrov LLP | 0.709 | 6 |
| 3 | Brown Law Firm | 0.697 | 4 |
| 4 | R L Wilson Law Firm | 0.647 | — |
| 5 | Kane Russell Coleman Logan | 0.610 | 9 |
| 6 | Law Offices of Craig W. Saunders | 0.597 | 1 |
| 7 | Clausewitz Reyes | 0.591 | 4 |
| 8 | Cutler Smith PC | 0.579 | 7 |
| 9 | The Biggers Law Firm, P.C. | 0.564 | 5 |
| 10 | BoyarMiller Attorneys at Law | 0.559 | 15 |
| 11 | Rogers & Whitley, L.L.P. | 0.552 | 3 |

**Decision artifacts:**

- `scripts/weight_sweep.py` (commit `87bc98c`) — offline weight sweep
- `scripts/topup_rescore.py` (commit `3589f7d`) — resumable rescore tool with math-only path for already-scored firms
- `scripts/threshold_sweep.py` (commit `9f10845`) — offline threshold sweep

---

## 2026-06-07 — Run `e81032ba`

ICP: `configs/icp_law_boutique.yaml` · limit 54 · provider: Cerebras `gpt-oss-120b`

**Validated two fixes shipped this session:**

- **`attorney_count` prompt carve-out** (commit `82e7198`): field-level
  description now permits counting named attorney bios as "stated in
  text" rather than guessing. Fill rate moved from 16/43 (37%) to
  40/43 (93%) of completed firms.
- **Scraper noise filter** (commit `44787be`): `select_relevant_links`
  now drops news/press/blog URLs (strict path-segment match) and
  award/press slug tokens before page-type assignment. BoyarMiller
  dropped 0.720 → 0.550 because the previous run picked a 2026 Chambers
  awards press release as the "practice" page (inflating
  `cre_specialization` to 6); the corrected run picks the real
  `/practices/` page, which shows BoyarMiller is a multi-practice
  business firm rather than a CRE specialist (`cre_specialization` → 3).
  The drop is an accuracy gain, not a regression.

**Stats:** 43 completed, 11 failed (scrape stage), 15 qualified at ≥0.55.
98 LLM calls · 398K tokens · 46m 50s wall (paced by Cerebras 5 RPM cap).

**Qualified firms (≥0.55):**

| # | Firm | Score | Attorneys |
|---|---|---|---|
| 1 | Golden Steves & Gordon, LLP | 0.850 | 17 |
| 2 | Law Offices of Craig W. Saunders | 0.730 | 1 |
| 3 | Law Offices of John S. Unell | 0.715 | 1 |
| 4 | Andrews Myers, P.C. | 0.645 | 68 |
| 5 | R L Wilson Law Firm | 0.640 | — |
| 6 | RattikinLaw | 0.635 | 2 |
| 7 | Sprigg-Novak Law Firm, PLLC | 0.625 | 2 |
| 8 | Johnson Petrov LLP | 0.625 | 6 |
| 9 | Wallace Law PLLC | 0.625 | 1 |
| 10 | Kane Russell Coleman Logan | 0.610 | 9 |
| 11 | Taylor & Coughlin, PLLC | 0.580 | 2 |
| 12 | Greenwald & Greenwald, PLLC | 0.575 | 2 |
| 13 | The Farah Law Firm, P.C. | 0.565 | 1 |
| 14 | Craddock Massey LLP | 0.560 | 2 |
| 15 | BoyarMiller Attorneys at Law | 0.550 | 15 |

CSV: `data/outputs/icp_law_boutique_20260607_214901.csv`.

**Known open issues:** KRCL's `attorney_count=9` is bounded by the
client-side rendering of its `/attorneys` page; the static HTML strips
to ~650 chars, so the model counts only what the homepage exposes.
JS-rendered key pages affect roughly 16% of firms in this run but do
not measurably depress scores; see `scripts/audit_thin_text.py`.

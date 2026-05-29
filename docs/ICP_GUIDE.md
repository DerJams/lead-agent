# Writing an ICP

An Ideal Customer Profile (ICP) is a single YAML file that tells the agent who to
look for and how to score them. The same code runs any vertical — **swap the YAML,
not the Python**. This guide walks the schema using the two shipped configs as
worked examples.

- `configs/icp_law_boutique.yaml` — small commercial real estate law firms.
- `configs/icp_cpa_firm.yaml` — small/mid CPA firms as AI-automation buyers.

Validate any ICP by loading it: `uv run lead-agent run --config <your.yaml> --limit 1`
parses and validates before doing any work, and the eval command does the same.

## Anatomy of an ICP

Eight top-level keys, all required:

| Key | Purpose |
|---|---|
| `name` | Human-readable label (used as the run's `icp_name`). |
| `description` | Short market summary; fed to the LLM during query gen, extraction, and scoring. |
| `search_queries` | Query templates, geographies, and noise keywords for discovery. |
| `extraction_schema` | Fields to pull from each firm site (becomes a Pydantic model). |
| `hard_filters` | Deterministic must-pass rules (the gate). |
| `soft_signals` | LLM-judged 1–10 factors with weights. |
| `scoring` | How filters + signals combine into qualify/not. |
| `output_fields` | Which columns appear in the ranked CSV. |

## Worked example 1: law boutique

```yaml
search_queries:
  templates:
    - "commercial real estate law firm {city} Texas"
  geo_focus: [Dallas, Fort Worth, Austin, Houston, San Antonio]
  negative_keywords: [martindale, avvo, findlaw, directory, news]

hard_filters:
  - field: attorney_count
    operator: between
    value: [3, 15]
  - field: practice_areas
    operator: contains
    value: commercial real estate

soft_signals:
  - name: cre_specialization
    weight: 0.35
    prompt: "...how specialized are they in commercial real estate? Return 1-10."
  # + deal_activity (0.25), owner_operated (0.20), texas_market_depth (0.20)

scoring:
  hard_filter_policy: gate
  min_qualify_score: 0.55
```

Reads as: *find 3–15-attorney Texas CRE firms; require a CRE practice area; then
rate specialization, deal activity, independence, and local depth, and qualify
anything scoring ≥ 0.55.*

## Worked example 2: CPA firm (a deliberately different vertical)

The CPA config is **not** a relabeled law config — it exercises the schema
differently:

| Aspect | Law boutique | CPA firm |
|---|---|---|
| Size gate | `attorney_count between [3,15]` | `employee_count between [10,75]` (bigger band, different field) |
| Domain gate | `practice_areas contains "commercial real estate"` | **none** — compliance fit is a *soft signal* instead |
| New field types | — | `hiring` is a **boolean**; `software_stack` a list — neither used by law |
| Soft signals | CRE-specific | automation-readiness: tech adoption, growth strain, advisory orientation, decision-maker access |

Why the CPA config moves "does compliance work" to a soft signal: the natural rule
is *tax OR bookkeeping OR payroll OR audit*, which the hard-filter operators can't
express (see [DECISIONS.md](DECISIONS.md) ADR-002). Routing it to the LLM sidesteps
that limit and keeps `employee_count` as the only deterministic gate.

This is the payoff of the pluggable design: a genuinely different buyer profile
dropped in as config, no code change.

## Field reference

### `search_queries`
- **`templates`** — search strings. The **only** template variable is `{city}`,
  expanded over `geo_focus` (so `N templates × M cities` queries). Templates without
  `{city}` are emitted once. (Single-variable templating is a known limit — ADR-002.)
- **`geo_focus`** — list of cities substituted into `{city}`.
- **`negative_keywords`** — matched (case-insensitive) against each result's host
  and title to drop directories/news/aggregators before the LLM filter.

### `extraction_schema`
A list of fields, each `{name, type, description, required}`. Types map to Python:

| `type` | Python type |
|---|---|
| `string` | `str` |
| `integer` | `int` |
| `list` | `list[str]` |
| `boolean` | `bool` |

Every field is **nullable** regardless of `required` — extraction returns `null`
when data is absent rather than hallucinating. `required` only signals the LLM that
a field is high-priority; the **hard filters** do the actual gating. `description`
is the per-field instruction the LLM follows, so write it well.

### `hard_filters`
Deterministic rules of `{field, operator, value}`. `field` must be an
`extraction_schema` field name.

| Operator | Meaning |
|---|---|
| `gte` / `lte` | numeric ≥ / ≤ (strings coerced) |
| `between` | numeric within `[lo, hi]` inclusive (value must be exactly 2 numbers) |
| `eq` | equality (case-insensitive for strings) |
| `in` | field value is one of `value` (a list) |
| `contains` | case-insensitive substring; if the field is a list, passes when **any** item contains the needle |

A **`None` field value always fails** its filter — that's the intended gate
behavior with lenient extraction. There is no OR / list-overlap operator yet
(ADR-002); model multi-value criteria as a soft signal.

### `soft_signals`
Each is `{name, description, weight, prompt}`:
- **`weight`** — the **weights must sum to 1.0 (± 0.05)** across all signals, or the
  config fails validation.
- **`prompt`** — must instruct the model to return a single integer **1–10**. The
  scorer rates all signals in one batched call, clamps to [1,10], normalizes as
  `rating / 10`, and combines by weight.
- Signals judge the **scraped website text only** — they cannot read the extracted
  profile (ADR-002). So a "uses modern software" signal must describe what to look
  for in the text, even if you also extract a `software_stack` field for output.

### `scoring`
- **`hard_filter_policy`** — `gate` (a hard-filter failure disqualifies and skips the
  scoring LLM call — the cheap, default behavior) or `weighted` (filters recorded
  but not blocking).
- **`soft_signal_normalization`** — `weighted_average` (default) or `sum`.
- **`min_qualify_score`** — a firm qualifies if `score ≥ this` (and, under `gate`,
  it passed the hard filters).

### `output_fields`
The CSV columns, in order. Each must be one of: an `extraction_schema` field name, a
`soft_signal` name, or the literal `score`. List-valued fields are joined with `; `.

## Authoring workflow

1. Copy a shipped config and edit it for your vertical.
2. Load it — validation runs on load and fails loudly with a clear message.
3. Build an eval set (`tests/eval/eval_set_<vertical>.yaml`) of 15–25 firms you've
   hand-labeled (see the shipped templates).
4. Calibrate: `uv run lead-agent eval --config <your.yaml> --eval-set <your_eval.yaml>`
   and adjust soft-signal prompts/weights and `min_qualify_score` against the
   precision/recall/MAE it reports.

## Validation & troubleshooting

The loader raises clear errors:

| Message | Fix |
|---|---|
| `soft_signals weights sum to X; expected 1.0 ± 0.05` | Rebalance weights to total 1.0. |
| `output_fields contains unknown fields [...]` | Every output field must be an extraction field, a soft-signal name, or `score`. |
| `operator 'between' requires a list of exactly 2 numbers` | Use `value: [lo, hi]` with two numbers. |
| `Invalid ICP config in <path>` | A Pydantic validation error — the detail names the offending field/type. |
| `ICP config must be a YAML mapping` | The file's top level must be a mapping, not a list/scalar. |

## Tips for a good ICP

- **Gate cheaply, judge richly.** Put hard, deterministic must-haves in
  `hard_filters` (they run first, free, and short-circuit the LLM); reserve
  `soft_signals` for nuanced fit you'd trust a human to judge from the website.
- **Weight by predictive value.** The signal that best separates good from bad leads
  should carry the most weight.
- **Write prompts like rubrics.** Anchor 1 and 10 explicitly ("10 = …, 1 = …") so
  ratings are consistent across firms.
- **Tune `min_qualify_score` against the eval set**, not by intuition.

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — how the config drives each pipeline stage.
- [DECISIONS.md](DECISIONS.md) — ADR-002, the four current ICP/engine limitations.
- `configs/icp_law_boutique.yaml`, `configs/icp_cpa_firm.yaml` — full reference configs.

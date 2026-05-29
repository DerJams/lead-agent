# Architecture Decision Records

Short ADRs for notable choices. Newest first.

## ADR-002: v1 known limitations surfaced by the second ICP (CPA firms)

**Status:** accepted (known limitations, not bugs)

**Context.** Adding a genuinely different vertical (`configs/icp_cpa_firm.yaml`, CPA firms
as AI-automation buyers) stress-tested the pluggable-ICP design. The dynamic extraction
model, hard-filter field references, soft-signal batching/weights, and output mapping all
generalized cleanly — nothing law-specific is baked into those paths. Four edges did surface.
We are accepting them for v1 and recording them here so they are deliberate, not forgotten.

1. **Single-dimension `{city}` geo placeholder.**
   - *Limitation:* `search.expand_templates` only substitutes `{city}` over `geo_focus`.
     There is no way to express a state/region/"nationwide-remote" geo unit, or a
     multi-variable template (e.g. `{city} × {service}`). CPA automation is largely sold
     remotely, so metros are used as a search proxy rather than a true buyer requirement.
   - *Why deferred:* the law and CPA ICPs are both adequately served by city-seeded search;
     generalizing the templating engine is not blocking for v1.
   - *Future fix:* support arbitrary named template variables, each backed by a list in the
     config (e.g. `template_vars: {city: [...], service: [...]}`), and take the Cartesian
     product over all variables present in a template.

2. **No OR / list-overlap hard-filter operator.**
   - *Limitation:* `contains` takes a single needle and `in` is scalar membership only.
     "Does compliance work" is naturally *tax OR bookkeeping OR payroll OR audit*, which
     cannot be expressed as a hard rule. (The law ICP's single phrase "commercial real
     estate" masked this.) The CPA ICP sidesteps it by routing compliance fit to a soft
     signal and gating only on `employee_count`.
   - *Why deferred:* the soft-signal route is a reasonable substitute, and most hard gates
     are single-valued.
   - *Future fix:* add a `contains_any` (substring-any) and/or `intersects` (list-overlap)
     operator to `HardFilter` and `scorer._apply_operator`, with `value` as a list.

3. **Built-in scrape blocklist is general+legal-tuned.**
   - *Limitation:* `search._DEFAULT_BLOCKED_DOMAINS` hardcodes legal directories
     (martindale, avvo, findlaw, justia, …) and omits accounting aggregators (thumbtack,
     clutch, upcity). CPA discovery leans on `negative_keywords` to compensate.
   - *Why deferred:* `negative_keywords` already covers most accounting noise, so quality is
     acceptable without a code change.
   - *Future fix:* make the blocklist per-ICP (a `blocked_domains` config key merged with a
     small shared default), so each vertical curates its own aggregator list.

4. **Soft signals cannot read the extracted profile.**
   - *Limitation:* `scorer.score_firm` passes only `combined_text` to the soft-signal LLM,
     never the extracted `profile`. So `tech_adoption_signals` re-judges the tech stack from
     raw text even though `software_stack` is already extracted; a signal cannot reason over
     a structured extracted value.
   - *Why deferred:* judging from raw text is robust and keeps extraction and scoring
     decoupled; the redundancy is cheap.
   - *Future fix:* optionally pass the extracted profile into the soft-signal prompt builder
     so signals can reference structured fields when useful.

**Consequences.** These are v1 acceptable trade-offs. Each has a concrete, isolated future
fix; none requires reworking the core pluggable design.

## ADR-001: Eval harness logic lives in the package, not under `tests/`

**Status:** accepted

**Context.** The build plan placed the eval harness at `tests/eval/run_evals.py`.
However, the `eval` CLI command (`lead-agent eval`) must invoke the harness, and the
installed package only ships `src/lead_agent` (`[tool.hatch.build.targets.wheel]
packages = ["src/lead_agent"]`). Code under `tests/` is not packaged, so an installed
CLI cannot import it without a `sys.path`/file-path hack.

**Decision.** The reusable harness logic — `EvalSet`/`EvalFirm` models, `load_eval_set`,
`compute_metrics`, and `evaluate` — lives in `src/lead_agent/eval.py`. The CLI imports it
directly. `tests/eval/run_evals.py` is reduced to a thin, skip-by-default pytest that runs
the harness against `tests/eval/eval_set_law.yaml` using the configured LLM and asserts
threshold metrics; it is gated on the `RUN_LLM_EVALS` environment variable so the normal
(offline) test suite stays fast and deterministic. Because pytest only auto-collects
`test_*.py`, `run_evals.py` is added to `python_files` in `pyproject.toml`.

**Consequences.** The harness is importable and unit-testable offline (`tests/test_eval.py`
injects a fake LLM). The `eval` command works from an installed package. This is a minor
deviation from the originally documented file layout, recorded here; `CLAUDE.md`'s structure
section has been updated to list `src/lead_agent/eval.py`.

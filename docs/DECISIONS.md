# Architecture Decision Records

Short ADRs for notable choices. Newest first.

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

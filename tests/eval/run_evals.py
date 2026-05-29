"""Gated scoring-quality regression: runs the eval harness against the real LLM.

Skipped unless RUN_LLM_EVALS is set, because it needs a configured Ollama/Groq
model and makes real, non-deterministic LLM calls. Thresholds are tunable via env
(EVAL_MIN_PRECISION, EVAL_MIN_RECALL, EVAL_MAX_MAE) so you can calibrate against a
real, hand-labeled eval_set_law.yaml. Run with:

    RUN_LLM_EVALS=1 uv run pytest tests/eval/run_evals.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lead_agent.config import load_icp
from lead_agent.eval import evaluate, load_eval_set
from lead_agent.llm import get_client

_CONFIG = Path("configs/icp_law_boutique.yaml")
_EVAL_SET = Path("tests/eval/eval_set_law.yaml")

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LLM_EVALS"),
    reason="set RUN_LLM_EVALS=1 to run LLM-dependent scoring evals",
)


def _threshold(name: str, default: str) -> float:
    return float(os.getenv(name, default))


async def test_law_boutique_scoring_meets_thresholds() -> None:
    icp = load_icp(_CONFIG)
    eval_set = load_eval_set(_EVAL_SET)
    report = await evaluate(icp, eval_set, get_client())
    metrics = report.metrics
    assert metrics.precision >= _threshold("EVAL_MIN_PRECISION", "0.6"), metrics
    assert metrics.recall >= _threshold("EVAL_MIN_RECALL", "0.6"), metrics
    assert metrics.mae <= _threshold("EVAL_MAX_MAE", "0.25"), metrics

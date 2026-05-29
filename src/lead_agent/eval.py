"""Eval harness: score hand-labeled firms and report precision/recall/F1/MAE.

Lives in the package (not tests/) so the CLI eval command can import it. Each eval
firm carries its data inline (a profile for the hard filters and representative
website text for the soft signals), so evaluation is reproducible and offline-ish:
it isolates the scorer, which is the part being calibrated. It still calls the
configured LLM for soft signals, so document acceptable run-to-run variance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from .pipeline import RunStats
from .scorer import score_firm

if TYPE_CHECKING:
    from .config import ICPConfig
    from .llm import LLMClient


# ---------------------------------------------------------------------------
# Eval set
# ---------------------------------------------------------------------------

class EvalFirm(BaseModel):
    name: str
    url: str = ""
    profile: dict[str, Any]
    text: str
    expected_score: float = Field(ge=0.0, le=1.0)
    expected_qualified: bool | None = None


class EvalSet(BaseModel):
    name: str = ""
    firms: list[EvalFirm] = Field(min_length=1)


def load_eval_set(path: Path) -> EvalSet:
    """Load and validate a hand-labeled eval set from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Eval set must be a YAML mapping, got {type(raw).__name__}: {path}")
    try:
        return EvalSet.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid eval set in {path}:\n{exc}") from exc


# ---------------------------------------------------------------------------
# Results and metrics
# ---------------------------------------------------------------------------

@dataclass
class FirmEval:
    name: str
    url: str
    system_score: float
    expected_score: float
    system_qualified: bool
    expected_qualified: bool
    abs_error: float


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    mae: float
    tp: int
    fp: int
    fn: int
    tn: int
    n: int


@dataclass
class EvalReport:
    metrics: Metrics
    results: list[FirmEval]
    stats: RunStats


def compute_metrics(results: list[FirmEval]) -> Metrics:
    """Precision/recall/F1 on the qualified decision, MAE on the score.

    Positive class is 'qualified'. precision/recall default to 1.0 when their
    denominator is 0 (no predicted/actual positives); F1 is 0.0 when P+R is 0.
    """
    tp = sum(1 for r in results if r.system_qualified and r.expected_qualified)
    fp = sum(1 for r in results if r.system_qualified and not r.expected_qualified)
    fn = sum(1 for r in results if not r.system_qualified and r.expected_qualified)
    tn = sum(1 for r in results if not r.system_qualified and not r.expected_qualified)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mae = sum(r.abs_error for r in results) / len(results) if results else 0.0

    return Metrics(
        precision=precision, recall=recall, f1=f1, mae=mae,
        tp=tp, fp=fp, fn=fn, tn=tn, n=len(results),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate(
    icp: ICPConfig,
    eval_set: EvalSet,
    client: LLMClient,
    *,
    concurrency: int = 5,
) -> EvalReport:
    """Score every eval firm and compare to its labels. Aggregates LLM cost/tokens."""
    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(firm: EvalFirm) -> tuple[EvalFirm, Any]:
        async with semaphore:
            return firm, await score_firm(firm.profile, firm.text, icp, client)

    pairs = await asyncio.gather(*(score_one(firm) for firm in eval_set.firms))

    stats = RunStats()
    results: list[FirmEval] = []
    for firm, result in pairs:
        stats.add_calls(result.stats)
        expected_qualified = firm.expected_qualified
        if expected_qualified is None:
            expected_qualified = firm.expected_score >= icp.scoring.min_qualify_score
        results.append(
            FirmEval(
                name=firm.name,
                url=firm.url,
                system_score=result.score,
                expected_score=firm.expected_score,
                system_qualified=result.qualified,
                expected_qualified=expected_qualified,
                abs_error=abs(result.score - firm.expected_score),
            )
        )

    return EvalReport(metrics=compute_metrics(results), results=results, stats=stats)

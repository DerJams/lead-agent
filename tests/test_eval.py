"""Offline tests for the eval harness: metrics, loading, evaluate, and the eval command."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lead_agent.cli import _async_eval, app
from lead_agent.config import ICPConfig
from lead_agent.eval import (
    EvalSet,
    FirmEval,
    compute_metrics,
    evaluate,
    load_eval_set,
)
from lead_agent.llm import CallStats, LLMResponse
from lead_agent.scorer import SignalRating, SignalRatings

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

_BASE_ICP: dict = {
    "name": "Test ICP",
    "description": "A minimal test ICP.",
    "search_queries": {"templates": ["{city} t"], "geo_focus": ["Dallas"], "negative_keywords": []},
    "extraction_schema": [
        {"name": "attorney_count", "type": "integer", "description": "Count"},
    ],
    "hard_filters": [{"field": "attorney_count", "operator": "between", "value": [3, 15]}],
    "soft_signals": [
        {"name": "sig_a", "description": "A", "weight": 0.5, "prompt": "Rate A 1-10."},
        {"name": "sig_b", "description": "B", "weight": 0.5, "prompt": "Rate B 1-10."},
    ],
    "scoring": {
        "hard_filter_policy": "gate",
        "soft_signal_normalization": "weighted_average",
        "min_qualify_score": 0.55,
    },
    "output_fields": ["score", "sig_a", "sig_b"],
}


def make_icp() -> ICPConfig:
    return ICPConfig.model_validate(copy.deepcopy(_BASE_ICP))


class FakeLLM:
    """Only soft-signal scoring is exercised by the harness; always returns ratings 8/8."""

    def __init__(self) -> None:
        self.calls = 0

    async def extract(
        self,
        prompt: str,
        response_model: type,
        system: str = "",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> LLMResponse[SignalRatings]:
        self.calls += 1
        content = SignalRatings(
            ratings=[SignalRating(name="sig_a", rating=8), SignalRating(name="sig_b", rating=8)]
        )
        stats = CallStats(
            model="fake", prompt_tokens=10, completion_tokens=5, cost_usd=0.0, duration_ms=1
        )
        return LLMResponse(content=content, stats=stats)


def _fe(*, sys_q: bool, exp_q: bool, abs_error: float = 0.0) -> FirmEval:
    return FirmEval(
        name="x", url="", system_score=0.0, expected_score=0.0,
        system_qualified=sys_q, expected_qualified=exp_q, abs_error=abs_error,
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_classification(self) -> None:
        results = [_fe(sys_q=True, exp_q=True), _fe(sys_q=False, exp_q=False)]
        m = compute_metrics(results)
        assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
        assert (m.tp, m.fp, m.fn, m.tn) == (1, 0, 0, 1)

    def test_all_wrong(self) -> None:
        results = [_fe(sys_q=False, exp_q=True), _fe(sys_q=True, exp_q=False)]
        m = compute_metrics(results)
        assert m.precision == 0.0  # tp=0, fp=1
        assert m.recall == 0.0  # tp=0, fn=1
        assert m.f1 == 0.0

    def test_mixed(self) -> None:
        results = [
            _fe(sys_q=True, exp_q=True),  # TP
            _fe(sys_q=True, exp_q=False),  # FP
            _fe(sys_q=False, exp_q=True),  # FN
            _fe(sys_q=False, exp_q=False),  # TN
        ]
        m = compute_metrics(results)
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)
        assert m.f1 == pytest.approx(0.5)

    def test_no_positives_guards_to_one(self) -> None:
        # No predicted or actual positives -> precision/recall default to 1.0
        results = [_fe(sys_q=False, exp_q=False), _fe(sys_q=False, exp_q=False)]
        m = compute_metrics(results)
        assert m.precision == 1.0
        assert m.recall == 1.0

    def test_mae(self) -> None:
        results = [
            _fe(sys_q=True, exp_q=True, abs_error=0.1),
            _fe(sys_q=True, exp_q=True, abs_error=0.3),
        ]
        assert compute_metrics(results).mae == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# load_eval_set
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


_VALID_FIRM = {
    "name": "Firm A",
    "profile": {"attorney_count": 7},
    "text": "website text",
    "expected_score": 0.8,
    "expected_qualified": True,
}


class TestLoadEvalSet:
    def test_loads_valid(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"name": "s", "firms": [_VALID_FIRM]})
        es = load_eval_set(path)
        assert len(es.firms) == 1
        assert es.firms[0].name == "Firm A"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_eval_set(tmp_path / "nope.yaml")

    def test_score_out_of_range_raises(self, tmp_path: Path) -> None:
        firm = {**_VALID_FIRM, "expected_score": 1.5}
        path = _write(tmp_path, {"firms": [firm]})
        with pytest.raises(ValueError, match="Invalid eval set"):
            load_eval_set(path)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        firm = {"name": "x", "profile": {}, "expected_score": 0.5}  # no 'text'
        path = _write(tmp_path, {"firms": [firm]})
        with pytest.raises(ValueError, match="Invalid eval set"):
            load_eval_set(path)

    def test_empty_firms_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"firms": []})
        with pytest.raises(ValueError, match="Invalid eval set"):
            load_eval_set(path)

    def test_expected_qualified_optional(self, tmp_path: Path) -> None:
        firm = {"name": "x", "profile": {"attorney_count": 7}, "text": "t", "expected_score": 0.8}
        path = _write(tmp_path, {"firms": [firm]})
        es = load_eval_set(path)
        assert es.firms[0].expected_qualified is None


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    async def test_scores_and_computes_metrics(self) -> None:
        icp = make_icp()
        eval_set = EvalSet.model_validate(
            {
                "firms": [
                    {  # passes hard filter, rated 0.8, labeled qualified -> TP
                        "name": "good", "profile": {"attorney_count": 7}, "text": "t",
                        "expected_score": 0.8, "expected_qualified": True,
                    },
                    {  # fails hard filter -> system not qualified, labeled not -> TN
                        "name": "toobig", "profile": {"attorney_count": 50}, "text": "t",
                        "expected_score": 0.1, "expected_qualified": False,
                    },
                ]
            }
        )
        llm = FakeLLM()
        report = await evaluate(icp, eval_set, llm)
        assert report.metrics.precision == 1.0
        assert report.metrics.recall == 1.0
        assert report.metrics.mae == pytest.approx(0.05)  # |0.8-0.8|=0, |0.0-0.1|=0.1
        # gate short-circuit: only the passing firm triggers a scoring LLM call
        assert llm.calls == 1
        assert report.stats.llm_calls == 1

    async def test_derives_expected_qualified_when_omitted(self) -> None:
        icp = make_icp()
        eval_set = EvalSet.model_validate(
            {
                "firms": [
                    {  # expected_qualified omitted; 0.8 >= 0.55 -> derived True
                        "name": "g", "profile": {"attorney_count": 7}, "text": "t",
                        "expected_score": 0.8,
                    }
                ]
            }
        )
        report = await evaluate(icp, eval_set, FakeLLM())
        assert report.results[0].expected_qualified is True
        assert report.metrics.tp == 1


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "icp.yaml"
    path.write_text(yaml.safe_dump(_BASE_ICP), encoding="utf-8")
    return path


class TestEvalCommand:
    async def test_async_eval_returns_report(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        eval_path = _write(tmp_path, {"firms": [_VALID_FIRM]})
        report = await _async_eval(config, eval_path, client=FakeLLM())
        assert report.metrics.n == 1
        assert report.results[0].name == "Firm A"

    def test_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        eval_path = _write(tmp_path, {"firms": [_VALID_FIRM]})
        result = CliRunner().invoke(
            app, ["eval", "--config", "nope.yaml", "--eval-set", str(eval_path)]
        )
        assert result.exit_code == 1

    def test_missing_eval_set_exits_nonzero(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        result = CliRunner().invoke(
            app, ["eval", "--config", str(config), "--eval-set", "nope.yaml"]
        )
        assert result.exit_code == 1

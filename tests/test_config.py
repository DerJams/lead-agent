"""Tests for ICP config loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from lead_agent.config import ICPConfig, load_icp


VALID: dict = {
    "name": "Test ICP",
    "description": "A minimal test ICP.",
    "search_queries": {
        "templates": ["{city} test query"],
        "geo_focus": ["Dallas"],
        "negative_keywords": [],
    },
    "extraction_schema": [
        {"name": "firm_name", "type": "string", "description": "Firm name"},
        {"name": "attorney_count", "type": "integer", "description": "Attorney count"},
        {"name": "practice_areas", "type": "list", "description": "Practice areas"},
    ],
    "hard_filters": [
        {"field": "attorney_count", "operator": "between", "value": [3, 15]},
    ],
    "soft_signals": [
        {
            "name": "specialization",
            "description": "How specialized is the firm",
            "weight": 1.0,
            "prompt": "Rate specialization 1-10.",
        }
    ],
    "scoring": {
        "hard_filter_policy": "gate",
        "soft_signal_normalization": "weighted_average",
        "min_qualify_score": 0.55,
    },
    "output_fields": ["firm_name", "attorney_count", "practice_areas", "score", "specialization"],
}


def patch(**overrides: object) -> dict:
    """Return a deep copy of VALID with top-level keys replaced."""
    d = copy.deepcopy(VALID)
    d.update(overrides)
    return d


class TestValidConfig:
    def test_loads_cleanly(self) -> None:
        cfg = ICPConfig.model_validate(VALID)
        assert cfg.name == "Test ICP"
        assert len(cfg.extraction_schema) == 3
        assert cfg.scoring.min_qualify_score == 0.55

    def test_negative_keywords_defaults_to_empty_list(self) -> None:
        d = copy.deepcopy(VALID)
        del d["search_queries"]["negative_keywords"]
        cfg = ICPConfig.model_validate(d)
        assert cfg.search_queries.negative_keywords == []

    def test_extraction_field_required_defaults_true(self) -> None:
        # VALID's firm_name entry has no explicit "required" key — default must kick in
        assert "required" not in VALID["extraction_schema"][0]
        cfg = ICPConfig.model_validate(VALID)
        assert cfg.extraction_schema[0].required is True


class TestMissingRequiredFields:
    @pytest.mark.parametrize(
        "field",
        [
            "name",
            "description",
            "search_queries",
            "extraction_schema",
            "hard_filters",
            "soft_signals",
            "scoring",
            "output_fields",
        ],
    )
    def test_missing_top_level_field_raises(self, field: str) -> None:
        d = copy.deepcopy(VALID)
        del d[field]
        with pytest.raises(ValidationError, match=field):
            ICPConfig.model_validate(d)


class TestInvalidTypes:
    def test_non_numeric_min_qualify_score_raises(self) -> None:
        d = patch(scoring={**VALID["scoring"], "min_qualify_score": "high"})
        with pytest.raises(ValidationError):
            ICPConfig.model_validate(d)

    def test_unknown_extraction_field_type_raises(self) -> None:
        d = copy.deepcopy(VALID)
        d["extraction_schema"][0]["type"] = "uuid"
        with pytest.raises(ValidationError):
            ICPConfig.model_validate(d)

    def test_unknown_operator_raises(self) -> None:
        d = patch(hard_filters=[{"field": "attorney_count", "operator": "fuzzy", "value": 5}])
        with pytest.raises(ValidationError):
            ICPConfig.model_validate(d)

    def test_weight_above_one_raises(self) -> None:
        d = copy.deepcopy(VALID)
        d["soft_signals"][0]["weight"] = 1.5
        with pytest.raises(ValidationError):
            ICPConfig.model_validate(d)

    def test_weight_below_zero_raises(self) -> None:
        d = copy.deepcopy(VALID)
        d["soft_signals"][0]["weight"] = -0.1
        with pytest.raises(ValidationError):
            ICPConfig.model_validate(d)


class TestSoftSignalWeights:
    def test_weights_too_far_from_one_raises(self) -> None:
        d = patch(
            soft_signals=[
                {"name": "a", "description": "A", "weight": 0.3, "prompt": "p"},
                {"name": "b", "description": "B", "weight": 0.3, "prompt": "p"},
            ],
            output_fields=["firm_name", "score", "a", "b"],
        )
        with pytest.raises(ValidationError, match="weights sum to"):
            ICPConfig.model_validate(d)

    def test_weights_within_tolerance_pass(self) -> None:
        # 0.52 + 0.52 = 1.04, within the ±0.05 tolerance
        d = patch(
            soft_signals=[
                {"name": "a", "description": "A", "weight": 0.52, "prompt": "p"},
                {"name": "b", "description": "B", "weight": 0.52, "prompt": "p"},
            ],
            output_fields=["firm_name", "score", "a", "b"],
        )
        cfg = ICPConfig.model_validate(d)
        assert len(cfg.soft_signals) == 2


class TestHardFilterBetween:
    def test_between_with_scalar_raises(self) -> None:
        d = patch(hard_filters=[{"field": "attorney_count", "operator": "between", "value": 5}])
        with pytest.raises(ValidationError, match="between"):
            ICPConfig.model_validate(d)

    def test_between_with_one_element_raises(self) -> None:
        d = patch(hard_filters=[{"field": "attorney_count", "operator": "between", "value": [5]}])
        with pytest.raises(ValidationError, match="between"):
            ICPConfig.model_validate(d)

    def test_between_with_string_values_raises(self) -> None:
        d = patch(
            hard_filters=[{"field": "attorney_count", "operator": "between", "value": ["a", "b"]}]
        )
        with pytest.raises(ValidationError, match="between"):
            ICPConfig.model_validate(d)

    def test_between_with_valid_range_passes(self) -> None:
        cfg = ICPConfig.model_validate(VALID)
        assert cfg.hard_filters[0].value == [3, 15]


class TestOutputFieldsCrossValidation:
    def test_unknown_output_field_raises(self) -> None:
        d = patch(output_fields=["firm_name", "nonexistent_field"])
        with pytest.raises(ValidationError, match="nonexistent_field"):
            ICPConfig.model_validate(d)

    def test_score_is_a_valid_output_field(self) -> None:
        cfg = ICPConfig.model_validate(VALID)
        assert "score" in cfg.output_fields

    def test_soft_signal_name_is_a_valid_output_field(self) -> None:
        cfg = ICPConfig.model_validate(VALID)
        assert "specialization" in cfg.output_fields

    def test_extraction_schema_name_is_a_valid_output_field(self) -> None:
        cfg = ICPConfig.model_validate(VALID)
        assert "firm_name" in cfg.output_fields


class TestLoadIcp:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_icp(tmp_path / "missing.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("{unclosed", encoding="utf-8")
        with pytest.raises(ValueError, match="parse YAML"):
            load_icp(bad)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_icp(bad)

    def test_invalid_config_yaml_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "invalid.yaml"
        bad.write_text("name: Test\n", encoding="utf-8")  # missing required fields
        with pytest.raises(ValueError, match="Invalid ICP config"):
            load_icp(bad)

    def test_canonical_law_boutique_loads_successfully(self) -> None:
        path = Path(__file__).parent.parent / "configs" / "icp_law_boutique.yaml"
        cfg = load_icp(path)
        assert cfg.name == "Small Commercial Real Estate Law Boutique (Dallas/Texas)"
        assert any(f.name == "attorney_count" for f in cfg.extraction_schema)
        assert any(f.field == "attorney_count" for f in cfg.hard_filters)
        assert len(cfg.soft_signals) == 4
        assert abs(sum(s.weight for s in cfg.soft_signals) - 1.0) <= 0.05
        assert "score" in cfg.output_fields

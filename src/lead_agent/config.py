"""ICP config loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class SearchQueriesConfig(BaseModel):
    templates: list[str]
    geo_focus: list[str]
    negative_keywords: list[str] = Field(default_factory=list)


class ExtractionField(BaseModel):
    name: str
    type: Literal["string", "integer", "list", "boolean"]
    description: str
    required: bool = True


class HardFilter(BaseModel):
    field: str
    operator: Literal["gte", "lte", "between", "eq", "in", "contains"]
    value: int | float | str | list[int | float | str]

    @model_validator(mode="after")
    def check_between_value(self) -> HardFilter:
        if self.operator == "between":
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError(
                    f"operator 'between' requires a list of exactly 2 numbers, got {self.value!r}"
                )
            if not all(isinstance(v, (int, float)) for v in self.value):
                raise ValueError(
                    f"operator 'between' requires numeric values, got {self.value!r}"
                )
        return self


class SoftSignal(BaseModel):
    name: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)
    prompt: str


class ScoringConfig(BaseModel):
    hard_filter_policy: Literal["gate", "weighted"] = "gate"
    soft_signal_normalization: Literal["weighted_average", "sum"] = "weighted_average"
    min_qualify_score: float = Field(ge=0.0, le=1.0)


class ICPConfig(BaseModel):
    name: str
    description: str
    search_queries: SearchQueriesConfig
    extraction_schema: list[ExtractionField]
    hard_filters: list[HardFilter]
    soft_signals: list[SoftSignal]
    scoring: ScoringConfig
    output_fields: list[str]

    @model_validator(mode="after")
    def check_soft_signal_weights(self) -> ICPConfig:
        total = sum(s.weight for s in self.soft_signals)
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"soft_signals weights sum to {total:.2f}; expected 1.0 ± 0.05"
            )
        return self

    @model_validator(mode="after")
    def check_output_fields(self) -> ICPConfig:
        valid = (
            {f.name for f in self.extraction_schema}
            | {s.name for s in self.soft_signals}
            | {"score"}
        )
        unknown = [f for f in self.output_fields if f not in valid]
        if unknown:
            raise ValueError(
                f"output_fields contains unknown fields {unknown}; "
                f"valid fields: {sorted(valid)}"
            )
        return self


def load_icp(path: Path) -> ICPConfig:
    """Load and validate an ICP config from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"ICP config not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(
            f"ICP config must be a YAML mapping, got {type(raw).__name__}: {path}"
        )

    try:
        return ICPConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid ICP config in {path}:\n{e}") from e

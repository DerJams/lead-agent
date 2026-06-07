"""LLM-based structured extraction via Instructor.

Builds a Pydantic response model at runtime from the ICP's extraction_schema and
extracts a firm profile from scraped website text. All fields are nullable
(lenient): absent data surfaces as null and is gated later by the scorer's
hard_filters, rather than forcing the model to invent values. Returns the profile
as a dict; the pipeline persists it via storage.update_firm_stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from .llm import LLMSettings

if TYPE_CHECKING:
    from .config import ExtractionField, ICPConfig
    from .llm import CallStats, LLMClient


_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "list": list[str],
    "boolean": bool,
}

_EXTRACT_SYSTEM = (
    "You extract structured firm information from the text of a firm's own website. "
    "Use only information present in the provided text. Never invent or guess values. "
    "If a field is not stated in the text, return null for it."
)


@dataclass
class ExtractionResult:
    profile: dict[str, Any] | None
    stats: CallStats | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.profile is not None


def build_extraction_model(schema: list[ExtractionField]) -> type[BaseModel]:
    """Build a Pydantic model from an extraction schema. Every field is nullable (default None)."""
    fields: dict[str, Any] = {}
    for field in schema:
        py_type = _TYPE_MAP[field.type]
        fields[field.name] = (py_type | None, Field(default=None, description=field.description))
    return create_model("ExtractedProfile", **fields)


def _build_extract_prompt(
    icp: ICPConfig, combined_text: str, source_url: str | None, *, max_chars: int
) -> str:
    lines = [f"Target market: {icp.name}"]
    if source_url:
        lines.append(f"Firm website: {source_url}")
    required = [f.name for f in icp.extraction_schema if f.required]
    if required:
        lines.append("High-priority fields to find if present: " + ", ".join(required))
    lines.extend(["", "Website text:", combined_text[:max_chars]])
    return "\n".join(lines)


async def extract_profile(
    combined_text: str,
    icp: ICPConfig,
    client: LLMClient,
    *,
    source_url: str | None = None,
    max_chars: int | None = None,
) -> ExtractionResult:
    """Extract a firm profile from scraped text per the ICP schema. Isolates per-firm failures."""
    if max_chars is None:
        max_chars = LLMSettings().llm_input_max_chars
    model = build_extraction_model(icp.extraction_schema)
    prompt = _build_extract_prompt(icp, combined_text, source_url, max_chars=max_chars)
    try:
        response = await client.extract(prompt, model, system=_EXTRACT_SYSTEM)
    except Exception as exc:
        return ExtractionResult(profile=None, error=str(exc))
    return ExtractionResult(profile=response.content.model_dump(), stats=response.stats)

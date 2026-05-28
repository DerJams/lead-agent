"""CLI entry point. Commands: run, eval."""

from __future__ import annotations

import asyncio
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import load_icp
from .pipeline import resume_run, run_pipeline
from .storage import Storage

if TYPE_CHECKING:
    from .config import ICPConfig
    from .llm import LLMClient
    from .pipeline import RunResult
    from .scraper import Scraper
    from .search import SearchProvider

app = typer.Typer(name="lead-agent", help="ICP-driven lead generation agent.")


class PathSettings(BaseSettings):
    db_path: Path = Path("data/lead_agent.db")
    output_dir: Path = Path("data/outputs")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ---------------------------------------------------------------------------
# CSV output (pure helpers)
# ---------------------------------------------------------------------------

def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_rows(
    result: RunResult, icp: ICPConfig, *, include_all: bool = False
) -> list[dict[str, str]]:
    """Map firms to CSV rows keyed by icp.output_fields, pulling from profile/score/signals."""
    firms = result.scored if include_all else result.qualified
    extraction_names = {f.name for f in icp.extraction_schema}
    signal_names = {s.name for s in icp.soft_signals}

    rows: list[dict[str, str]] = []
    for firm in firms:
        profile = firm.get("extracted_profile") or {}
        soft = (firm.get("score_breakdown") or {}).get("soft_signals") or {}
        row: dict[str, str] = {}
        for name in icp.output_fields:
            if name == "score":
                value: Any = firm.get("score")
            elif name in extraction_names:
                value = profile.get(name)
            elif name in signal_names:
                value = (soft.get(name) or {}).get("rating")
            else:
                value = None
            row[name] = _format_value(value)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _default_output_path(config: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return PathSettings().output_dir / f"{config.stem}_{timestamp}.csv"


# ---------------------------------------------------------------------------
# Run core (injectable for tests)
# ---------------------------------------------------------------------------

async def _async_run(
    config: Path,
    *,
    db_path: Path,
    limit: int | None,
    output: Path | None,
    resume: str | None,
    augment: bool,
    include_all: bool,
    client: LLMClient | None = None,
    provider: SearchProvider | None = None,
    scraper: Scraper | None = None,
) -> tuple[RunResult, Path]:
    icp = load_icp(config)
    async with Storage(db_path) as db:
        if resume:
            result = await resume_run(
                resume, icp, storage=db, client=client, scraper=scraper, limit=limit
            )
        else:
            result = await run_pipeline(
                icp,
                storage=db,
                client=client,
                provider=provider,
                scraper=scraper,
                limit=limit,
                augment_queries=augment,
            )
    out_path = output or _default_output_path(config)
    write_csv(build_rows(result, icp, include_all=include_all), icp.output_fields, out_path)
    return result, out_path


def _print_summary(result: RunResult, out_path: Path) -> None:
    stats = result.stats
    typer.echo(f"Run {result.run_id}")
    typer.echo(f"  stages:        {result.stage_counts}")
    typer.echo(f"  qualified:     {len(result.qualified)}")
    typer.echo(f"  scored:        {len(result.scored)}")
    typer.echo(
        f"  llm calls:     {stats.llm_calls}  tokens: {stats.total_tokens}  "
        f"cost: ${stats.cost_usd:.4f}"
    )
    typer.echo(f"  scraped bytes: {stats.bytes_scraped}")
    typer.echo(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to ICP YAML config."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max firms to process."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output CSV path."),
    db: Path | None = typer.Option(None, "--db", help="SQLite DB path (default from .env)."),
    resume: str | None = typer.Option(None, "--resume", help="Resume an existing run by run_id."),
    augment: bool = typer.Option(
        True, "--augment/--no-augment", help="LLM-augment search queries."
    ),
    include_all: bool = typer.Option(
        False, "--all", help="Include all scored firms, not just qualified."
    ),
) -> None:
    """Run the full lead generation pipeline and write a ranked CSV."""
    db_path = db or PathSettings().db_path
    try:
        result, out_path = asyncio.run(
            _async_run(
                config,
                db_path=db_path,
                limit=limit,
                output=output,
                resume=resume,
                augment=augment,
                include_all=include_all,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    _print_summary(result, out_path)


@app.command()
def eval(
    config: Path = typer.Option(..., "--config", "-c", help="Path to ICP YAML config."),
) -> None:
    """Run the eval harness against hand-labeled firms."""
    typer.echo(f"Eval harness not yet implemented (step 11). Config: {config}")


if __name__ == "__main__":
    app()

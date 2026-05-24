"""CLI entry point. Commands: run, eval."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="lead-agent", help="ICP-driven lead generation agent.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to ICP YAML config."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max firms to process."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path."),
) -> None:
    """Run the full lead generation pipeline."""
    typer.echo(f"Pipeline not yet implemented. Config: {config}, limit: {limit}")


@app.command()
def eval(
    config: Path = typer.Option(..., "--config", "-c", help="Path to ICP YAML config."),
) -> None:
    """Run the eval harness against hand-labeled firms."""
    typer.echo(f"Eval harness not yet implemented. Config: {config}")


if __name__ == "__main__":
    app()

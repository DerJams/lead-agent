# Lead Sourcing Agent

A configurable, open-source lead generation agent for professional services firms. Define your Ideal Customer Profile in YAML and the agent discovers, scores, and ranks qualified leads end-to-end.

> **Status:** Active development. v1 (lead discovery and scoring) in progress.

## What It Does

Given an ICP config, the agent:

1. Generates search queries tailored to your target market
2. Discovers candidate firms via search APIs
3. Scrapes public websites for structured firm profiles
4. Scores each firm against your ICP using hybrid rule-based and LLM-judged criteria
5. Outputs a ranked CSV with personalization hooks ready for outreach

## Why

Most lead-sourcing tools are closed-source vertical SaaS. This is a transparent alternative: configure once for your target market, run as often as you like, swap LLMs based on your cost-quality tradeoff. Built to run free on local LLMs (Ollama) for development and cheaply on cloud LLMs (Groq) for production.

## Quick Start

```bash
# Install
git clone https://github.com/DerJams/lead-agent.git
cd lead-agent
uv sync

# Configure
cp .env.example .env
# Edit .env with your API keys (Tavily, Groq)

# Optional: run fully local with Ollama
ollama pull llama3.1:8b
export LLM_PROVIDER=ollama

# Run against the example ICP
uv run lead-agent run --config configs/icp_law_boutique.yaml --limit 25
```

Output appears in `data/outputs/` as a ranked CSV.

## Configuration

ICPs are defined in YAML. See `configs/icp_law_boutique.yaml` for a worked example, and `docs/ICP_GUIDE.md` for instructions on writing your own.

## Supported LLMs

| Provider | Use Case | Setup |
|---|---|---|
| Ollama (local) | Free dev runs, privacy-sensitive workloads | `ollama pull llama3.1:8b` |
| Groq | Production batches; fast and cheap | `GROQ_API_KEY` in `.env` |
| Anthropic | High-stakes tasks (planned for v2) | `ANTHROPIC_API_KEY` in `.env` |

Swap providers via the `LLM_PROVIDER` env variable. All LLM calls flow through a single adapter (`src/lead_agent/llm.py`).

## Architecture

See `docs/ARCHITECTURE.md` for full details. High-level flow:

```
ICP config → query generation → search → candidate filtering →
  scrape → structured extraction → ICP scoring → ranked CSV
```

State is persisted in SQLite so runs are resumable.

## Roadmap

- **v1 (in progress):** end-to-end lead discovery and scoring
- **v2:** email drafting, sending infrastructure, reply classification
- **v3:** CRM integration, web UI, multi-tenant support

## License

MIT

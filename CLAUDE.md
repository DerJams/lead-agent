# Lead Sourcing Agent

## What This Is

A configurable, ICP-driven lead generation agent for professional services firms. Given an Ideal Customer Profile defined in YAML, the agent discovers candidate firms via web search, scrapes their public sites, extracts structured profiles, scores them against the ICP, and outputs a ranked CSV of qualified leads with personalization hooks.

## Why It Exists

Built as both a working tool for a consulting business and an open-source portfolio piece demonstrating modern AI agent design. The pluggable ICP architecture means the same codebase targets law firms, accounting firms, architecture firms, or any vertical with a defined fit profile. Same Python, swap the YAML.

## Architecture

End-to-end pipeline. ICP in, ranked leads out.

1. **Query generation**: LLM reads the ICP config, generates a set of search queries tailored to the target market.
2. **Search execution**: Agent hits a search API (Tavily by default, Serper as alternative), dedupes results.
3. **Candidate filtering**: LLM reads search result snippets and flags real firm websites versus directories, news articles, or noise.
4. **Site scraping**: Polite async scraping of relevant pages (homepage, about, attorneys, practice areas).
5. **Structured extraction**: LLM extracts firm profile per the ICP's extraction schema.
6. **ICP scoring**: Hybrid scoring combining hard rule-based filters with LLM-judged soft signals.
7. **Output**: Ranked CSV with firm profiles and personalization hooks.

The pipeline is sequential and stateful. Each firm progresses through stages (pending, searched, scraped, extracted, scored, completed), persisted in SQLite so runs are resumable.

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for AI tooling, broad ecosystem |
| Packaging | uv | Fast, modern, handles ARM64 Windows reliably |
| LLM adapter | LiteLLM | One interface across Ollama, Groq, Anthropic |
| Local LLM (dev) | Ollama with Llama 3.1 8B | Free, runs on ARM64 Windows |
| Cloud LLM (prod) | Groq with Llama 3.3 70B | Fast, cheap, free tier covers dev and small batches |
| Structured outputs | Pydantic + Instructor | Type-safe JSON extraction |
| HTTP / scraping | httpx (async) + BeautifulSoup | No JS execution needed for typical firm sites |
| Search API | Tavily (default), Serper (alt) | AI-friendly snippets, generous free tiers |
| Storage | SQLite | Zero infra, handles cache and run state |
| CLI | Typer | Clean, type-hinted |
| Tests | pytest + custom eval harness | Real metrics, not vibes |
| Lint / format | Ruff | Fast, opinionated, single tool |

No agent framework (LangChain, LangGraph, CrewAI). Plain Python keeps the agent readable and debuggable, which matters for both learning and portfolio purposes.

## v1 Scope

### In scope
- ICP-driven query generation
- Search-based candidate discovery
- Polite async scraping of firm websites
- LLM-based structured extraction
- Hybrid ICP scoring (rules plus LLM soft signals)
- Ranked CSV output with personalization hooks
- Pluggable ICP via YAML config files
- Two reference ICPs shipped (small commercial law boutique, small CPA firm)
- Eval harness with 15-25 hand-labeled firms per ICP
- Cost tracking (tokens and dollars per run)

### Out of scope (deferred to v2)
- Email drafting
- Sending infrastructure
- Reply classification and handling
- CRM integration
- LinkedIn or social signal enrichment
- Web UI

## Project Structure

```
lead-agent/
├── README.md
├── CLAUDE.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── icp_law_boutique.yaml
│   └── icp_cpa_firm.yaml
├── src/
│   └── lead_agent/
│       ├── __init__.py
│       ├── cli.py            # Typer CLI entry point
│       ├── pipeline.py       # Orchestrates the full pipeline
│       ├── search.py         # Query generation + search API calls
│       ├── scraper.py        # Async site scraping
│       ├── extractor.py      # LLM-based structured extraction
│       ├── scorer.py         # Hybrid ICP scoring
│       ├── storage.py        # SQLite cache and run state
│       ├── llm.py            # LiteLLM adapter, single point for all LLM calls
│       ├── eval.py           # Eval harness logic (in package so the CLI can import it)
│       └── config.py         # ICP config loading and validation
├── tests/
│   ├── test_scorer.py
│   ├── test_config.py
│   └── eval/
│       ├── eval_set_law.yaml
│       └── run_evals.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DECISIONS.md          # ADRs for major choices
│   ├── ICP_GUIDE.md          # How to write your own ICP
│   └── COST_ANALYSIS.md
└── data/
    ├── cache/                # Scraped content cache
    └── outputs/              # Ranked CSVs
```

## ICP Config Schema

Each ICP is a YAML file with these top-level keys:

- `name`: human-readable label
- `description`: short summary of the target market
- `search_queries`: query templates and hints for the LLM query generator
- `extraction_schema`: fields to extract from firm sites (maps to Pydantic models)
- `hard_filters`: must-pass rules (e.g., attorney_count between 3 and 15)
- `soft_signals`: LLM-judged factors with descriptions and weights
- `scoring`: how filters and signals combine into a final score
- `output_fields`: which fields appear in the ranked CSV

The canonical example to build first: `configs/icp_law_boutique.yaml`.

## Coding Conventions

- Type hints throughout, enforced via Ruff
- Async by default for I/O
- Small, testable functions; modules under roughly 200 LOC where practical
- Configuration over hardcoding; secrets via `.env` only, never committed
- All LLM calls go through `llm.py`; nothing else imports LiteLLM directly
- Polite scraping: per-domain rate limits, respect `robots.txt` where present, identifiable `User-Agent`

## Development Workflow

1. Copy `.env.example` to `.env` and fill in API keys (Groq, Tavily)
2. `uv sync` to install dependencies
3. For local dev: `ollama pull llama3.1:8b`, then set `LLM_PROVIDER=ollama`
4. For production-quality runs: set `LLM_PROVIDER=groq`
5. Validate against eval set: `uv run lead-agent eval --config configs/icp_law_boutique.yaml`
6. Run a real batch: `uv run lead-agent run --config configs/icp_law_boutique.yaml --limit 50`

## Testing

- Unit tests for scorer logic and config parsing
- Integration tests for the pipeline against a small mock dataset
- Eval set of 15-25 hand-labeled real firms per ICP, with expected fit scores
- Eval harness reports precision, recall, and mean absolute error on scores
- LLM-dependent tests pin a deterministic local model; document acceptable variance

## Cost Targets

- Tavily and Groq free tiers sufficient for dev and 50-firm batches
- Per-batch cost on paid Groq tier: under $0.25 for 50 firms
- Document actual costs in `docs/COST_ANALYSIS.md` as the project matures

## Open Items

- Compare Tavily vs Serper on real firm-site snippet quality before locking in default
- Iterate the law-boutique soft signals against the eval set
- Calibrate LLM-judged scoring prompts; may need multiple iterations

## Build Order for Claude Code

When implementing, work in this order so each stage is testable:

1. Project scaffolding (`pyproject.toml`, `.gitignore`, `.env.example`, basic CLI)
2. `config.py`: ICP YAML loader with Pydantic validation
3. `llm.py`: LiteLLM adapter supporting Ollama + Groq
4. `storage.py`: SQLite schema and basic CRUD
5. `search.py`: query generation + Tavily integration
6. `scraper.py`: async polite scraper
7. `extractor.py`: structured extraction via Instructor
8. `scorer.py`: hybrid scoring
9. `pipeline.py`: orchestration that ties it all together
10. `cli.py`: `run` and `eval` commands
11. Eval harness in `tests/eval/`
12. First ICP config: `configs/icp_law_boutique.yaml`
13. Docs: `ARCHITECTURE.md`, `DECISIONS.md`, `ICP_GUIDE.md`, `COST_ANALYSIS.md`

Each step should have a passing test or runnable demo before moving on.

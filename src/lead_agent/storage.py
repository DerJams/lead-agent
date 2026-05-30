"""SQLite cache and run state via aiosqlite."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    icp_name     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running',
    started_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS firms (
    firm_id           TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    url               TEXT NOT NULL,
    stage             TEXT NOT NULL DEFAULT 'pending',
    scraped_at        TEXT,
    extracted_profile TEXT,
    score             REAL,
    score_breakdown   TEXT,
    error             TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE(run_id, url)
);

CREATE INDEX IF NOT EXISTS idx_firms_run_stage ON firms(run_id, stage);
CREATE INDEX IF NOT EXISTS idx_firms_stage ON firms(stage);

CREATE TABLE IF NOT EXISTS scrape_cache (
    url        TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    scraped_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_cache (
    query     TEXT PRIMARY KEY,
    results   TEXT NOT NULL,
    cached_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS filter_decisions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    icp_name   TEXT NOT NULL,
    url        TEXT NOT NULL,
    title      TEXT NOT NULL,
    snippet    TEXT NOT NULL,
    is_firm    INTEGER NOT NULL,
    reason     TEXT NOT NULL DEFAULT '',
    decided_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_filter_decisions_run ON filter_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_filter_decisions_url ON filter_decisions(url);

CREATE TABLE IF NOT EXISTS filter_cache (
    icp_name   TEXT NOT NULL,
    batch_hash TEXT NOT NULL,
    decisions  TEXT NOT NULL,
    cached_at  TEXT NOT NULL,
    PRIMARY KEY (icp_name, batch_hash)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _deserialize_firm(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("extracted_profile", "score_breakdown"):
        if row.get(field) is not None:
            row[field] = json.loads(row[field])
    return row


class Storage:
    """
    Async context manager wrapping a single aiosqlite connection for the pipeline lifetime.

        async with Storage(Path("data/lead_agent.db")) as db:
            run_id = await db.create_run("law_boutique")
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def __aenter__(self) -> Storage:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # -- Run lifecycle -------------------------------------------------------

    async def create_run(self, icp_name: str) -> str:
        """Insert a new run record; return run_id (UUID)."""
        run_id = str(uuid.uuid4())
        await self._conn.execute(
            "INSERT INTO runs (run_id, icp_name, status, started_at) VALUES (?, ?, 'running', ?)",
            (run_id, icp_name, _now()),
        )
        await self._conn.commit()
        return run_id

    async def complete_run(self, run_id: str, *, status: str = "completed") -> None:
        """Set completed_at = now and status on the run record."""
        await self._conn.execute(
            "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
            (status, _now(), run_id),
        )
        await self._conn.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a run record by ID; None if not found."""
        async with self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    # -- Firm tracking -------------------------------------------------------

    async def add_firm(self, run_id: str, url: str) -> str:
        """Insert firm in 'pending' stage; idempotent — returns existing firm_id on duplicate."""
        firm_id = str(uuid.uuid4())
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO firms (firm_id, run_id, url, stage, created_at)
            VALUES (?, ?, ?, 'pending', ?)
            """,
            (firm_id, run_id, url, _now()),
        )
        await self._conn.commit()
        async with self._conn.execute(
            "SELECT firm_id FROM firms WHERE run_id = ? AND url = ?", (run_id, url)
        ) as cursor:
            row = await cursor.fetchone()
        return row["firm_id"]

    async def update_firm_stage(
        self,
        firm_id: str,
        stage: str,
        *,
        scraped_at: str | None = None,
        extracted_profile: dict[str, Any] | None = None,
        score: float | None = None,
        score_breakdown: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update stage and any provided payload fields; unset kwargs leave columns unchanged."""
        fields = ["stage = ?"]
        values: list[Any] = [stage]
        if scraped_at is not None:
            fields.append("scraped_at = ?")
            values.append(scraped_at)
        if extracted_profile is not None:
            fields.append("extracted_profile = ?")
            values.append(json.dumps(extracted_profile))
        if score is not None:
            fields.append("score = ?")
            values.append(score)
        if score_breakdown is not None:
            fields.append("score_breakdown = ?")
            values.append(json.dumps(score_breakdown))
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        values.append(firm_id)
        await self._conn.execute(
            f"UPDATE firms SET {', '.join(fields)} WHERE firm_id = ?",
            values,
        )
        await self._conn.commit()

    async def get_firms_by_stage(self, run_id: str, stage: str) -> list[dict[str, Any]]:
        """All firms for a run at a given stage; JSON fields deserialized."""
        async with self._conn.execute(
            "SELECT * FROM firms WHERE run_id = ? AND stage = ?", (run_id, stage)
        ) as cursor:
            rows = await cursor.fetchall()
        return [_deserialize_firm(_row_to_dict(row)) for row in rows]

    async def get_firm(self, firm_id: str) -> dict[str, Any] | None:
        """Single firm record by ID; JSON fields deserialized."""
        async with self._conn.execute(
            "SELECT * FROM firms WHERE firm_id = ?", (firm_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _deserialize_firm(_row_to_dict(row)) if row else None

    async def count_firms_by_stage(self, run_id: str) -> dict[str, int]:
        """Return {stage: count} for all stages present in this run."""
        async with self._conn.execute(
            "SELECT stage, COUNT(*) AS count FROM firms WHERE run_id = ? GROUP BY stage",
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {row["stage"]: row["count"] for row in rows}

    # -- Scrape cache --------------------------------------------------------

    async def cache_scrape(self, url: str, content: str) -> None:
        """Insert or replace a scrape cache entry."""
        await self._conn.execute(
            "INSERT OR REPLACE INTO scrape_cache (url, content, scraped_at) VALUES (?, ?, ?)",
            (url, content, _now()),
        )
        await self._conn.commit()

    async def get_cached_scrape(self, url: str, ttl_hours: float = 24.0) -> str | None:
        """Return cached content if within TTL; None on miss or expiry."""
        async with self._conn.execute(
            "SELECT content, scraped_at FROM scrape_cache WHERE url = ?", (url,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        age = datetime.now(UTC) - datetime.fromisoformat(row["scraped_at"])
        if age.total_seconds() > ttl_hours * 3600:
            return None
        return row["content"]

    # -- Search cache --------------------------------------------------------

    async def cache_search(self, query: str, results: list[dict[str, Any]]) -> None:
        """Insert or replace cached search results (JSON) for a query."""
        await self._conn.execute(
            "INSERT OR REPLACE INTO search_cache (query, results, cached_at) VALUES (?, ?, ?)",
            (query, json.dumps(results), _now()),
        )
        await self._conn.commit()

    async def get_cached_search(
        self, query: str, ttl_hours: float = 168.0
    ) -> list[dict[str, Any]] | None:
        """Return cached results if within TTL (default 7 days); None on miss or expiry."""
        async with self._conn.execute(
            "SELECT results, cached_at FROM search_cache WHERE query = ?", (query,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        age = datetime.now(UTC) - datetime.fromisoformat(row["cached_at"])
        if age.total_seconds() > ttl_hours * 3600:
            return None
        return json.loads(row["results"])

    # -- Filter decisions / cache --------------------------------------------

    async def get_cached_filter(
        self, icp_name: str, batch_hash: str
    ) -> list[dict[str, Any]] | None:
        """Return cached LLM decisions for a (icp_name, batch_hash) key; None on miss."""
        async with self._conn.execute(
            "SELECT decisions FROM filter_cache WHERE icp_name = ? AND batch_hash = ?",
            (icp_name, batch_hash),
        ) as cursor:
            row = await cursor.fetchone()
        return json.loads(row["decisions"]) if row else None

    async def cache_filter(
        self, icp_name: str, batch_hash: str, decisions: list[dict[str, Any]]
    ) -> None:
        """Insert or replace cached filter decisions for a (icp_name, batch_hash) key."""
        await self._conn.execute(
            "INSERT OR REPLACE INTO filter_cache "
            "(icp_name, batch_hash, decisions, cached_at) VALUES (?, ?, ?, ?)",
            (icp_name, batch_hash, json.dumps(decisions), _now()),
        )
        await self._conn.commit()

    async def log_filter_decisions(
        self, run_id: str, icp_name: str, decisions: list[dict[str, Any]]
    ) -> None:
        """Bulk-append filter decisions for audit.

        Each dict needs url/title/snippet/is_firm/reason.
        """
        if not decisions:
            return
        now = _now()
        await self._conn.executemany(
            "INSERT INTO filter_decisions "
            "(run_id, icp_name, url, title, snippet, is_firm, reason, decided_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    icp_name,
                    d["url"],
                    d["title"],
                    d["snippet"],
                    int(bool(d["is_firm"])),
                    d["reason"],
                    now,
                )
                for d in decisions
            ],
        )
        await self._conn.commit()

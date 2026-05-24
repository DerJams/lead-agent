"""Tests for SQLite storage: schema, run lifecycle, firm stage transitions, cache, resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lead_agent.storage import Storage


class TestSchemaInit:
    async def test_tables_created_on_enter(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            async with db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                tables = {row[0] for row in await cursor.fetchall()}
        assert {"runs", "firms", "scrape_cache"} <= tables

    async def test_indexes_created(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            async with db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ) as cursor:
                indexes = {row[0] for row in await cursor.fetchall()}
        assert "idx_firms_run_stage" in indexes
        assert "idx_firms_stage" in indexes

    async def test_init_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "test.db"
        async with Storage(path):
            pass
        async with Storage(path):  # second open must not raise
            pass

    async def test_parent_directories_created_if_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c" / "test.db"
        async with Storage(path) as db:
            run_id = await db.create_run("test")
        assert path.exists()
        assert run_id


class TestRunLifecycle:
    async def test_create_run_returns_uuid(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("law_boutique")
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID4 canonical form

    async def test_get_run_returns_correct_fields(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("law_boutique")
            run = await db.get_run(run_id)
        assert run is not None
        assert run["run_id"] == run_id
        assert run["icp_name"] == "law_boutique"
        assert run["status"] == "running"
        assert run["started_at"] is not None
        assert run["completed_at"] is None

    async def test_get_run_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            result = await db.get_run("00000000-0000-0000-0000-000000000000")
        assert result is None

    async def test_complete_run_sets_status_and_timestamp(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("law_boutique")
            await db.complete_run(run_id)
            run = await db.get_run(run_id)
        assert run["status"] == "completed"
        assert run["completed_at"] is not None

    async def test_complete_run_with_failed_status(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("law_boutique")
            await db.complete_run(run_id, status="failed")
            run = await db.get_run(run_id)
        assert run["status"] == "failed"
        assert run["completed_at"] is not None

    async def test_multiple_independent_runs(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            id1 = await db.create_run("icp_a")
            id2 = await db.create_run("icp_b")
        assert id1 != id2


class TestFirmStageTransitions:
    async def test_add_firm_returns_uuid(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
        assert isinstance(firm_id, str)
        assert len(firm_id) == 36

    async def test_new_firm_starts_in_pending_stage(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            firm = await db.get_firm(firm_id)
        assert firm is not None
        assert firm["stage"] == "pending"
        assert firm["url"] == "https://example.com"
        assert firm["run_id"] == run_id

    async def test_get_firm_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            result = await db.get_firm("00000000-0000-0000-0000-000000000000")
        assert result is None

    async def test_update_firm_stage_only(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            await db.update_firm_stage(firm_id, "searched")
            firm = await db.get_firm(firm_id)
        assert firm["stage"] == "searched"

    async def test_update_firm_with_scraped_at(self, tmp_path: Path) -> None:
        ts = datetime.now(UTC).isoformat()
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            await db.update_firm_stage(firm_id, "scraped", scraped_at=ts)
            firm = await db.get_firm(firm_id)
        assert firm["stage"] == "scraped"
        assert firm["scraped_at"] == ts

    async def test_update_firm_with_extracted_profile_deserializes_json(
        self, tmp_path: Path
    ) -> None:
        profile = {"firm_name": "Acme Law", "attorney_count": 7, "practice_areas": ["CRE"]}
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            await db.update_firm_stage(firm_id, "extracted", extracted_profile=profile)
            firm = await db.get_firm(firm_id)
        assert firm["stage"] == "extracted"
        assert firm["extracted_profile"] == profile

    async def test_update_firm_with_score_and_breakdown(self, tmp_path: Path) -> None:
        breakdown = {"cre_specialization": 8, "deal_activity": 6, "owner_operated": 9}
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            await db.update_firm_stage(
                firm_id, "scored", score=0.74, score_breakdown=breakdown
            )
            firm = await db.get_firm(firm_id)
        assert firm["score"] == pytest.approx(0.74)
        assert firm["score_breakdown"] == breakdown

    async def test_update_firm_with_error(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            await db.update_firm_stage(firm_id, "failed", error="Connection timeout after 15s")
            firm = await db.get_firm(firm_id)
        assert firm["stage"] == "failed"
        assert firm["error"] == "Connection timeout after 15s"

    async def test_unset_kwargs_leave_columns_unchanged(self, tmp_path: Path) -> None:
        profile = {"firm_name": "Acme Law"}
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            await db.update_firm_stage(firm_id, "extracted", extracted_profile=profile)
            # Advance stage without touching extracted_profile
            await db.update_firm_stage(firm_id, "scored", score=0.8)
            firm = await db.get_firm(firm_id)
        assert firm["stage"] == "scored"
        assert firm["extracted_profile"] == profile  # must survive the second update
        assert firm["score"] == pytest.approx(0.8)

    async def test_get_firms_by_stage_returns_correct_subset(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            id_a = await db.add_firm(run_id, "https://a.com")
            id_b = await db.add_firm(run_id, "https://b.com")
            await db.add_firm(run_id, "https://c.com")  # stays pending
            await db.update_firm_stage(id_a, "scraped")
            await db.update_firm_stage(id_b, "scraped")
            scraped = await db.get_firms_by_stage(run_id, "scraped")
            pending = await db.get_firms_by_stage(run_id, "pending")
        assert len(scraped) == 2
        assert len(pending) == 1
        assert {f["firm_id"] for f in scraped} == {id_a, id_b}

    async def test_get_firms_by_stage_isolated_to_run(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_a = await db.create_run("test")
            run_b = await db.create_run("test")
            await db.add_firm(run_a, "https://a.com")
            await db.add_firm(run_b, "https://b.com")
            firms_a = await db.get_firms_by_stage(run_a, "pending")
            firms_b = await db.get_firms_by_stage(run_b, "pending")
        assert len(firms_a) == 1
        assert firms_a[0]["url"] == "https://a.com"
        assert len(firms_b) == 1
        assert firms_b[0]["url"] == "https://b.com"

    async def test_count_firms_by_stage(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            id_a = await db.add_firm(run_id, "https://a.com")
            id_b = await db.add_firm(run_id, "https://b.com")
            await db.add_firm(run_id, "https://c.com")
            await db.update_firm_stage(id_a, "scraped")
            await db.update_firm_stage(id_b, "scraped")
            counts = await db.count_firms_by_stage(run_id)
        assert counts["scraped"] == 2
        assert counts["pending"] == 1
        assert "extracted" not in counts  # absent stages not included


class TestScrapeCache:
    async def test_miss_returns_none(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            result = await db.get_cached_scrape("https://notcached.com")
        assert result is None

    async def test_hit_returns_content(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            await db.cache_scrape("https://example.com", "<html>content</html>")
            result = await db.get_cached_scrape("https://example.com")
        assert result == "<html>content</html>"

    async def test_expired_entry_returns_none(self, tmp_path: Path) -> None:
        old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
        async with Storage(tmp_path / "test.db") as db:
            await db._conn.execute(
                "INSERT INTO scrape_cache (url, content, scraped_at) VALUES (?, ?, ?)",
                ("https://old.com", "stale content", old_ts),
            )
            await db._conn.commit()
            result = await db.get_cached_scrape("https://old.com", ttl_hours=24.0)
        assert result is None

    async def test_entry_within_ttl_returned(self, tmp_path: Path) -> None:
        recent_ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        async with Storage(tmp_path / "test.db") as db:
            await db._conn.execute(
                "INSERT INTO scrape_cache (url, content, scraped_at) VALUES (?, ?, ?)",
                ("https://recent.com", "fresh content", recent_ts),
            )
            await db._conn.commit()
            result = await db.get_cached_scrape("https://recent.com", ttl_hours=24.0)
        assert result == "fresh content"

    async def test_cache_scrape_replaces_existing(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            await db.cache_scrape("https://example.com", "old content")
            await db.cache_scrape("https://example.com", "new content")
            result = await db.get_cached_scrape("https://example.com")
        assert result == "new content"

    async def test_custom_ttl_boundary(self, tmp_path: Path) -> None:
        ts_2h_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        async with Storage(tmp_path / "test.db") as db:
            await db._conn.execute(
                "INSERT INTO scrape_cache (url, content, scraped_at) VALUES (?, ?, ?)",
                ("https://example.com", "content", ts_2h_ago),
            )
            await db._conn.commit()
            assert await db.get_cached_scrape("https://example.com", ttl_hours=1.0) is None
            assert await db.get_cached_scrape("https://example.com", ttl_hours=3.0) == "content"


class TestResumeFromPartialRun:
    async def test_add_firm_idempotent_same_url(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run_id = await db.create_run("test")
            id1 = await db.add_firm(run_id, "https://example.com")
            id2 = await db.add_firm(run_id, "https://example.com")
        assert id1 == id2

    async def test_same_url_different_runs_are_distinct(self, tmp_path: Path) -> None:
        async with Storage(tmp_path / "test.db") as db:
            run1 = await db.create_run("test")
            run2 = await db.create_run("test")
            id1 = await db.add_firm(run1, "https://example.com")
            id2 = await db.add_firm(run2, "https://example.com")
        assert id1 != id2

    async def test_firm_state_survives_reconnect(self, tmp_path: Path) -> None:
        path = tmp_path / "test.db"
        async with Storage(path) as db:
            run_id = await db.create_run("test")
            firm_id = await db.add_firm(run_id, "https://example.com")
            await db.update_firm_stage(
                firm_id,
                "extracted",
                extracted_profile={"firm_name": "Acme Law"},
            )

        async with Storage(path) as db:
            firm = await db.get_firm(firm_id)
        assert firm["stage"] == "extracted"
        assert firm["extracted_profile"] == {"firm_name": "Acme Law"}

    async def test_pending_firms_queryable_after_partial_run(self, tmp_path: Path) -> None:
        path = tmp_path / "test.db"
        async with Storage(path) as db:
            run_id = await db.create_run("test")
            await db.add_firm(run_id, "https://a.com")
            await db.add_firm(run_id, "https://b.com")
            done_id = await db.add_firm(run_id, "https://c.com")
            await db.update_firm_stage(done_id, "completed")

        async with Storage(path) as db:
            pending = await db.get_firms_by_stage(run_id, "pending")
            completed = await db.get_firms_by_stage(run_id, "completed")
        assert len(pending) == 2
        assert len(completed) == 1

    async def test_new_run_does_not_see_prior_run_firms(self, tmp_path: Path) -> None:
        path = tmp_path / "test.db"
        async with Storage(path) as db:
            run1 = await db.create_run("test")
            await db.add_firm(run1, "https://a.com")
            await db.add_firm(run1, "https://b.com")

        async with Storage(path) as db:
            run2 = await db.create_run("test")
            pending = await db.get_firms_by_stage(run2, "pending")
        assert len(pending) == 0

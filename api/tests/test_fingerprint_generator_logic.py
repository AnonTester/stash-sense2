"""Tests for fingerprint_generator.py dataclasses and enums."""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from fingerprint_generator import (
    GeneratorStatus,
    GeneratorProgress,
    FingerprintResult,
    SceneFingerprintGenerator,
    StashUnavailableError,
    STASH_RETRY_BUDGET_SECONDS,
)


class TestGeneratorStatus:
    def test_enum_values(self):
        assert GeneratorStatus.IDLE == "idle"
        assert GeneratorStatus.RUNNING == "running"
        assert GeneratorStatus.PAUSED == "paused"
        assert GeneratorStatus.STOPPING == "stopping"
        assert GeneratorStatus.COMPLETED == "completed"
        assert GeneratorStatus.ERROR == "error"

    def test_all_statuses_present(self):
        expected = {"idle", "running", "paused", "stopping", "completed", "error"}
        actual = {s.value for s in GeneratorStatus}
        assert actual == expected


class TestGeneratorProgress:
    def test_progress_pct_zero_when_total_zero(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.IDLE,
            total_scenes=0,
            processed_scenes=0,
            successful=0,
            failed=0,
            skipped=0,
        )
        assert progress.progress_pct == 0.0

    def test_progress_pct_correct_percentage(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=200,
            processed_scenes=50,
            successful=40,
            failed=5,
            skipped=5,
        )
        assert progress.progress_pct == 25.0

    def test_progress_pct_100_when_complete(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.COMPLETED,
            total_scenes=100,
            processed_scenes=100,
            successful=90,
            failed=5,
            skipped=5,
        )
        assert progress.progress_pct == 100.0

    def test_to_dict_all_keys_present(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=500,
            processed_scenes=125,
            successful=100,
            failed=10,
            skipped=15,
            current_scene_id=42,
            current_scene_title="Test Scene",
            error_message=None,
        )
        d = progress.to_dict()

        expected_keys = {
            "status", "total_scenes", "processed_scenes", "successful",
            "failed", "skipped", "progress_pct", "current_scene_id",
            "current_scene_title", "error_message",
        }
        assert set(d.keys()) == expected_keys


class TestGetScenesWithRetry:
    """_get_scenes_with_retry() should absorb a brief Stash connectivity
    outage (container restart, transient network blip) rather than letting
    it kill an overnight run immediately -- see the incident this was
    added for: a daily auto-update cron restarted Stash mid-run and the
    job died on the very next scene-list fetch."""

    def _generator(self):
        stash = MagicMock()
        stash.base_url = "http://192.168.1.2:9997"
        stash.get_scenes_for_fingerprinting = AsyncMock()
        rec_db = MagicMock()
        return SceneFingerprintGenerator(stash_client=stash, rec_db=rec_db, db_version="2026.01.01"), stash

    @pytest.mark.asyncio
    async def test_succeeds_immediately_without_retry(self):
        gen, stash = self._generator()
        stash.get_scenes_for_fingerprinting.return_value = ([{"id": "1"}], 1)

        result = await gen._get_scenes_with_retry(limit=100, offset=0)

        assert result == ([{"id": "1"}], 1)
        stash.get_scenes_for_fingerprinting.assert_awaited_once_with(limit=100, offset=0)

    @pytest.mark.asyncio
    async def test_retries_through_connect_error_then_succeeds(self, monkeypatch):
        gen, stash = self._generator()
        stash.get_scenes_for_fingerprinting.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
            ([{"id": "1"}], 1),
        ]
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr("fingerprint_generator.asyncio.sleep", fake_sleep)

        result = await gen._get_scenes_with_retry(limit=100, offset=0)

        assert result == ([{"id": "1"}], 1)
        assert stash.get_scenes_for_fingerprinting.await_count == 3
        assert len(sleep_calls) == 2

    @pytest.mark.parametrize("exc", [
        httpx.ConnectError("Connection refused"),
        httpx.ConnectTimeout("timed out"),
        httpx.ReadTimeout("timed out"),
    ])
    @pytest.mark.asyncio
    async def test_retries_all_connectivity_exception_types(self, monkeypatch, exc):
        gen, stash = self._generator()
        stash.get_scenes_for_fingerprinting.side_effect = [exc, ([{"id": "1"}], 1)]
        monkeypatch.setattr("fingerprint_generator.asyncio.sleep", AsyncMock())

        result = await gen._get_scenes_with_retry(limit=100, offset=0)

        assert result == ([{"id": "1"}], 1)
        assert stash.get_scenes_for_fingerprinting.await_count == 2

    @pytest.mark.asyncio
    async def test_non_connectivity_exception_propagates_without_retry(self):
        gen, stash = self._generator()
        stash.get_scenes_for_fingerprinting.side_effect = RuntimeError("query malformed")

        with pytest.raises(RuntimeError, match="query malformed"):
            await gen._get_scenes_with_retry(limit=100, offset=0)

        stash.get_scenes_for_fingerprinting.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gives_up_after_budget_exhausted_with_stash_specific_message(self, monkeypatch):
        gen, stash = self._generator()
        stash.get_scenes_for_fingerprinting.side_effect = httpx.ConnectError("Connection refused")

        # Fake clock: each check of time.monotonic() advances past the
        # retry budget after a couple of iterations, so the test doesn't
        # actually wait STASH_RETRY_BUDGET_SECONDS in real time.
        clock = {"t": 0.0}

        def fake_monotonic():
            return clock["t"]

        async def fake_sleep(seconds):
            clock["t"] += STASH_RETRY_BUDGET_SECONDS  # jump straight past the deadline

        monkeypatch.setattr("fingerprint_generator.time.monotonic", fake_monotonic)
        monkeypatch.setattr("fingerprint_generator.asyncio.sleep", fake_sleep)

        with pytest.raises(StashUnavailableError) as exc_info:
            await gen._get_scenes_with_retry(limit=100, offset=0)

        message = str(exc_info.value)
        assert "Stash" in message
        assert "192.168.1.2:9997" in message
        assert stash.get_scenes_for_fingerprinting.await_count == 2

    def test_to_dict_correct_types(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=500,
            processed_scenes=125,
            successful=100,
            failed=10,
            skipped=15,
        )
        d = progress.to_dict()

        assert isinstance(d["status"], str)
        assert isinstance(d["total_scenes"], int)
        assert isinstance(d["processed_scenes"], int)
        assert isinstance(d["successful"], int)
        assert isinstance(d["failed"], int)
        assert isinstance(d["skipped"], int)
        assert isinstance(d["progress_pct"], float)

    def test_to_dict_status_is_string_value(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.COMPLETED,
            total_scenes=10,
            processed_scenes=10,
            successful=10,
            failed=0,
            skipped=0,
        )
        d = progress.to_dict()
        assert d["status"] == "completed"

    def test_to_dict_progress_pct_rounded(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=3,
            processed_scenes=1,
            successful=1,
            failed=0,
            skipped=0,
        )
        d = progress.to_dict()
        # 1/3 * 100 = 33.333... -> rounded to 33.3
        assert d["progress_pct"] == 33.3


class TestFingerprintResult:
    def test_creation_with_defaults(self):
        result = FingerprintResult(scene_id=42, success=True)
        assert result.scene_id == 42
        assert result.success is True
        assert result.fingerprint_id is None
        assert result.performers_found == 0
        assert result.frames_analyzed == 0
        assert result.error is None

    def test_creation_with_values(self):
        result = FingerprintResult(
            scene_id=99,
            success=True,
            fingerprint_id=5,
            performers_found=3,
            frames_analyzed=60,
        )
        assert result.fingerprint_id == 5
        assert result.performers_found == 3
        assert result.frames_analyzed == 60

    def test_creation_with_error(self):
        result = FingerprintResult(
            scene_id=42,
            success=False,
            error="Timeout",
        )
        assert result.success is False
        assert result.error == "Timeout"

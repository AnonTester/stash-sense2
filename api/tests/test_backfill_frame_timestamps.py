"""Tests for backfill_frame_timestamps.py -- reconstructing real per-face
jump-to-frame timestamps for already-cached scene_face_embeddings rows
without any re-detection/re-embedding. See that module's own docstring for
the full rationale.
"""
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from backfill_frame_timestamps import (
    backfill_sprite_frame_indices,
    backfill_video_timestamps,
    run_video_timestamp_backfill_once,
)
from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "test.db"))


@pytest.fixture
def conn(db):
    c = sqlite3.connect(db.db_path)
    yield c
    c.close()


class TestBackfillSpriteFrameIndices:
    def test_reassigns_shared_frame_index_to_unique_values(self, db, conn):
        db.replace_face_embeddings(1, [
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a", "timestamp_sec": 5.0},
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"b", "timestamp_sec": 15.0},
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"c", "timestamp_sec": 25.0},
        ], is_sprite=True)

        updated = backfill_sprite_frame_indices(conn, dry_run=False)
        conn.commit()

        assert updated == 3
        rows = db.get_face_embeddings(1, is_sprite=True)
        indices = [r["frame_index"] for r in rows]
        assert len(set(indices)) == 3
        assert all(i <= -2 for i in indices)
        # Each row's own real timestamp must survive the reassignment untouched.
        assert {r["timestamp_sec"] for r in rows} == {5.0, 15.0, 25.0}

    def test_already_unique_scene_left_untouched(self, db, conn):
        db.replace_face_embeddings(1, [
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a", "timestamp_sec": 5.0},
            {"frame_index": -3, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"b", "timestamp_sec": 15.0},
        ], is_sprite=True)

        updated = backfill_sprite_frame_indices(conn, dry_run=False)

        assert updated == 0

    def test_dry_run_makes_no_changes(self, db, conn):
        db.replace_face_embeddings(1, [
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a", "timestamp_sec": 5.0},
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"b", "timestamp_sec": 15.0},
        ], is_sprite=True)

        updated = backfill_sprite_frame_indices(conn, dry_run=True)

        assert updated == 2
        rows = db.get_face_embeddings(1, is_sprite=True)
        assert {r["frame_index"] for r in rows} == {-2}  # unchanged

    def test_multiple_scenes_handled_independently(self, db, conn):
        db.replace_face_embeddings(1, [
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a", "timestamp_sec": 1.0},
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"b", "timestamp_sec": 2.0},
        ], is_sprite=True)
        db.replace_face_embeddings(2, [
            {"frame_index": -2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"c", "timestamp_sec": 3.0},
        ], is_sprite=True)

        updated = backfill_sprite_frame_indices(conn, dry_run=False)

        assert updated == 2  # scene 2's single row was already unique, untouched
        assert len({r["frame_index"] for r in db.get_face_embeddings(1, is_sprite=True)}) == 2


class TestBackfillVideoTimestamps:
    async def test_reconstructs_timestamp_from_cached_extraction_params(self, db, conn):
        db.save_scene_signal_cache(
            1, num_frames=5, min_face_size=80, min_face_confidence=0.5,
            start_offset_pct=0.0, end_offset_pct=1.0, frames_analyzed=5,
        )
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
            {"frame_index": 2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"b"},
            {"frame_index": 4, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"c"},
        ], is_sprite=False)

        with patch(
            "backfill_frame_timestamps._fetch_duration",
            AsyncMock(return_value=100.0),
        ):
            updated, skipped = await backfill_video_timestamps(conn, "http://stash", "key", dry_run=False)

        assert updated == 3
        assert skipped == 0
        rows = {r["frame_index"]: r["timestamp_sec"] for r in db.get_face_embeddings(1, is_sprite=False)}
        # duration=100, offsets 0..1 -> interval = 100/4 = 25s/frame
        assert rows[0] == pytest.approx(0.0)
        assert rows[2] == pytest.approx(50.0)
        assert rows[4] == pytest.approx(100.0)

    async def test_respects_nonzero_offsets(self, db, conn):
        db.save_scene_signal_cache(
            1, num_frames=3, min_face_size=80, min_face_confidence=0.5,
            start_offset_pct=0.1, end_offset_pct=0.9, frames_analyzed=3,
        )
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
            {"frame_index": 1, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"b"},
            {"frame_index": 2, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"c"},
        ], is_sprite=False)

        with patch(
            "backfill_frame_timestamps._fetch_duration",
            AsyncMock(return_value=200.0),
        ):
            await backfill_video_timestamps(conn, "http://stash", "key", dry_run=False)

        rows = {r["frame_index"]: r["timestamp_sec"] for r in db.get_face_embeddings(1, is_sprite=False)}
        # start=20, end=180, interval=(180-20)/2=80
        assert rows[0] == pytest.approx(20.0)
        assert rows[1] == pytest.approx(100.0)
        assert rows[2] == pytest.approx(180.0)

    async def test_scene_with_no_cache_meta_is_skipped(self, db, conn):
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
        ], is_sprite=False)

        updated, skipped = await backfill_video_timestamps(conn, "http://stash", "key", dry_run=False)

        assert updated == 0
        assert skipped == 1

    async def test_duration_fetch_failure_skips_scene_without_raising(self, db, conn):
        db.save_scene_signal_cache(
            1, num_frames=3, min_face_size=80, min_face_confidence=0.5,
            start_offset_pct=0.0, end_offset_pct=1.0, frames_analyzed=3,
        )
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
        ], is_sprite=False)

        with patch(
            "backfill_frame_timestamps._fetch_duration",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            updated, skipped = await backfill_video_timestamps(conn, "http://stash", "key", dry_run=False)

        assert updated == 0
        assert skipped == 1

    async def test_dry_run_makes_no_changes(self, db, conn):
        db.save_scene_signal_cache(
            1, num_frames=3, min_face_size=80, min_face_confidence=0.5,
            start_offset_pct=0.0, end_offset_pct=1.0, frames_analyzed=3,
        )
        db.replace_face_embeddings(1, [
            {"frame_index": 1, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
        ], is_sprite=False)

        with patch(
            "backfill_frame_timestamps._fetch_duration",
            AsyncMock(return_value=100.0),
        ):
            updated, skipped = await backfill_video_timestamps(conn, "http://stash", "key", dry_run=True)

        assert updated == 1
        rows = db.get_face_embeddings(1, is_sprite=False)
        assert rows[0]["timestamp_sec"] is None  # unchanged


class TestRunVideoTimestampBackfillOnce:
    """The main.py startup wrapper: runs at most once per install (gated by
    a user_settings flag), never blocks/raises out to the caller, and
    leaves the flag unset on failure so the next startup retries."""

    async def test_runs_and_sets_the_done_flag(self, db):
        db.save_scene_signal_cache(
            1, num_frames=3, min_face_size=80, min_face_confidence=0.5,
            start_offset_pct=0.0, end_offset_pct=1.0, frames_analyzed=3,
        )
        db.replace_face_embeddings(1, [
            {"frame_index": 1, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
        ], is_sprite=False)

        with patch(
            "backfill_frame_timestamps._fetch_duration",
            AsyncMock(return_value=100.0),
        ):
            await run_video_timestamp_backfill_once(db, "http://stash", "key")

        assert db.get_user_setting("_video_timestamp_backfill_v1_done") is True
        rows = db.get_face_embeddings(1, is_sprite=False)
        assert rows[0]["timestamp_sec"] == pytest.approx(50.0)

    async def test_second_call_is_a_no_op(self, db):
        db.set_user_setting("_video_timestamp_backfill_v1_done", True)
        db.save_scene_signal_cache(
            1, num_frames=3, min_face_size=80, min_face_confidence=0.5,
            start_offset_pct=0.0, end_offset_pct=1.0, frames_analyzed=3,
        )
        db.replace_face_embeddings(1, [
            {"frame_index": 1, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
        ], is_sprite=False)

        fetch_mock = AsyncMock(return_value=100.0)
        with patch("backfill_frame_timestamps._fetch_duration", fetch_mock):
            await run_video_timestamp_backfill_once(db, "http://stash", "key")

        fetch_mock.assert_not_called()
        rows = db.get_face_embeddings(1, is_sprite=False)
        assert rows[0]["timestamp_sec"] is None  # never touched -- flag already set

    async def test_failure_leaves_flag_unset_for_retry_next_startup(self, db):
        db.save_scene_signal_cache(
            1, num_frames=3, min_face_size=80, min_face_confidence=0.5,
            start_offset_pct=0.0, end_offset_pct=1.0, frames_analyzed=3,
        )
        db.replace_face_embeddings(1, [
            {"frame_index": 1, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a"},
        ], is_sprite=False)

        with patch(
            "backfill_frame_timestamps.backfill_video_timestamps",
            AsyncMock(side_effect=RuntimeError("db exploded")),
        ):
            await run_video_timestamp_backfill_once(db, "http://stash", "key")  # must not raise

        assert db.get_user_setting("_video_timestamp_backfill_v1_done") is None

    async def test_no_stash_url_is_a_no_op(self, db):
        fetch_mock = AsyncMock(return_value=100.0)
        with patch("backfill_frame_timestamps._fetch_duration", fetch_mock):
            await run_video_timestamp_backfill_once(db, "", "key")

        fetch_mock.assert_not_called()
        assert db.get_user_setting("_video_timestamp_backfill_v1_done") is None

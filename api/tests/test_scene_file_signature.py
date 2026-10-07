"""Cached faces belong to the video they were taken from: a scene whose primary file is swapped or merged must not be
served from its old cache (scene_file_signature.py)."""
import numpy as np
import pytest

from recommendations_db import RecommendationsDB
from scene_file_signature import (
    STALE_FILE_MARKER,
    FileSignature,
    ensure_scene_cache_matches_file,
    legacy_cache_is_stale,
    reconcile_scene_signatures,
    signature_from_scene,
    signatures_match,
)


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "rec.db"))


def _cache_scene(db, scene_id, timestamps=(9.0, 171.0), end_pct=0.95, sprite=True):
    db.save_scene_signal_cache(scene_id, num_frames=60, min_face_size=40, min_face_confidence=0.5,
                               start_offset_pct=0.05, end_offset_pct=end_pct, frames_analyzed=60)
    db.replace_face_embeddings(scene_id, [
        {"frame_index": i, "bbox": {"x": 0}, "confidence": 0.9, "embedding": np.ones(4, dtype=np.float32).tobytes(),
         "timestamp_sec": ts} for i, ts in enumerate(timestamps)])
    if sprite:
        db.mark_sprite_cache_checked(scene_id)


def _has_cache(db, scene_id):
    return bool(db.get_scene_signal_cache(scene_id)) or bool(db.get_face_embeddings(scene_id)) \
        or db.is_sprite_cache_checked(scene_id)


class TestPureHelpers:
    def test_signature_from_scene_uses_the_primary_file(self):
        sig = signature_from_scene({"files": [{"id": 7, "duration": 14091.1}, {"id": 8, "duration": 180}]})
        assert sig == FileSignature("7", 14091.1)
        assert signature_from_scene({"files": []}) is None and signature_from_scene(None) is None

    def test_match_rules(self):
        cur = FileSignature("7", 100.0)
        assert signatures_match({"file_id": "7", "duration_sec": 100.2}, cur)
        assert not signatures_match({"file_id": "8", "duration_sec": 100.0}, cur)           # another file
        assert not signatures_match({"file_id": "7", "duration_sec": 180.0}, cur)           # same id, replaced content
        assert signatures_match({"file_id": None, "duration_sec": None}, cur)               # nothing known: not held against it
        assert signatures_match({"file_id": "7", "duration_sec": 14091.0}, FileSignature("7", 14091.5))

    def test_legacy_rule(self):
        # frames spread to 95% of a ~180 s video; the file is now 3.9 hours
        assert legacy_cache_is_stale(171.2, 0.95, 14091.0)
        assert not legacy_cache_is_stale(171.2, 0.95, 180.2)
        assert not legacy_cache_is_stale(160.0, 0.95, 180.2)         # last frame a little short of the end: normal
        assert not legacy_cache_is_stale(282.0, 0.95, 375.1)         # real unchanged scene: last frame at 75% of the video
        assert legacy_cache_is_stale(500.0, 0.95, 120.0)             # shorter than a frame taken from it
        assert not legacy_cache_is_stale(0, 0.95, 14091.0) and not legacy_cache_is_stale(171.0, 0.95, None)


class TestEnsureSceneCacheMatchesFile:
    async def test_new_scene_just_records_its_file(self, db):
        assert await ensure_scene_cache_matches_file(db, 1, FileSignature("7", 100.0)) is False
        assert db.get_scene_file_signature(1) == {"file_id": "7", "duration_sec": 100.0}

    async def test_unchanged_file_keeps_the_cache(self, db):
        _cache_scene(db, 1)
        await ensure_scene_cache_matches_file(db, 1, FileSignature("7", 180.0))
        assert await ensure_scene_cache_matches_file(db, 1, FileSignature("7", 180.3)) is False
        assert _has_cache(db, 1)

    async def test_swapped_primary_file_drops_the_cache(self, db):
        _cache_scene(db, 1)
        await ensure_scene_cache_matches_file(db, 1, FileSignature("7", 180.0))
        assert await ensure_scene_cache_matches_file(db, 1, FileSignature("9", 14091.0)) is True
        assert not _has_cache(db, 1)
        assert db.get_scene_file_signature(1) == {"file_id": "9", "duration_sec": 14091.0}

    async def test_old_cache_without_a_signature_is_judged_by_its_frames(self, db):
        _cache_scene(db, 1)                               # a 3-minute video's frames, written before signatures
        assert await ensure_scene_cache_matches_file(db, 1, FileSignature("9", 14091.0)) is True
        assert not _has_cache(db, 1)

    async def test_old_cache_that_still_fits_is_adopted(self, db):
        _cache_scene(db, 1)
        assert await ensure_scene_cache_matches_file(db, 1, FileSignature("7", 180.2)) is False
        assert _has_cache(db, 1) and db.get_scene_file_signature(1)["file_id"] == "7"

    async def test_unknown_current_file_changes_nothing(self, db):
        _cache_scene(db, 1)
        assert await ensure_scene_cache_matches_file(db, 1, None) is False
        assert _has_cache(db, 1) and db.get_scene_file_signature(1) is None


class TestReconcile:
    def test_changed_scene_loses_its_cache_and_is_marked_outdated(self, db):
        _cache_scene(db, 1)
        _cache_scene(db, 2)
        db.set_scene_file_signatures({1: ("7", 180.0), 2: ("8", 600.0)})
        for sid in (1, 2):
            db.create_scene_fingerprint(sid, 3, 60, fingerprint_status="complete", db_version="2026.10.05")
        result = reconcile_scene_signatures(db, {1: FileSignature("9", 14091.0), 2: FileSignature("8", 600.0)})
        assert result.changed == [1] and result.checked == 2
        # only the changed scene's stored result is outdated, so Refresh Outdated redoes just that one
        assert db.get_scene_fingerprint(1)["db_version"] == STALE_FILE_MARKER
        assert db.get_scene_fingerprint(2)["db_version"] == "2026.10.05"
        assert not _has_cache(db, 1) and _has_cache(db, 2)
        assert db.get_scene_file_signature(1)["file_id"] == "9"

    def test_scene_without_signature_is_adopted_or_caught_by_the_legacy_rule(self, db):
        _cache_scene(db, 1)                                   # 3-minute frames, file is now 3.9 h
        _cache_scene(db, 2, timestamps=(30.0, 570.0))         # fits a 600 s file
        result = reconcile_scene_signatures(db, {1: FileSignature("9", 14091.0), 2: FileSignature("8", 600.0),
                                                 3: FileSignature("5", 90.0)})
        assert result.changed == [1] and result.adopted == 2
        assert not _has_cache(db, 1) and _has_cache(db, 2)
        assert set(db.get_all_scene_file_signatures()) == {1, 2, 3}

    def test_second_run_is_a_no_op(self, db):
        _cache_scene(db, 1)
        current = {1: FileSignature("9", 14091.0)}
        reconcile_scene_signatures(db, current)
        again = reconcile_scene_signatures(db, current)
        assert again.changed == [] and again.adopted == 0

    def test_stale_marker_value(self):
        assert STALE_FILE_MARKER

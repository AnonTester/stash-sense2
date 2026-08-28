"""Tests for SceneFaceMatchAnalyzer (Face Recommendations) -- reads Face
Identification's stored per-scene match data instead of re-running
detect+embed+match, with an on-demand sprite top-up for scenes whose stored
data predates sprite coverage. See analyzers/scene_face_match.py's docstring.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from analyzers.scene_face_match import SceneFaceMatchAnalyzer


def _scene(scene_id, duration=60, width=1920, height=1080, updated_at="2026-01-01T00:00:00Z"):
    return {
        "id": str(scene_id), "title": f"Scene {scene_id}", "updated_at": updated_at,
        "files": [{"duration": duration, "width": width, "height": height}],
    }


def _match_row(universal_id, person_id=0, frame_count=10, is_best_match=True, confidence=0.7):
    return {
        "person_id": person_id, "frame_count": frame_count, "is_best_match": is_best_match,
        "universal_id": universal_id, "stashdb_id": "abc", "name": "Performer",
        "confidence": confidence, "distance": 1 - confidence, "country": None,
        "image_url": None, "endpoint": "stashdb.org", "local_performer_id": None,
        "source": None, "catalogue_url": None, "profile_url": None, "top_timestamps_sec": [],
    }


def _analyzer(scenes, fingerprints):
    """scenes: list of scene dicts. fingerprints: {scene_id_int: fp_dict_or_None}."""
    stash = MagicMock()
    stash.get_scenes_without_performers = AsyncMock(side_effect=[(scenes, len(scenes)), ([], len(scenes))])
    rec_db = MagicMock()
    rec_db.get_watermark.return_value = None
    rec_db.is_dismissed.return_value = False
    rec_db.get_scene_fingerprint.side_effect = lambda sid: fingerprints.get(sid)
    rec_db.create_recommendation.side_effect = lambda **kwargs: 1  # always "created"
    return SceneFaceMatchAnalyzer(stash=stash, rec_db=rec_db, run_id=None), stash, rec_db


def _patch_common(**overrides):
    import identification_router
    defaults = dict(
        require_db_available=AsyncMock(),
    )
    defaults.update(overrides)
    return [patch.object(identification_router, name, value, create=True) for name, value in defaults.items()]


class TestStoredDataPath:
    async def test_scene_with_sprite_coverage_reads_stored_matches_no_identify_call(self):
        fp = {"id": 5, "fingerprint_status": "complete", "used_sprite": 1}
        analyzer, stash, rec_db = _analyzer([_scene(1)], {1: fp})
        rec_db.get_fingerprint_matches.return_value = [_match_row("stashdb.org:perf-1")]

        patchers = _patch_common()
        for p in patchers:
            p.start()
        try:
            with patch("scene_batch_orchestrator.identify_scenes_batched") as batched:
                result = await analyzer.run(incremental=False)
        finally:
            for p in patchers:
                p.stop()

        # Called with an empty spec list (harmless no-op) since this scene
        # doesn't need a top-up -- the real behavior under test is that no
        # fresh identify work happened for it.
        assert batched.call_args.args[0] == []
        rec_db.get_fingerprint_matches.assert_called_once_with(5)
        assert result.recommendations_created == 1
        assert result.items_processed == 1
        created_kwargs = rec_db.create_recommendation.call_args.kwargs
        assert created_kwargs["target_id"] == "1|stashdb.org:perf-1"
        assert created_kwargs["details"]["is_best_match"] is True

    async def test_scene_with_no_fingerprint_is_skipped(self):
        analyzer, stash, rec_db = _analyzer([_scene(1)], {1: None})

        patchers = _patch_common()
        for p in patchers:
            p.start()
        try:
            with patch("scene_batch_orchestrator.identify_scenes_batched") as batched:
                result = await analyzer.run(incremental=False)
        finally:
            for p in patchers:
                p.stop()

        assert batched.call_args.args[0] == []
        rec_db.create_recommendation.assert_not_called()
        assert result.items_processed == 1
        assert result.recommendations_created == 0

    async def test_scene_with_error_status_is_skipped(self):
        fp = {"id": 5, "fingerprint_status": "error", "used_sprite": 0}
        analyzer, stash, rec_db = _analyzer([_scene(1)], {1: fp})

        patchers = _patch_common()
        for p in patchers:
            p.start()
        try:
            with patch("scene_batch_orchestrator.identify_scenes_batched") as batched:
                result = await analyzer.run(incremental=False)
        finally:
            for p in patchers:
                p.stop()

        assert batched.call_args.args[0] == []
        rec_db.create_recommendation.assert_not_called()
        assert result.items_processed == 1


class TestSpriteTopUpPath:
    async def test_scene_without_sprite_coverage_triggers_topup_and_creates_recommendation(self):
        fp = {"id": 7, "fingerprint_status": "complete", "used_sprite": 0}
        analyzer, stash, rec_db = _analyzer([_scene(1)], {1: fp})

        match = SimpleNamespace(
            stashdb_id="abc", name="Performer", confidence=0.9, distance=0.1, country=None,
            image_url=None, endpoint="stashdb.org", local_performer_id=None, source=None,
            catalogue_url=None, profile_url=None, top_timestamps_sec=[],
        )
        person = SimpleNamespace(person_id=0, frame_count=12, best_match=match, all_matches=[match])
        response = SimpleNamespace(persons=[person])

        async def _fake_batched(specs, is_stop_requested=None, before_scene=None):
            assert len(specs) == 1
            assert specs[0].request.use_sprite is True
            assert specs[0].request.matching_mode == "hybrid"
            yield "1", response

        patchers = _patch_common()
        for p in patchers:
            p.start()
        try:
            with patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_fake_batched):
                result = await analyzer.run(incremental=False)
        finally:
            for p in patchers:
                p.stop()

        rec_db.get_fingerprint_matches.assert_not_called()
        assert result.recommendations_created == 1
        created_kwargs = rec_db.create_recommendation.call_args.kwargs
        assert created_kwargs["target_id"] == "1|stashdb.org:abc"

    async def test_topup_failure_is_recorded_as_error_not_raised(self):
        fp = {"id": 7, "fingerprint_status": "complete", "used_sprite": 0}
        analyzer, stash, rec_db = _analyzer([_scene(1)], {1: fp})

        async def _fake_batched(specs, is_stop_requested=None, before_scene=None):
            yield "1", RuntimeError("boom")

        patchers = _patch_common()
        for p in patchers:
            p.start()
        try:
            with patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_fake_batched):
                result = await analyzer.run(incremental=False)
        finally:
            for p in patchers:
                p.stop()

        assert result.recommendations_created == 0
        assert any("boom" in e for e in result.errors)

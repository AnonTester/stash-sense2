"""Tests for fingerprint_generator.py's in-process identify path (replacing
the old self-HTTP loopback to /identify/scene) and its use of
scene_batch_orchestrator.py for the bulk generate_all() loop.

Mocks identification_router._identify_scene_impl/require_db_available and
scene_batch_orchestrator.identify_scenes_batched -- these tests are about
fingerprint_generator.py's own bookkeeping (skip logic, pre-emptive error
rows, shifted-retry consolidation, progress/cursor semantics), not the
underlying face-recognition pipeline.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fingerprint_generator import FingerprintResult, SceneFingerprintGenerator


def _generator():
    stash = MagicMock()
    stash.get_all_scene_ids = AsyncMock(return_value={1})
    rec_db = MagicMock()
    rec_db.get_scene_fingerprint.return_value = None
    gen = SceneFingerprintGenerator(stash_client=stash, rec_db=rec_db, db_version="2026.01.01")
    return gen, stash, rec_db


def _response(faces_after_filter=1, fingerprint_saved=True, fingerprint_error=None, has_match=True, frames_analyzed=10):
    person = SimpleNamespace(best_match=SimpleNamespace() if has_match else None)
    return SimpleNamespace(
        persons=[person] if has_match else [],
        faces_after_filter=faces_after_filter,
        fingerprint_saved=fingerprint_saved,
        fingerprint_error=fingerprint_error,
        frames_analyzed=frames_analyzed,
    )


class TestResponseToResult:
    def test_success_maps_fields(self):
        gen, _, rec_db = _generator()
        response = _response(faces_after_filter=3, frames_analyzed=42)

        result = gen._response_to_result(1, response)

        assert result.success is True
        assert result.performers_found == 1
        assert result.faces_found == 3
        assert result.frames_analyzed == 42
        rec_db.create_scene_fingerprint.assert_not_called()

    def test_save_failed_with_matches_is_a_failure(self):
        gen, _, rec_db = _generator()
        response = _response(fingerprint_saved=False, fingerprint_error="disk full")

        result = gen._response_to_result(1, response)

        assert result.success is False
        assert "disk full" in result.error
        assert result.performers_found == 1

    def test_zero_matches_unsaved_is_still_success(self):
        # No performers found -> nothing to save -- fingerprint_saved=False
        # here just means "there was nothing to persist", not a failure.
        gen, _, rec_db = _generator()
        response = _response(fingerprint_saved=False, has_match=False, faces_after_filter=0)

        result = gen._response_to_result(1, response)

        assert result.success is True
        assert result.performers_found == 0

    def test_http_exception_is_mapped_and_recorded(self):
        from fastapi import HTTPException
        gen, _, rec_db = _generator()

        result = gen._response_to_result(1, HTTPException(status_code=404, detail="Scene not found"))

        assert result.success is False
        assert "Scene not found" in result.error
        rec_db.create_scene_fingerprint.assert_called_once()
        assert rec_db.create_scene_fingerprint.call_args.kwargs["fingerprint_status"] == "error"

    def test_generic_exception_is_mapped_and_recorded(self):
        gen, _, rec_db = _generator()

        result = gen._response_to_result(1, RuntimeError("boom"))

        assert result.success is False
        assert "boom" in result.error
        rec_db.create_scene_fingerprint.assert_called_once()


class TestIdentifyScene:
    async def test_no_retry_when_faces_found(self):
        gen, _, _ = _generator()
        with patch("identification_router._identify_scene_impl", AsyncMock(return_value=_response(faces_after_filter=2))), \
             patch("identification_router.require_db_available", AsyncMock()):
            result = await gen._identify_scene(1)

        assert result.success is True
        assert result.faces_found == 2
        assert result.retried_with_shifted_frames is False

    async def test_retries_and_uses_retry_result_when_it_succeeds(self):
        gen, _, _ = _generator()
        responses = [_response(faces_after_filter=0), _response(faces_after_filter=1)]
        impl = AsyncMock(side_effect=lambda req: responses.pop(0))
        with patch("identification_router._identify_scene_impl", impl), \
             patch("identification_router.require_db_available", AsyncMock()):
            result = await gen._identify_scene(1)

        assert impl.await_count == 2
        assert result.success is True
        assert result.faces_found == 1
        assert result.retried_with_shifted_frames is True

    async def test_retry_failure_keeps_original_zero_face_result(self):
        gen, _, _ = _generator()
        impl = AsyncMock(side_effect=[_response(faces_after_filter=0), RuntimeError("retry boom")])
        with patch("identification_router._identify_scene_impl", impl), \
             patch("identification_router.require_db_available", AsyncMock()):
            result = await gen._identify_scene(1)

        assert result.success is True
        assert result.faces_found == 0
        assert result.retried_with_shifted_frames is False


def _batched_results(pairs):
    async def _gen(specs, is_stop_requested=None, before_scene=None, on_scene_start=None):
        for scene_id, outcome in pairs:
            if on_scene_start:
                await on_scene_start(scene_id)
            yield scene_id, outcome
    return _gen


class TestUseSpriteAlwaysRequested:
    """Sprite-tile embeddings are now cached the same way video-frame
    embeddings already are (scene_face_embeddings, is_sprite=1 -- see
    identification_router.py's scene_sprite_cache_status), so requesting
    them costs real detection time only the first time for a given scene,
    then nothing on every call after. A bulk run requests it by default --
    missing scenes, scenes refreshed for being outdated, and scenes that
    already had sprite coverage all get (or keep) it -- unless the "Use
    Sprite Tiles For Detection" setting is off (see
    TestUseSpriteGatedBySetting below); these tests don't call
    settings.init_settings() at all, so generate_all()'s own
    RuntimeError/KeyError fallback (settings unavailable -> default True)
    is what's actually exercised here, same net effect as the setting
    being on."""

    async def test_missing_scene_requests_use_sprite(self):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[([_scene], 1), ([_scene], 1), ([], 1)]
        )
        rec_db.get_scene_fingerprint.return_value = None
        captured_specs = []

        def _batched(specs, is_stop_requested=None, before_scene=None, on_scene_start=None):
            captured_specs.extend(specs)
            return _batched_results([("1", _response(faces_after_filter=1))])(specs)

        with patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_batched):
            [p async for p in gen.generate_all(batch_size=100)]

        assert captured_specs[0].request.use_sprite is True

    async def test_refresh_of_scene_with_existing_sprite_coverage_requests_use_sprite(self):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[([_scene], 1), ([_scene], 1), ([], 1)]
        )
        rec_db.get_scene_fingerprint.return_value = {
            "fingerprint_status": "complete", "db_version": "2025.01.01", "used_sprite": 1,
        }
        captured_specs = []

        def _batched(specs, is_stop_requested=None, before_scene=None, on_scene_start=None):
            captured_specs.extend(specs)
            return _batched_results([("1", _response(faces_after_filter=1))])(specs)

        with patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_batched):
            [p async for p in gen.generate_all(batch_size=100, refresh_outdated=True)]

        assert captured_specs[0].request.use_sprite is True

    async def test_refresh_of_scene_without_prior_sprite_coverage_requests_use_sprite(self):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[([_scene], 1), ([_scene], 1), ([], 1)]
        )
        rec_db.get_scene_fingerprint.return_value = {
            "fingerprint_status": "complete", "db_version": "2025.01.01", "used_sprite": 0,
        }
        captured_specs = []

        def _batched(specs, is_stop_requested=None, before_scene=None, on_scene_start=None):
            captured_specs.extend(specs)
            return _batched_results([("1", _response(faces_after_filter=1))])(specs)

        with patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_batched):
            [p async for p in gen.generate_all(batch_size=100, refresh_outdated=True)]

        assert captured_specs[0].request.use_sprite is True


class TestUseSpriteGatedBySetting:
    """generate_all() reads the "Use Sprite Tiles For Detection" setting
    once per run (see its own docstring for why per-run, not per-scene) and
    threads that value into every scene's use_sprite request -- explicit
    coverage for both settings values, on top of TestUseSpriteAlwaysRequested's
    settings-unavailable fallback case above."""

    async def _run_with_setting(self, value: bool):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[([_scene], 1), ([_scene], 1), ([], 1)]
        )
        rec_db.get_scene_fingerprint.return_value = None
        captured_specs = []

        def _batched(specs, is_stop_requested=None, before_scene=None, on_scene_start=None):
            captured_specs.extend(specs)
            return _batched_results([("1", _response(faces_after_filter=1))])(specs)

        with patch("settings.get_setting", return_value=value), \
                patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_batched):
            [p async for p in gen.generate_all(batch_size=100)]

        return captured_specs

    async def test_setting_enabled_requests_use_sprite(self):
        specs = await self._run_with_setting(True)
        assert specs[0].request.use_sprite is True

    async def test_setting_disabled_does_not_request_use_sprite(self):
        specs = await self._run_with_setting(False)
        assert specs[0].request.use_sprite is False


class TestGenerateAllBatchedLoop:
    async def test_skips_already_complete_scene_without_calling_orchestrator(self):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[
                ([_scene], 1),  # total-count fetch (limit=1) -- only `total` is used
                ([_scene], 1),  # actual batch fetch
                ([], 1),
            ]
        )
        rec_db.get_scene_fingerprint.return_value = {"fingerprint_status": "complete", "db_version": "2026.01.01"}

        with patch("scene_batch_orchestrator.identify_scenes_batched") as batched:
            progresses = [p async for p in gen.generate_all(batch_size=100)]

        batched.assert_not_called()
        final = progresses[-1]
        assert final.skipped == 1
        # A skipped scene is not progress: nothing needed doing, so 0 of 0.
        assert final.processed_scenes == 0
        assert final.total_scenes == 0

    async def test_processes_new_scene_via_orchestrator_and_marks_complete(self):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[
                ([_scene], 1),  # total-count fetch (limit=1) -- only `total` is used
                ([_scene], 1),  # actual batch fetch
                ([], 1),
            ]
        )

        with patch(
            "scene_batch_orchestrator.identify_scenes_batched",
            _batched_results([("1", _response(faces_after_filter=2))]),
        ):
            progresses = [p async for p in gen.generate_all(batch_size=100)]

        # on_scene_start's lazy pre-emptive error row (see
        # scene_batch_orchestrator.py) goes through create_scene_fingerprint
        # too, so at least one call happened for scene 1.
        assert rec_db.create_scene_fingerprint.call_count >= 1
        final = progresses[-1]
        assert final.successful == 1
        assert final.processed_scenes == 1

    async def test_zero_face_result_triggers_batched_retry_pass(self):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[
                ([_scene], 1),  # total-count fetch (limit=1) -- only `total` is used
                ([_scene], 1),  # actual batch fetch
                ([], 1),
            ]
        )

        call_log = []

        def _batched(specs, is_stop_requested=None, before_scene=None, on_scene_start=None):
            call_log.append([s.scene_id for s in specs])
            if len(call_log) == 1:
                return _batched_results([("1", _response(faces_after_filter=0))])(specs)
            return _batched_results([("1", _response(faces_after_filter=1))])(specs)

        with patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_batched):
            progresses = [p async for p in gen.generate_all(batch_size=100)]

        assert len(call_log) == 2  # first pass + retry pass
        assert call_log[0] == ["1"]
        assert call_log[1] == ["1"]
        final = progresses[-1]
        assert final.successful == 1
        assert final.processed_scenes == 1  # counted once, not twice

    async def test_retry_pass_failure_keeps_first_pass_zero_face_success(self):
        gen, stash, rec_db = _generator()
        _scene = {"id": "1", "title": "S1", "files": [{"duration": 10, "width": 1920, "height": 1080}]}
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[
                ([_scene], 1),  # total-count fetch (limit=1) -- only `total` is used
                ([_scene], 1),  # actual batch fetch
                ([], 1),
            ]
        )

        def _batched(specs, is_stop_requested=None, before_scene=None, on_scene_start=None):
            if specs and specs[0].scene_id == "1" and not hasattr(_batched, "_called"):
                _batched._called = True
                return _batched_results([("1", _response(faces_after_filter=0))])(specs)
            return _batched_results([("1", RuntimeError("retry failed"))])(specs)

        with patch("scene_batch_orchestrator.identify_scenes_batched", side_effect=_batched):
            progresses = [p async for p in gen.generate_all(batch_size=100)]

        final = progresses[-1]
        # Retry errored -- original zero-face success is kept, not counted as failed.
        assert final.successful == 1
        assert final.failed == 0


class TestProgressCoversOnlyScenesThatNeedWork:
    """Progress is "x of <scenes missing a fingerprint>", not a crawl through every scene in
    the library that only slows down once it reaches the real work."""

    @staticmethod
    def _scene(sid):
        return {"id": str(sid), "title": f"S{sid}", "files": [{"duration": 10, "width": 1920, "height": 1080}]}

    def _setup(self, scenes_in_stash, complete_ids, *, batches=None):
        gen, stash, rec_db = _generator()
        stash.get_all_scene_ids = AsyncMock(return_value=set(scenes_in_stash))
        scenes = [self._scene(s) for s in scenes_in_stash]
        # total-count fetch, then the pages, then an empty page
        stash.get_scenes_for_fingerprinting = AsyncMock(
            side_effect=[(scenes[:1], len(scenes))] + (batches or [(scenes, len(scenes))]) + [([], len(scenes))]
        )
        rec_db.get_scene_fingerprint.side_effect = lambda sid: (
            {"fingerprint_status": "complete", "db_version": "2026.01.01"} if sid in complete_ids else None
        )
        return gen, stash, rec_db

    async def test_total_is_only_the_missing_scenes(self):
        gen, _, _ = self._setup([1, 2, 3, 4], complete_ids={1, 2, 3})
        with patch("scene_batch_orchestrator.identify_scenes_batched",
                   _batched_results([("4", _response(faces_after_filter=1))])):
            progresses = [p async for p in gen.generate_all(batch_size=100)]

        final = progresses[-1]
        assert final.total_scenes == 1
        assert final.processed_scenes == 1
        assert final.successful == 1
        assert final.skipped == 3

    async def test_skipped_scenes_do_not_advance_progress(self):
        gen, _, _ = self._setup([1, 2, 3, 4], complete_ids={1, 2, 3})
        seen = []  # the generator yields one mutable progress object, so snapshot values as they arrive
        with patch("scene_batch_orchestrator.identify_scenes_batched",
                   _batched_results([("4", _response(faces_after_filter=1))])):
            async for p in gen.generate_all(batch_size=100):
                seen.append((p.processed_scenes, p.total_scenes))

        # progress only ever moves when the one missing scene finishes, and never exceeds the total
        assert {processed for processed, _ in seen} == {0, 1}
        assert all(processed <= total for processed, total in seen)

    async def test_stops_paging_once_the_last_needed_scene_is_done(self):
        # 4 scenes, batch_size=2: scene 1 (missing) is in the first page, so the second page
        # (which would only be scanned to skip 3 and 4) must never be fetched.
        page1 = ([self._scene(1), self._scene(2)], 4)
        page2 = ([self._scene(3), self._scene(4)], 4)
        gen, stash, _ = self._setup([1, 2, 3, 4], complete_ids={2, 3, 4}, batches=[page1, page2])
        with patch("scene_batch_orchestrator.identify_scenes_batched",
                   _batched_results([("1", _response(faces_after_filter=1))])):
            progresses = [p async for p in gen.generate_all(batch_size=2)]

        assert stash.get_scenes_for_fingerprinting.await_count == 2  # the count fetch + page 1 only
        assert progresses[-1].total_scenes == 1
        assert progresses[-1].processed_scenes == 1

    async def test_nothing_missing_completes_immediately_with_zero_total(self):
        gen, _, _ = self._setup([1, 2], complete_ids={1, 2})
        with patch("scene_batch_orchestrator.identify_scenes_batched") as batched:
            progresses = [p async for p in gen.generate_all(batch_size=100)]

        batched.assert_not_called()
        final = progresses[-1]
        assert final.total_scenes == 0
        assert final.processed_scenes == 0
        assert final.status.value == "completed"

    async def test_resumed_total_includes_scenes_done_before_the_resume(self):
        gen, _, _ = self._setup([1, 2, 3], complete_ids={1, 2})
        first = None
        with patch("scene_batch_orchestrator.identify_scenes_batched",
                   _batched_results([("3", _response(faces_after_filter=1))])):
            async for p in gen.generate_all(batch_size=100, start_offset=0, start_processed=40):
                if first is None:
                    first = (p.processed_scenes, p.total_scenes)
                final = p

        assert first == (40, 41)  # 40 done before the resume + 1 left
        assert final.processed_scenes == 41

    async def test_refresh_outdated_counts_outdated_scenes_as_needing_work(self):
        gen, _, rec_db = self._setup([1, 2], complete_ids={1, 2})
        rec_db.get_scene_fingerprint.side_effect = lambda sid: {
            "fingerprint_status": "complete", "db_version": "2025.01.01" if sid == 2 else "2026.01.01"}
        with patch("scene_batch_orchestrator.identify_scenes_batched",
                   _batched_results([("2", _response(faces_after_filter=1))])):
            progresses = [p async for p in gen.generate_all(batch_size=100, refresh_outdated=True)]
        assert progresses[-1].total_scenes == 1

        gen2, _, rec_db2 = self._setup([1, 2], complete_ids={1, 2})
        rec_db2.get_scene_fingerprint.side_effect = rec_db.get_scene_fingerprint.side_effect
        with patch("scene_batch_orchestrator.identify_scenes_batched") as batched:
            progresses = [p async for p in gen2.generate_all(batch_size=100, refresh_outdated=False)]
        batched.assert_not_called()
        assert progresses[-1].total_scenes == 0  # "Fingerprint Missing" never touches outdated ones

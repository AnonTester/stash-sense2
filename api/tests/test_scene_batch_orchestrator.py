"""Tests for scene_batch_orchestrator.py's identify_scenes_batched() --
the two-pass sequencing that keeps VAAPI decode and ROCm compute from
overlapping for >=4K scenes while leaving normal-res scenes on the
existing unbatched path.

Mocks identification_router's pieces (imported lazily inside
identify_scenes_batched to avoid a circular import) rather than exercising
the real pipeline -- this is a sequencing/bookkeeping test, not a face-
recognition test.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scene_batch_orchestrator import SceneBatchSpec, identify_scenes_batched


class _FakeResponse(SimpleNamespace):
    """Stand-in for SceneIdentifyResponse -- a distinct type from
    _FakePrepared so the orchestrator's isinstance() check (which
    distinguishes an already-final response from a still-needs-decode
    bundle) behaves correctly under mocking."""


class _FakePrepared(SimpleNamespace):
    """Stand-in for PreparedSceneIdentify -- see _FakeResponse."""


def _spec(scene_id: str, width=None, height=None) -> SceneBatchSpec:
    return SceneBatchSpec(scene_id=scene_id, width=width, height=height, request=SimpleNamespace(scene_id=scene_id))


def _response(scene_id: str):
    return _FakeResponse(scene_id=scene_id)


def _prepared(scene_id: str):
    return _FakePrepared(
        num_frames=60, match_config=object(), scene_id_int=int(scene_id),
        sprite_extra_results=[], sprite_timestamps={}, t_start=0.0,
    )


def _patch_router(**overrides):
    defaults = dict(
        _identify_scene_impl=AsyncMock(side_effect=lambda req: _response(req.scene_id)),
        _prepare_scene_identify=AsyncMock(side_effect=lambda req: _prepared(req.scene_id)),
        _extract_scene_frames=AsyncMock(side_effect=lambda req, num_frames, t_start: SimpleNamespace(scene_id=req.scene_id)),
        _identify_scene_compute=AsyncMock(
            side_effect=lambda req, bundle, num_frames, match_config, scene_id_int, sprite, t_start, sprite_timestamps=None: _response(req.scene_id)
        ),
    )
    defaults.update(overrides)

    import identification_router
    patchers = [patch.object(identification_router, name, value, create=True) for name, value in defaults.items()]
    # PreparedSceneIdentify/SceneIdentifyResponse are used for isinstance()
    # checks inside the orchestrator -- patch them to the same distinct
    # stand-in types _prepared()/_response() actually construct, so
    # isinstance(prepared_result, SceneIdentifyResponse) correctly
    # separates "already a final response" from "needs decode+compute".
    patchers.append(patch.object(identification_router, "SceneIdentifyResponse", _FakeResponse, create=True))
    patchers.append(patch.object(identification_router, "PreparedSceneIdentify", _FakePrepared, create=True))
    return patchers


class TestIdentifyScenesBatched:
    async def test_normal_res_scenes_use_unbatched_path(self):
        specs = [_spec("1", 1920, 1080), _spec("2", 1280, 720)]
        patchers = _patch_router()
        for p in patchers:
            p.start()
        try:
            results = {sid: r async for sid, r in identify_scenes_batched(specs)}
        finally:
            for p in patchers:
                p.stop()

        assert set(results) == {"1", "2"}
        for sid, r in results.items():
            assert r.scene_id == sid

    async def test_vaapi_scenes_decode_fully_before_any_compute(self):
        specs = [_spec("1", 3840, 2160), _spec("2", 4096, 2160)]
        call_order = []

        async def _extract(req, num_frames, t_start):
            call_order.append(("extract", req.scene_id))
            return SimpleNamespace(scene_id=req.scene_id)

        async def _compute(req, bundle, num_frames, match_config, scene_id_int, sprite, t_start, sprite_timestamps=None):
            call_order.append(("compute", req.scene_id))
            return _response(req.scene_id)

        patchers = _patch_router(
            _extract_scene_frames=AsyncMock(side_effect=_extract),
            _identify_scene_compute=AsyncMock(side_effect=_compute),
        )
        for p in patchers:
            p.start()
        try:
            results = {sid: r async for sid, r in identify_scenes_batched(specs)}
        finally:
            for p in patchers:
                p.stop()

        assert set(results) == {"1", "2"}
        # Both scenes fit in one VAAPI_BATCH_SIZE=2 batch -- every extract
        # must happen before any compute.
        extract_positions = [i for i, (kind, _) in enumerate(call_order) if kind == "extract"]
        compute_positions = [i for i, (kind, _) in enumerate(call_order) if kind == "compute"]
        assert max(extract_positions) < min(compute_positions)

    async def test_mixed_batch_processes_normal_scenes_first(self):
        specs = [_spec("1", 3840, 2160), _spec("2", 1920, 1080)]
        patchers = _patch_router()
        for p in patchers:
            p.start()
        try:
            order = [sid async for sid, _ in identify_scenes_batched(specs)]
        finally:
            for p in patchers:
                p.stop()

        assert order == ["2", "1"]  # normal-res scene 2 processed before vaapi scene 1

    async def test_one_scene_failure_does_not_abort_others(self):
        specs = [_spec("1", 1920, 1080), _spec("2", 1920, 1080)]

        async def _impl(req):
            if req.scene_id == "1":
                raise RuntimeError("boom")
            return _response(req.scene_id)

        patchers = _patch_router(_identify_scene_impl=AsyncMock(side_effect=_impl))
        for p in patchers:
            p.start()
        try:
            results = {sid: r async for sid, r in identify_scenes_batched(specs)}
        finally:
            for p in patchers:
                p.stop()

        assert isinstance(results["1"], RuntimeError)
        assert results["2"].scene_id == "2"

    async def test_stop_requested_halts_before_next_scene(self):
        specs = [_spec("1", 1920, 1080), _spec("2", 1920, 1080), _spec("3", 1920, 1080)]
        patchers = _patch_router()
        for p in patchers:
            p.start()
        try:
            seen = []
            async for sid, r in identify_scenes_batched(specs, is_stop_requested=lambda: len(seen) >= 1):
                seen.append(sid)
        finally:
            for p in patchers:
                p.stop()

        assert seen == ["1"]

    async def test_before_scene_hook_called_for_each_unit_of_work(self):
        specs = [_spec("1", 1920, 1080), _spec("2", 3840, 2160)]
        hook = AsyncMock()
        patchers = _patch_router()
        for p in patchers:
            p.start()
        try:
            results = {sid: r async for sid, r in identify_scenes_batched(specs, before_scene=hook)}
        finally:
            for p in patchers:
                p.stop()

        assert set(results) == {"1", "2"}
        # 1 normal-scene call + 2 vaapi calls (prepare doesn't get the hook,
        # but extract and compute each do) = 3.
        assert hook.call_count == 3

    async def test_on_scene_start_called_once_per_scene_before_work_begins(self):
        specs = [_spec("1", 1920, 1080), _spec("2", 3840, 2160)]
        started = []

        async def _on_start(scene_id):
            started.append(scene_id)

        patchers = _patch_router()
        for p in patchers:
            p.start()
        try:
            results = {sid: r async for sid, r in identify_scenes_batched(specs, on_scene_start=_on_start)}
        finally:
            for p in patchers:
                p.stop()

        assert set(results) == {"1", "2"}
        # Once for the normal scene, once for the vaapi scene's decode step
        # (not again for its compute step) -- see on_scene_start's docstring.
        assert started == ["1", "2"]

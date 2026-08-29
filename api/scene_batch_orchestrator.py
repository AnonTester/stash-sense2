"""Two-pass scene-identify orchestration for batch jobs (scene_face_match.py
and, where wired in, fingerprinting), keeping VAAPI GPU video-decode and
ROCm/HIP GPU compute from ever running at the same time.

Background: the sidecar's ROCm container was found to destabilize when
FFMPEG_HWACCEL=vaapi hardware video-decode (used for >=4K scenes specifically
to keep large-frame decode+resize off the CPU/RAM path -- see
identification_router.py's _scene_needs_vaapi docstring) runs concurrently
with ROCm/HIP inference on the same iGPU. Below-4K scenes already always
decode on CPU (per _scene_needs_vaapi), so an ordinary /identify/scene call
for one of those is unaffected either way and can run through the existing,
unbatched _identify_scene_impl path -- CPU decode and GPU compute share no
driver subsystem, so overlapping them (as already happens) is safe. This
module only changes sequencing for the >=4K minority: those scenes are
grouped into small batches, and for each batch every scene's frames are
fully decoded (_extract_scene_frames) before compute (_identify_scene_compute)
starts for any scene in that batch -- so VAAPI decode for one batch never
overlaps compute for another, and never overlaps its own batch's compute
either.

No disk cache -- extracted frames are already downsized by the time VAAPI
hands them back (the OOM risk this VAAPI path exists to avoid was
specifically the decode+resize step itself needing large buffers, not
holding a couple of videos' worth of already-extracted frames afterward), so
an in-memory hold for one small batch at a time is sufficient. Revisit with
an actual on-disk cache only if real memory pressure shows up in testing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, AsyncIterator, Callable, Optional, Union

logger = logging.getLogger(__name__)

# Deliberately small -- each scene in a batch holds its full extracted-frame
# set in memory for the whole batch's decode phase (see module docstring for
# why that's fine memory-wise), but a bigger batch also means a longer
# stretch where the GPU sits idle during decode before any compute runs.
VAAPI_BATCH_SIZE = 2


@dataclass
class SceneBatchSpec:
    """One scene to identify, with the resolution info needed to decide
    whether it needs the >=4K VAAPI-separated path. Callers already have
    width/height from their own scene-listing GraphQL query (alongside
    duration), so no extra per-scene fetch is needed here."""
    scene_id: str
    width: Optional[int]
    height: Optional[int]
    request: "SceneIdentifyRequest"


async def identify_scenes_batched(
    specs: list[SceneBatchSpec],
    is_stop_requested: Optional[Callable[[], bool]] = None,
    before_scene: Optional[Callable[[], Awaitable[None]]] = None,
    on_scene_start: Optional[Callable[[str], Awaitable[None]]] = None,
) -> AsyncIterator[tuple[str, Union["SceneIdentifyResponse", Exception]]]:
    """Identifies every scene in `specs`, yielding (scene_id, result) as
    each completes -- result is either a SceneIdentifyResponse or the
    Exception raised while processing that scene (never re-raised, so one
    scene's failure doesn't abort the rest of the batch, matching how
    callers already handled per-scene errors around a plain
    _identify_scene_impl loop).

    Processes normal-res scenes first, one at a time in submission order
    (unchanged behavior -- these never touch VAAPI). Then processes >=4K
    scenes afterward in VAAPI_BATCH_SIZE batches: decodes every scene in a
    batch, then computes every scene in that batch, then moves on.

    `before_scene`, if given, is awaited before each individual unit of
    work (once per normal scene; once per vaapi-batch scene's decode step
    and again before its compute step). Callers should pass
    require_db_available here -- a long batch job calling straight into
    _identify_scene_impl/_extract_scene_frames/_identify_scene_compute
    bypasses the FastAPI Depends() that normally re-touches the idle timer
    on every request, so without this a scan running longer than the idle-
    unload timeout gets the face recognition model evicted mid-run (see
    scene_face_match.py's own comment on this, confirmed live previously).

    `on_scene_start`, if given, is awaited exactly once per scene, right
    before that scene's actual work begins (before _identify_scene_impl for
    a normal scene; before decode for a vaapi-batch scene). This exists so a
    caller doing its own crash-safety bookkeeping (see
    fingerprint_generator.py's create_scene_fingerprint pre-write) can mark
    a scene as "in flight" lazily, one at a time, instead of bulk-marking
    every spec in `specs` before any of them actually start -- bulk-marking
    made every not-yet-reached scene in a batch look identically broken for
    the whole batch's duration (confirmed live: a 100-scene page produced a
    visibly regressing fingerprint-coverage count in the Settings UI).
    """
    from identification_router import (
        PreparedSceneIdentify,
        SceneIdentifyResponse,
        _extract_scene_frames,
        _identify_scene_compute,
        _identify_scene_impl,
        _prepare_scene_identify,
        _scene_needs_vaapi,
    )

    normal_specs = [s for s in specs if not _scene_needs_vaapi(s.width, s.height)]
    vaapi_specs = [s for s in specs if _scene_needs_vaapi(s.width, s.height)]

    for spec in normal_specs:
        if is_stop_requested and is_stop_requested():
            return
        try:
            if before_scene:
                await before_scene()
            if on_scene_start:
                await on_scene_start(spec.scene_id)
            response = await _identify_scene_impl(spec.request)
            yield spec.scene_id, response
        except Exception as e:
            yield spec.scene_id, e

    for batch_start in range(0, len(vaapi_specs), VAAPI_BATCH_SIZE):
        if is_stop_requested and is_stop_requested():
            return
        batch = vaapi_specs[batch_start:batch_start + VAAPI_BATCH_SIZE]

        # Settings/sprite/cache/skip resolution first (cheap, no video
        # decode involved -- sprites come from Stash's own pre-generated
        # JPEG, not this scene's video file) -- some scenes may already be
        # fully answered here (skip_frame_extraction or a cache hit) and
        # never need decode at all.
        prepared: list[tuple[SceneBatchSpec, PreparedSceneIdentify]] = []
        for spec in batch:
            try:
                result = await _prepare_scene_identify(spec.request)
            except Exception as e:
                yield spec.scene_id, e
                continue
            if isinstance(result, SceneIdentifyResponse):
                yield spec.scene_id, result
            else:
                prepared.append((spec, result))

        if not prepared:
            continue

        # Decode phase -- every scene in this batch, fully, before any
        # compute starts for any of them.
        bundles = {}
        decode_failed: set[str] = set()
        for spec, p in prepared:
            try:
                if before_scene:
                    await before_scene()
                if on_scene_start:
                    await on_scene_start(spec.scene_id)
                bundles[spec.scene_id] = await _extract_scene_frames(
                    spec.request, p.num_frames, p.t_start,
                )
            except Exception as e:
                decode_failed.add(spec.scene_id)
                yield spec.scene_id, e

        # Compute phase -- only now does anything touch ROCm/HIP inference
        # for this batch.
        for spec, p in prepared:
            if spec.scene_id in decode_failed:
                continue
            try:
                if before_scene:
                    await before_scene()
                response = await _identify_scene_compute(
                    spec.request, bundles[spec.scene_id], p.num_frames,
                    p.match_config, p.scene_id_int, p.sprite_extra_results, p.t_start,
                    sprite_timestamps=p.sprite_timestamps,
                )
                yield spec.scene_id, response
            except Exception as e:
                yield spec.scene_id, e

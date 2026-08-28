"""
Scene Fingerprint Generator

Generates face fingerprints for scenes in the Stash library.
Supports checkpointing for restart resilience and rate limiting.

This generator calls straight into identification_router.py's
_identify_scene_impl (bulk path: via scene_batch_orchestrator.py, so >=4K
scenes get VAAPI-decode/ROCm-compute batching -- see that module's
docstring), which handles frame extraction, face detection, matching, and
fingerprint persistence automatically. It used to loop back through this
same sidecar's own /identify/scene HTTP endpoint instead -- that went
through the same process either way, so the HTTP round trip was pure
overhead once bulk batching needed a hook into the decode/compute split.

Usage:
    generator = SceneFingerprintGenerator(
        stash_client=stash,
        rec_db=db,
        db_version="2026.01.30",
    )

    # Generate fingerprints for all scenes
    async for progress in generator.generate_all():
        print(f"Progress: {progress.processed}/{progress.total}")
"""

import asyncio
import httpx
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, AsyncIterator
from enum import Enum

import face_config

if TYPE_CHECKING:
    from stash_client_unified import StashClientUnified
    from recommendations_db import RecommendationsDB


logger = logging.getLogger(__name__)

# How long to keep retrying a Stash connectivity failure (container
# restart, brief network blip) before giving up on the whole run --
# see get_scenes_for_fingerprinting()'s own retry wrapper below. An
# overnight batch run shouldn't die over an outage this short.
STASH_RETRY_BUDGET_SECONDS = 300
STASH_RETRY_INTERVAL_SECONDS = 15


class StashUnavailableError(RuntimeError):
    """Raised when Stash stays unreachable past STASH_RETRY_BUDGET_SECONDS."""


class GeneratorStatus(str, Enum):
    """Status of the fingerprint generator."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class GeneratorProgress:
    """Progress information for fingerprint generation."""
    status: GeneratorStatus
    total_scenes: int
    processed_scenes: int
    successful: int
    failed: int
    skipped: int  # Already have current-version fingerprint
    current_scene_id: Optional[int] = None
    current_scene_title: Optional[str] = None
    error_message: Optional[str] = None
    # Cursor support: set at the end of each batch so the job can checkpoint.
    batch_completed: bool = False   # True only on the extra yield after a full batch
    current_offset: int = 0         # Pagination offset after this batch

    @property
    def progress_pct(self) -> float:
        if self.total_scenes == 0:
            return 0.0
        return (self.processed_scenes / self.total_scenes) * 100

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "total_scenes": self.total_scenes,
            "processed_scenes": self.processed_scenes,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "progress_pct": round(self.progress_pct, 1),
            "current_scene_id": self.current_scene_id,
            "current_scene_title": self.current_scene_title,
            "error_message": self.error_message,
        }


@dataclass
class FingerprintResult:
    """Result of fingerprinting a single scene."""
    scene_id: int
    success: bool
    fingerprint_id: Optional[int] = None
    performers_found: int = 0
    frames_analyzed: int = 0
    faces_found: int = 0
    retried_with_shifted_frames: bool = False
    error: Optional[str] = None


class SceneFingerprintGenerator:
    """
    Generates face fingerprints for scenes with checkpointing.

    Features:
    - Processes scenes one at a time
    - Calls /identify/scene which saves fingerprint automatically
    - Respects rate limiting
    - Can be stopped gracefully
    - Skips scenes with up-to-date fingerprints
    - Supports cursor-based resumption via start_offset / start_processed
    """

    def __init__(
        self,
        stash_client: "StashClientUnified",
        rec_db: "RecommendationsDB",
        db_version: str,
        num_frames: int = face_config.NUM_FRAMES,
        min_face_size: int = face_config.MIN_FACE_SIZE,
        max_distance: float = face_config.MAX_DISTANCE,
        start_offset_pct: float = face_config.START_OFFSET_PCT,
        end_offset_pct: float = face_config.END_OFFSET_PCT,
    ):
        self.stash = stash_client
        self.rec_db = rec_db
        self.db_version = db_version

        # Identification config
        self.num_frames = num_frames
        self.min_face_size = min_face_size
        self.max_distance = max_distance
        self.start_offset_pct = start_offset_pct
        self.end_offset_pct = end_offset_pct

        # State
        self._status = GeneratorStatus.IDLE
        self._stop_requested = False
        self._progress = GeneratorProgress(
            status=GeneratorStatus.IDLE,
            total_scenes=0,
            processed_scenes=0,
            successful=0,
            failed=0,
            skipped=0,
        )

    @property
    def status(self) -> GeneratorStatus:
        return self._status

    @property
    def progress(self) -> GeneratorProgress:
        return self._progress

    def request_stop(self):
        """Request graceful stop. Generator will finish current scene then stop."""
        if self._status == GeneratorStatus.RUNNING:
            self._stop_requested = True
            self._status = GeneratorStatus.STOPPING
            self._progress.status = GeneratorStatus.STOPPING
            logger.info("Stop requested, will finish current scene")

    async def _get_scenes_with_retry(self, limit: int, offset: int) -> tuple[list, int]:
        """Fetch a page of scenes from Stash, retrying through a brief
        connectivity outage (Stash container restart, transient network
        blip) instead of letting the whole overnight run die on the first
        hiccup. Gives up after STASH_RETRY_BUDGET_SECONDS and raises
        StashUnavailableError with a message that names Stash specifically,
        not a generic network error.
        """
        deadline = time.monotonic() + STASH_RETRY_BUDGET_SECONDS
        attempt = 0
        while True:
            try:
                return await self.stash.get_scenes_for_fingerprinting(limit=limit, offset=offset)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                attempt += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StashUnavailableError(
                        f"Could not connect to the Stash instance at {self.stash.base_url} "
                        f"after retrying for {STASH_RETRY_BUDGET_SECONDS // 60} minutes "
                        f"({attempt} attempts) -- last error: {e}"
                    ) from e
                wait = min(STASH_RETRY_INTERVAL_SECONDS, remaining)
                logger.warning(
                    "Stash unreachable while fetching scenes for fingerprinting "
                    "(attempt %d, %.0fs left before giving up): %s -- retrying in %.0fs",
                    attempt, remaining, e, wait,
                )
                await asyncio.sleep(wait)

    async def generate_all(
        self,
        refresh_outdated: bool = True,
        batch_size: int = 100,
        start_offset: int = 0,
        start_processed: int = 0,
        skip_errors: bool = False,
    ) -> AsyncIterator[GeneratorProgress]:
        """
        Generate fingerprints for all scenes that need them.

        Args:
            refresh_outdated: Also regenerate fingerprints from older DB versions
            batch_size: Number of scenes to query at a time
            start_offset: Resume pagination from this offset (0 = start fresh)
            start_processed: Cumulative processed count before this run (from cursor)
            skip_errors: When True, skip scenes whose last attempt recorded an error.
                Use when resuming from a crash cursor to avoid infinite retry loops.

        Yields:
            GeneratorProgress after each scene and once more at each batch boundary
            (batch_completed=True) so the job can save a resumption cursor.
        """
        self._status = GeneratorStatus.RUNNING
        self._progress.status = GeneratorStatus.RUNNING
        self._stop_requested = False
        self._progress.processed_scenes = start_processed
        self._progress.current_offset = start_offset

        resuming = start_offset > 0 or start_processed > 0

        try:
            # Get total scene count
            _, total = await self._get_scenes_with_retry(limit=1, offset=0)
            self._progress.total_scenes = total

            if resuming:
                logger.warning(
                    "Fingerprint generation resuming from offset %d "
                    "(previously processed: %d / %d, db_version=%s)",
                    start_offset, start_processed, total, self.db_version,
                )
            else:
                logger.warning(
                    "Fingerprint generation starting: %d total scenes, db_version=%s, "
                    "refresh_outdated=%s, batch_size=%d",
                    total, self.db_version, refresh_outdated, batch_size,
                )

            yield self._progress

            offset = start_offset
            while offset < total and not self._stop_requested:
                # Fetch batch of scenes
                scenes, _ = await self._get_scenes_with_retry(
                    limit=batch_size,
                    offset=offset,
                )

                if not scenes:
                    logger.debug(
                        "Batch at offset=%d returned no scenes (total=%d); stopping.",
                        offset, total,
                    )
                    break

                batch_successful = 0
                batch_failed = 0
                batch_skipped = 0

                logger.debug(
                    "Processing batch: offset=%d, scenes_in_batch=%d, "
                    "cumulative_processed=%d/%d (%.1f%%)",
                    offset, len(scenes),
                    self._progress.processed_scenes, total,
                    self._progress.progress_pct,
                )

                to_process: list[dict] = []
                scene_titles: dict[int, str] = {}
                # Per-scene use_sprite decision -- see _spec()'s docstring
                # below for why this can't just be a single job-wide setting.
                use_sprite_by_scene: dict[int, bool] = {}
                for scene in scenes:
                    if self._stop_requested:
                        break

                    scene_id = int(scene["id"])
                    scene_title = scene.get("title") or f"Scene {scene_id}"
                    scene_titles[scene_id] = scene_title

                    # Check if we need to process this scene
                    existing = self.rec_db.get_scene_fingerprint(scene_id)
                    if existing:
                        status = existing.get("fingerprint_status")
                        if status == "complete":
                            if not refresh_outdated or existing.get("db_version") == self.db_version:
                                logger.debug(
                                    "Scene %d (%s): skipped — already fingerprinted "
                                    "(db_version=%s, current=%s)",
                                    scene_id, scene_title,
                                    existing.get("db_version"), self.db_version,
                                )
                                self._progress.current_scene_id = scene_id
                                self._progress.current_scene_title = scene_title
                                self._progress.batch_completed = False
                                self._progress.skipped += 1
                                self._progress.processed_scenes += 1
                                batch_skipped += 1
                                yield self._progress
                                continue
                        elif status == "error" and skip_errors:
                            logger.debug(
                                "Scene %d (%s): skipped — previous attempt failed "
                                "(error record present, skip_errors=True)",
                                scene_id, scene_title,
                            )
                            self._progress.current_scene_id = scene_id
                            self._progress.current_scene_title = scene_title
                            self._progress.batch_completed = False
                            self._progress.skipped += 1
                            self._progress.processed_scenes += 1
                            batch_skipped += 1
                            yield self._progress
                            continue

                        logger.debug(
                            "Scene %d (%s): re-fingerprinting "
                            "(status=%s, db_version=%s → %s)",
                            scene_id, scene_title,
                            status,
                            existing.get("db_version"), self.db_version,
                        )
                    else:
                        logger.debug("Scene %d (%s): no existing fingerprint", scene_id, scene_title)

                    # Bulk runs default to no sprite processing (real added
                    # cost per scene -- see _build_identify_request) EXCEPT
                    # when refreshing a scene that already has sprite
                    # coverage: Face Recommendations may have separately
                    # paid that cost for this scene (its own on-demand
                    # top-up, see scene_face_match.py), and a routine bulk
                    # refresh must not silently discard it.
                    use_sprite_by_scene[scene_id] = bool(existing and existing.get("used_sprite"))
                    to_process.append(scene)

                if to_process and not self._stop_requested:
                    from identification_router import require_db_available
                    from scene_batch_orchestrator import SceneBatchSpec, identify_scenes_batched

                    def _spec(scene: dict, start_offset_pct: float, end_offset_pct: float) -> SceneBatchSpec:
                        sid = int(scene["id"])
                        file_info = (scene.get("files") or [{}])[0]
                        return SceneBatchSpec(
                            scene_id=str(sid),
                            width=file_info.get("width"),
                            height=file_info.get("height"),
                            request=self._build_identify_request(
                                sid, start_offset_pct, end_offset_pct,
                                use_sprite=use_sprite_by_scene.get(sid, False),
                            ),
                        )

                    # First pass, batched -- normal-res scenes through the
                    # existing unbatched path, >=4K scenes decoded-then-
                    # computed in small batches (see scene_batch_orchestrator.py).
                    first_pass_specs = [_spec(s, self.start_offset_pct, self.end_offset_pct) for s in to_process]
                    retry_pending: dict[int, FingerprintResult] = {}

                    async for scene_id_str, outcome in identify_scenes_batched(
                        first_pass_specs,
                        is_stop_requested=lambda: self._stop_requested,
                        before_scene=require_db_available,
                        on_scene_start=self._mark_scene_started,
                    ):
                        scene_id = int(scene_id_str)
                        result = self._response_to_result(scene_id, outcome)
                        if result.success and result.faces_found == 0:
                            # See _identify_scene's docstring -- shifted
                            # retry, batched together below for every scene
                            # that needs one, rather than one at a time.
                            retry_pending[scene_id] = result
                            continue
                        batch_successful, batch_failed = self._finalize_scene_result(
                            scene_id, scene_titles[scene_id], result, batch_successful, batch_failed,
                        )
                        yield self._progress

                    if retry_pending and not self._stop_requested:
                        scenes_by_id = {int(s["id"]): s for s in to_process}
                        shifted_start, shifted_end = self._shifted_offsets(self.start_offset_pct, self.end_offset_pct)
                        retry_specs = [
                            _spec(scenes_by_id[sid], shifted_start, shifted_end) for sid in retry_pending
                        ]
                        logger.debug(
                            "Retrying %d scene(s) with frames shifted (%.4f-%.4f) -> (%.4f-%.4f)",
                            len(retry_specs), self.start_offset_pct, self.end_offset_pct,
                            shifted_start, shifted_end,
                        )
                        async for scene_id_str, outcome in identify_scenes_batched(
                            retry_specs,
                            is_stop_requested=lambda: self._stop_requested,
                            before_scene=require_db_available,
                            on_scene_start=self._mark_scene_started,
                        ):
                            scene_id = int(scene_id_str)
                            retry_result = self._response_to_result(scene_id, outcome)
                            if retry_result.success:
                                retry_result.retried_with_shifted_frames = True
                                final = retry_result
                            else:
                                # Retry itself errored -- keep the good first
                                # result rather than discarding a confirmed
                                # "0 faces" for it.
                                final = retry_pending[scene_id]
                            del retry_pending[scene_id]
                            batch_successful, batch_failed = self._finalize_scene_result(
                                scene_id, scene_titles[scene_id], final, batch_successful, batch_failed,
                            )
                            yield self._progress

                    # Anything still pending here means a stop was requested
                    # mid-retry-pass -- finalize with the good first-pass
                    # result rather than leaving it silently uncounted.
                    for scene_id, result in retry_pending.items():
                        batch_successful, batch_failed = self._finalize_scene_result(
                            scene_id, scene_titles[scene_id], result, batch_successful, batch_failed,
                        )
                        yield self._progress

                offset += batch_size
                self._progress.current_offset = offset

                logger.debug(
                    "Batch complete: offset_now=%d, "
                    "batch_ok=%d skipped=%d failed=%d | "
                    "total processed=%d/%d (%.1f%%)",
                    offset,
                    batch_successful, batch_skipped, batch_failed,
                    self._progress.processed_scenes, total,
                    self._progress.progress_pct,
                )

                # Yield once more with batch_completed=True so the job can
                # save a resumption cursor without writing per-scene.
                self._progress.batch_completed = True
                self._progress.current_scene_id = None
                self._progress.current_scene_title = None
                yield self._progress
                self._progress.batch_completed = False

            if self._stop_requested:
                self._status = GeneratorStatus.PAUSED
                self._progress.status = GeneratorStatus.PAUSED
                logger.warning(
                    "Fingerprint generation paused at offset %d "
                    "(%d/%d processed, %d ok, %d skipped, %d failed)",
                    offset,
                    self._progress.processed_scenes, total,
                    self._progress.successful,
                    self._progress.skipped,
                    self._progress.failed,
                )
            else:
                self._status = GeneratorStatus.COMPLETED
                self._progress.status = GeneratorStatus.COMPLETED
                logger.warning(
                    "Fingerprint generation complete: %d/%d processed "
                    "(%d successful, %d skipped, %d failed), db_version=%s",
                    self._progress.processed_scenes, total,
                    self._progress.successful,
                    self._progress.skipped,
                    self._progress.failed,
                    self.db_version,
                )

        except Exception as e:
            self._status = GeneratorStatus.ERROR
            self._progress.status = GeneratorStatus.ERROR
            self._progress.error_message = str(e)
            logger.error(
                "Fingerprint generation error at offset ~%d (%d processed so far): %s",
                self._progress.current_offset,
                self._progress.processed_scenes,
                e,
                exc_info=True,
            )
            raise

        finally:
            self._progress.current_scene_id = None
            self._progress.current_scene_title = None
            self._progress.batch_completed = False
            yield self._progress

    async def generate_for_scene(self, scene_id: int) -> FingerprintResult:
        """Generate fingerprint for a single scene."""
        return await self._identify_scene(scene_id)

    def _build_identify_request(
        self, scene_id: int, start_offset_pct: float, end_offset_pct: float,
        use_sprite: bool = False,
    ) -> "SceneIdentifyRequest":
        from identification_router import SceneIdentifyRequest
        return SceneIdentifyRequest(
            scene_id=str(scene_id),
            num_frames=self.num_frames,
            min_face_size=self.min_face_size,
            max_distance=self.max_distance,
            start_offset_pct=start_offset_pct,
            end_offset_pct=end_offset_pct,
            matching_mode="hybrid",
            # top_k standardized to match the live Identify button / Face
            # Recommendations -- this job's stored output is now the
            # canonical source both read from instead of re-running their
            # own identify pass. use_sprite stays off by default here (real
            # added per-scene cost, and this job runs over the whole
            # library including scenes nobody needs a new recommendation
            # for) -- callers pass use_sprite=True only to preserve
            # existing sprite coverage on a refresh; see generate_all()'s
            # use_sprite_by_scene.
            top_k=5,
            use_sprite=use_sprite,
        )

    async def _mark_scene_started(self, scene_id_str: str) -> None:
        """Marks one scene as in-flight, right before its actual work
        begins -- see scene_batch_orchestrator.py's on_scene_start docstring.
        If the sidecar is SIGKILL'd right after this, this row correctly
        shows the scene as never-completed rather than never-attempted; a
        successful run overwrites it with "complete" (INSERT OR REPLACE, see
        _finalize_scene_result/_response_to_result)."""
        self.rec_db.create_scene_fingerprint(
            stash_scene_id=int(scene_id_str), total_faces=0, frames_analyzed=0,
            fingerprint_status="error", db_version=self.db_version,
        )

    def _response_to_result(
        self, scene_id: int, outcome: "SceneIdentifyResponse | Exception",
    ) -> FingerprintResult:
        """Maps a SceneIdentifyResponse (or the Exception raised while
        producing one) into this generator's own FingerprintResult,
        including writing the scene_fingerprints error row on failure --
        shared by both the single-scene path (_call_identify) and the
        batched bulk path (generate_all(), via
        scene_batch_orchestrator.py)."""
        from fastapi import HTTPException

        if isinstance(outcome, Exception):
            detail = outcome.detail if isinstance(outcome, HTTPException) else str(outcome)
            logger.warning(f"Scene {scene_id} identification failed: {detail}")
            self.rec_db.create_scene_fingerprint(
                stash_scene_id=scene_id, total_faces=0, frames_analyzed=0,
                fingerprint_status="error", db_version=self.db_version,
            )
            return FingerprintResult(scene_id=scene_id, success=False, error=str(detail))

        response = outcome
        performers_found = sum(1 for p in response.persons if p.best_match)
        faces_found = response.faces_after_filter

        if not response.fingerprint_saved and performers_found > 0:
            # Identification succeeded but save failed
            error_msg = response.fingerprint_error or "Fingerprint save failed"
            logger.warning(f"Scene {scene_id} fingerprint save failed: {error_msg}")
            return FingerprintResult(
                scene_id=scene_id,
                success=False,
                error=f"Save failed: {error_msg}",
                performers_found=performers_found,
                frames_analyzed=response.frames_analyzed,
                faces_found=faces_found,
            )

        return FingerprintResult(
            scene_id=scene_id,
            success=True,
            performers_found=performers_found,
            frames_analyzed=response.frames_analyzed,
            faces_found=faces_found,
        )

    def _finalize_scene_result(
        self, scene_id: int, scene_title: str, result: FingerprintResult,
        batch_successful: int, batch_failed: int,
    ) -> tuple[int, int]:
        """Records one scene's final outcome into self._progress (for the
        caller to yield) and the batch-local tallies generate_all() logs at
        the end of each Stash-fetched batch. Called exactly once per scene
        that reached _identify_scene_impl (skipped scenes are counted
        inline in generate_all()'s own skip-scan loop, not here)."""
        self._progress.current_scene_id = scene_id
        self._progress.current_scene_title = scene_title
        self._progress.batch_completed = False
        self._progress.processed_scenes += 1
        if result.success:
            self._progress.successful += 1
            batch_successful += 1
            logger.debug(
                "Scene %d (%s): fingerprinted — performers_found=%d, faces_found=%d, "
                "frames=%d%s",
                scene_id, scene_title,
                result.performers_found, result.faces_found, result.frames_analyzed,
                " (retried with shifted frames)" if result.retried_with_shifted_frames else "",
            )
        else:
            self._progress.failed += 1
            batch_failed += 1
            logger.debug(
                "Scene %d (%s): fingerprint failed — %s",
                scene_id, scene_title, result.error,
            )
        return batch_successful, batch_failed

    def _shifted_offsets(self, start_offset_pct: float, end_offset_pct: float) -> tuple[float, float]:
        """Shifts a sampling window by half a sampling interval -- see
        _identify_scene's docstring for why."""
        interval_pct = (end_offset_pct - start_offset_pct) / max(1, self.num_frames - 1)
        half_shift = interval_pct / 2
        return min(1.0, start_offset_pct + half_shift), min(1.0, end_offset_pct + half_shift)

    async def _identify_scene(self, scene_id: int) -> FingerprintResult:
        """Identify a single scene, retrying once with a shifted sampling
        grid if the first pass finds no usable faces at all.

        Frame sampling is deterministic (uniform timestamps derived purely
        from num_frames/start_offset_pct/end_offset_pct), so a scene whose
        only faces fall between sampled instants would report "no faces"
        forever no matter how many times it's reprocessed with the same
        settings. The retry keeps num_frames the same but shifts the whole
        sampling grid by half a sampling interval, landing on entirely
        different timestamps -- interleaved with the first pass's. If that
        also finds nothing, the scene is accepted as having no identifiable
        faces (matches success, just faces_found=0).
        """
        result = await self._call_identify(scene_id, self.start_offset_pct, self.end_offset_pct)

        if result.success and result.faces_found == 0:
            shifted_start, shifted_end = self._shifted_offsets(self.start_offset_pct, self.end_offset_pct)
            logger.debug(
                "Scene %d: no faces in first pass, retrying with frames shifted "
                "(%.4f-%.4f) -> (%.4f-%.4f)",
                scene_id, self.start_offset_pct, self.end_offset_pct, shifted_start, shifted_end,
            )
            retry_result = await self._call_identify(scene_id, shifted_start, shifted_end)
            if retry_result.success:
                # Whatever this pass found (or didn't) is now the final,
                # authoritative result -- give up after one retry either way.
                retry_result.retried_with_shifted_frames = True
                return retry_result
            # Retry itself errored (timeout, etc.) -- keep the good first
            # result rather than discarding a confirmed "0 faces" for it.

        return result

    async def _call_identify(
        self, scene_id: int, start_offset_pct: float, end_offset_pct: float,
    ) -> FingerprintResult:
        """Single, unbatched identify pass for one scene, calling straight
        into _identify_scene_impl instead of looping back through this
        sidecar's own /identify/scene HTTP endpoint -- this generator always
        runs inside the same process as that endpoint, so the HTTP round
        trip was pure overhead. A single scene doesn't need
        scene_batch_orchestrator.py's VAAPI/compute batching (its own
        decode-then-compute sequencing already keeps the two from
        overlapping for just one scene) -- see generate_all() for the
        batched bulk path this doesn't cover.
        See _identify_scene for the retry wrapper around this."""
        from identification_router import _identify_scene_impl, require_db_available

        request = self._build_identify_request(scene_id, start_offset_pct, end_offset_pct)
        try:
            # See scene_batch_orchestrator.py's before_scene docstring --
            # calling _identify_scene_impl directly bypasses the FastAPI
            # Depends() that normally re-touches the idle-unload timer on
            # every request, so this must be re-checked on every call.
            await require_db_available()
            response = await _identify_scene_impl(request)
        except Exception as e:
            return self._response_to_result(scene_id, e)

        return self._response_to_result(scene_id, response)

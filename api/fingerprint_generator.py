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
    total_scenes: int      # scenes to fingerprint in this run (missing/outdated) -- NOT every scene in Stash
    processed_scenes: int  # of those, how many were attempted (successful + failed); skipped scenes don't count
    successful: int
    failed: int
    skipped: int  # Passed over: already have a current-version fingerprint (not part of the progress)
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
    - Skips scenes with up-to-date fingerprints; progress (and so its ETA) covers only the
      scenes that actually need work, decided up front, not every scene in the library
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
        # Scene ids still to fingerprint in the current generate_all() run; drained as scenes
        # finish so the paging loop can stop as soon as the last needed scene is done.
        self._remaining_ids: set[int] = set()
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
        """One page of scenes from Stash, retried through a brief outage (see _with_stash_retry)."""
        return await self._with_stash_retry(
            lambda: self.stash.get_scenes_for_fingerprinting(limit=limit, offset=offset)
        )

    async def _with_stash_retry(self, call):
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
                return await call()
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

    def _needs_work(self, scene_id: int, refresh_outdated: bool, skip_errors: bool) -> bool:
        """Whether generate_all() should fingerprint this scene in this run."""
        existing = self.rec_db.get_scene_fingerprint(scene_id)
        if not existing:
            return True
        status = existing.get("fingerprint_status")
        if status == "complete":
            return bool(refresh_outdated) and existing.get("db_version") != self.db_version
        if status == "error" and skip_errors:
            return False
        return True

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

        # Read once per run (a resumed job re-enters generate_all(), so this
        # is also re-read on every resume) rather than per-scene/per-batch --
        # see the "Use Sprite Tiles For Detection" setting's own description
        # for why this granularity is the intended behavior: pausing a run,
        # flipping the setting, and resuming picks up the new value; toggling
        # it mid-run without pausing does not.
        try:
            from settings import get_setting
            use_sprite = bool(get_setting("sprite_detection_enabled"))
        except (RuntimeError, KeyError):
            use_sprite = True
        logger.warning("Fingerprint generation: sprite_detection_enabled=%s", use_sprite)

        # Same once-per-run/re-read-on-resume granularity as use_sprite above.
        try:
            from settings import get_setting
            scoped_refresh = bool(get_setting("refresh_outdated_scoped"))
        except (RuntimeError, KeyError):
            scoped_refresh = True

        try:
            # Total scene count (only drives the paging below)
            _, total = await self._get_scenes_with_retry(limit=1, offset=0)

            all_scene_ids = await self._with_stash_retry(self.stash.get_all_scene_ids)

            # Delta-scoped Refresh Outdated (see delta_scope.py): among the
            # scenes that are db_version-outdated but already have a
            # COMPLETE fingerprint, find the ones a recent delta could not
            # possibly have affected and mark them current in place --
            # skipped entirely below (never enters pending_ids/_needs_work
            # for them), instead of every outdated scene paying for a real
            # rematch just because the version number moved. Only applies
            # to refresh_outdated=True runs -- Fingerprint Missing
            # (refresh_outdated=False) never considers outdated-but-complete
            # scenes in the first place, so there's nothing to scope there.
            bumped_ids: set[int] = set()
            if refresh_outdated and scoped_refresh:
                all_scene_ids_set = set(all_scene_ids)
                outdated_complete = [
                    fp for fp in self.rec_db.get_all_scene_fingerprints(status="complete")
                    if fp["stash_scene_id"] in all_scene_ids_set and fp.get("db_version") != self.db_version
                ]
                if outdated_complete:
                    from delta_scope import scenes_needing_rematch
                    must_rematch, safe_to_bump = scenes_needing_rematch(self.rec_db, outdated_complete)
                    for sid in safe_to_bump:
                        self.rec_db.bump_fingerprint_db_version(sid, self.db_version)
                    bumped_ids = safe_to_bump
                    logger.warning(
                        "Refresh Outdated: delta-scoped %d/%d outdated scene(s) marked current without "
                        "a rematch (%d still need one), db_version=%s",
                        len(safe_to_bump), len(outdated_complete), len(must_rematch), self.db_version,
                    )

            # Decide up front which scenes actually need fingerprinting, so progress and its ETA
            # are measured against only those (25 missing -> "x of 25") instead of crawling every
            # scene in the library and only slowing down once it reaches the real work. IDs only,
            # one request; skipped scenes are then passed over without counting as progress.
            pending_ids = {
                sid for sid in all_scene_ids
                if sid not in bumped_ids and self._needs_work(sid, refresh_outdated, skip_errors)
            }
            self._remaining_ids = set(pending_ids)
            # Scenes that need nothing this run (already fingerprinted with the current version, or
            # errored earlier when resuming): tallied exactly, up front, without paging through them.
            self._progress.skipped = len(all_scene_ids) - len(pending_ids)
            # On a resume, start_processed scenes were already attempted before this run.
            self._progress.total_scenes = start_processed + len(pending_ids)

            if resuming:
                logger.warning(
                    "Fingerprint generation resuming from offset %d "
                    "(previously processed: %d, still to do: %d, db_version=%s)",
                    start_offset, start_processed, len(pending_ids), self.db_version,
                )
            else:
                logger.warning(
                    "Fingerprint generation starting: %d of %d scenes need fingerprinting, "
                    "db_version=%s, refresh_outdated=%s, batch_size=%d",
                    len(pending_ids), total, self.db_version, refresh_outdated, batch_size,
                )

            yield self._progress

            offset = start_offset
            # Stops early once every scene that needed work has been done -- the remaining pages
            # would only be scanned to skip them.
            while offset < total and not self._stop_requested and self._remaining_ids:
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
                    self._progress.processed_scenes, self._progress.total_scenes,
                    self._progress.progress_pct,
                )

                to_process: list[dict] = []
                scene_titles: dict[int, str] = {}
                for scene in scenes:
                    if self._stop_requested:
                        break

                    scene_id = int(scene["id"])
                    scene_title = scene.get("title") or f"Scene {scene_id}"
                    scene_titles[scene_id] = scene_title

                    # Not in the pending set: already has what this run would produce (or, when resuming,
                    # a previous attempt errored). Passed over; not part of the progress.
                    if scene_id not in pending_ids:
                        logger.debug("Scene %d (%s): skipped -- nothing to do this run", scene_id, scene_title)
                        batch_skipped += 1
                        continue

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
                            # Sprite tiles cache their own embeddings the
                            # same way video frames do (scene_face_embeddings,
                            # is_sprite=1 -- see identification_router.py's
                            # scene_sprite_cache_status), so requesting them
                            # here only ever pays real detection cost once
                            # per scene, ever: a scene that already has
                            # sprite coverage just reuses that cache, and one
                            # that doesn't gets it computed and cached now
                            # instead of leaving that for a later on-demand
                            # top-up in scene_face_match.py. Gated on the
                            # "Use Sprite Tiles For Detection" setting
                            # (use_sprite local var, read once above) --
                            # when off, no sprite work is requested and
                            # nothing gets generated for scenes that don't
                            # already have it cached.
                            request=self._build_identify_request(
                                sid, start_offset_pct, end_offset_pct, use_sprite=use_sprite,
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
                    self._progress.processed_scenes, self._progress.total_scenes,
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
                    self._progress.processed_scenes, self._progress.total_scenes,
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
                    self._progress.processed_scenes, self._progress.total_scenes,
                    self._progress.successful,
                    self._progress.skipped,
                    self._progress.failed,
                    self.db_version,
                )
                # A clean (non-paused) refresh_outdated pass with scoping
                # on means every currently-complete scene is now either
                # freshly rematched or provably safe-to-bump at
                # self.db_version -- nothing depends on dirty tracking
                # older than that anymore, so it's safe to drain (see
                # delta_scope.finalize_dirty_state's own docstring). Never
                # drains when scoping was off this run (bumped_ids/the
                # scoping check above never ran, so nothing was actually
                # consumed) or on Fingerprint Missing runs (refresh_outdated
                # =False never looks at dirty state at all).
                if refresh_outdated and scoped_refresh:
                    from delta_scope import finalize_dirty_state
                    finalize_dirty_state(self.rec_db, self.db_version)

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
            # own identify pass. use_sprite defaults to off here (only
            # generate_all()'s bulk scan currently passes True) since sprite
            # detection is a real cost the first time it runs for a scene --
            # cheap ever after, once cached (see identification_router.py's
            # scene_sprite_cache_status).
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
        that reached _identify_scene_impl (skipped scenes are tallied
        inline in generate_all()'s own skip-scan loop and are not progress)."""
        self._progress.current_scene_id = scene_id
        self._progress.current_scene_title = scene_title
        self._progress.batch_completed = False
        self._progress.processed_scenes += 1
        self._remaining_ids.discard(scene_id)
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

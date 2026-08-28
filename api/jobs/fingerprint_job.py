"""Fingerprint generation as a queue job."""
from __future__ import annotations

import json
import logging
from typing import Optional

from base_job import BaseJob, JobContext
from fingerprint_generator import GeneratorStatus, SceneFingerprintGenerator
from recommendations_router import get_db_version, get_rec_db, get_stash_client

logger = logging.getLogger(__name__)


class FingerprintGenerationJob(BaseJob):
    """Wraps SceneFingerprintGenerator as a queue-managed job.

    Backs two distinct, separately-tracked queue job types that used to be
    one job type with the scope hidden inside its cursor -- that made the
    Operations tab's single "Face Identification" Quick Action ambiguous
    (which scope would it run?), and made the Settings tab's two buttons
    both show "in progress" off one shared "is *a* fingerprint_generation
    job running" flag even when only one of the two scopes was actually
    active. See queue_manager.py's _create_job_instance -- the two type_ids
    ("fingerprint_generation" / "fingerprint_refresh_outdated") each
    construct this class with `refresh_outdated` fixed at dispatch time, so
    scope is now a property of *which job type ran*, not something buried
    in a cursor.

    Cursor format (JSON string): {"offset": <int>, "processed": <int>} --
    purely a resume checkpoint now, saved after each batch of 100 scenes.
    """

    def __init__(self, refresh_outdated: bool):
        self.refresh_outdated = refresh_outdated

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return None

        db_version = get_db_version()
        if db_version is None:
            raise RuntimeError("No face recognition database loaded; cannot generate fingerprints")

        # Parse resumption cursor
        start_offset = 0
        start_processed = 0
        if cursor:
            try:
                c = json.loads(cursor)
                start_offset = int(c.get("offset", 0))
                start_processed = int(c.get("processed", 0))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning(
                    "Fingerprint job: could not parse cursor %r — starting from offset 0",
                    cursor,
                )

        resuming = start_offset > 0 or start_processed > 0

        logger.warning(
            "Fingerprint generation job starting (job_id=%d, db_version=%s, "
            "start_offset=%d, start_processed=%d, refresh_outdated=%s)",
            context.job_id, db_version, start_offset, start_processed, self.refresh_outdated,
        )

        stash = get_stash_client()
        db = get_rec_db()
        generator = SceneFingerprintGenerator(
            stash_client=stash,
            rec_db=db,
            db_version=db_version,
        )

        # When resuming from a crash cursor, skip scenes that previously errored out
        # so a permanently-broken scene cannot block the job indefinitely.
        async for progress in generator.generate_all(
            start_offset=start_offset,
            start_processed=start_processed,
            skip_errors=resuming,
            refresh_outdated=self.refresh_outdated,
        ):
            if progress.batch_completed:
                # Persist cursor after each full batch for crash recovery
                new_cursor = json.dumps({
                    "offset": progress.current_offset,
                    "processed": progress.processed_scenes,
                })
                await context.checkpoint(
                    cursor=new_cursor,
                    items_processed=progress.processed_scenes,
                )
                logger.debug(
                    "Fingerprint job checkpoint: offset=%d processed=%d/%d",
                    progress.current_offset,
                    progress.processed_scenes,
                    progress.total_scenes,
                )
            else:
                await context.report_progress(
                    progress.processed_scenes,
                    progress.total_scenes,
                )

            if context.is_stop_requested():
                generator.request_stop()
                break

        final = generator.progress
        logger.warning(
            "Fingerprint generation job finished (job_id=%d, status=%s): "
            "%d processed, %d successful, %d skipped, %d failed",
            context.job_id, final.status.value,
            final.processed_scenes,
            final.successful,
            final.skipped,
            final.failed,
        )

        if final.status in (GeneratorStatus.PAUSED, GeneratorStatus.STOPPING):
            # Job was stopped before finishing all scenes. Return a cursor so the
            # queue re-queues it rather than marking it completed.
            resume_cursor = json.dumps({
                "offset": final.current_offset,
                "processed": final.processed_scenes,
            })
            logger.warning(
                "Fingerprint job interrupted (job_id=%d): "
                "re-queuing from offset=%d (processed=%d/%d, failed=%d)",
                context.job_id,
                final.current_offset, final.processed_scenes, final.total_scenes,
                final.failed,
            )
            return resume_cursor

        return None

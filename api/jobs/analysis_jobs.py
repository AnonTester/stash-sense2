"""Wraps existing analyzers as BaseJob subclasses."""
from __future__ import annotations

import logging
from typing import Optional

from base_job import BaseJob, JobContext
from recommendations_router import ANALYZERS, get_rec_db, get_stash_client

logger = logging.getLogger(__name__)

FULL_RUN_CURSOR = "__full__"


class AnalysisJob(BaseJob):
    """Generic wrapper that runs any registered analyzer as a queue job."""

    def __init__(self, type_id: str):
        self._type_id = type_id

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return cursor

        db = get_rec_db()
        stash = get_stash_client()
        analyzer_class = ANALYZERS.get(self._type_id)
        if not analyzer_class:
            raise ValueError(f"Unknown analyzer type: {self._type_id}")

        # Create a real analysis_runs entry (FK target for duplicate_candidates.run_id)
        run_id = db.start_analysis_run(self._type_id)
        logger.warning("Starting analysis job %s (run_id=%d, job_id=%d)", self._type_id, run_id, context.job_id)

        # Bridge progress from analyzer → job queue for frontend polling
        def progress_callback(items_processed: int, items_total: Optional[int]) -> None:
            try:
                # Write queue progress synchronously for deterministic UI updates.
                # Fire-and-forget async tasks can drop/interleave updates under load.
                db.update_job_progress(
                    context.job_id,
                    items_processed=items_processed,
                    items_total=items_total,
                )
            except Exception:
                logger.warning(
                    "Failed to persist queue progress for job %s (%s)",
                    context.job_id,
                    self._type_id,
                    exc_info=True,
                )

        def label_callback(label: str) -> None:
            try:
                db.update_job_progress(context.job_id, label=label)
            except Exception:
                pass

        analyzer = analyzer_class(stash, db, run_id=run_id)
        analyzer._job_progress_callback = progress_callback
        analyzer._job_label_callback = label_callback

        # Wire stop signal from job context to the analyzer LIVE -- context's
        # flag can flip True at any point *during* analyzer.run() (the user
        # clicking Stop mid-run), not just before it starts, so a one-time
        # snapshot check here (as this used to be) would only ever see
        # False, since a stop is never requested before the job has even
        # started. Delegating means every self.is_stop_requested() call the
        # analyzer makes reflects the live context state.
        analyzer.is_stop_requested = context.is_stop_requested

        force_full = cursor == FULL_RUN_CURSOR
        incremental = not force_full

        try:
            result = await analyzer.run(incremental=incremental)

            db.complete_analysis_run(run_id, result.recommendations_created)
            final_total = result.items_processed
            if analyzer._items_total is not None and result.items_processed <= analyzer._items_total:
                final_total = analyzer._items_total
            await context.report_progress(result.items_processed, final_total)
            summary = (
                f"{result.items_processed:,} item(s) checked, "
                f"{result.recommendations_created:,} recommendation(s) added"
            )
            if result.recommendations_updated:
                summary += f", {result.recommendations_updated:,} refreshed"
            context.set_result_summary(summary)
            logger.warning(
                "Analysis job %s completed (run_id=%d): %d processed, %d recommendations",
                self._type_id, run_id, result.items_processed, result.recommendations_created,
            )
        except Exception:
            db.fail_analysis_run(run_id, "Analysis job failed with exception")
            logger.warning("Analysis job %s failed (run_id=%d)", self._type_id, run_id, exc_info=True)
            raise

        return None

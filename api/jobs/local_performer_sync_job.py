"""Local performer index sync as a queue job.

Builds/updates a usearch index from this Stash instance's own performer
cover images (see local_performer_index.py), diffed against the index's
own persisted JSON mapping -- no separate SQL tracking table needed, the
mapping file already records each performer's last-synced image content
hash. Every performer with a real (non-placeholder) cover gets its image
fetched and hash-compared each run; only ones whose bytes actually
changed pay for face detection + embedding. A URL-based shortcut to skip
the fetch too was tried and reverted -- Stash's image URL cache-busting
param tracks updated_at, which changes on *any* field edit, not just an
image swap, so it couldn't tell "nothing changed" from "renamed."

A performer with no custom cover image still returns an image_path, but
Stash marks it with a "default=true" query param -- that's how "no image
yet" is detected, without needing to fetch and fail to decode a
placeholder icon.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from base_job import BaseJob, JobContext
from config import DatabaseConfig
from embeddings import FaceEmbeddingGenerator
from local_performer_index import LocalPerformerIndex, sync_one_performer
from recommendations_router import get_stash_client

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Save the index to disk (and let a resumed run pick up here) after this
# many performers processed in a batch, mirroring fingerprint_job.py's
# per-100-scenes checkpoint cadence.
CHECKPOINT_BATCH_SIZE = 50


class LocalPerformerSyncJob(BaseJob):
    """Diffs current Stash performers against the local index, embedding
    new/changed performers and removing ones no longer present.

    Cursor format (JSON string): {"position": <int>} -- index into the
    stable-sorted (by id) performer list to resume from. The index is
    saved to disk at the same checkpoints, so a crash never loses already-
    embedded work, and a resumed run naturally skips anything already
    up to date (same image fingerprint) even without the cursor.
    """

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return None

        start_position = 0
        if cursor:
            try:
                start_position = int(json.loads(cursor).get("position", 0))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning(
                    "Local performer sync: could not parse cursor %r — starting from 0", cursor,
                )

        stash = get_stash_client()
        db_config = DatabaseConfig(data_dir=Path(DATA_DIR))
        index = LocalPerformerIndex(
            db_config.local_embedding_index_path,
            db_config.local_faces_json_path,
        )
        generator = FaceEmbeddingGenerator()

        performers = await stash.get_all_performers()
        performers.sort(key=lambda p: int(p["id"]))  # stable order across runs
        total = len(performers)

        logger.warning(
            "Local performer sync starting (job_id=%d, %d performers, start_position=%d, "
            "index currently has %d)",
            context.job_id, total, start_position, len(index),
        )

        current_ids = {int(p["id"]) for p in performers}
        removed = 0
        # Remove performers no longer in Stash (only on a fresh run -- a
        # resumed run already did this pass before it was interrupted).
        if start_position == 0:
            for stale_id_str in list(index.mapping.keys()):
                if int(stale_id_str) not in current_ids:
                    index.remove(int(stale_id_str))
                    removed += 1
            if removed:
                logger.warning("Local performer sync: removed %d stale entries", removed)

        added = 0
        updated = 0
        skipped_no_image = 0
        errored = 0

        for i in range(start_position, total):
            if context.is_stop_requested():
                break

            performer = performers[i]
            performer_id = int(performer["id"])

            # Cheap pre-check against the already-fetched bulk list: only
            # skips performers with no image_path at all (rare -- Stash
            # normally returns a placeholder URL even with no custom
            # cover). Everything else goes through sync_one_performer,
            # which fetches and content-hashes the image to decide whether
            # it actually needs re-embedding.
            if not performer.get("image_path"):
                await context.report_progress(i + 1, total)
                continue

            try:
                status = await sync_one_performer(stash, generator, index, performer_id, "update")
                if status == "added":
                    added += 1
                elif status == "updated":
                    updated += 1
                elif status == "skipped_no_image":
                    skipped_no_image += 1
            except Exception as e:
                logger.warning("Local performer sync: failed on performer %d (%s): %s",
                                performer_id, performer.get("name"), e)
                errored += 1

            await context.report_progress(i + 1, total)

            if (i + 1) % CHECKPOINT_BATCH_SIZE == 0:
                index.save()
                await context.checkpoint(
                    cursor=json.dumps({"position": i + 1}),
                    items_processed=i + 1,
                )
                logger.debug("Local performer sync checkpoint: position=%d/%d", i + 1, total)

        index.save()

        stopped_early = context.is_stop_requested()
        logger.warning(
            "Local performer sync finished (job_id=%d, stopped=%s): "
            "%d added, %d updated, %d removed, %d skipped (no image), %d errored, index now has %d",
            context.job_id, stopped_early, added, updated, removed,
            skipped_no_image, errored, len(index),
        )

        # Force the live recognizer to pick up the updated index on its next
        # use, same mechanism the database updater uses for main-DB hot-swaps.
        try:
            from resource_manager import get_resource_manager
            get_resource_manager().unload("face_recognition")
        except (RuntimeError, KeyError):
            pass  # not initialized / not currently loaded -- nothing to unload

        if stopped_early:
            return json.dumps({"position": i + 1})

        summary = (
            f"{added} added, {updated} updated, {removed} removed "
            f"({skipped_no_image} had no usable image)"
        )
        if errored:
            summary += f", {errored} failed to fetch/decode"
        context.set_result_summary(summary)
        return None

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

The one bulk `get_all_performers()` call in `run()` already returns every
field (image_path, stash_ids, name) that a per-performer GraphQL query
would -- `_fetch_one()` uses that directly rather than re-fetching each
performer individually. That per-performer re-fetch used to go through
the shared RateLimiter that guards the local Stash instance's API (5
req/sec by default) on every one of them -- confirmed live, that alone
turned a run over ~1600 already-unchanged performers into 5+ minutes,
regardless of FETCH_CONCURRENCY, since the limiter serializes across all
concurrent fetches. The image download itself bypasses that GraphQL
client/rate limiter entirely (a plain httpx call), so removing the
redundant re-fetch leaves this job bound only by that.

A performer with no custom cover image still returns an image_path, but
Stash marks it with a "default=true" query param -- that's how "no image
yet" is detected, without needing to fetch and fail to decode a
placeholder icon.

## Concurrency

Fetch (network) is async with bounded concurrency, sharing one
httpx.AsyncClient; detect+embed (CPU-bound) runs on a small pool of real
worker threads, each its own FaceEmbeddingGenerator -- same split as
stash-sense2-data-gen's build/backfill_genderage.py, and for the same
reason (fetches are I/O-bound, detection is CPU-bound around each ONNX
call). Kept far more modest than that backfill's 32/5: this job's own
library is typically a few hundred to a few thousand performers, not
660k, and unlike the backfill this runs *inside* a long-lived service
process's shared event loop, not a one-shot batch container -- the main
coordinator loop below is async and never blocks synchronously, so other
requests/jobs keep running while this one is in flight.

`LocalPerformerIndex.upsert()`/`remove()` touch a usearch C++ Index
object and are not thread-safe, so exactly one thread (the main
coordinator, i.e. this job's own `run()`) ever calls them -- the async
fetch stage only ever *reads* `index.get_image_hash()` (a plain dict
lookup, safe under the GIL, and it never touches the usearch Index
object at all), and reports anything requiring a write back to the
coordinator via a queue instead of writing it itself.

Uses `sync_one_performer()`'s pure helpers (`_image_fingerprint`,
`_relative_image_url`, `STASHDB_ENDPOINT`) but not the function itself --
`sync_one_performer()` stays untouched since the single-performer Stash
hook handler also depends on its exact current (simple, sequential)
behavior.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, Optional

import httpx

from base_job import BaseJob, JobContext
from config import DatabaseConfig
from embeddings import FaceEmbeddingGenerator, load_image, GPU_COMPUTE_LOCK
from local_performer_index import (
    STASHDB_ENDPOINT,
    LocalPerformerIndex,
    _image_fingerprint,
    _relative_image_url,
)
from recommendations_router import get_stash_client

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Save the index to disk (and let a resumed run pick up here) after this
# many performers processed in a batch, mirroring fingerprint_job.py's
# per-100-scenes checkpoint cadence.
CHECKPOINT_BATCH_SIZE = 50

# How often to push a live progress update (JobContext.report_progress).
# Each call is a synchronous SQLite UPDATE (open connection, write, commit,
# close) on the coordinator's own event loop -- confirmed live, calling it
# on every single completion (as this job originally did) turned a run
# over ~1600 already-unchanged performers (a few seconds of real fetch
# work at FETCH_CONCURRENCY=8) into 5+ minutes, entirely spent on ~1600
# individual DB commits with no GPU/CPU work behind them at all. A small
# interval keeps the displayed progress/ETA responsive without paying that
# cost per item.
PROGRESS_REPORT_INTERVAL = 20

FETCH_CONCURRENCY = 8
EMBED_WORKERS = 2


async def _fetch_one(
    stash: Any, client: httpx.AsyncClient, index: LocalPerformerIndex,
    performer: dict, semaphore: asyncio.Semaphore, position: int,
    event_queue: "queue.Queue", stop_event: threading.Event,
) -> None:
    """Fetches + fingerprints one performer's cover image and reports the
    outcome onto `event_queue` for the coordinator to act on. Never
    writes to `index` -- only reads `get_image_hash()` (plain dict
    lookup, safe from this thread; see module docstring)."""
    if stop_event.is_set():
        return  # never dispatched -- position stays incomplete, picked up on resume
    performer_id = int(performer["id"])
    # event_queue.put() below is wrapped in asyncio.to_thread() throughout --
    # a plain blocking call here would, if the queue is momentarily full,
    # stall this whole producer thread's event loop (all FETCH_CONCURRENCY
    # in-flight fetches), not just this one task. Real embed-worker threads
    # (see _embed_worker) don't need this -- a blocking put() there only
    # blocks that one dedicated thread, which is fine/expected backpressure.
    async with semaphore:
        # `performer` already has everything get_performer() would return --
        # get_all_performers() and get_performer() fetch the exact same
        # field set (see stash_client_unified.py). Re-fetching it here
        # per-performer was pure waste, and a costly one: every such call
        # goes through RateLimiter (5 req/sec by default, protecting the
        # user's own Stash instance), serializing all FETCH_CONCURRENCY
        # fetches onto one shared budget -- confirmed live, this alone
        # accounted for the ~5 minutes a run over ~1600 performers took
        # regardless of the report_progress throttling fix above (that fix
        # was real but not the dominant cost). The image download below
        # bypasses the GraphQL client/rate limiter entirely (plain
        # client.get), so removing this call leaves fetches bound only by
        # that -- confirmed fast (single-digit milliseconds) even under
        # concurrency.
        image_path = performer.get("image_path")
        has_custom_image = bool(image_path) and "default=true" not in image_path
        if not has_custom_image:
            await asyncio.to_thread(event_queue.put, ("no_image", performer_id, position, None))
            return

        try:
            resp = await client.get(image_path, headers={"ApiKey": stash.api_key})
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            logger.warning("Local performer sync: failed to fetch image for performer %d: %s", performer_id, e)
            await asyncio.to_thread(event_queue.put, ("fetch_error", performer_id, position, None))
            return

        fingerprint = _image_fingerprint(image_bytes)
        current_urls = performer.get("urls") or []
        if index.get_image_hash(performer_id) == fingerprint:
            # Cover unchanged, but `urls` may not be -- e.g. right after the
            # identity-resolution flow in recommendations_router.py writes a
            # new profile URL onto an existing performer, which doesn't
            # touch the cover. Payload carries the current urls so the
            # consumer loop (which owns `index`) can refresh them in place
            # without a re-embed. See local_performer_index.update_urls().
            await asyncio.to_thread(event_queue.put, ("unchanged", performer_id, position, current_urls))
            return

        stashdb_id = next(
            (sid["stash_id"] for sid in (performer.get("stash_ids") or [])
             if sid.get("endpoint") == STASHDB_ENDPOINT),
            None,
        )
        meta = {
            "name": performer["name"], "stashdb_id": stashdb_id,
            "image_url": _relative_image_url(image_path), "urls": performer.get("urls") or [],
        }
        await asyncio.to_thread(
            event_queue.put, ("needs_embed", performer_id, position, (image_bytes, fingerprint, meta)),
        )


async def _produce(
    stash: Any, index: LocalPerformerIndex, performers: list[dict], start_position: int,
    event_queue: "queue.Queue", stop_event: threading.Event,
) -> None:
    semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)
    async with httpx.AsyncClient(timeout=15.0) as client:
        tasks = [
            asyncio.create_task(
                _fetch_one(stash, client, index, performers[i], semaphore, i, event_queue, stop_event)
            )
            for i in range(start_position, len(performers))
        ]
        await asyncio.gather(*tasks)
    event_queue.put(("producer_done", None, None, None))


def _embed_worker(embed_queue: "queue.Queue", event_queue: "queue.Queue") -> None:
    """Runs in its own OS thread with its own FaceEmbeddingGenerator --
    pulls fetched images off embed_queue, does detect+embed, and reports
    the outcome onto event_queue for the coordinator to apply."""
    generator = FaceEmbeddingGenerator()
    while True:
        item = embed_queue.get()
        if item is None:
            break
        performer_id, position, image_bytes, fingerprint, meta = item
        try:
            image = load_image(image_bytes)
            # GPU_COMPUTE_LOCK: a plain threading.Lock, acquired directly
            # since this runs in its own raw OS thread (no event loop) --
            # see embeddings.py's docstring. Serializes against
            # identification_router.py's identify calls too, not just
            # other _embed_worker threads -- confirmed live, an in-flight
            # GPU job's detection concurrent with a live identify request
            # produced real ROCm/MIOpen failures on this hardware.
            with GPU_COMPUTE_LOCK:
                faces = generator.detect_faces(image, min_confidence=0.5)
        except Exception as e:
            logger.warning("Local performer sync: decode/detect failed for performer %d: %s", performer_id, e)
            event_queue.put(("embed_error", performer_id, position, None))
            continue

        if not faces:
            event_queue.put(("no_face", performer_id, position, None))
            continue

        best_face = max(faces, key=lambda f: f.bbox["w"] * f.bbox["h"])
        embedding = generator.get_embedding(best_face)
        event_queue.put(("embedded", performer_id, position, (embedding.embedding, fingerprint, meta)))

    event_queue.put(("worker_done", None, None, None))


class LocalPerformerSyncJob(BaseJob):
    """Diffs current Stash performers against the local index, embedding
    new/changed performers and removing ones no longer present.

    Cursor format (JSON string): {"position": <int>} -- index into the
    stable-sorted (by id) performer list to resume from; specifically the
    highest N such that positions [0, N) are *all* complete (not just
    "highest position seen"), since fetch/embed now complete out of
    order under concurrency. The index is saved to disk at the same
    checkpoints, so a crash never loses already-embedded work, and a
    resumed run naturally skips anything already up to date (same image
    fingerprint) even without the cursor.
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

        if start_position >= total:
            context.set_result_summary(f"0 added, 0 updated, {removed} removed (nothing new to sync)")
            return None

        event_queue: "queue.Queue" = queue.Queue(maxsize=500)
        embed_queue: "queue.Queue" = queue.Queue(maxsize=200)
        stop_event = threading.Event()

        embed_threads = [
            threading.Thread(target=_embed_worker, args=(embed_queue, event_queue), daemon=True)
            for _ in range(EMBED_WORKERS)
        ]
        for t in embed_threads:
            t.start()

        producer_thread = threading.Thread(
            target=lambda: asyncio.run(_produce(stash, index, performers, start_position, event_queue, stop_event)),
            daemon=True,
        )
        producer_thread.start()

        completed_positions: set[int] = set()
        next_expected = start_position
        producer_finished = False
        worker_sentinels = 0
        since_checkpoint = 0
        since_progress_report = 0
        # Real completions, in whatever order they actually finish -- unlike
        # next_expected (only the *contiguous* prefix, since that's what the
        # resume cursor needs), this is what progress/ETA should be measured
        # against. Confirmed live: with next_expected alone, one slow
        # straggler position pinned the displayed progress at "2" for
        # several minutes while ~93 later positions had already completed
        # concurrently, making a healthy run look stuck with a wildly wrong
        # ETA. Starts at start_position, same as next_expected, so a
        # resumed run's displayed progress is absolute (out of `total`),
        # not reset to 0 for just the remaining work.
        completed_count = start_position

        def _advance_and_report(position: int) -> None:
            nonlocal next_expected, since_checkpoint, since_progress_report, completed_count
            completed_positions.add(position)
            while next_expected in completed_positions:
                completed_positions.discard(next_expected)
                next_expected += 1
            since_checkpoint += 1
            since_progress_report += 1
            completed_count += 1

        while True:
            kind, performer_id, position, payload = await asyncio.to_thread(event_queue.get)

            if kind == "producer_done":
                producer_finished = True
                for _ in range(EMBED_WORKERS):
                    embed_queue.put(None)
                continue
            if kind == "worker_done":
                worker_sentinels += 1
                if producer_finished and worker_sentinels >= EMBED_WORKERS:
                    break
                continue

            if kind == "unchanged":
                if index.update_urls(performer_id, payload or []):
                    updated += 1
            elif kind == "fetch_error":
                errored += 1
            elif kind == "no_image":
                was_present = performer_id in index
                index.remove(performer_id)
                if was_present:
                    removed += 1
                else:
                    skipped_no_image += 1
            elif kind == "needs_embed":
                image_bytes, fingerprint, meta = payload
                await asyncio.to_thread(embed_queue.put, (performer_id, position, image_bytes, fingerprint, meta))
                continue  # not complete yet -- don't advance the cursor until the embed result comes back
            elif kind in ("no_face", "embed_error"):
                was_present = performer_id in index
                index.remove(performer_id)
                if kind == "embed_error":
                    errored += 1
                elif was_present:
                    removed += 1
                else:
                    skipped_no_image += 1
            elif kind == "embedded":
                embedding, fingerprint, meta = payload
                was_present = performer_id in index
                index.upsert(
                    performer_id=performer_id, name=meta["name"], stashdb_id=meta["stashdb_id"],
                    image_hash=fingerprint, image_url=meta["image_url"], embedding=embedding,
                    urls=meta["urls"],
                )
                if was_present:
                    updated += 1
                else:
                    added += 1

            _advance_and_report(position)
            if since_progress_report >= PROGRESS_REPORT_INTERVAL:
                since_progress_report = 0
                await context.report_progress(completed_count, total)

            if since_checkpoint >= CHECKPOINT_BATCH_SIZE:
                since_checkpoint = 0
                index.save()
                # cursor stays next_expected (the contiguous-prefix resume
                # point -- correctness-critical), but items_processed is
                # completed_count (real progress -- display-critical). These
                # can legitimately differ for a while under concurrency; see
                # completed_count's own comment above.
                await context.checkpoint(cursor=json.dumps({"position": next_expected}), items_processed=completed_count)
                logger.debug(
                    "Local performer sync checkpoint: resume_position=%d, completed=%d/%d",
                    next_expected, completed_count, total,
                )

            if context.is_stop_requested() and not stop_event.is_set():
                stop_event.set()  # prevents new fetches from starting; in-flight ones still complete and get applied

        producer_thread.join()
        for t in embed_threads:
            t.join()
        index.save()
        # Final report regardless of the throttle above -- otherwise a total
        # not evenly divisible by PROGRESS_REPORT_INTERVAL leaves the last
        # few completions never reflected in the displayed progress.
        await context.report_progress(completed_count, total)

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

        if stopped_early and next_expected < total:
            return json.dumps({"position": next_expected})

        summary = (
            f"{added} added, {updated} updated, {removed} removed "
            f"({skipped_no_image} had no usable image)"
        )
        if errored:
            summary += f", {errored} failed to fetch/decode"
        context.set_result_summary(summary)
        return None

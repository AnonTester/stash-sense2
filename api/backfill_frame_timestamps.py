"""Backfill: reconstruct real per-face timestamp_sec for scene_face_embeddings
rows cached before frame-timestamp tracking was fixed (see
identification_router.py's _identify_scene_compute/_process_sprite_frames)
-- restores "jump to frame" data for already-fingerprinted scenes without
re-running detection/embedding (no ffmpeg, no GPU).

Two independent fixes:

1. Sprite-tile rows (is_sprite=1): each row's own timestamp_sec was always
   written correctly, but every sprite face for a scene shared
   frame_index=-2, so only one timestamp per scene was ever resolvable via
   the frame_index -> timestamp lookup the matching pipeline builds.
   Reassigns each existing row within a scene a distinct frame_index
   (-2, -3, -4, ...) -- the value only needs to be unique, not meaningful
   (nothing besides this lookup reads it). This half runs automatically as
   schema migration v18 (recommendations_db.py) -- pure local SQL, so it
   fits the normal synchronous migration path and needs no separate wiring
   here; `backfill_sprite_frame_indices` below exists for the standalone
   CLI/dry-run path (and as the migration's own reference implementation).

2. Video-frame rows (is_sprite=0) with timestamp_sec IS NULL: the live
   pipeline always uses uniform-interval sampling
   (frame_extractor.calculate_extraction_timestamps -- burst/weighted modes
   are unused dead code for /identify/scene, only sampling_experiment.py's
   offline tooling calls them), so
       timestamp = duration_sec*start_offset_pct + frame_index*interval
       interval = (duration_sec*end_offset_pct - duration_sec*start_offset_pct)
                  / (num_frames - 1)
   num_frames/start_offset_pct/end_offset_pct are already stored per-scene
   in scene_signal_cache; duration_sec needs one cheap GraphQL call per
   scene to Stash (metadata only, no video decode). This can't be a
   synchronous schema migration (needs a live network call), so
   `run_video_timestamp_backfill_once` runs it as a background asyncio
   task at app startup instead (see main.py), gated by a user_settings
   flag so it only ever runs once per install.

Commits after each scene (not once at the end) -- both backfills below run
against the live production DB while the sidecar is serving real traffic,
and a single long-held write transaction across many scenes (each with a
network round trip for the video half's duration lookup) would block the
app's own writes for the whole run.

Standalone CLI usage (inside the sidecar container, where DATA_DIR/
STASH_URL/STASH_API_KEY already match the running app's own environment) --
useful for a dry-run preview, or to force the video backfill immediately
rather than waiting for the next startup:
    python backfill_frame_timestamps.py --dry-run
    python backfill_frame_timestamps.py
"""
import argparse
import asyncio
import logging
import os
import sqlite3
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_VIDEO_BACKFILL_DONE_SETTING = "_video_timestamp_backfill_v1_done"


def backfill_sprite_frame_indices(conn: sqlite3.Connection, dry_run: bool) -> int:
    scene_ids = [row[0] for row in conn.execute(
        "SELECT DISTINCT stash_scene_id FROM scene_face_embeddings WHERE is_sprite = 1"
    )]
    updated = 0
    for scene_id in scene_ids:
        rows = conn.execute(
            "SELECT id, frame_index FROM scene_face_embeddings WHERE stash_scene_id = ? AND is_sprite = 1 ORDER BY id",
            (scene_id,),
        ).fetchall()
        # Already distinct (e.g. re-cached since the code fix landed) --
        # nothing to reassign for this scene.
        if len(rows) == len({r[1] for r in rows}):
            continue
        for i, (row_id, _old_index) in enumerate(rows):
            new_index = -2 - i
            if not dry_run:
                conn.execute(
                    "UPDATE scene_face_embeddings SET frame_index = ? WHERE id = ?",
                    (new_index, row_id),
                )
            updated += 1
        if not dry_run:
            conn.commit()
    return updated


async def _fetch_duration(client: httpx.AsyncClient, base_url: str, api_key: str, scene_id: int) -> float | None:
    resp = await client.post(
        f"{base_url}/graphql",
        json={"query": f'{{ findScene(id: "{scene_id}") {{ files {{ duration }} }} }}'},
        headers={"ApiKey": api_key, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    scene = (payload.get("data") or {}).get("findScene")
    if not scene:
        return None
    files = scene.get("files") or []
    return files[0]["duration"] if files and files[0].get("duration") else None


async def backfill_video_timestamps(
    conn: sqlite3.Connection, base_url: str, api_key: str, dry_run: bool,
) -> tuple[int, int]:
    scene_ids = [row[0] for row in conn.execute(
        "SELECT DISTINCT stash_scene_id FROM scene_face_embeddings WHERE is_sprite = 0 AND timestamp_sec IS NULL"
    )]
    updated = 0
    skipped = 0
    async with httpx.AsyncClient(timeout=15.0) as client:
        for scene_id in scene_ids:
            cache_row = conn.execute(
                "SELECT num_frames, start_offset_pct, end_offset_pct FROM scene_signal_cache WHERE stash_scene_id = ?",
                (scene_id,),
            ).fetchone()
            if not cache_row or not cache_row[0] or cache_row[0] <= 1:
                skipped += 1
                continue
            num_frames, start_pct, end_pct = cache_row

            try:
                duration = await _fetch_duration(client, base_url, api_key, scene_id)
            except Exception as e:
                print(f"  scene {scene_id}: failed to fetch duration ({e}), skipping")
                skipped += 1
                continue
            if not duration:
                skipped += 1
                continue

            start_sec = duration * start_pct
            end_sec = duration * end_pct
            interval = (end_sec - start_sec) / (num_frames - 1)

            rows = conn.execute(
                "SELECT id, frame_index FROM scene_face_embeddings "
                "WHERE stash_scene_id = ? AND is_sprite = 0 AND timestamp_sec IS NULL",
                (scene_id,),
            ).fetchall()
            for row_id, frame_index in rows:
                if frame_index is None or frame_index < 0 or frame_index >= num_frames:
                    continue  # not a plain uniformly-sampled video frame -- leave untouched
                ts = start_sec + frame_index * interval
                if not dry_run:
                    conn.execute(
                        "UPDATE scene_face_embeddings SET timestamp_sec = ? WHERE id = ?",
                        (ts, row_id),
                    )
                updated += 1
            if not dry_run:
                conn.commit()
    return updated, skipped


async def run_video_timestamp_backfill_once(rec_db, stash_url: str, api_key: str) -> None:
    """Run the video-frame timestamp_sec backfill exactly once per install,
    as a non-blocking background task launched from main.py's startup --
    see this module's own docstring for why this half can't be a
    synchronous schema migration the way the sprite half is (v18).

    Best-effort and silent-on-failure by design (matches this codebase's
    convention for background warm/refresh tasks, e.g. release_info's own
    refresh loop): a scene the world's own duration-lookup transiently
    fails for is already handled per-scene inside backfill_video_timestamps
    (skipped, not retried within this run), and if the whole run raises
    (e.g. Stash unreachable at this exact moment) the "done" flag is
    deliberately left unset so the very next startup just tries again --
    no harm in retrying, since re-running is idempotent (already-filled
    rows are skipped by definition, only timestamp_sec IS NULL rows are
    ever touched)."""
    if rec_db.get_user_setting(_VIDEO_BACKFILL_DONE_SETTING):
        return
    if not stash_url:
        return

    conn = sqlite3.connect(str(rec_db.db_path))
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        updated, skipped = await backfill_video_timestamps(conn, stash_url, api_key, dry_run=False)
        logger.warning(
            "Video-frame timestamp_sec backfill complete: %d row(s) updated, %d scene(s) skipped",
            updated, skipped,
        )
        rec_db.set_user_setting(_VIDEO_BACKFILL_DONE_SETTING, True)
    except Exception:
        logger.exception("Video-frame timestamp_sec backfill failed -- will retry on next startup")
    finally:
        conn.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    args = parser.parse_args()

    db_path = Path(args.data_dir) / "stash_sense.db"
    if not db_path.exists():
        raise SystemExit(f"Database not found at {db_path}")

    stash_url = os.environ.get("STASH_URL", "").rstrip("/")
    api_key = os.environ.get("STASH_API_KEY", "")
    if not stash_url:
        raise SystemExit("STASH_URL env var not set")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 30000")  # wait up to 30s if the live app holds a lock
    try:
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Backfilling sprite frame_index uniqueness...")
        sprite_updated = backfill_sprite_frame_indices(conn, args.dry_run)
        print(f"  {sprite_updated} sprite row(s) reassigned a unique frame_index")

        print(f"{'[DRY RUN] ' if args.dry_run else ''}Backfilling video-frame timestamp_sec...")
        video_updated, video_skipped = await backfill_video_timestamps(conn, stash_url, api_key, args.dry_run)
        print(f"  {video_updated} video-frame row(s) given a reconstructed timestamp_sec")
        print(f"  {video_skipped} scene(s) skipped (no cache-meta, or duration lookup failed)")

        if args.dry_run:
            print("Dry run -- no changes written.")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())

"""Knowing which video a scene's cached faces came from.

Everything expensive about identifying a scene (frame extraction, detection, embedding, the sprite sheet) is cached per
scene id and reused forever. A scene id outlives its video, though: merge a longer file into a scene and make it the
primary, or replace the file, and the cached faces still describe the old video -- every later "re-scan" is then a
cache hit that re-matches the old faces and never looks at the new video.

Each scene therefore carries a signature of the primary file the caches were built from (its Stash id and duration,
table ``scene_file_signature``). It is compared with the scene's current primary file before any cached face is
reused; on a mismatch the scene's face caches are dropped, so the identify that follows extracts the new video.

* ``ensure_scene_cache_matches_file`` -- the per-scene check every identify call makes (identification_router.py).
* ``reconcile_scene_signatures`` -- the library-wide pass (startup, and the start of Fingerprint Missing / Refresh
  Outdated): drops the caches of every changed scene, marks their stored results stale so Refresh Outdated redoes
  them, and records the signature of scenes that have none yet.

Caches written before this existed have no signature. Their frames still tell how long the video was: they were
sampled between ``start_offset_pct`` and ``end_offset_pct`` of its duration, so the latest cached timestamp bounds it.
``legacy_cache_is_stale`` only calls a cache stale when the current duration is clearly outside that bound.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Two durations of one file differ by container rounding at most; a replaced or merged video differs by far more.
DURATION_TOLERANCE_SEC = 1.0
DURATION_TOLERANCE_RATIO = 0.005
# A legacy cache is only called stale when the current duration is this much longer than its frames can come from.
# Frames are spread over [start_offset_pct, end_offset_pct] of the video, but with zones and an intro skip the latest
# one can land well short of that: measured on a real library, many unchanged scenes' last frame sits at ~75% of the
# video (current duration 1.26x what end_offset_pct alone predicts). A replaced or merged video is off by far more.
LEGACY_LONGER_RATIO = 2.0

# db_version written on a stored result whose video changed, so Refresh Outdated treats it as outdated.
STALE_FILE_MARKER = "stale-file"


@dataclass(frozen=True)
class FileSignature:
    file_id: Optional[str]
    duration_sec: Optional[float]


def signature_from_scene(scene: Optional[dict]) -> Optional[FileSignature]:
    """The signature of a Stash scene's primary file (``files[0]``, as Stash orders them), or None without one."""
    files = (scene or {}).get("files") or []
    if not files:
        return None
    primary = files[0]
    duration = primary.get("duration")
    file_id = primary.get("id")
    return FileSignature(
        file_id=str(file_id) if file_id is not None else None,
        duration_sec=float(duration) if duration else None,
    )


def signatures_match(stored: dict, current: FileSignature) -> bool:
    """Whether a stored {file_id, duration_sec} still describes the current primary file. A value missing on either
    side proves nothing and is not held against the cache."""
    stored_id = stored.get("file_id")
    if stored_id and current.file_id and str(stored_id) != current.file_id:
        return False
    stored_duration = stored.get("duration_sec")
    if stored_duration and current.duration_sec:
        allowed = max(DURATION_TOLERANCE_SEC, DURATION_TOLERANCE_RATIO * max(stored_duration, current.duration_sec))
        if abs(stored_duration - current.duration_sec) > allowed:
            return False
    return True


def legacy_cache_is_stale(max_cached_timestamp: float, end_offset_pct: float, current_duration: Optional[float]) -> bool:
    """For a cache with no stored signature: do its frames fit the video as it is now?

    Stale when the current video is shorter than a frame that was taken from it, or much longer than frames spread up
    to ``end_offset_pct`` of it could cover."""
    if not current_duration or not max_cached_timestamp or max_cached_timestamp <= 0:
        return False
    if max_cached_timestamp > current_duration + DURATION_TOLERANCE_SEC:
        return True
    covered = max_cached_timestamp / max(end_offset_pct or 1.0, 0.05)
    return current_duration > covered * LEGACY_LONGER_RATIO


async def fetch_scene_signature(stash_url: str, api_key: str, scene_id: int) -> Optional[FileSignature]:
    """One small GraphQL call for a scene's current primary file. None when Stash cannot be asked."""
    import httpx
    query = f'{{ findScene(id: "{scene_id}") {{ files {{ id duration }} }} }}'
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{stash_url.rstrip('/')}/graphql", json={"query": query},
                headers={"ApiKey": api_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return signature_from_scene((resp.json().get("data") or {}).get("findScene"))
    except Exception as e:  # noqa: BLE001 -- the check is best effort; never block an identify on it
        logger.warning("Could not read scene %s's file from Stash (cache kept as is): %s", scene_id, e)
        return None


async def ensure_scene_cache_matches_file(
    rec_db: Any, scene_id: int, current: Optional[FileSignature],
) -> bool:
    """Drops this scene's cached faces if they came from a different video than its current primary file, and records
    the current file as the one any new cache belongs to. Returns True when caches were dropped.

    `current` None (no file known) does nothing."""
    if rec_db is None or current is None:
        return False
    stored = rec_db.get_scene_file_signature(scene_id)
    purged = False
    if stored is not None:
        purged = not signatures_match(stored, current)
    else:
        extent = rec_db.get_cached_video_extent(scene_id)
        purged = bool(extent) and legacy_cache_is_stale(extent[0], extent[1], current.duration_sec)
    if purged:
        logger.warning(
            "Scene %s: its video changed (now file %s, %.0fs) -- cached faces dropped, extracting again",
            scene_id, current.file_id, current.duration_sec or 0,
        )
        rec_db.purge_scene_face_caches([scene_id])
    if purged or stored is None:
        rec_db.set_scene_file_signatures({scene_id: (current.file_id, current.duration_sec)})
    return purged


@dataclass
class ReconcileResult:
    changed: list[int]      # scenes whose video changed: caches dropped, stored result marked outdated
    adopted: int            # scenes that had no signature and now have one
    checked: int


def reconcile_scene_signatures(rec_db: Any, current: dict[int, FileSignature]) -> ReconcileResult:
    """Library-wide version of ensure_scene_cache_matches_file for `current` ({scene id: signature}, every scene in
    Stash). Synchronous and DB-only (callers run it on a worker thread)."""
    stored = rec_db.get_all_scene_file_signatures()
    cached = rec_db.get_scene_ids_with_face_cache()
    fingerprinted = rec_db.get_fingerprinted_scene_ids()
    extents = rec_db.get_cached_video_extents()

    changed: list[int] = []
    to_record: dict[int, tuple[Optional[str], Optional[float]]] = {}
    adopted = 0
    for scene_id, sig in current.items():
        old = stored.get(scene_id)
        if old is not None:
            if signatures_match(old, sig):
                continue
            changed.append(scene_id)
        else:
            has_data = scene_id in cached or scene_id in fingerprinted
            extent = extents.get(scene_id)
            if has_data and extent and legacy_cache_is_stale(extent[0], extent[1], sig.duration_sec):
                changed.append(scene_id)
            else:
                adopted += 1
        to_record[scene_id] = (sig.file_id, sig.duration_sec)

    if changed:
        rec_db.purge_scene_face_caches(changed)
        rec_db.mark_fingerprints_stale([s for s in changed if s in fingerprinted], STALE_FILE_MARKER)
        logger.warning("%d scene(s) have a different video than their cached faces came from: %s",
                       len(changed), changed[:20])
    rec_db.set_scene_file_signatures(to_record)
    return ReconcileResult(changed=sorted(changed), adopted=adopted, checked=len(current))


async def reconcile_on_startup(rec_db: Any, stash_url: str, api_key: str) -> None:
    """Background task: compare every scene's primary file with the one its cached faces came from. Never raises."""
    import asyncio
    try:
        from stash_client_unified import StashClientUnified
        client = StashClientUnified(stash_url, api_key)
        current = await client.get_all_scene_file_signatures()
        result = await asyncio.to_thread(reconcile_scene_signatures, rec_db, current)
        logger.warning(
            "Scene file check: %d scenes compared, %d with a changed video (cache dropped, marked outdated), "
            "%d newly recorded", result.checked, len(result.changed), result.adopted,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Scene file check at startup skipped: %s", e)

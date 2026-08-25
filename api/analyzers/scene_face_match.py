"""Analyzer: identify performers in scenes that have none assigned yet.

Extends BaseAnalyzer (not BaseUpstreamAnalyzer) -- like
scene_fingerprint_match.py, this finds new candidates for scenes rather than
diffing already-linked entities. Reuses the same manual-identify pipeline
the scene-page "Identify" button calls (identification_router.py's
_identify_scene_impl) instead of reimplementing frame extraction/detection/
matching, so results and caching behavior stay identical to a manual run.
"""

import logging
from typing import Optional

from fastapi import HTTPException

from .base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

# Matches per detected person/face-cluster. The plain "cluster" matching_mode
# (unlike hybrid/frequency) doesn't drop single-appearance persons, which is
# the "plus single frame matches" behavior this job is meant to surface.
TOP_K_PER_PERSON = 5

PAGE_SIZE = 100


def _match_universal_id(match) -> Optional[str]:
    """Reconstruct the universal_id a PerformerMatchResponse was built from.

    The response model only carries the decomposed parts (endpoint,
    stashdb_id, local_performer_id), not universal_id itself. Per
    recognizer.py's recognize_face_v2: for a local-index match,
    `stashdb_id` may hold that performer's *linked* StashDB uuid rather than
    their local id, so local_performer_id (always the local id for a local
    match) must be checked first -- endpoint+stashdb_id alone would silently
    reconstruct the wrong key. For stashbox/catalogue matches, endpoint+
    stashdb_id already exactly reproduces the original "<endpoint>:<id>".
    """
    if match.local_performer_id:
        return f"local:{match.local_performer_id}"
    if match.endpoint and match.stashdb_id:
        return f"{match.endpoint}:{match.stashdb_id}"
    return None


class SceneFaceMatchAnalyzer(BaseAnalyzer):
    type = "scene_face_match"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        # Lazy import to avoid a circular dependency: recommendations_router
        # imports this module (via the ANALYZERS registry) before it has
        # finished defining the names identification_router itself imports
        # back from recommendations_router. See scene_matcher.py for the
        # same pattern/reasoning.
        from identification_router import (
            _identify_scene_impl,
            require_db_available,
            SceneIdentifyRequest,
        )

        try:
            await require_db_available()
        except HTTPException as e:
            logger.warning("[scene_face_match] Face recognition unavailable, skipping run: %s", e.detail)
            return AnalysisResult(items_processed=0, recommendations_created=0, errors=[str(e.detail)])

        # Manual/full scans should rebuild pending recommendations from
        # scratch so stale entries disappear, matching scene_fingerprint_match.
        if not incremental:
            cleared = self.rec_db.delete_pending_recommendations_by_type(self.type)
            if cleared:
                logger.warning("[scene_face_match] Cleared %d stale pending recommendations before full scan", cleared)

        watermark_key = "scene_face_match"
        watermark_ts = None
        if incremental:
            wm = self.rec_db.get_watermark(watermark_key)
            if wm:
                watermark_ts = wm.get("last_stash_updated_at")

        scenes_to_scan: list[dict] = []
        offset = 0
        latest_updated = watermark_ts
        skipped_no_duration = 0

        while True:
            if self.is_stop_requested():
                logger.warning("[scene_face_match] Stop requested while fetching candidate scenes")
                break

            scenes, total = await self.stash.get_scenes_without_performers(
                updated_after=watermark_ts, limit=PAGE_SIZE, offset=offset,
            )
            if not scenes:
                break

            for scene in scenes:
                updated_at = scene.get("updated_at")
                if updated_at and (latest_updated is None or updated_at > latest_updated):
                    latest_updated = updated_at

                files = scene.get("files") or []
                if not files or not files[0].get("duration"):
                    # identify_scene requires scene duration to plan ffmpeg extraction
                    skipped_no_duration += 1
                    continue
                scenes_to_scan.append(scene)

            offset += len(scenes)
            if offset >= total:
                break

        logger.warning(
            "[scene_face_match] Scan summary: candidates=%d, skipped_no_duration=%d, "
            "incremental=%s, watermark=%s",
            len(scenes_to_scan), skipped_no_duration, incremental, watermark_ts or "-",
        )

        if not scenes_to_scan:
            if latest_updated:
                self.rec_db.set_watermark(watermark_key, last_stash_updated_at=latest_updated)
            self.update_progress(0, 0)
            return AnalysisResult(items_processed=0, recommendations_created=0)

        self.set_items_total(len(scenes_to_scan))
        created = 0
        processed = 0
        errors: list[str] = []

        for scene in scenes_to_scan:
            if self.is_stop_requested():
                logger.warning(
                    "[scene_face_match] Stop requested after %d/%d scenes",
                    processed, len(scenes_to_scan),
                )
                break

            scene_id = str(scene["id"])
            scene_title = scene.get("title") or f"Scene {scene_id}"

            try:
                request = SceneIdentifyRequest(
                    scene_id=scene_id,
                    top_k=TOP_K_PER_PERSON,
                    matching_mode="cluster",
                    use_cache=True,
                    # Matches the manual "Identify" button's full-video flow
                    # (stash-sense.js's handleIdentifyFullVideo), which always
                    # merges sprite/VTT scrubber-bar tile faces alongside
                    # ffmpeg video-frame faces -- without this, a scene whose
                    # sprite catches a face the sampled video frames miss
                    # would silently get weaker results here than manual
                    # identify would find for the same scene.
                    use_sprite=True,
                )
                response = await _identify_scene_impl(request)
            except Exception as e:
                logger.warning("[scene_face_match] Failed to identify scene %s: %s", scene_id, e)
                errors.append(f"scene {scene_id}: {e}")
                processed += 1
                self.update_progress(processed, created)
                continue

            for person in response.persons:
                best_uid = _match_universal_id(person.best_match) if person.best_match else None
                for match in person.all_matches:
                    universal_id = _match_universal_id(match)
                    if not universal_id:
                        continue

                    target_id = f"{scene_id}|{universal_id}"
                    details = {
                        "scene_id": scene_id,
                        "scene_title": scene_title,
                        "person_id": person.person_id,
                        "frame_count": person.frame_count,
                        "is_best_match": universal_id == best_uid,
                        "universal_id": universal_id,
                        "stashdb_id": match.stashdb_id,
                        "name": match.name,
                        "confidence": match.confidence,
                        "distance": match.distance,
                        "country": match.country,
                        "image_url": match.image_url,
                        "endpoint": match.endpoint,
                        "local_performer_id": match.local_performer_id,
                        "source": match.source,
                        "catalogue_url": match.catalogue_url,
                        "profile_url": match.profile_url,
                        "top_timestamps_sec": match.top_timestamps_sec,
                    }

                    rec_id = self.create_recommendation(
                        target_type="scene",
                        target_id=target_id,
                        details=details,
                        confidence=match.confidence,
                    )
                    if rec_id:
                        created += 1

            processed += 1
            self.update_progress(processed, created)

        logger.warning(
            "[scene_face_match] Complete: %d matches found from %d/%d scenes scanned",
            created, processed, len(scenes_to_scan),
        )

        # Only advance the watermark past a fully-completed sweep -- mirrors
        # scene_fingerprint_match, so an interrupted run resumes the same
        # incremental window next time rather than silently skipping scenes
        # it never actually reached.
        if latest_updated and not self.is_stop_requested():
            self.rec_db.set_watermark(watermark_key, last_stash_updated_at=latest_updated)

        return AnalysisResult(
            items_processed=processed, recommendations_created=created, errors=errors,
        )

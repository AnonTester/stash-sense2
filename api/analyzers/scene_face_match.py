"""Analyzer: identify performers in scenes that have none assigned yet.

Extends BaseAnalyzer (not BaseUpstreamAnalyzer) -- like
scene_fingerprint_match.py, this finds new candidates for scenes rather than
diffing already-linked entities.

Reads Face Identification's stored per-scene match data
(scene_fingerprints/scene_fingerprint_matches, see identification_router.py's
save_scene_fingerprint) instead of re-running detect+embed+match itself --
that data is already there for any scene Face Identification has covered, so
recreating it here would just be redundant work against the same underlying
pipeline. The one thing that data might be missing is sprite-tile detection
(Face Identification's bulk runs default to no sprites, for cost -- see
fingerprint_generator.py); since a sprite-only face is exactly the kind of
thing that turns a performerless scene into a recommendation, this analyzer
does a small on-demand single-scene top-up (fresh identify, use_sprite=True)
for exactly the scenes that still need one, which also upgrades that scene's
stored data for next time (via the same save_scene_fingerprint path) --
after the first pass over a given scene, subsequent runs are a pure DB read.
"""

import logging

from fastapi import HTTPException

from scene_matcher import match_universal_id as _match_universal_id

from .base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

PAGE_SIZE = 100


def _make_details(
    scene_id: str, scene_title: str, person_id: int, frame_count: int, is_best_match: bool,
    universal_id: str, stashdb_id, name, confidence, distance, country, image_url, endpoint,
    local_performer_id, source, catalogue_url, profile_url, top_timestamps_sec,
) -> dict:
    """Shared recommendation `details` shape -- built the same way whether
    the match came from stored data or a fresh top-up identify."""
    return {
        "scene_id": scene_id,
        "scene_title": scene_title,
        "person_id": person_id,
        "frame_count": frame_count,
        "is_best_match": is_best_match,
        "universal_id": universal_id,
        "stashdb_id": stashdb_id,
        "name": name,
        "confidence": confidence,
        "distance": distance,
        "country": country,
        "image_url": image_url,
        "endpoint": endpoint,
        "local_performer_id": local_performer_id,
        "source": source,
        "catalogue_url": catalogue_url,
        "profile_url": profile_url,
        "top_timestamps_sec": top_timestamps_sec,
    }


class SceneFaceMatchAnalyzer(BaseAnalyzer):
    type = "scene_face_match"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        # Lazy import to avoid a circular dependency: recommendations_router
        # imports this module (via the ANALYZERS registry) before it has
        # finished defining the names identification_router itself imports
        # back from recommendations_router. See scene_matcher.py for the
        # same pattern/reasoning.
        from identification_router import require_db_available, SceneIdentifyRequest
        from scene_batch_orchestrator import SceneBatchSpec, identify_scenes_batched

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
        skipped_no_data = 0
        errors: list[str] = []

        scene_titles = {str(scene["id"]): scene.get("title") or f"Scene {scene['id']}" for scene in scenes_to_scan}

        # Partition: scenes whose stored data already has sprite coverage
        # are a pure DB read (fast); everything else needs a small on-demand
        # top-up identify (fresh sprite pass, cached video-frame detection).
        stored_scenes: list[tuple[str, dict]] = []  # (scene_id, fingerprint row)
        topup_specs = []
        for scene in scenes_to_scan:
            scene_id = str(scene["id"])
            fp = self.rec_db.get_scene_fingerprint(int(scene_id))
            if not fp or fp.get("fingerprint_status") != "complete":
                skipped_no_data += 1
                processed += 1
                self.update_progress(processed, created)
                continue
            if fp.get("used_sprite"):
                stored_scenes.append((scene_id, fp))
                continue
            file_info = (scene.get("files") or [{}])[0]
            topup_specs.append(SceneBatchSpec(
                scene_id=scene_id,
                width=file_info.get("width"),
                height=file_info.get("height"),
                request=SceneIdentifyRequest(
                    scene_id=scene_id,
                    matching_mode="hybrid",
                    top_k=5,
                    use_cache=True,
                    use_sprite=True,
                ),
            ))

        if skipped_no_data:
            logger.warning(
                "[scene_face_match] %d scene(s) have no completed Face Identification "
                "data yet -- run that first to cover them",
                skipped_no_data,
            )

        for scene_id, fp in stored_scenes:
            scene_title = scene_titles[scene_id]
            for row in self.rec_db.get_fingerprint_matches(fp["id"]):
                details = _make_details(
                    scene_id, scene_title, row["person_id"], row["frame_count"], row["is_best_match"],
                    row["universal_id"], row["stashdb_id"], row["name"], row["confidence"], row["distance"],
                    row["country"], row["image_url"], row["endpoint"], row["local_performer_id"],
                    row["source"], row["catalogue_url"], row["profile_url"], row["top_timestamps_sec"],
                )
                rec_id = self.create_recommendation(
                    target_type="scene", target_id=f"{scene_id}|{row['universal_id']}",
                    details=details, confidence=row["confidence"],
                )
                if rec_id:
                    created += 1
            processed += 1
            self.update_progress(processed, created)

        # require_db_available is passed through as identify_scenes_batched's
        # before_scene hook rather than called once up front -- it's both
        # the lazy-loader AND the idle-timer touch (see its own docstring),
        # normally re-run on every request via FastAPI's Depends() on the
        # real /identify/scene endpoint. This scan calls straight into the
        # underlying identify functions, bypassing that dependency, so on a
        # run long enough to cross the idle timeout (30 min default) the
        # face recognition model got unloaded mid-scan and never reloaded --
        # every remaining scene failed for the rest of the run with
        # "'NoneType' object has no attribute 'generator'". Confirmed live:
        # a 1144-scene full scan failed on 820 of them (72%) this way,
        # including the exact scene a user reported a confidence bug on, so
        # it never got a fresh recommendation from that run at all.
        # Re-checking before every scene reloads if evicted and resets the
        # idle timer so it doesn't happen again for the rest of a long scan.
        async for scene_id, result in identify_scenes_batched(
            topup_specs, is_stop_requested=self.is_stop_requested, before_scene=require_db_available,
        ):
            scene_title = scene_titles[scene_id]

            if isinstance(result, Exception):
                logger.warning("[scene_face_match] Sprite top-up failed for scene %s: %s", scene_id, result)
                errors.append(f"scene {scene_id}: {result}")
                processed += 1
                self.update_progress(processed, created)
                continue

            response = result
            for person in response.persons:
                best_uid = _match_universal_id(person.best_match) if person.best_match else None
                for match in person.all_matches:
                    universal_id = _match_universal_id(match)
                    if not universal_id:
                        continue
                    details = _make_details(
                        scene_id, scene_title, person.person_id, person.frame_count,
                        universal_id == best_uid, universal_id, match.stashdb_id, match.name,
                        match.confidence, match.distance, match.country, match.image_url, match.endpoint,
                        match.local_performer_id, match.source, match.catalogue_url, match.profile_url,
                        match.top_timestamps_sec,
                    )
                    rec_id = self.create_recommendation(
                        target_type="scene", target_id=f"{scene_id}|{universal_id}",
                        details=details, confidence=match.confidence,
                    )
                    if rec_id:
                        created += 1

            processed += 1
            self.update_progress(processed, created)

        if self.is_stop_requested() and processed < len(scenes_to_scan):
            logger.warning(
                "[scene_face_match] Stop requested after %d/%d scenes",
                processed, len(scenes_to_scan),
            )

        logger.warning(
            "[scene_face_match] Complete: %d matches found from %d/%d scenes scanned "
            "(%d from stored data, %d sprite top-ups, %d skipped -- no Face Identification data)",
            created, processed, len(scenes_to_scan),
            len(stored_scenes), len(topup_specs), skipped_no_data,
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

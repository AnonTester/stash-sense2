"""Analyzer: match local scenes to stash-box entries via fingerprints.

Extends BaseAnalyzer (not BaseUpstreamAnalyzer) because this finds
unlinked scenes rather than diffing already-linked entities.
"""

import logging
from typing import Optional

from .base import BaseAnalyzer, AnalysisResult
from scene_fingerprint_scoring import score_match, is_high_confidence
from stashbox_client import StashBoxClient

logger = logging.getLogger(__name__)

# Max scenes per stash-box batch query (matches Stash tagger batching)
BATCH_SIZE = 40


class SceneFingerprintMatchAnalyzer(BaseAnalyzer):
    type = "scene_fingerprint_match"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        connections = await self.stash.get_stashbox_connections()
        if not connections:
            return AnalysisResult(items_processed=0, recommendations_created=0)

        # Manual/full scans should rebuild pending recommendations from scratch
        # so stale entries (e.g. scenes linked after previous runs) disappear.
        if not incremental:
            cleared_raw = self.rec_db.delete_pending_recommendations_by_type(self.type)
            try:
                cleared = int(cleared_raw or 0)
            except Exception:
                cleared = 0
            if cleared > 0:
                logger.warning(
                    "[%s] Cleared %d stale pending recommendations before full scan",
                    self.type, cleared,
                )

        # Load user-configurable thresholds
        min_count = self._get_setting("scene_fp_min_count", 2)
        min_percentage = self._get_setting("scene_fp_min_percentage", 66)

        total_processed = 0
        total_created = 0

        for conn in connections:
            if self.is_stop_requested():
                logger.warning("Stop requested, halting scene fingerprint scan before next endpoint")
                break

            endpoint = conn["endpoint"]
            api_key = conn.get("api_key", "")
            endpoint_name = conn.get("name", endpoint)

            processed, created = await self._process_endpoint(
                endpoint, api_key, endpoint_name,
                incremental, min_count, min_percentage,
            )
            total_processed += processed
            total_created += created

        return AnalysisResult(
            items_processed=total_processed,
            recommendations_created=total_created,
        )

    async def _process_endpoint(
        self,
        endpoint: str,
        api_key: str,
        endpoint_name: str,
        incremental: bool,
        min_count: int,
        min_percentage: int,
    ) -> tuple[int, int]:
        """Process one stash-box endpoint. Returns (processed, created)."""
        watermark_key = f"scene_fp_match_{endpoint}"

        # Get watermark for incremental mode
        watermark_ts = None
        if incremental:
            wm = self.rec_db.get_watermark(watermark_key)
            if wm:
                watermark_ts = wm.get("last_stash_updated_at")

        # Fetch all local scenes with fingerprint data
        scenes_needing_match = []
        offset = 0
        latest_updated = watermark_ts
        fetched_scenes = 0
        skipped_with_links = 0
        skipped_without_fingerprints = 0
        stale_deleted = 0
        linked_scene_ids: set[str] = set()

        while True:
            if self.is_stop_requested():
                logger.warning("[%s] Stop requested while fetching local scenes", endpoint_name)
                break

            scenes, total = await self.stash.get_scenes_with_fingerprints(
                updated_after=watermark_ts, limit=100, offset=offset,
            )
            if not scenes:
                break
            fetched_scenes += len(scenes)

            for scene in scenes:
                # Track latest updated_at for watermark
                updated_at = scene.get("updated_at")
                if updated_at and (latest_updated is None or updated_at > latest_updated):
                    latest_updated = updated_at

                # Skip scenes that already have any stash-box link.
                # Scene Stash-Box Tagger should only work on completely unlinked scenes.
                existing_stash_ids = scene.get("stash_ids") or []
                if existing_stash_ids:
                    skipped_with_links += 1
                    linked_scene_ids.add(str(scene["id"]))
                    continue

                # Collect fingerprints from all files
                fingerprints = []
                duration = None
                for f in scene.get("files") or []:
                    if duration is None and f.get("duration"):
                        duration = f["duration"]
                    for fp in f.get("fingerprints") or []:
                        fingerprints.append({
                            "hash": fp["value"],
                            "algorithm": fp["type"].upper(),
                        })

                if fingerprints:
                    scenes_needing_match.append({
                        "scene": scene,
                        "fingerprints": fingerprints,
                        "duration": duration,
                    })
                else:
                    skipped_without_fingerprints += 1

            offset += len(scenes)
            if offset >= total:
                break

        # Clean up stale pending recommendations for scenes that are now linked.
        # This keeps Scene Stash-Box Tagger results limited to truly unlinked scenes.
        for linked_scene_id in linked_scene_ids:
            stale_deleted += self.rec_db.delete_pending_scene_fingerprint_for_scene(
                scene_id=linked_scene_id
            )

        logger.warning(
            "[%s] Scene fingerprint scan summary: fetched=%d, candidates=%d, "
            "skipped_linked=%d, skipped_no_fingerprints=%d, stale_deleted=%d, "
            "incremental=%s, watermark=%s",
            endpoint_name,
            fetched_scenes,
            len(scenes_needing_match),
            skipped_with_links,
            skipped_without_fingerprints,
            stale_deleted,
            incremental,
            watermark_ts or "-",
        )

        if not scenes_needing_match:
            if latest_updated:
                self.rec_db.set_watermark(watermark_key, last_stash_updated_at=latest_updated)
            self.update_progress(0, 0)
            return 0, 0

        # Batch query stash-box
        stashbox = StashBoxClient(endpoint, api_key)
        self.set_items_total(len(scenes_needing_match), label=endpoint_name)
        logger.warning(
            "[%s] Starting scan of %d scenes with fingerprints",
            endpoint_name, len(scenes_needing_match),
        )
        created = 0

        for batch_start in range(0, len(scenes_needing_match), BATCH_SIZE):
            if self.is_stop_requested():
                logger.warning(
                    "[%s] Stop requested after %d/%d scenes scanned",
                    endpoint_name, batch_start, len(scenes_needing_match),
                )
                break

            batch = scenes_needing_match[batch_start:batch_start + BATCH_SIZE]
            fp_sets = [item["fingerprints"] for item in batch]

            results = await stashbox.find_scenes_by_fingerprints(fp_sets)

            for i, matches in enumerate(results):
                item = batch[i]
                scene = item["scene"]
                local_fps = item["fingerprints"]
                local_duration = item["duration"]
                is_ambiguous = len(matches) > 1
                local_hashes_norm = {
                    str(fp.get("hash", "")).strip().lower()
                    for fp in local_fps
                    if fp.get("hash")
                }

                for match in matches:
                    # Build composite target_id for pair-based dismissal
                    target_id = f"{scene['id']}|{endpoint}|{match['id']}"

                    if self.is_dismissed("scene", target_id):
                        continue

                    # Find which local fingerprints matched this stash-box scene
                    matching_fps = [
                        fp for fp in match.get("fingerprints", [])
                        if str(fp.get("hash", "")).strip().lower() in local_hashes_norm
                    ]

                    # Defensive filter: if stash-box returned a candidate but no
                    # overlapping fingerprints remain after normalization, skip it.
                    # This avoids suggestions with effectively no shared signal.
                    if not matching_fps:
                        continue

                    score_result = score_match(
                        matching_fingerprints=matching_fps,
                        total_local_fingerprints=len(local_fps),
                        local_duration=local_duration or 0,
                    )

                    high_conf = (
                        not is_ambiguous
                        and is_high_confidence(
                            score_result["match_count"],
                            score_result["match_percentage"],
                            min_count=min_count,
                            min_percentage=min_percentage,
                        )
                    )

                    performer_links = []
                    for p in (match.get("performers") or []):
                        perf = p.get("performer")
                        if not perf:
                            continue
                        performer_links.append({
                            "id": str(perf.get("id")) if perf.get("id") is not None else None,
                            "name": perf.get("name"),
                        })
                    performers = [p["name"] for p in performer_links if p.get("name")]
                    studio = match.get("studio")
                    images = match.get("images") or []

                    details = {
                        "local_scene_id": scene["id"],
                        "local_scene_title": scene.get("title") or f"Scene {scene['id']}",
                        "endpoint": endpoint,
                        "endpoint_name": endpoint_name,
                        "stashbox_scene_id": match["id"],
                        "stashbox_scene_title": match.get("title"),
                        "stashbox_studio": studio.get("name") if studio else None,
                        "stashbox_studio_id": str(studio.get("id")) if studio and studio.get("id") is not None else None,
                        "stashbox_performers": performers,
                        "stashbox_performer_links": performer_links,
                        # release_date is the canonical StashBox field; date kept as legacy fallback
                        "stashbox_date": match.get("release_date") or match.get("date"),
                        "stashbox_cover_url": images[0]["url"] if images else None,
                        "matching_fingerprints": matching_fps,
                        "total_local_fingerprints": len(local_fps),
                        "match_count": score_result["match_count"],
                        "match_percentage": score_result["match_percentage"],
                        "has_exact_hash": score_result["has_exact_hash"],
                        "duration_local": local_duration,
                        "duration_remote": match.get("duration"),
                        "duration_agreement": score_result["duration_agreement"],
                        "duration_diff": score_result["duration_diff"],
                        "total_submissions": score_result["total_submissions"],
                        "high_confidence": high_conf,
                    }

                    confidence = score_result["match_percentage"] / 100.0

                    rec_id = self.create_recommendation(
                        target_type="scene",
                        target_id=target_id,
                        details=details,
                        confidence=confidence,
                    )
                    if rec_id:
                        created += 1

            batch_end = min(batch_start + BATCH_SIZE, len(scenes_needing_match))
            logger.warning(
                "[%s] Processed %d/%d scenes, %d matches found",
                endpoint_name, batch_end, len(scenes_needing_match), created,
            )
            self.update_progress(batch_end, created)

        processed = len(scenes_needing_match)

        logger.warning(
            "[%s] Complete: %d matches found from %d scenes",
            endpoint_name, created, len(scenes_needing_match),
        )

        if latest_updated:
            self.rec_db.set_watermark(watermark_key, last_stash_updated_at=latest_updated)

        return processed, created

    def _get_setting(self, key: str, default):
        """Read a user setting with fallback."""
        val = self.rec_db.get_user_setting(key)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

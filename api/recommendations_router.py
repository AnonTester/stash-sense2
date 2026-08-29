"""
Recommendations API Router

Endpoints for managing recommendations, running analysis, and configuration.
"""

from copy import deepcopy
import json
import logging
from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

import face_config
from recommendations_db import Recommendation, RecommendationsDB
from stash_client_unified import StashClientUnified
from analyzers import DuplicatePerformerAnalyzer, DuplicateSceneFilesAnalyzer, DuplicateScenesAnalyzer, UpstreamPerformerAnalyzer, UpstreamTagAnalyzer, UpstreamStudioAnalyzer, UpstreamSceneAnalyzer
from analyzers.scene_fingerprint_match import SceneFingerprintMatchAnalyzer
from analyzers.scene_face_match import SceneFaceMatchAnalyzer

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Global instances (initialized by main app)
rec_db: Optional[RecommendationsDB] = None
stash_client: Optional[StashClientUnified] = None


def init_recommendations(db_path: str, stash_url: str, stash_api_key: str):
    """Initialize recommendations database and stash client."""
    global rec_db, stash_client
    rec_db = RecommendationsDB(db_path)
    # Clean up any analysis runs left as 'running' from a previous sidecar session
    stale = rec_db.fail_stale_analysis_runs()
    if stale:
        logger.warning("Marked %d stale analysis run(s) as failed", stale)
    # Clean up orphaned candidates from broken runs that passed run_id=None
    orphans = rec_db.clear_orphaned_candidates()
    if orphans:
        logger.warning("Cleaned up %d orphaned duplicate candidates (NULL run_id)", orphans)
    if stash_url:
        stash_client = StashClientUnified(stash_url, stash_api_key)


def get_rec_db() -> RecommendationsDB:
    if rec_db is None:
        raise HTTPException(status_code=503, detail="Recommendations database not initialized")
    return rec_db


def get_stash_client() -> StashClientUnified:
    if stash_client is None:
        raise HTTPException(status_code=503, detail="Stash connection not configured. Set STASH_URL env var.")
    return stash_client


# --- Entity name cache for alias dedup ---
_entity_name_cache: dict[str, set[str]] = {}
_entity_name_cache_loaded: dict[str, bool] = {}


async def _get_entity_names(entity_type: str) -> set[str]:
    """Get cached set of all entity primary names (lowercased) for cross-entity dedup."""
    if not _entity_name_cache_loaded.get(entity_type):
        stash = get_stash_client()
        if entity_type == "tags":
            all_entities = await stash.get_all_tags()
        elif entity_type == "performers":
            all_entities = await stash.get_all_performers()
        else:
            return set()
        _entity_name_cache[entity_type] = {
            e["name"].lower() for e in all_entities if e.get("name")
        }
        _entity_name_cache_loaded[entity_type] = True
    return _entity_name_cache[entity_type]


def _invalidate_entity_name_cache(entity_type: str):
    """Invalidate cache after an entity is updated (name may have changed)."""
    _entity_name_cache_loaded[entity_type] = False


def save_scene_fingerprint(
    scene_id: int,
    frames_analyzed: int,
    persons: list,
    db_version: Optional[str] = None,
    used_sprite: bool = False,
) -> tuple[Optional[int], Optional[str]]:
    """
    Persist a scene fingerprint to the database -- both the coverage/status
    row (scene_fingerprints) and the full per-person candidate match list
    (scene_fingerprint_matches, every entry of each person's all_matches,
    not just the best one). Storing the full list is what lets recommendation
    generation and the live Identify button be served from this stored data
    instead of re-running detect+embed+match themselves.

    Args:
        scene_id: Stash scene ID
        frames_analyzed: Number of frames analyzed
        persons: list[PersonResult] (identification_router.py) -- the
            already-matched persons for this scene.
        db_version: Face recognition DB version
        used_sprite: Whether this identify included sprite-tile detection
            (request.use_sprite) -- persisted so a later bulk refresh can
            tell whether it's safe to skip sprites for this scene without
            regressing coverage. See fingerprint_generator.py's
            use_sprite_by_scene and scene_face_match.py's sprite top-up.

    Returns:
        Tuple of (fingerprint_id, error_message). On success, error is None.
        On failure, fingerprint_id is None and error contains the message.
    """
    from scene_matcher import match_universal_id

    if rec_db is None:
        return None, "Recommendations database not initialized"

    try:
        # Matches pre-existing semantics: total_faces is the sum of
        # frame_count across persons that *have* a best match, not every
        # detected person -- get_fingerprints_with_faces' proportion
        # calculation (face_count / total_faces) depends on this same
        # denominator.
        total_faces = sum(p.frame_count for p in persons if p.best_match)

        fingerprint_id = rec_db.create_scene_fingerprint(
            stash_scene_id=scene_id,
            total_faces=total_faces,
            frames_analyzed=frames_analyzed,
            fingerprint_status="complete",
            db_version=db_version,
            used_sprite=used_sprite,
        )

        match_rows = []
        for person in persons:
            best_uid = match_universal_id(person.best_match) if person.best_match else None
            for rank, match in enumerate(person.all_matches):
                universal_id = match_universal_id(match)
                if not universal_id:
                    continue
                match_rows.append({
                    "person_id": person.person_id,
                    "frame_count": person.frame_count,
                    "match_rank": rank,
                    "is_best_match": universal_id == best_uid,
                    "universal_id": universal_id,
                    "stashdb_id": match.stashdb_id,
                    "name": match.name,
                    "confidence": match.confidence,
                    "distance": match.distance,
                    "country": match.country,
                    "image_url": match.image_url,
                    "endpoint": match.endpoint,
                    "already_tagged": match.already_tagged,
                    "local_performer_id": match.local_performer_id,
                    "source": match.source,
                    "catalogue_url": match.catalogue_url,
                    "profile_url": match.profile_url,
                    "top_timestamps_sec": match.top_timestamps_sec,
                })
        rec_db.replace_fingerprint_matches(fingerprint_id, match_rows)

        return fingerprint_id, None
    except Exception as e:
        error_msg = str(e)
        logger.error("save_scene_fingerprint failed: %s", error_msg, exc_info=True)
        return None, error_msg


# ==================== Scene signal cache ====================
#
# Caches the DB-independent, expensive part of scene analysis (frame
# extraction + face detection+embedding) separately from
# scene_fingerprint_matches' match results, so a performer-database version
# bump can redo just the cheap matching/re-ranking step instead of the
# whole pipeline. See identification_router.py's /identify/scene.

def get_scene_signal_cache(scene_id: int) -> Optional[dict]:
    """Cached detection params for a scene, or None if this scene has
    never been analyzed."""
    if rec_db is None:
        return None
    return rec_db.get_scene_signal_cache(scene_id)


def is_scene_cache_compatible(
    cache_meta: dict, num_frames: int, min_face_size: int, min_face_confidence: float,
    start_offset_pct: float, end_offset_pct: float,
) -> bool:
    """Detection-affecting params must match for a cache to be reusable —
    these determine which frames got sampled and which faces got detected
    in the first place. Matching-time-only params (max_distance/top_k/
    cluster_threshold/matching_mode) are deliberately not checked here:
    they only affect how cached data gets matched/clustered, and the fast
    path always redoes that step fresh against the current request/DB
    regardless, so different values there don't invalidate the cache.
    """
    return (
        cache_meta["num_frames"] == num_frames
        and cache_meta["min_face_size"] == min_face_size
        and abs(cache_meta["min_face_confidence"] - min_face_confidence) < 1e-6
        and abs(cache_meta["start_offset_pct"] - start_offset_pct) < 1e-6
        and abs(cache_meta["end_offset_pct"] - end_offset_pct) < 1e-6
    )


def save_scene_signal_cache(
    scene_id: int, *, num_frames: int, min_face_size: int, min_face_confidence: float,
    start_offset_pct: float, end_offset_pct: float, frames_analyzed: int,
) -> None:
    if rec_db is None:
        return
    rec_db.save_scene_signal_cache(
        scene_id, num_frames=num_frames, min_face_size=min_face_size,
        min_face_confidence=min_face_confidence, start_offset_pct=start_offset_pct,
        end_offset_pct=end_offset_pct, frames_analyzed=frames_analyzed,
    )


def save_face_signal_cache(scene_id: int, faces: list[dict], *, is_sprite: bool = False) -> None:
    if rec_db is None:
        return
    rec_db.replace_face_embeddings(scene_id, faces, is_sprite=is_sprite)


def load_face_signal_cache(scene_id: int, *, is_sprite: Optional[bool] = None) -> list[dict]:
    if rec_db is None:
        return []
    return rec_db.get_face_embeddings(scene_id, is_sprite=is_sprite)


def is_sprite_cache_checked(scene_id: int) -> bool:
    if rec_db is None:
        return False
    return rec_db.is_sprite_cache_checked(scene_id)


def mark_sprite_cache_checked(scene_id: int) -> None:
    if rec_db is None:
        return
    rec_db.mark_sprite_cache_checked(scene_id)


def save_image_fingerprint(
    image_id: str,
    gallery_id: Optional[str],
    faces: list,
    image_shape: tuple[int, int],
    db_version: Optional[str] = None,
) -> tuple[Optional[int], Optional[str]]:
    """Save image identification results as a fingerprint."""
    if rec_db is None:
        return None, "Recommendations database not initialized"

    try:
        img_h, img_w = image_shape

        fp_id = rec_db.create_image_fingerprint(
            stash_image_id=image_id,
            gallery_id=gallery_id,
            faces_detected=len(faces),
            db_version=db_version,
        )

        # Clear old face data
        rec_db.delete_image_fingerprint_faces(image_id)

        # Save each detected face's best match
        for result in faces:
            if result.matches:
                best = result.matches[0]
                bbox = result.face.bbox  # dict with x, y, w, h in pixels
                rec_db.add_image_fingerprint_face(
                    stash_image_id=image_id,
                    performer_id=best.stashdb_id,
                    confidence=max(0.0, min(1.0, 1.0 - best.combined_score)),
                    distance=best.combined_score,
                    bbox_x=bbox["x"] / img_w if img_w > 0 else 0,
                    bbox_y=bbox["y"] / img_h if img_h > 0 else 0,
                    bbox_w=bbox["w"] / img_w if img_w > 0 else 0,
                    bbox_h=bbox["h"] / img_h if img_h > 0 else 0,
                )

        return fp_id, None
    except Exception as e:
        error_msg = str(e)
        logger.error("save_image_fingerprint failed: %s", error_msg, exc_info=True)
        return None, error_msg


# ==================== Pydantic Models ====================

class RecommendationResponse(BaseModel):
    """A single recommendation."""
    id: int
    type: str
    status: str
    target_type: str
    target_id: str
    details: dict
    confidence: Optional[float]
    created_at: str
    updated_at: str


class RecommendationListResponse(BaseModel):
    """List of recommendations."""
    recommendations: list[RecommendationResponse]
    total: int


class RecommendationCountsResponse(BaseModel):
    """Counts by type and status."""
    counts: dict[str, dict[str, int]]
    total_pending: int


class UserSettingRequest(BaseModel):
    """Request to set a user setting value."""
    value: Any


class ResolveRequest(BaseModel):
    """Request to resolve a recommendation."""
    action: str = Field(description="Action taken: 'merged', 'deleted', 'linked', etc.")
    details: Optional[dict] = Field(None, description="Action-specific details")


class DismissRequest(BaseModel):
    """Request to dismiss a recommendation."""
    reason: Optional[str] = Field(None, description="Why this was dismissed")


class MergeDuplicateSceneGroupRequest(BaseModel):
    """Merge selected duplicate-scene matches into the source scene."""
    source_scene_id: str
    selected_match_scene_ids: list[str]
    selected_recommendation_ids: list[int]
    unselected_recommendation_ids: list[int] = Field(default_factory=list)


class DeleteDuplicateSceneGroupRequest(BaseModel):
    """Delete the source scene for a grouped duplicate-scene review."""
    source_scene_id: str
    recommendation_ids: list[int]
    delete_file: bool = False


class DismissDuplicateSceneGroupRequest(BaseModel):
    """Dismiss all raw duplicate-scene recommendations in a grouped review."""
    recommendation_ids: list[int]
    reason: Optional[str] = None


class DeleteDuplicateSceneMatchRequest(BaseModel):
    """Delete one matched scene from a grouped duplicate-scene review."""
    source_scene_id: str
    match_scene_id: str
    recommendation_id: int
    delete_file: bool = False


class DuplicateSceneMatchCleanupItem(BaseModel):
    """One matched scene to clean up after a source->match merge."""
    recommendation_id: int
    scene_id: str


class MergeSourceIntoDuplicateSceneMatchRequest(BaseModel):
    """Merge the grouped source scene into one matched scene."""
    source_scene_id: str
    keeper_match_scene_id: str
    keeper_recommendation_id: int
    other_matches: list[DuplicateSceneMatchCleanupItem] = Field(default_factory=list)


class AnalysisRunResponse(BaseModel):
    """An analysis run."""
    id: int
    type: str
    status: str
    started_at: str
    completed_at: Optional[str]
    items_total: Optional[int]
    items_processed: Optional[int]
    recommendations_created: int
    error_message: Optional[str]


class RunAnalysisResponse(BaseModel):
    """Response when starting an analysis run."""
    run_id: int
    message: str


class StashStatusResponse(BaseModel):
    """Stash connection status."""
    connected: bool
    url: Optional[str]
    error: Optional[str]


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


class AnalysisTypeInfo(BaseModel):
    """Info about a single analysis type."""
    type: str
    enabled: bool
    description: Optional[str]


class AnalysisTypesResponse(BaseModel):
    """List of available analysis types."""
    types: list[AnalysisTypeInfo]


class FingerprintRefreshResponse(BaseModel):
    """Response for marking fingerprints for refresh."""
    marked_for_refresh: int
    scene_ids: list[int]


class FingerprintRefreshAllResponse(BaseModel):
    """Response for marking all fingerprints for refresh."""
    marked_for_refresh: int
    message: str


class FieldConfigEntry(BaseModel):
    """A single field config entry."""
    enabled: bool
    label: str


class FieldConfigResponse(BaseModel):
    """Response for field monitoring config."""
    endpoint: str
    fields: dict[str, FieldConfigEntry]


# ==================== Recommendation Endpoints ====================

def _extract_scene_fp_local_scene_id(rec) -> Optional[str]:
    """Extract local scene ID from a scene_fingerprint_match recommendation."""
    try:
        details = rec.details or {}
        scene_id = str(details.get("local_scene_id", "")).strip()
        if scene_id:
            return scene_id
    except Exception:
        pass

    try:
        target_id = str(rec.target_id or "")
        if "|" in target_id:
            scene_id = target_id.split("|", 1)[0].strip()
            return scene_id or None
    except Exception:
        pass

    return None


def _normalize_scene_id(value) -> Optional[str]:
    """Normalize a scene ID candidate to non-empty string."""
    if value is None:
        return None
    scene_id = str(value).strip()
    return scene_id or None


def _extract_scene_ids_from_recommendation(rec) -> list[str]:
    """Extract referenced local scene IDs from a recommendation."""
    details = rec.details or {}
    scene_ids: list[str] = []

    def add(value):
        sid = _normalize_scene_id(value)
        if sid:
            scene_ids.append(sid)

    rec_type = str(rec.type or "")
    if rec_type == "duplicate_scenes":
        add(details.get("scene_a_id"))
        add(details.get("scene_b_id"))
        target = str(rec.target_id or "")
        if ":" in target:
            left, right = target.split(":", 1)
            add(left)
            add(right)
    elif rec_type == "duplicate_scene_files":
        add(details.get("scene_id"))
        add(rec.target_id)
    elif rec_type == "upstream_scene_changes":
        add(details.get("scene_id"))
        add(rec.target_id)
    elif rec_type == "scene_fingerprint_match":
        add(_extract_scene_fp_local_scene_id(rec))
    elif rec_type == "scene_face_match":
        add(_extract_scene_face_match_scene_id(rec))
    elif str(rec.target_type or "") == "scene":
        add(rec.target_id)

    # Stable dedupe preserving insertion order
    deduped: list[str] = []
    seen: set[str] = set()
    for sid in scene_ids:
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(sid)
    return deduped


def _parse_duplicate_scene_confidence_percent(rec: Recommendation) -> float:
    """Normalize duplicate-scene confidence to a 0-100 percentage."""
    details = rec.details or {}
    raw = details.get("top_confidence", details.get("confidence", rec.confidence))
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 1:
        numeric *= 100.0
    return max(0.0, numeric)


def _extract_duplicate_scene_source_id(rec: Recommendation) -> Optional[str]:
    """Get the source/left-hand scene ID for a duplicate-scenes recommendation."""
    details = rec.details or {}
    source_id = _normalize_scene_id(details.get("source_scene_id"))
    if source_id:
        return source_id
    source_id = _normalize_scene_id(details.get("scene_a_id"))
    if source_id:
        return source_id
    target = str(rec.target_id or "")
    if ":" in target:
        left, _ = target.split(":", 1)
        return _normalize_scene_id(left)
    return None


def _extract_duplicate_scene_match_id(rec: Recommendation) -> Optional[str]:
    """Get the matched/right-hand scene ID for a duplicate-scenes recommendation."""
    details = rec.details or {}
    match_id = _normalize_scene_id(details.get("match_scene_id"))
    if match_id:
        return match_id
    match_id = _normalize_scene_id(details.get("scene_b_id"))
    if match_id:
        return match_id
    target = str(rec.target_id or "")
    if ":" in target:
        _, right = target.split(":", 1)
        return _normalize_scene_id(right)
    return None


def _duplicate_match_sort_key(match: dict[str, Any]) -> tuple[float, int]:
    """Sort duplicate-scene matches by confidence desc, then rec ID desc."""
    return (
        float(match.get("confidence") or 0.0),
        int(match.get("recommendation_id") or 0),
    )


def _group_duplicate_scene_recommendations(recs: list[Recommendation]) -> list[Recommendation]:
    """Group duplicate-scene recommendations by source scene and top confidence."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for rec in recs:
        source_id = _extract_duplicate_scene_source_id(rec)
        match_id = _extract_duplicate_scene_match_id(rec)
        if not source_id or not match_id:
            continue

        details = rec.details or {}
        group_key = (str(rec.status or ""), source_id)
        group = grouped.setdefault(group_key, {
            "source_scene_id": source_id,
            "source_summary": deepcopy(details.get("source_summary") or details.get("scene_a_summary") or {}),
            "matches": [],
            "top_rec": None,
            "top_confidence": -1.0,
        })

        confidence = _parse_duplicate_scene_confidence_percent(rec)
        match_entry = {
            "recommendation_id": rec.id,
            "target_id": rec.target_id,
            "match_scene_id": match_id,
            "confidence": confidence,
            "reasoning": deepcopy(details.get("reasoning") or []),
            "signal_breakdown": deepcopy(details.get("signal_breakdown") or {}),
            "source_summary": deepcopy(details.get("source_summary") or details.get("scene_a_summary") or {}),
            "match_summary": deepcopy(details.get("match_summary") or details.get("scene_b_summary") or {}),
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "resolution_action": rec.resolution_action,
        }
        group["matches"].append(match_entry)

        current_top = group["top_rec"]
        if (
            current_top is None
            or confidence > group["top_confidence"]
            or (confidence == group["top_confidence"] and rec.id > current_top.id)
        ):
            group["top_rec"] = rec
            group["top_confidence"] = confidence
            if not group["source_summary"]:
                group["source_summary"] = deepcopy(match_entry["source_summary"])

    grouped_recs: list[Recommendation] = []
    for group in grouped.values():
        matches = sorted(group["matches"], key=_duplicate_match_sort_key, reverse=True)
        if not matches:
            continue

        top_rec: Recommendation = group["top_rec"]
        top_match = matches[0]
        merged_details = deepcopy(top_rec.details or {})
        merged_details.update({
            "grouped": True,
            "source_scene_id": group["source_scene_id"],
            "scene_a_id": group["source_scene_id"],
            "scene_b_id": top_match["match_scene_id"],
            "source_summary": deepcopy(group["source_summary"]),
            "scene_a_summary": deepcopy(group["source_summary"]),
            "scene_b_summary": deepcopy(top_match.get("match_summary") or {}),
            "duplicate_matches": matches,
            "match_count": len(matches),
            "top_confidence": group["top_confidence"],
            "confidence": group["top_confidence"],
            "reasoning": deepcopy(top_match.get("reasoning") or []),
            "signal_breakdown": deepcopy(top_match.get("signal_breakdown") or {}),
        })

        grouped_recs.append(Recommendation(
            id=top_rec.id,
            type=top_rec.type,
            status=top_rec.status,
            target_type=top_rec.target_type,
            target_id=top_rec.target_id,
            details=merged_details,
            resolution_action=top_rec.resolution_action,
            resolution_details=top_rec.resolution_details,
            resolved_at=top_rec.resolved_at,
            confidence=group["top_confidence"] / 100.0,
            source_analysis_id=top_rec.source_analysis_id,
            created_at=top_rec.created_at,
            updated_at=top_rec.updated_at,
        ))

    first_status = grouped_recs[0].status if grouped_recs else None
    if first_status in ("dismissed", "resolved"):
        grouped_recs.sort(
            key=lambda rec: (
                rec.resolved_at or rec.updated_at or "",
                int(rec.id or 0),
            ),
            reverse=True,
        )
    else:
        grouped_recs.sort(
            key=lambda rec: (
                _parse_duplicate_scene_confidence_percent(rec),
                int(rec.id or 0),
            ),
            reverse=True,
        )
    return grouped_recs


def _extract_scene_face_match_scene_id(rec) -> Optional[str]:
    """Extract local scene ID from a scene_face_match recommendation."""
    try:
        details = rec.details or {}
        scene_id = str(details.get("scene_id", "")).strip()
        if scene_id:
            return scene_id
    except Exception:
        pass

    try:
        target_id = str(rec.target_id or "")
        if "|" in target_id:
            scene_id = target_id.split("|", 1)[0].strip()
            return scene_id or None
    except Exception:
        pass

    return None


def _scene_face_match_candidate(rec: Recommendation) -> dict[str, Any]:
    """Build one candidate entry (a single scene+performer recommendation row)
    for the grouped scene_face_match view."""
    details = rec.details or {}
    return {
        "recommendation_id": rec.id,
        "person_id": details.get("person_id"),
        "frame_count": details.get("frame_count"),
        "is_best_match": bool(details.get("is_best_match")),
        "universal_id": details.get("universal_id"),
        "stashdb_id": details.get("stashdb_id"),
        "name": details.get("name"),
        "confidence": rec.confidence if rec.confidence is not None else details.get("confidence"),
        "distance": details.get("distance"),
        "country": details.get("country"),
        "image_url": details.get("image_url"),
        "endpoint": details.get("endpoint"),
        "local_performer_id": details.get("local_performer_id"),
        "source": details.get("source"),
        "catalogue_url": details.get("catalogue_url"),
        "profile_url": details.get("profile_url"),
        "top_timestamps_sec": details.get("top_timestamps_sec") or [],
        "status": rec.status,
        "resolution_action": rec.resolution_action,
    }


def _group_scene_face_match_recommendations(recs: list[Recommendation]) -> list[Recommendation]:
    """Group scene_face_match recommendations (one row per scene+performer
    candidate) into one synthetic Recommendation per scene, with candidates
    nested under their detected person/face-cluster. Mirrors
    _group_duplicate_scene_recommendations's shape/conventions."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for rec in recs:
        scene_id = _extract_scene_face_match_scene_id(rec)
        if not scene_id:
            continue

        details = rec.details or {}
        group_key = (str(rec.status or ""), scene_id)
        group = grouped.setdefault(group_key, {
            "scene_id": scene_id,
            "scene_title": details.get("scene_title") or f"Scene {scene_id}",
            "candidates": [],
            "top_rec": None,
            "top_confidence": -1.0,
        })

        candidate = _scene_face_match_candidate(rec)
        group["candidates"].append(candidate)

        confidence = float(candidate.get("confidence") or 0.0)
        current_top = group["top_rec"]
        if (
            current_top is None
            or confidence > group["top_confidence"]
            or (confidence == group["top_confidence"] and rec.id > current_top.id)
        ):
            group["top_rec"] = rec
            group["top_confidence"] = confidence

    grouped_recs: list[Recommendation] = []
    for group in grouped.values():
        # is_best_match first (not raw confidence) -- the analyzer picks
        # is_best_match via a frame-count-weighted score (rewards a
        # performer appearing consistently across the tracked cluster,
        # not just whoever's single best frame happens to be a hair
        # closer), so it can legitimately differ from whichever candidate
        # has the single highest confidence value. Sorting by confidence
        # alone put a higher-confidence, non-preselected candidate ahead
        # of the actually-preselected one -- confirmed live (a 59.9%
        # candidate listed first while a 59.2% candidate was the one
        # pre-checked) -- so the preselected candidate must always sort
        # first, with confidence only as the tiebreaker among the rest.
        candidates = sorted(
            group["candidates"],
            key=lambda c: (
                int(c.get("person_id") or 0),
                not c.get("is_best_match"),
                -(float(c.get("confidence") or 0.0)),
            ),
        )
        if not candidates:
            continue

        persons: dict[int, dict[str, Any]] = {}
        for c in candidates:
            person_id = int(c.get("person_id") or 0)
            person = persons.setdefault(person_id, {
                "person_id": person_id,
                "frame_count": c.get("frame_count"),
                "candidates": [],
            })
            person["candidates"].append(c)

        top_rec: Recommendation = group["top_rec"]
        merged_details = deepcopy(top_rec.details or {})
        merged_details.update({
            "grouped": True,
            "scene_id": group["scene_id"],
            "scene_title": group["scene_title"],
            "persons": [persons[pid] for pid in sorted(persons.keys())],
            "candidates": candidates,
            "candidate_count": len(candidates),
            "top_confidence": group["top_confidence"],
            "confidence": group["top_confidence"],
        })

        grouped_recs.append(Recommendation(
            id=top_rec.id,
            type=top_rec.type,
            status=top_rec.status,
            target_type=top_rec.target_type,
            target_id=top_rec.target_id,
            details=merged_details,
            resolution_action=top_rec.resolution_action,
            resolution_details=top_rec.resolution_details,
            resolved_at=top_rec.resolved_at,
            confidence=group["top_confidence"],
            source_analysis_id=top_rec.source_analysis_id,
            created_at=top_rec.created_at,
            updated_at=top_rec.updated_at,
        ))

    first_status = grouped_recs[0].status if grouped_recs else None
    if first_status in ("dismissed", "resolved"):
        grouped_recs.sort(
            key=lambda rec: (
                rec.resolved_at or rec.updated_at or "",
                int(rec.id or 0),
            ),
            reverse=True,
        )
    else:
        grouped_recs.sort(
            key=lambda rec: (
                float(rec.details.get("top_confidence") or 0.0),
                int(rec.id or 0),
            ),
            reverse=True,
        )
    return grouped_recs


def _build_scene_face_match_group_for_recommendation(
    db: RecommendationsDB,
    rec: Recommendation,
) -> Recommendation:
    """Expand one scene_face_match recommendation into its grouped scene view."""
    scene_id = _extract_scene_face_match_scene_id(rec)
    if not scene_id:
        return rec

    grouped_recs = _group_scene_face_match_recommendations(
        _load_all_recommendations(
            db,
            status=rec.status,
            type="scene_face_match",
            target_type=rec.target_type,
        )
    )
    for grouped_rec in grouped_recs:
        if _extract_scene_face_match_scene_id(grouped_rec) == scene_id:
            return grouped_rec
    return rec


def _load_all_recommendations(
    db: RecommendationsDB,
    *,
    status: Optional[str] = None,
    type: Optional[str] = None,
    target_type: Optional[str] = None,
    page_size: int = 500,
) -> list[Recommendation]:
    """Load all recommendations matching a filter in stable paginated batches."""
    all_recs: list[Recommendation] = []
    offset = 0
    while True:
        batch = db.get_recommendations(
            status=status,
            type=type,
            target_type=target_type,
            limit=page_size,
            offset=offset,
        )
        if not batch:
            break
        all_recs.extend(batch)
        if len(batch) < page_size:
            break
        offset += len(batch)
    return all_recs


def _build_duplicate_scene_group_for_recommendation(
    db: RecommendationsDB,
    rec: Recommendation,
) -> Recommendation:
    """Expand one duplicate-scene recommendation into its grouped source-scene view."""
    source_scene_id = _extract_duplicate_scene_source_id(rec)
    if not source_scene_id:
        return rec

    grouped_recs = _group_duplicate_scene_recommendations(
        _load_all_recommendations(
            db,
            status=rec.status,
            type="duplicate_scenes",
            target_type=rec.target_type,
        )
    )
    for grouped_rec in grouped_recs:
        if _extract_duplicate_scene_source_id(grouped_rec) == source_scene_id:
            return grouped_rec
    return rec


async def _validate_and_prune_missing_scene_recommendation(rec, stash, db) -> list[str]:
    """Delete recommendation if any referenced scene no longer exists.

    Returns list of missing scene IDs. Empty list means all referenced scenes exist
    (or recommendation has no scene references).
    """
    scene_ids = _extract_scene_ids_from_recommendation(rec)
    if not scene_ids:
        return []

    missing_scene_ids: list[str] = []
    for scene_id in scene_ids:
        try:
            scene = await stash.get_scene_by_id(scene_id)
        except Exception as exc:
            logger.warning(
                "Failed scene existence check for recommendation %s scene %s: %s",
                rec.id,
                scene_id,
                exc,
            )
            continue
        if not scene:
            missing_scene_ids.append(scene_id)

    if missing_scene_ids:
        db.delete_recommendation(rec.id)
        logger.info(
            "Deleted stale recommendation %s (%s): missing scene(s) %s",
            rec.id,
            rec.type,
            ", ".join(missing_scene_ids),
        )

    return missing_scene_ids


async def _validate_and_prune_duplicate_scene_group_recommendations(
    rec: Recommendation,
    stash,
    db: RecommendationsDB,
) -> tuple[set[int], list[str]]:
    """Prune stale duplicate-scene recommendations across the whole source-scene group."""
    source_scene_id = _extract_duplicate_scene_source_id(rec)
    if not source_scene_id:
        missing_scene_ids = await _validate_and_prune_missing_scene_recommendation(rec, stash, db)
        return ({rec.id} if missing_scene_ids else set(), missing_scene_ids)

    group_recs = [
        candidate
        for candidate in _load_all_recommendations(
            db,
            status=rec.status,
            type="duplicate_scenes",
            target_type=rec.target_type,
        )
        if _extract_duplicate_scene_source_id(candidate) == source_scene_id
    ]
    if not group_recs:
        return set(), []

    scene_exists: dict[str, Optional[bool]] = {}
    deleted_rec_ids: set[int] = set()
    missing_scene_ids: list[str] = []
    seen_missing_scene_ids: set[str] = set()

    for candidate in group_recs:
        candidate_missing_scene_ids: list[str] = []
        for scene_id in _extract_scene_ids_from_recommendation(candidate):
            if scene_id not in scene_exists:
                try:
                    scene = await stash.get_scene_by_id(scene_id)
                except Exception as exc:
                    logger.warning(
                        "Failed scene existence check for recommendation %s scene %s: %s",
                        candidate.id,
                        scene_id,
                        exc,
                    )
                    scene_exists[scene_id] = None
                else:
                    scene_exists[scene_id] = bool(scene)

            if scene_exists.get(scene_id) is False:
                candidate_missing_scene_ids.append(scene_id)
                if scene_id not in seen_missing_scene_ids:
                    seen_missing_scene_ids.add(scene_id)
                    missing_scene_ids.append(scene_id)

        if candidate_missing_scene_ids:
            db.delete_recommendation(candidate.id)
            deleted_rec_ids.add(candidate.id)
            logger.info(
                "Deleted stale duplicate-scene recommendation %s: missing scene(s) %s",
                candidate.id,
                ", ".join(candidate_missing_scene_ids),
            )

    return deleted_rec_ids, missing_scene_ids


async def _validate_and_prune_scene_face_match_group_recommendations(
    rec: Recommendation,
    stash,
    db: RecommendationsDB,
) -> tuple[set[int], list[str]]:
    """Prune stale scene_face_match recommendations across the whole
    per-scene group, not just the single row `rec` was loaded from.

    Every raw scene_face_match row in a group (one per detected
    person/candidate) references the same scene_id -- `get_recommendation`
    is called with the group's synthetic representative id
    (`_group_scene_face_match_recommendations`'s `top_rec`), so deleting
    only that one row leaves its siblings behind. Those survivors then
    get re-grouped into a "new" entry for the same (now-nonexistent)
    scene on the next list load, which is exactly the bug this mirrors
    the duplicate-scenes group prune above to fix.
    """
    scene_id = _extract_scene_face_match_scene_id(rec)
    if not scene_id:
        return set(), []

    group_recs = [
        candidate
        for candidate in _load_all_recommendations(
            db,
            status=rec.status,
            type="scene_face_match",
            target_type=rec.target_type,
        )
        if _extract_scene_face_match_scene_id(candidate) == scene_id
    ]
    if not group_recs:
        return set(), []

    try:
        scene = await stash.get_scene_by_id(scene_id)
    except Exception as exc:
        logger.warning(
            "Failed scene existence check for recommendation %s scene %s: %s",
            rec.id,
            scene_id,
            exc,
        )
        return set(), []

    if scene:
        return set(), []

    deleted_rec_ids: set[int] = set()
    for candidate in group_recs:
        db.delete_recommendation(candidate.id)
        deleted_rec_ids.add(candidate.id)
    logger.info(
        "Deleted %d stale scene_face_match recommendation(s): missing scene %s",
        len(deleted_rec_ids),
        scene_id,
    )

    return deleted_rec_ids, [scene_id]


@router.get("/settings")
async def get_all_user_settings():
    """Get all user settings stored in the recommendations DB."""
    db = get_rec_db()
    return {"settings": db.get_all_user_settings()}


@router.get("/settings/{key}")
async def get_user_setting(key: str):
    """Get a single user setting by key."""
    db = get_rec_db()
    value = db.get_user_setting(key)
    return {"key": key, "value": value}


@router.post("/settings/{key}")
async def set_user_setting(key: str, request: UserSettingRequest):
    """Set a user setting value."""
    db = get_rec_db()
    db.set_user_setting(key, request.value)
    return {"key": key, "value": request.value}


@router.get("", response_model=RecommendationListResponse)
async def list_recommendations(
    status: Optional[str] = None,
    type: Optional[str] = None,
    target_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List recommendations with optional filtering."""
    db = get_rec_db()
    if type in ("duplicate_scenes", "scene_face_match"):
        group_fn = (
            _group_duplicate_scene_recommendations if type == "duplicate_scenes"
            else _group_scene_face_match_recommendations
        )
        grouped = group_fn(
            _load_all_recommendations(
                db,
                status=status,
                type=type,
                target_type=target_type,
            )
        )
        paged = grouped[offset:offset + limit]
        return RecommendationListResponse(
            recommendations=[
                RecommendationResponse(
                    id=r.id,
                    type=r.type,
                    status=r.status,
                    target_type=r.target_type,
                    target_id=r.target_id,
                    details=r.details,
                    confidence=r.confidence,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                )
                for r in paged
            ],
            total=len(grouped),
        )

    recs = db.get_recommendations(
        status=status,
        type=type,
        target_type=target_type,
        limit=limit,
        offset=offset,
    )

    # Scene Stash-Box Tagger only applies to scenes without any stash_id.
    # If stale pending recommendations exist for now-linked/deleted scenes, auto-remove
    # them as part of list rendering so users don't see invalid entries.
    if status == "pending" and recs and (type is None or type == "scene_fingerprint_match"):
        stash = get_stash_client()
        scene_ids = {
            _extract_scene_fp_local_scene_id(r) or ""
            for r in recs
            if r.type == "scene_fingerprint_match"
        }
        scene_ids.discard("")
        cleaned = 0
        for scene_id in scene_ids:
            try:
                scene = await stash.get_scene_by_id(scene_id)
            except Exception:
                continue
            if not scene:
                cleaned += db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)
                continue
            if scene.get("stash_ids"):
                cleaned += db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)
        if cleaned:
            recs = db.get_recommendations(
                status=status,
                type=type,
                target_type=target_type,
                limit=limit,
                offset=offset,
            )

    return RecommendationListResponse(
        recommendations=[
            RecommendationResponse(
                id=r.id,
                type=r.type,
                status=r.status,
                target_type=r.target_type,
                target_id=r.target_id,
                details=r.details,
                confidence=r.confidence,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in recs
        ],
        total=db.count_recommendations(status=status, type=type, target_type=target_type),
    )


@router.get("/counts", response_model=RecommendationCountsResponse)
async def get_recommendation_counts():
    """Get recommendation counts by type and status."""
    db = get_rec_db()
    counts = db.get_recommendation_counts()
    raw_duplicate_recs = _load_all_recommendations(db, type="duplicate_scenes", target_type="scene")
    if raw_duplicate_recs:
        grouped_duplicate_counts: dict[str, int] = {}
        for rec in _group_duplicate_scene_recommendations(raw_duplicate_recs):
            grouped_duplicate_counts[rec.status] = grouped_duplicate_counts.get(rec.status, 0) + 1
        counts["duplicate_scenes"] = grouped_duplicate_counts
    raw_sfm_recs = _load_all_recommendations(db, type="scene_face_match", target_type="scene")
    if raw_sfm_recs:
        grouped_sfm_counts: dict[str, int] = {}
        for rec in _group_scene_face_match_recommendations(raw_sfm_recs):
            grouped_sfm_counts[rec.status] = grouped_sfm_counts.get(rec.status, 0) + 1
        counts["scene_face_match"] = grouped_sfm_counts
    total_pending = sum(
        type_counts.get("pending", 0)
        for type_counts in counts.values()
    )
    return RecommendationCountsResponse(counts=counts, total_pending=total_pending)


@router.get("/{rec_id}", response_model=RecommendationResponse)
async def get_recommendation(rec_id: int):
    """Get a single recommendation."""
    db = get_rec_db()
    rec = db.get_recommendation(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    stash = get_stash_client()
    if rec.type == "duplicate_scenes":
        deleted_rec_ids, missing_scene_ids = await _validate_and_prune_duplicate_scene_group_recommendations(rec, stash, db)
        if rec_id in deleted_rec_ids:
            missing_list = ", ".join(missing_scene_ids)
            raise HTTPException(
                status_code=404,
                detail=(
                    "Recommendation removed because referenced scene no longer exists: "
                    f"{missing_list}"
                ),
            )
        rec = db.get_recommendation(rec_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Recommendation not found")
    elif rec.type == "scene_face_match":
        deleted_rec_ids, missing_scene_ids = await _validate_and_prune_scene_face_match_group_recommendations(rec, stash, db)
        if rec_id in deleted_rec_ids:
            missing_list = ", ".join(missing_scene_ids)
            raise HTTPException(
                status_code=404,
                detail=(
                    "Recommendation removed because referenced scene no longer exists: "
                    f"{missing_list}"
                ),
            )
        rec = db.get_recommendation(rec_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Recommendation not found")
    else:
        missing_scene_ids = await _validate_and_prune_missing_scene_recommendation(rec, stash, db)
        if missing_scene_ids:
            missing_list = ", ".join(missing_scene_ids)
            raise HTTPException(
                status_code=404,
                detail=(
                    "Recommendation removed because referenced scene no longer exists: "
                    f"{missing_list}"
                ),
            )

    if rec.type == "duplicate_scenes":
        rec = _build_duplicate_scene_group_for_recommendation(db, rec)
        if not rec.details.get("duplicate_matches"):
            raise HTTPException(status_code=404, detail="Recommendation not found")
    elif rec.type == "scene_face_match":
        rec = _build_scene_face_match_group_for_recommendation(db, rec)
        if not rec.details.get("candidates"):
            raise HTTPException(status_code=404, detail="Recommendation not found")

    return RecommendationResponse(
        id=rec.id,
        type=rec.type,
        status=rec.status,
        target_type=rec.target_type,
        target_id=rec.target_id,
        details=rec.details,
        confidence=rec.confidence,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )


@router.post("/{rec_id}/resolve", response_model=SuccessResponse)
async def resolve_recommendation(rec_id: int, request: ResolveRequest):
    """Mark a recommendation as resolved."""
    logger.debug("Action: resolve rec_id=%s action=%s", rec_id, request.action)
    db = get_rec_db()
    success = db.resolve_recommendation(rec_id, request.action, request.details)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True}


@router.post("/{rec_id}/dismiss", response_model=SuccessResponse)
async def dismiss_recommendation(rec_id: int, request: DismissRequest = None):
    """Dismiss a recommendation (won't be re-created)."""
    logger.debug("Action: dismiss rec_id=%s", rec_id)
    db = get_rec_db()
    reason = request.reason if request else None
    success = db.dismiss_recommendation(rec_id, reason)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True}


# ==================== Analysis Endpoints ====================

ANALYZERS = {
    "duplicate_performer": DuplicatePerformerAnalyzer,
    "duplicate_scene_files": DuplicateSceneFilesAnalyzer,
    "duplicate_scenes": DuplicateScenesAnalyzer,
    "upstream_performer_changes": UpstreamPerformerAnalyzer,
    "upstream_tag_changes": UpstreamTagAnalyzer,
    "upstream_studio_changes": UpstreamStudioAnalyzer,
    "upstream_scene_changes": UpstreamSceneAnalyzer,
    "scene_fingerprint_match": SceneFingerprintMatchAnalyzer,
    "scene_face_match": SceneFaceMatchAnalyzer,
}

FORCE_FULL_SCAN_USER_ANALYSIS_TYPES = {"scene_fingerprint_match", "upstream_scene_changes"}


@router.get("/analysis/types", response_model=AnalysisTypesResponse)
async def list_analysis_types():
    """List available analysis types."""
    db = get_rec_db()
    types = []
    for type_name in ANALYZERS.keys():
        settings = db.get_settings(type_name)
        types.append({
            "type": type_name,
            "enabled": settings.enabled if settings else True,
            "description": ANALYZERS[type_name].__doc__.strip().split("\n")[0] if ANALYZERS[type_name].__doc__ else None,
        })
    return {"types": types}


@router.post("/analysis/{type}/run", response_model=RunAnalysisResponse)
async def run_analysis(type: str, full: bool = False):
    """Start an analysis. Now delegates to the job queue."""
    if type not in ANALYZERS:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {type}")

    from queue_router import _queue_manager
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")

    from job_models import JobPriority
    from jobs.analysis_jobs import FULL_RUN_CURSOR
    force_full = full or (type in FORCE_FULL_SCAN_USER_ANALYSIS_TYPES)
    job_id = _queue_manager.submit(
        type_id=type,
        triggered_by="user",
        priority=JobPriority.HIGH,
        cursor=FULL_RUN_CURSOR if force_full else None,
    )
    if job_id is None:
        raise HTTPException(status_code=409, detail=f"Analysis '{type}' is already running or queued")

    return RunAnalysisResponse(run_id=job_id, message=f"Analysis '{type}' queued (job #{job_id})")


@router.get("/analysis/runs", response_model=list[AnalysisRunResponse])
async def list_analysis_runs(type: Optional[str] = None, limit: int = 20):
    """List recent analysis runs."""
    db = get_rec_db()
    runs = db.get_recent_analysis_runs(type=type, limit=limit)
    return [
        AnalysisRunResponse(
            id=r.id,
            type=r.type,
            status=r.status,
            started_at=r.started_at,
            completed_at=r.completed_at,
            items_total=r.items_total,
            items_processed=r.items_processed,
            recommendations_created=r.recommendations_created,
            error_message=r.error_message,
        )
        for r in runs
    ]


@router.get("/analysis/runs/{run_id}", response_model=AnalysisRunResponse)
async def get_analysis_run(run_id: int):
    """Get status of an analysis run."""
    db = get_rec_db()
    run = db.get_analysis_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return AnalysisRunResponse(
        id=run.id,
        type=run.type,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        items_total=run.items_total,
        items_processed=run.items_processed,
        recommendations_created=run.recommendations_created,
        error_message=run.error_message,
    )


# ==================== Stash Connection ====================

@router.get("/stash/status", response_model=StashStatusResponse)
async def stash_status():
    """Check Stash connection status."""
    if stash_client is None:
        return StashStatusResponse(
            connected=False,
            url=None,
            error="STASH_URL environment variable not set",
        )

    try:
        await stash_client.test_connection()
        return StashStatusResponse(
            connected=True,
            url=stash_client.base_url,
            error=None,
        )
    except Exception as e:
        return StashStatusResponse(
            connected=False,
            url=stash_client.base_url,
            error=str(e),
        )


# ==================== Actions ====================

class MergePerformersRequest(BaseModel):
    """Request to merge duplicate performers."""
    destination_id: str
    source_ids: list[str]


@router.post("/actions/merge-performers")
async def merge_performers(request: MergePerformersRequest):
    """Execute a performer merge. Deletes source performers after merge."""
    logger.debug("Action: merge-performers source_ids=%s destination_id=%s", request.source_ids, request.destination_id)
    stash = get_stash_client()
    try:
        result = await stash.merge_performers(request.source_ids, request.destination_id)
        # Delete orphaned source performers after merge
        for source_id in request.source_ids:
            try:
                await stash.destroy_performer(source_id)
            except Exception as del_err:
                logger.warning(f"Failed to delete performer {source_id} after merge: {del_err}")
        return {"success": True, "merged_into": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteSceneFilesRequest(BaseModel):
    """Request to delete duplicate scene files."""
    scene_id: str
    file_ids_to_delete: list[str]
    keep_file_id: str
    all_file_ids: list[str]


@router.post("/actions/delete-scene-files")
async def delete_scene_files(request: DeleteSceneFilesRequest):
    """Delete files from a scene, keeping the specified file."""
    logger.debug("Action: delete-scene-files scene_id=%s keep_file_id=%s delete_file_ids=%s", request.scene_id, request.keep_file_id, request.file_ids_to_delete)
    stash = get_stash_client()
    try:
        result = await stash.delete_scene_files(
            scene_id=request.scene_id,
            file_ids_to_delete=request.file_ids_to_delete,
            keep_file_id=request.keep_file_id,
            all_file_ids=request.all_file_ids,
        )
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MergeScenesRequest(BaseModel):
    """Request to merge duplicate scenes."""
    destination_id: str
    source_ids: list[str]


@router.post("/actions/merge-scenes")
async def merge_scenes(request: MergeScenesRequest):
    """Execute a scene merge via Stash's sceneMerge mutation."""
    logger.debug("Action: merge-scenes source_ids=%s destination_id=%s", request.source_ids, request.destination_id)
    stash = get_stash_client()
    db = get_rec_db()
    try:
        result = await stash.merge_scenes(request.source_ids, request.destination_id)
        scene_ids_to_cleanup = [str(request.destination_id)] + [str(sid) for sid in (request.source_ids or [])]
        for scene_id in scene_ids_to_cleanup:
            db.delete_pending_duplicate_scene_recommendations_for_scene(scene_id=scene_id)
            db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/merge-duplicate-scene-group")
async def merge_duplicate_scene_group(request: MergeDuplicateSceneGroupRequest):
    """Merge selected matched scenes into the grouped source scene."""
    logger.debug("Action: merge-duplicate-scene-group source_scene_id=%s match_scene_ids=%s rec_ids=%s", request.source_scene_id, request.selected_match_scene_ids, request.selected_recommendation_ids)
    if not request.source_scene_id or not request.selected_match_scene_ids:
        raise HTTPException(status_code=400, detail="source_scene_id and selected_match_scene_ids are required")

    stash = get_stash_client()
    db = get_rec_db()
    try:
        result = await stash.merge_scenes(request.selected_match_scene_ids, request.source_scene_id)

        merged_details = {
            "keeper_id": request.source_scene_id,
            "source_ids": request.selected_match_scene_ids,
        }
        for rec_id in request.selected_recommendation_ids:
            db.resolve_recommendation(rec_id, action="merged", details=merged_details)

        for rec_id in request.unselected_recommendation_ids:
            db.resolve_recommendation(
                rec_id,
                action="not_duplicate",
                details={"source_scene_id": request.source_scene_id},
            )
            db.add_recommendation_target_dismissal(
                rec_id,
                reason=f"Marked not duplicate while merging into scene {request.source_scene_id}",
            )

        scene_ids_to_cleanup = [str(request.source_scene_id)] + [str(sid) for sid in request.selected_match_scene_ids]
        for scene_id in scene_ids_to_cleanup:
            db.delete_pending_duplicate_scene_recommendations_for_scene(scene_id=scene_id)
            db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)

        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteSceneRequest(BaseModel):
    """Request to delete a scene."""
    scene_id: str
    delete_file: bool = False


@router.post("/actions/delete-scene")
async def delete_scene(request: DeleteSceneRequest):
    """Delete a scene from Stash."""
    logger.debug("Action: delete-scene scene_id=%s delete_file=%s", request.scene_id, request.delete_file)
    stash = get_stash_client()
    db = get_rec_db()
    try:
        result = await stash.destroy_scene(request.scene_id, delete_file=request.delete_file)
        if result:
            scene_id = str(request.scene_id)
            db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)
            db.delete_pending_duplicate_scene_recommendations_for_scene(scene_id=scene_id)
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/delete-duplicate-scene-match")
async def delete_duplicate_scene_match(request: DeleteDuplicateSceneMatchRequest):
    """Delete one matched scene and resolve its grouped duplicate-scene recommendation."""
    logger.debug("Action: delete-duplicate-scene-match source_scene_id=%s match_scene_id=%s rec_id=%s", request.source_scene_id, request.match_scene_id, request.recommendation_id)
    if not request.source_scene_id or not request.match_scene_id:
        raise HTTPException(status_code=400, detail="source_scene_id and match_scene_id are required")

    stash = get_stash_client()
    db = get_rec_db()
    try:
        result = await stash.destroy_scene(request.match_scene_id, delete_file=request.delete_file)
        if result:
            db.resolve_recommendation(
                request.recommendation_id,
                action="deleted_match",
                details={
                    "source_scene_id": request.source_scene_id,
                    "deleted_scene_id": request.match_scene_id,
                },
            )
            scene_id = str(request.match_scene_id)
            db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)
            db.delete_pending_duplicate_scene_recommendations_for_scene(scene_id=scene_id)
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/delete-duplicate-scene-group")
async def delete_duplicate_scene_group(request: DeleteDuplicateSceneGroupRequest):
    """Delete the grouped source scene and resolve the reviewed duplicate set."""
    logger.debug("Action: delete-duplicate-scene-group source_scene_id=%s rec_ids=%s", request.source_scene_id, request.recommendation_ids)
    if not request.source_scene_id:
        raise HTTPException(status_code=400, detail="source_scene_id is required")

    stash = get_stash_client()
    db = get_rec_db()
    try:
        result = await stash.destroy_scene(request.source_scene_id, delete_file=request.delete_file)
        if result:
            for index, rec_id in enumerate(request.recommendation_ids):
                action = "deleted_source" if index == 0 else "not_duplicate_after_source_delete"
                db.resolve_recommendation(
                    rec_id,
                    action=action,
                    details={"deleted_scene_id": request.source_scene_id},
                )

            scene_id = str(request.source_scene_id)
            db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)
            db.delete_pending_duplicate_scene_recommendations_for_scene(scene_id=scene_id)
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/merge-source-into-duplicate-scene-match")
async def merge_source_into_duplicate_scene_match(request: MergeSourceIntoDuplicateSceneMatchRequest):
    """Keep one matched scene, merge the source into it, then close sibling matches."""
    logger.debug("Action: merge-source-into-duplicate-scene-match source_scene_id=%s keeper_match_scene_id=%s", request.source_scene_id, request.keeper_match_scene_id)
    if not request.source_scene_id or not request.keeper_match_scene_id:
        raise HTTPException(status_code=400, detail="source_scene_id and keeper_match_scene_id are required")

    stash = get_stash_client()
    db = get_rec_db()
    try:
        result = await stash.merge_scenes([request.source_scene_id], request.keeper_match_scene_id)

        db.resolve_recommendation(
            request.keeper_recommendation_id,
            action="merged_source_into_match",
            details={
                "keeper_id": request.keeper_match_scene_id,
                "source_ids": [request.source_scene_id],
                "deleted_match_scene_ids": [match.scene_id for match in request.other_matches],
            },
        )

        delete_failures: list[dict[str, str]] = []
        for other_match in request.other_matches:
            resolution_action = "deleted_match_after_source_merge"
            resolution_details: dict[str, Any] = {
                "source_scene_id": request.source_scene_id,
                "keeper_id": request.keeper_match_scene_id,
                "deleted_scene_id": other_match.scene_id,
            }

            deleted = False
            try:
                deleted = await stash.destroy_scene(other_match.scene_id, delete_file=False)
                if not deleted:
                    resolution_action = "closed_after_source_merge"
                    resolution_details["delete_error"] = "Delete returned false"
                    delete_failures.append({
                        "scene_id": str(other_match.scene_id),
                        "error": "Delete returned false",
                    })
            except Exception as exc:
                resolution_action = "closed_after_source_merge"
                resolution_details["delete_error"] = str(exc)
                delete_failures.append({
                    "scene_id": str(other_match.scene_id),
                    "error": str(exc),
                })

            # Resolve *before* the pending-only cleanup sweep below: this
            # recommendation is still 'pending' at this point, and
            # delete_pending_duplicate_scene_recommendations_for_scene()
            # deletes any pending row referencing this scene -- including
            # this one -- which would silently drop the resolution instead
            # of recording what happened to it.
            db.resolve_recommendation(
                other_match.recommendation_id,
                action=resolution_action,
                details=resolution_details,
            )

            if deleted:
                scene_id = str(other_match.scene_id)
                db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)
                db.delete_pending_duplicate_scene_recommendations_for_scene(scene_id=scene_id)

        for scene_id in [str(request.source_scene_id), str(request.keeper_match_scene_id)]:
            db.delete_pending_duplicate_scene_recommendations_for_scene(scene_id=scene_id)
            db.delete_pending_scene_fingerprint_for_scene(scene_id=scene_id)

        return {"success": True, "result": result, "delete_failures": delete_failures}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/dismiss-duplicate-scene-group")
async def dismiss_duplicate_scene_group(request: DismissDuplicateSceneGroupRequest):
    """Dismiss all duplicate-scene recommendations in a grouped source-scene review."""
    db = get_rec_db()
    dismissed_count = 0
    for rec_id in request.recommendation_ids:
        if db.dismiss_recommendation(rec_id, reason=request.reason or "Marked not duplicates"):
            dismissed_count += 1
    return {"success": True, "dismissed_count": dismissed_count}


@router.get("/scene/{scene_id}")
async def get_scene_detail(scene_id: str):
    """Get scene details for the duplicate scenes detail view."""
    stash = get_stash_client()
    try:
        scene = await stash.get_scene_by_id(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        return scene
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Fingerprints ====================

_current_db_version: Optional[str] = None


def set_db_version(version: str):
    """Set the current face recognition DB version (called from main.py startup)."""
    global _current_db_version
    _current_db_version = version


def get_db_version() -> Optional[str]:
    """Get the current face recognition DB version."""
    return _current_db_version


class FingerprintStatusResponse(BaseModel):
    """Response for fingerprint status endpoint."""
    total_fingerprints: int
    complete_fingerprints: int
    pending_fingerprints: int
    error_fingerprints: int
    current_db_version: Optional[str]
    current_version_count: Optional[int] = None
    needs_refresh_count: Optional[int] = None
    outdated_count: Optional[int] = None
    total_scenes: Optional[int] = None


class FingerprintGenerateRequest(BaseModel):
    """Request to start fingerprint generation. Defaults from face_config.py."""
    refresh_outdated: bool = True
    num_frames: int = face_config.NUM_FRAMES
    min_face_size: int = face_config.MIN_FACE_SIZE
    max_distance: float = face_config.MAX_DISTANCE


@router.get("/fingerprints/status", response_model=FingerprintStatusResponse)
async def get_fingerprint_status():
    """Get fingerprint coverage statistics."""
    db = get_rec_db()

    stats = db.get_fingerprint_stats(_current_db_version)

    # Total scene count from Stash itself, so the UI can show "missing"
    # without conflating it with the performer database's own counts. This
    # also self-heals scene_fingerprints (and the signal-cache tables)
    # against scenes deleted from Stash: their fingerprint rows never get
    # cleaned up on their own, so a naive "total - complete" subtraction
    # can go negative once enough scenes are deleted (complete_fingerprints
    # includes orphans, so it can end up larger than the live scene count).
    total_scenes = None
    complete_fingerprints = stats.get("complete_fingerprints", 0)
    try:
        stash = get_stash_client()
        stash_scene_ids = await stash.get_all_scene_ids()
        total_scenes = len(stash_scene_ids)

        # Orphan check covers every fingerprint row regardless of status --
        # an 'error' row for a deleted scene is just as orphaned as a
        # 'complete' one.
        all_fingerprinted_ids = db.get_all_fingerprint_scene_ids()
        orphaned_ids = all_fingerprinted_ids - stash_scene_ids
        if orphaned_ids:
            db.delete_fingerprints_for_scenes(list(orphaned_ids))
            # Orphans are gone from the table now, so a fresh count is accurate.
            complete_fingerprints = len(db.get_fingerprinted_scene_ids())
            logger.warning(
                "Pruned %d orphaned scene fingerprint(s) for scenes no longer in Stash",
                len(orphaned_ids),
            )
    except Exception:
        logger.exception("Failed to reconcile fingerprint coverage against Stash scene list")

    return FingerprintStatusResponse(
        total_fingerprints=stats.get("total_fingerprints", 0),
        complete_fingerprints=complete_fingerprints,
        pending_fingerprints=stats.get("pending_fingerprints", 0),
        error_fingerprints=stats.get("error_fingerprints", 0),
        current_db_version=_current_db_version,
        current_version_count=stats.get("current_version_count"),
        needs_refresh_count=stats.get("needs_refresh_count"),
        outdated_count=stats.get("outdated_count"),
        total_scenes=total_scenes,
    )


class SceneFingerprintCheckResponse(BaseModel):
    """Response for the per-scene fingerprint existence check."""
    exists: bool
    status: Optional[str] = None
    frames_analyzed: Optional[int] = None
    total_faces: Optional[int] = None
    db_version: Optional[str] = None
    updated_at: Optional[str] = None


@router.get("/fingerprints/scene/{scene_id}", response_model=SceneFingerprintCheckResponse)
async def get_scene_fingerprint_status(scene_id: int):
    """Check whether a single scene has been fingerprinted yet -- used by
    the scene page's "Identify full video" flow to decide whether to
    prompt the user to fingerprint before identifying."""
    db = get_rec_db()
    fp = db.get_scene_fingerprint(scene_id)
    if not fp:
        return SceneFingerprintCheckResponse(exists=False)

    return SceneFingerprintCheckResponse(
        exists=True,
        status=fp.get("fingerprint_status"),
        frames_analyzed=fp.get("frames_analyzed"),
        total_faces=fp.get("total_faces"),
        db_version=fp.get("db_version"),
        updated_at=fp.get("updated_at"),
    )


def _match_row_to_response_dict(row: dict) -> dict:
    """A stored scene_fingerprint_matches row -> a plain dict matching
    identification_router.py's PerformerMatchResponse field shape. No
    response_model on the route using this (see get_scene_identify_result)
    -- importing that model at module level here would be circular
    (identification_router.py already imports this module at its own
    module level), so this is returned as plain JSON instead."""
    return {
        "stashdb_id": row["stashdb_id"],
        "name": row["name"],
        "confidence": row["confidence"],
        "distance": row["distance"],
        "country": row["country"],
        "image_url": row["image_url"],
        "endpoint": row["endpoint"],
        "already_tagged": bool(row["already_tagged"]),
        "local_performer_id": row["local_performer_id"],
        "source": row["source"],
        "catalogue_url": row["catalogue_url"],
        "profile_url": row["profile_url"],
        "top_timestamps_sec": row["top_timestamps_sec"],
    }


@router.get("/fingerprints/scene/{scene_id}/result")
async def get_scene_identify_result(scene_id: int):
    """Stored identification result for a scene, shaped exactly like
    SceneIdentifyResponse (identification_router.py) so the scene page's
    Identify button can render it directly with the same code it already
    uses for a fresh /identify/scene call, instead of always re-running
    detect+embed+match. Always 200 -- {"available": false} (no `result` key)
    if Face Identification hasn't covered this scene yet, matching the
    existing exists:false convention on the plain fingerprint-check
    endpoint above rather than a 404, so the caller can tell "nothing
    stored" apart from a real backend error using the same JSON-shape
    check it already does. Caller falls back to the existing
    fingerprint-prompt flow in that case (see stash-sense.js's
    handleIdentifyFullVideo).

    already_tagged on each match reflects whatever was true the last time
    this scene was identified, not necessarily right now -- same staleness
    tradeoff as everything else served from storage; a Re-identify gets a
    fresh answer.
    """
    db = get_rec_db()
    fp = db.get_scene_fingerprint(scene_id)
    if not fp or fp.get("fingerprint_status") != "complete":
        return {"available": False}

    rows = db.get_fingerprint_matches(fp["id"])
    persons_by_id: dict[int, list[dict]] = {}
    for row in rows:
        persons_by_id.setdefault(row["person_id"], []).append(row)

    persons = []
    for person_id in sorted(persons_by_id):
        match_rows = sorted(persons_by_id[person_id], key=lambda r: r["match_rank"])
        all_matches = [_match_row_to_response_dict(r) for r in match_rows]
        best_match = next(
            (m for m, r in zip(all_matches, match_rows) if r["is_best_match"]), None,
        )
        persons.append({
            "person_id": person_id,
            "frame_count": match_rows[0]["frame_count"],
            "best_match": best_match,
            "all_matches": all_matches,
        })

    return {
        "available": True,
        "result": {
            "scene_id": str(scene_id),
            "frames_analyzed": fp.get("frames_analyzed", 0),
            "frames_requested": fp.get("frames_analyzed", 0),
            "faces_detected": fp.get("total_faces", 0),
            "faces_after_filter": fp.get("total_faces", 0),
            "persons": persons,
            "errors": [],
            "fingerprint_saved": True,
            "fingerprint_error": None,
            "timing": None,
            "used_cache": True,
        },
    }


@router.post("/fingerprints/generate")
async def start_fingerprint_generation(request: FingerprintGenerateRequest):
    """Start fingerprint generation. Delegates to the job queue.

    request.refresh_outdated selects which of the two distinct job types
    to submit: True -> "fingerprint_refresh_outdated" (re-matches already-
    identified scenes against the current performer database, cheap).
    False -> "fingerprint_generation" (detects/identifies scenes that have
    never been processed). These used to be one job type with the scope
    hidden in its cursor -- split so the Operations tab and each of the
    Settings tab's two buttons show independent progress/labels instead of
    both reacting to "is *a* fingerprint job running" (see
    fingerprint_job.py's docstring).
    """
    from queue_router import _queue_manager
    from job_models import JobPriority
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")

    type_id = "fingerprint_refresh_outdated" if request.refresh_outdated else "fingerprint_generation"
    job_id = _queue_manager.submit(
        type_id=type_id,
        triggered_by="user",
        priority=JobPriority.HIGH,
    )
    if job_id is None:
        raise HTTPException(status_code=409, detail="Fingerprint generation already running or queued")

    return {"job_id": job_id, "message": "Fingerprint generation queued"}


@router.post("/fingerprints/stop", response_model=MessageResponse)
async def stop_fingerprint_generation():
    """Stop whichever fingerprint job is running -- either
    fingerprint_generation (Fingerprint Missing) or
    fingerprint_refresh_outdated (Refresh Outdated); see
    fingerprint_job.py's docstring for why these are two job types."""
    from queue_router import _queue_manager
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")

    running = (
        _queue_manager.get_jobs(status="running", type="fingerprint_generation")
        + _queue_manager.get_jobs(status="running", type="fingerprint_refresh_outdated")
    )
    if not running:
        return {"message": "No fingerprint generation running"}
    for job in running:
        _queue_manager.cancel(job["id"])
    return {"message": "Stop requested"}


@router.post("/fingerprints/refresh", response_model=FingerprintRefreshResponse)
async def mark_fingerprints_for_refresh(scene_ids: Optional[list[int]] = None):
    """
    Mark fingerprints for refresh by clearing their db_version.
    If scene_ids is provided, only those scenes are marked.
    If scene_ids is None, ALL fingerprints are marked for refresh.
    """
    db = get_rec_db()

    if scene_ids is None:
        # Require explicit confirmation to refresh all
        raise HTTPException(
            status_code=400,
            detail="Must provide scene_ids list. To refresh all, use /fingerprints/refresh-all",
        )

    count = db.mark_fingerprints_for_refresh(scene_ids)
    return {
        "marked_for_refresh": count,
        "scene_ids": scene_ids,
    }


@router.post("/fingerprints/refresh-all", response_model=FingerprintRefreshAllResponse)
async def mark_all_fingerprints_for_refresh(confirm: bool = False):
    """Mark ALL fingerprints for refresh. Requires confirm=true."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to refresh all fingerprints",
        )

    db = get_rec_db()
    count = db.mark_fingerprints_for_refresh(None)

    return {
        "marked_for_refresh": count,
        "message": f"All {count} fingerprints marked for refresh",
    }


@router.post("/fingerprints/reset")
async def reset_fingerprints_with_backup():
    """Back up all scene fingerprints, then mark every one for refresh.

    For a detection-affecting settings change (e.g. detection_size) where
    the user wants existing fingerprints regenerated under the new
    setting, but with a safety copy of the pre-change data first -- see
    RecommendationsDB.reset_scene_fingerprints_with_backup().
    """
    db = get_rec_db()
    return db.reset_scene_fingerprints_with_backup()


# ==================== Local Performer Database ====================
#
# No dedicated "sync-all" endpoint here -- a full sync is just the
# "local_performer_sync" job type (registered in job_models.py), started
# the same generic way every other Quick Action/Settings-triggered job is:
# the plugin JS calls queue_submit({type: "local_performer_sync"}), same
# pattern as fingerprint generation's Settings-tab "Start Fingerprinting"
# button. Only the single-performer hook path below needs a bespoke
# endpoint, since it's a fast synchronous operation, not a queued job.


def _extract_performer_ids_from_recommendation(rec) -> list[str]:
    """Extract referenced local performer IDs from a recommendation.

    duplicate_performer's target_id is only the suggested-keeper anchor --
    the other performers in the group live in details.performers (a list
    of {"id": ...} dicts), so a deleted non-anchor performer must be found
    there too, not just via target_id. upstream_performer_changes has no
    such grouping -- target_id is the whole story.
    """
    details = rec.details or {}
    performer_ids: list[str] = []

    def add(value):
        if value is None:
            return
        pid = str(value).strip()
        if pid:
            performer_ids.append(pid)

    if str(rec.type or "") == "duplicate_performer":
        for p in details.get("performers") or []:
            add(p.get("id"))
        add(rec.target_id)
    elif str(rec.target_type or "") == "performer":
        add(rec.target_id)

    deduped: list[str] = []
    seen: set[str] = set()
    for pid in performer_ids:
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(pid)
    return deduped


def _extract_simple_target_id(rec, target_type: str) -> list[str]:
    """For recommendation types with a single scalar target_id and no
    grouped/composite ids to unpack (studio: upstream_studio_changes, tag:
    upstream_tag_changes) -- unlike performer's duplicate_performer group
    or scene's several composite-id types."""
    if str(rec.target_type or "") == target_type and rec.target_id:
        return [str(rec.target_id)]
    return []


class EntityCleanupRequest(BaseModel):
    """Request from a Stash performer/studio/tag destroy hook to remove
    every recommendation referencing that entity. Scenes have their own
    dedicated cleanup-scene endpoint (several recommendation types there
    use composite/grouped target_ids needing per-type extraction) --
    performer/studio/tag recommendation types are simpler, so one generic
    endpoint covers all three via target_type."""
    target_type: str  # "performer" | "studio" | "tag"
    entity_id: str


@router.post("/cleanup-entity")
async def cleanup_entity_recommendations(request: EntityCleanupRequest):
    """Fast single-entity cleanup for the performer/studio/tag destroy
    hooks. Deletes every recommendation (any status) referencing this
    entity id. Runs inline, same rationale as cleanup-scene."""
    if request.target_type not in ("performer", "studio", "tag"):
        raise HTTPException(status_code=400, detail=f"Unsupported target_type: {request.target_type}")

    db = get_rec_db()
    entity_id = str(request.entity_id)
    extractor = (
        _extract_performer_ids_from_recommendation
        if request.target_type == "performer"
        else lambda rec: _extract_simple_target_id(rec, request.target_type)
    )

    recs = _load_all_recommendations(db, target_type=request.target_type)
    deleted = 0
    for rec in recs:
        if entity_id in extractor(rec):
            db.delete_recommendation(rec.id)
            deleted += 1

    return {"target_type": request.target_type, "entity_id": entity_id, "deleted": deleted}


class SceneCleanupRequest(BaseModel):
    """Request from the Stash scene destroy hook to remove every
    recommendation referencing a scene that no longer exists. Covers both
    a plain delete and a merge (Stash has no separate Scene.Merge.Post
    hook -- merging folds the source scene's data into the destination and
    destroys the source, which fires this same Scene.Destroy.Post hook for
    the source scene id)."""
    scene_id: str


@router.post("/cleanup-scene")
async def cleanup_scene_recommendations(request: SceneCleanupRequest):
    """Fast single-scene cleanup for the Scene.Destroy.Post hook handler.
    Deletes every recommendation (any status) referencing this scene id --
    reuses the same extraction logic get_recommendation's lazy per-item
    prune uses (_extract_scene_ids_from_recommendation), just applied
    proactively for one already-known-gone scene instead of waiting for a
    user to open a stale recommendation. Runs inline (a handful of sqlite
    deletes at most) since the caller has its own local retry-cache for
    when this endpoint is unreachable, same as the performer sync hook."""
    db = get_rec_db()
    scene_id = str(request.scene_id)

    recs = _load_all_recommendations(db, target_type="scene")
    deleted = 0
    for rec in recs:
        if scene_id in _extract_scene_ids_from_recommendation(rec):
            db.delete_recommendation(rec.id)
            deleted += 1

    return {"scene_id": scene_id, "deleted": deleted}


class LocalPerformerSyncOneRequest(BaseModel):
    """Request from the Stash performer create/update/destroy hook to sync
    a single performer into the local index. Deliberately synchronous
    (not queued) so the caller can rely on a short, bounded turnaround."""
    performer_id: int
    event_type: str  # "create" | "update" | "destroy"


@router.post("/local-performers/sync-one")
async def sync_one_local_performer(request: LocalPerformerSyncOneRequest):
    """Fast single-performer sync for the hook handler. Runs inline
    (embeds one image at most) rather than going through the job queue,
    since the caller (a Stash plugin hook) expects a quick response and
    has its own local retry-cache for when this endpoint is unreachable."""
    from config import DatabaseConfig
    from embeddings import FaceEmbeddingGenerator
    from local_performer_index import LocalPerformerIndex, sync_one_performer
    import os
    from pathlib import Path

    stash = get_stash_client()
    db_config = DatabaseConfig(data_dir=Path(os.environ.get("DATA_DIR", "./data")))
    index = LocalPerformerIndex(
        db_config.local_embedding_index_path,
        db_config.local_faces_json_path,
    )
    generator = FaceEmbeddingGenerator()

    status = await sync_one_performer(
        stash, generator, index, request.performer_id, request.event_type,
    )
    index.save()

    if status in ("added", "updated", "removed"):
        try:
            from resource_manager import get_resource_manager
            get_resource_manager().unload("face_recognition")
        except (RuntimeError, KeyError):
            pass  # not initialized / not currently loaded -- nothing to unload

    return {"performer_id": request.performer_id, "status": status}


@router.get("/local-performers/stats")
async def get_local_performer_stats():
    """Count of performers currently in the local performer index (this
    Stash instance's own performer cover images -- see
    local_performer_index.py's module docstring), for the Settings tab's
    Identification Database section. One vector per performer in this
    index (unlike the main database, which has many faces per performer),
    so performer count and face count are the same number here.

    Reads the index fresh from disk rather than reusing recognizer.py's
    in-memory copy deliberately -- face recognition is lazy-loaded (see
    resource_manager.py) and this stat should be visible without forcing
    that load just to report a count. The index itself is small (this
    Stash instance's own performer library, not the ~450k-performer main
    database), so a fresh read here is cheap."""
    from config import DatabaseConfig
    import os
    from pathlib import Path

    db_config = DatabaseConfig(data_dir=Path(os.environ.get("DATA_DIR", "./data")))
    if not db_config.local_faces_json_path or not db_config.local_faces_json_path.exists():
        return {"performer_count": 0}

    from local_performer_index import LocalPerformerIndex
    index = LocalPerformerIndex(db_config.local_embedding_index_path, db_config.local_faces_json_path)
    return {"performer_count": len(index)}


# ==================== Upstream Sync Actions ====================


def deduplicate_aliases(
    aliases: list[str | None],
    entity_name: str,
    other_entity_names: set[str],
) -> list[str]:
    """Remove duplicate, self-referencing, and cross-entity-conflicting aliases.

    Rules (applied in order per alias):
    1. Filter out None/empty/whitespace-only values
    2. Remove aliases matching entity's own name (case-insensitive)
    3. Remove duplicate aliases (case-insensitive, keep first occurrence)
    4. Remove aliases matching another entity's primary name (case-insensitive)

    Note: other_entity_names may include the entity's own name — rule 2 handles
    that case before rule 4, so no alias is incorrectly dropped.
    """
    entity_name_lower = entity_name.lower().strip() if entity_name else ""
    result = []
    seen = set()
    for alias in aliases:
        if alias is None or not str(alias).strip():
            continue
        alias_str = str(alias).strip()
        alias_lower = alias_str.lower()
        if alias_lower == entity_name_lower:
            continue
        if alias_lower in seen:
            continue
        if alias_lower in other_entity_names:
            continue
        seen.add(alias_lower)
        result.append(alias_str)
    return result


class UpdatePerformerRequest(BaseModel):
    """Request to apply upstream changes to a performer."""
    performer_id: str
    fields: dict


@router.post("/actions/update-performer")
async def update_performer_fields(request: UpdatePerformerRequest, entity_type: str = "performer"):
    """Apply selected upstream changes to a performer.

    Translates diff-engine field names (StashBox-style) to Stash PerformerUpdateInput names.
    Compound fields (measurements, career_length) are smart-merged with existing values.

    Args:
        entity_type: Entity type (default: "performer"). Reserved for future entity types.
    """
    from upstream_field_mapper import parse_measurements, parse_career_length

    logger.debug("Action: update-performer performer_id=%s fields=%s", request.performer_id, sorted(request.fields.keys()))
    stash = get_stash_client()
    fields = dict(request.fields)
    performer_id = request.performer_id

    # Lazy-fetch current performer data (only when needed for smart merge)
    current_performer = None

    async def get_current():
        nonlocal current_performer
        if current_performer is None:
            current_performer = await stash.get_performer(performer_id) or {}
        return current_performer

    # --- Simple field renames ---
    FIELD_RENAME = {
        "aliases": "alias_list",
        "height": "height_cm",
        "breast_type": "fake_tits",
    }
    for old_name, new_name in FIELD_RENAME.items():
        if old_name in fields:
            fields[new_name] = fields.pop(old_name)

    # --- Career years → career_length (smart merge) ---
    career_start = fields.pop("career_start_year", None)
    career_end = fields.pop("career_end_year", None)
    if career_start is not None or career_end is not None:
        current = await get_current()
        existing = parse_career_length(current.get("career_length"))
        start_val = str(career_start) if career_start is not None else (
            str(existing["career_start_year"]) if existing["career_start_year"] else ""
        )
        end_val = str(career_end) if career_end is not None else (
            str(existing["career_end_year"]) if existing["career_end_year"] else ""
        )
        if start_val and end_val:
            fields["career_length"] = f"{start_val}-{end_val}"
        elif start_val:
            fields["career_length"] = f"{start_val}-"
        elif end_val:
            fields["career_length"] = f"-{end_val}"

    # --- Measurement fields → measurements string (smart merge) ---
    cup = fields.pop("cup_size", None)
    band = fields.pop("band_size", None)
    waist = fields.pop("waist_size", None)
    hip = fields.pop("hip_size", None)
    if any(v is not None for v in [cup, band, waist, hip]):
        current = await get_current()
        existing = parse_measurements(current.get("measurements"))
        # Overlay: accepted value wins, else keep existing
        final_band = str(band) if band is not None else (
            str(existing["band_size"]) if existing["band_size"] else ""
        )
        final_cup = str(cup) if cup is not None else (
            existing["cup_size"] or ""
        )
        final_waist = str(waist) if waist is not None else (
            str(existing["waist_size"]) if existing["waist_size"] else ""
        )
        final_hip = str(hip) if hip is not None else (
            str(existing["hip_size"]) if existing["hip_size"] else ""
        )
        bust = f"{final_band}{final_cup}"
        measurements = "-".join([bust, final_waist, final_hip])
        fields["measurements"] = measurements.rstrip("-") or None

    # --- Integer coercion ---
    if "height_cm" in fields and isinstance(fields["height_cm"], str):
        try:
            fields["height_cm"] = int(fields["height_cm"])
        except ValueError:
            pass

    # --- Alias merge (_alias_add meta-key) ---
    alias_additions = fields.pop("_alias_add", None)
    if alias_additions:
        current = await get_current()
        existing_aliases = current.get("alias_list", [])
        seen = {a.lower() for a in existing_aliases}
        merged = list(existing_aliases)
        for alias in alias_additions:
            if alias.lower() not in seen:
                merged.append(alias)
                seen.add(alias.lower())
        fields["alias_list"] = merged

    # --- Deduplicate aliases ---
    if "alias_list" in fields:
        new_name = fields.get("name") or (await get_current()).get("name", "")
        other_names = await _get_entity_names("performers")
        fields["alias_list"] = deduplicate_aliases(
            fields["alias_list"], new_name, other_names
        )

    try:
        result = await stash.update_performer(performer_id, **fields)
        _invalidate_entity_name_cache("performers")
        return {"success": True, "performer": result}
    except Exception as e:
        error_msg = str(e)
        # Detect name conflict: "Name X already used by performer Y"
        if "name" in fields and "already used by performer" in error_msg:
            # Get current performer's disambiguation for merge safety check
            current = await get_current()
            dest_disambig = fields.get("disambiguation") or current.get("disambiguation") or ""
            merged = await _auto_merge_conflicting_performer(
                stash, performer_id, fields["name"],
                destination_disambiguation=dest_disambig,
            )
            if merged and merged.get("blocked_by_disambiguation"):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Name conflict: a performer with this name exists but has a "
                        "different disambiguation — they are different people and "
                        "cannot be auto-merged. Use 'Update Fields Only' to apply "
                        "other changes without the name."
                    ),
                )
            if merged:
                # Retry the update after merging the conflicting performer
                try:
                    result = await stash.update_performer(performer_id, **fields)
                    _invalidate_entity_name_cache("performers")
                    return {
                        "success": True,
                        "performer": result,
                        "auto_merged": True,
                        "merged_performer_id": merged["merged_id"],
                        "merged_performer_name": merged["merged_name"],
                    }
                except Exception as retry_err:
                    raise HTTPException(status_code=500, detail=str(retry_err))

        raise HTTPException(status_code=500, detail=error_msg)


async def _auto_merge_conflicting_performer(
    stash: StashClientUnified,
    destination_id: str,
    conflicting_name: str,
    destination_disambiguation: str = "",
) -> dict | None:
    """Find and merge a performer that conflicts with the given name.

    Searches for performers with the conflicting name, finds the one that
    isn't the destination, and merges it into the destination.

    Skips merge when the performers have different disambiguations — they are
    explicitly marked as distinct people sharing a name. Allows merge when
    neither is disambiguated (unqualified duplicates) or when both share the
    same disambiguation (likely true duplicates).

    Returns:
        dict with merged_id/merged_name on success,
        dict with blocked_by_disambiguation=True when skipped due to disambiguation,
        or None if no matching performer found.
    """
    matches = await stash.search_performers(conflicting_name, limit=10)
    conflicting = None
    blocked_by_disambiguation = False
    name_lower = conflicting_name.strip().lower()
    dest_disambig = (destination_disambiguation or "").strip().lower()
    for match in matches:
        if str(match["id"]) == str(destination_id):
            continue
        if (match.get("name") or "").strip().lower() == name_lower:
            match_disambig = (match.get("disambiguation") or "").strip().lower()
            # Skip if disambiguations differ (different people sharing a name).
            # Allow merge when both have the same disambiguation (likely true
            # duplicates) or when neither is disambiguated.
            has_disambig = match_disambig or dest_disambig
            same_disambig = match_disambig == dest_disambig
            if has_disambig and not same_disambig:
                logger.warning(
                    f"Skipping auto-merge: disambiguations differ "
                    f"(dest='{dest_disambig}', match='{match_disambig}') "
                    f"for performer '{conflicting_name}'"
                )
                blocked_by_disambiguation = True
                continue
            conflicting = match
            break

    if not conflicting:
        if blocked_by_disambiguation:
            return {"blocked_by_disambiguation": True}
        return None

    logger.warning(
        f"Auto-merging conflicting performer '{conflicting['name']}' "
        f"(ID: {conflicting['id']}) into performer {destination_id}"
    )

    try:
        await stash.merge_performers([conflicting["id"]], destination_id)
    except Exception as e:
        err_msg = str(e).lower()
        if "not found" in err_msg or "does not exist" in err_msg:
            logger.warning(f"Conflicting performer {conflicting['id']} already deleted, skipping merge")
            return {"merged_id": conflicting["id"], "merged_name": conflicting["name"]}
        raise

    # Delete the now-orphaned source performer
    try:
        await stash.destroy_performer(conflicting["id"])
        logger.warning(f"Deleted orphaned performer {conflicting['id']} after merge")
    except Exception as e:
        logger.warning(f"Failed to delete performer {conflicting['id']} after merge: {e}")

    return {"merged_id": conflicting["id"], "merged_name": conflicting["name"]}


class UpdateTagRequest(BaseModel):
    """Request to apply upstream changes to a tag."""
    tag_id: str
    fields: dict


@router.post("/actions/update-tag")
async def update_tag_fields(request: UpdateTagRequest):
    """Apply selected upstream changes to a tag.

    Tags have simple 1:1 field mapping — no compound fields like performers.
    Stash TagUpdateInput accepts: name, description, aliases directly.
    """
    logger.debug("Action: update-tag tag_id=%s fields=%s", request.tag_id, sorted(request.fields.keys()))
    stash = get_stash_client()
    fields = dict(request.fields)
    tag_id = request.tag_id

    # Lazy-fetch current tag data (shared by alias merge and dedup)
    current_tag = None

    async def get_current_tag():
        nonlocal current_tag
        if current_tag is None:
            query = """
            query GetTag($id: ID!) {
              findTag(id: $id) { id name aliases }
            }
            """
            data = await stash._execute(query, {"id": tag_id})
            current_tag = data.get("findTag") or {}
        return current_tag

    # Alias merge (_alias_add meta-key)
    alias_additions = fields.pop("_alias_add", None)
    if alias_additions:
        tag_data = await get_current_tag()
        existing_aliases = tag_data.get("aliases") or []
        seen = {a.lower() for a in existing_aliases}
        merged = list(existing_aliases)
        for alias in alias_additions:
            if alias.lower() not in seen:
                merged.append(alias)
                seen.add(alias.lower())
        fields["aliases"] = merged

    # --- Deduplicate aliases ---
    if "aliases" in fields:
        new_name = fields.get("name") or (await get_current_tag()).get("name", "")
        other_names = await _get_entity_names("tags")
        fields["aliases"] = deduplicate_aliases(
            fields["aliases"], new_name, other_names
        )

    try:
        result = await stash.update_tag(tag_id, **fields)
        _invalidate_entity_name_cache("tags")
        return {"success": True, "tag": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateStudioRequest(BaseModel):
    """Request to apply upstream changes to a studio."""
    studio_id: str
    fields: dict
    endpoint: str = ""  # Needed for parent studio resolution


@router.post("/actions/update-studio")
async def update_studio_fields(request: UpdateStudioRequest):
    """Apply selected upstream changes to a studio.

    Studios have simple field mapping — no compound fields like performers.
    Special handling for parent_studio: resolves StashBox parent UUID to
    local studio ID, auto-importing the parent if not found locally.
    """
    logger.debug("Action: update-studio studio_id=%s fields=%s", request.studio_id, sorted(request.fields.keys()))
    stash = get_stash_client()
    fields = dict(request.fields)
    studio_id = request.studio_id

    current_studio: Optional[dict] = None

    async def get_current_studio():
        nonlocal current_studio
        if current_studio is None:
            query = """
            query GetStudio($id: ID!) {
              findStudio(id: $id) { id name aliases }
            }
            """
            data = await stash._execute(query, {"id": studio_id})
            current_studio = data.get("findStudio") or {}
        return current_studio

    # --- Alias merge (_alias_add meta-key) --- mirrors update_tag_fields/
    # update_performer_fields: studios do have an `aliases` field (already
    # queried elsewhere, e.g. _resolve_stashbox_studio_to_local's alias-match
    # lookup below), this endpoint just never translated the meta-key into
    # it -- _alias_add reached stash.update_studio(**fields) unhandled,
    # which isn't a real StudioUpdateInput field, failing the whole
    # mutation for every studio name-normalization change sent through
    # "Accept All" (computeBatchChanges always attaches _alias_add for a
    # name merge_type change, regardless of entity type).
    alias_additions = fields.pop("_alias_add", None)
    if alias_additions:
        studio_data = await get_current_studio()
        existing_aliases = studio_data.get("aliases") or []
        seen = {a.lower() for a in existing_aliases}
        merged = list(existing_aliases)
        for alias in alias_additions:
            if alias.lower() not in seen:
                merged.append(alias)
                seen.add(alias.lower())
        fields["aliases"] = merged

    # --- Parent studio resolution ---
    parent_stashbox_id = fields.pop("parent_studio", None)
    if parent_stashbox_id:
        local_parent_id = await _resolve_stashbox_studio_to_local(
            stash, parent_stashbox_id, request.endpoint, exclude_studio_id=studio_id
        )
        # None means no distinct studio could be resolved (self-reference
        # case) -- leave parent_id untouched rather than clearing it.
        if local_parent_id is not None:
            fields["parent_id"] = local_parent_id

    try:
        result = await stash.update_studio(studio_id, **fields)
        return {"success": True, "studio": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _resolve_stashbox_studio_to_local(
    stash: StashClientUnified,
    stashbox_id: str,
    endpoint: str,
    exclude_studio_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve a StashBox studio UUID to a local Stash studio ID.

    If the studio doesn't exist locally, import it from StashBox
    (name + first URL + stash_ids link) and return the new local ID.

    `exclude_studio_id`: the studio this resolution is *for* (i.e. whose
    parent_id is about to be set to whatever this returns), when known.
    Both lookup paths below match on name/stash_id alone, with no
    inherent reason they can't return that same studio -- confirmed live
    when a studio's name coincided with its own StashBox-reported parent
    name (e.g. renaming "Devil's Film (Network)" while StashBox reports
    its parent as a studio also named "Devil's Film (Network)"), which
    the name-match fallback resolved right back to the studio itself,
    and Stash's own studioUpdate mutation correctly rejects ("studio
    cannot be an ancestor of itself") but with no indication why to the
    caller. Filtering self-matches here means a genuinely different
    parent (existing or newly imported) gets used instead, rather than
    the whole update failing. Returns None (rather than raising) in the
    rarer case where the only local studio holding that name/alias *is*
    the excluded one -- no distinct studio to link to, and creating a
    second studio under the same name would just fail on Stash's own
    unique-name constraint. Callers should leave parent_id untouched
    (not set it to None) when this happens.
    """
    # Try to find locally by stash_box_id
    query = """
    query FindStudioByStashBoxID($studio_filter: StudioFilterType) {
      findStudios(studio_filter: $studio_filter, filter: { per_page: 10 }) {
        studios { id name }
      }
    }
    """
    data = await stash._execute(query, {
        "studio_filter": {
            "stash_id_endpoint": {
                "endpoint": endpoint,
                "stash_id": stashbox_id,
                "modifier": "EQUALS",
            }
        }
    })
    studios = [
        s for s in data["findStudios"]["studios"]
        if exclude_studio_id is None or s["id"] != exclude_studio_id
    ]
    if studios:
        return studios[0]["id"]

    # Not found by stash_id — fetch upstream info and try name match
    from stashbox_connection_manager import get_connection_manager
    mgr = get_connection_manager()
    sbc = mgr.get_client(endpoint)
    if not sbc:
        from stashbox_client import StashBoxClient
        sbc = StashBoxClient(endpoint)

    upstream = await sbc.get_studio(stashbox_id)
    if not upstream:
        raise HTTPException(
            status_code=404,
            detail=f"Parent studio {stashbox_id} not found on StashBox"
        )

    # Try to find locally by exact name or alias match before creating
    local_matches = await stash.search_studios(upstream["name"], limit=10)
    upstream_name_lower = upstream["name"].strip().lower()
    self_holds_this_name = False
    for match in local_matches:
        match_name = (match.get("name") or "").strip().lower()
        match_aliases = {a.strip().lower() for a in (match.get("aliases") or [])}
        is_name_match = match_name == upstream_name_lower or upstream_name_lower in match_aliases
        if not is_name_match:
            continue
        if exclude_studio_id is not None and match["id"] == exclude_studio_id:
            # The studio being resolved *for* is itself the only local
            # holder of this name -- can't link to it (self-reference) and
            # can't create a new studio under the same name either (Stash
            # enforces unique studio names, so that would just swap one
            # blocking error for another). Keep looking in case a
            # *different* studio also matches; otherwise this parent link
            # can't be resolved to a distinct studio at all right now.
            self_holds_this_name = True
            continue
        # Name or alias match on a different studio — link the stash_id to it
        existing_stash_ids = match.get("stash_ids") or []
        existing_stash_ids.append({"endpoint": endpoint, "stash_id": stashbox_id})
        await stash.update_studio(match["id"], stash_ids=existing_stash_ids)
        logger.warning(
            f"Linked existing studio '{match['name']}' (ID: {match['id']}) "
            f"to StashBox {stashbox_id} on {endpoint}"
        )
        return match["id"]

    if self_holds_this_name:
        logger.warning(
            f"Parent studio {stashbox_id} on {endpoint} resolves to the same "
            f"name ('{upstream['name']}') as the studio it would be set as "
            f"parent of (local ID {exclude_studio_id}) -- skipping rather "
            f"than self-reference or collide with it on create."
        )
        return None

    # No local match by name — auto-import from StashBox
    urls = []
    for u in (upstream.get("urls") or []):
        urls.append(u.get("url") if isinstance(u, dict) else u)

    new_studio = await stash.create_studio(
        name=upstream["name"],
        urls=urls if urls else None,
        stash_ids=[{"endpoint": endpoint, "stash_id": stashbox_id}],
    )
    logger.warning(
        f"Auto-imported parent studio '{upstream['name']}' (local ID: {new_studio['id']}) "
        f"from StashBox {stashbox_id}"
    )
    return new_studio["id"]


def _get_stashbox_client(endpoint: str):
    """Get or create a StashBoxClient for the given endpoint."""
    from stashbox_connection_manager import get_connection_manager
    mgr = get_connection_manager()
    sbc = mgr.get_client(endpoint)
    if not sbc:
        from stashbox_client import StashBoxClient
        sbc = StashBoxClient(endpoint)
    return sbc


async def _enrich_performer_after_create(
    stash, performer_id: str, stashbox_id: str, endpoint: str
) -> None:
    """Fetch full performer details from stashbox and update the newly created local performer."""
    try:
        sbc = _get_stashbox_client(endpoint)
        upstream = await sbc.get_performer(stashbox_id)
        if not upstream:
            return

        from upstream_field_mapper import normalize_upstream_performer
        normalized = normalize_upstream_performer(upstream)

        update_fields: dict = {}

        for field in ("disambiguation", "gender", "country", "eye_color", "hair_color",
                      "ethnicity", "birthdate", "death_date", "tattoos", "piercings"):
            if normalized.get(field) is not None:
                update_fields[field] = normalized[field]

        if normalized.get("aliases"):
            update_fields["alias_list"] = normalized["aliases"]

        if normalized.get("height") is not None:
            try:
                update_fields["height_cm"] = int(normalized["height"])
            except (TypeError, ValueError):
                pass

        if normalized.get("breast_type") is not None:
            update_fields["fake_tits"] = normalized["breast_type"]

        # Measurement fields → compound string "38F-24-35"
        cup = normalized.get("cup_size") or ""
        band = normalized.get("band_size")
        waist = normalized.get("waist_size")
        hip = normalized.get("hip_size")
        if any(x is not None and x != "" for x in (cup, band, waist, hip)):
            bust = f"{band or ''}{cup}"
            parts = [bust, str(waist) if waist is not None else "", str(hip) if hip is not None else ""]
            measurements = "-".join(parts).rstrip("-")
            if measurements:
                update_fields["measurements"] = measurements

        # career years → compound "YYYY-YYYY"
        start_year = normalized.get("career_start_year")
        end_year = normalized.get("career_end_year")
        if start_year or end_year:
            update_fields["career_length"] = f"{start_year or ''}-{end_year or ''}".rstrip("-")

        if normalized.get("urls"):
            update_fields["urls"] = normalized["urls"]

        images = upstream.get("images") or []
        if images:
            update_fields["image"] = images[0]["url"]

        if update_fields:
            await stash.update_performer(performer_id, **update_fields)
            logger.debug(
                "Enriched performer %s from stashbox %s (fields: %s)",
                performer_id, stashbox_id, ", ".join(update_fields.keys()),
            )
    except Exception as exc:
        logger.warning(
            "Failed to enrich performer %s from stashbox %s: %s",
            performer_id, stashbox_id, exc,
        )


async def _enrich_studio_after_create(
    stash, studio_id: str, stashbox_id: str, endpoint: str
) -> None:
    """Fetch full studio details from stashbox and update the newly created local studio."""
    try:
        sbc = _get_stashbox_client(endpoint)
        upstream = await sbc.get_studio(stashbox_id)
        if not upstream:
            return

        update_fields: dict = {}

        images = upstream.get("images") or []
        if images:
            update_fields["image"] = images[0]["url"]

        urls = [u.get("url") if isinstance(u, dict) else u for u in (upstream.get("urls") or [])]
        if urls:
            update_fields["urls"] = urls

        parent = upstream.get("parent")
        if parent and parent.get("id"):
            local_parent_id = await _resolve_stashbox_studio_to_local(
                stash, parent["id"], endpoint, exclude_studio_id=studio_id
            )
            if local_parent_id:
                update_fields["parent_id"] = local_parent_id

        if update_fields:
            await stash.update_studio(studio_id, **update_fields)
            logger.debug(
                "Enriched studio %s from stashbox %s (fields: %s)",
                studio_id, stashbox_id, ", ".join(update_fields.keys()),
            )
    except Exception as exc:
        logger.warning(
            "Failed to enrich studio %s from stashbox %s: %s",
            studio_id, stashbox_id, exc,
        )


async def _find_or_create_local_tag_by_stashbox_id(
    stash, endpoint: str, stashbox_tag_id: str, tag_name: str
) -> Optional[str]:
    """Find a local tag by stashbox ID (or name), creating it minimally if absent."""
    query = """
    query FindTagByStashBoxID($tag_filter: TagFilterType) {
      findTags(tag_filter: $tag_filter, filter: { per_page: 1 }) {
        tags { id name }
      }
    }
    """
    data = await stash._execute(query, {
        "tag_filter": {
            "stash_id_endpoint": {
                "endpoint": endpoint,
                "stash_id": stashbox_tag_id,
                "modifier": "EQUALS",
            }
        }
    })
    tags = (data.get("findTags") or {}).get("tags") or []
    if tags:
        return tags[0]["id"]

    existing = await _link_existing_tag_by_name_or_alias(stash, tag_name, endpoint, stashbox_tag_id)
    if existing:
        return existing["id"]

    try:
        created = await stash.create_tag(
            name=tag_name,
            stash_ids=[{"endpoint": endpoint, "stash_id": stashbox_tag_id}],
        )
        logger.debug("Created parent tag '%s' (local ID: %s)", tag_name, created.get("id"))
        return created["id"]
    except Exception as exc:
        logger.warning("Failed to create parent tag '%s' (stashbox %s): %s", tag_name, stashbox_tag_id, exc)
        return None


async def _enrich_tag_after_create(
    stash, tag_id: str, stashbox_id: str, endpoint: str
) -> None:
    """Fetch full tag details from stashbox and update the newly created local tag."""
    try:
        sbc = _get_stashbox_client(endpoint)
        upstream = await sbc.get_tag(stashbox_id)
        if not upstream:
            return

        update_fields: dict = {}

        images = upstream.get("images") or []
        if images:
            update_fields["image"] = images[0]["url"]

        if upstream.get("description"):
            update_fields["description"] = upstream["description"]

        if upstream.get("aliases"):
            update_fields["aliases"] = upstream["aliases"]

        parent_tags = upstream.get("parent_tags") or []
        parent_ids = []
        for ptag in parent_tags:
            ptag_stashbox_id = ptag.get("id")
            ptag_name = ptag.get("name", "")
            if ptag_stashbox_id:
                local_id = await _find_or_create_local_tag_by_stashbox_id(
                    stash, endpoint, ptag_stashbox_id, ptag_name
                )
                if local_id:
                    parent_ids.append(local_id)
        if parent_ids:
            update_fields["parent_ids"] = parent_ids

        if update_fields:
            await stash.update_tag(tag_id, **update_fields)
            logger.debug(
                "Enriched tag %s from stashbox %s (fields: %s)",
                tag_id, stashbox_id, ", ".join(update_fields.keys()),
            )
    except Exception as exc:
        logger.warning(
            "Failed to enrich tag %s from stashbox %s: %s",
            tag_id, stashbox_id, exc,
        )


class CreatePerformerRequest(BaseModel):
    """Request to create a performer from StashBox data."""
    stashbox_data: dict
    endpoint: str
    stashbox_id: str


class CreateTagRequest(BaseModel):
    """Request to create a tag from StashBox data."""
    stashbox_data: dict
    endpoint: str
    stashbox_id: str


class CreateStudioRequest(BaseModel):
    """Request to create a studio from StashBox data."""
    stashbox_data: dict
    endpoint: str
    stashbox_id: str


class SearchEntitiesRequest(BaseModel):
    """Request to search local entities by name."""
    entity_type: str  # "performer", "tag", "studio"
    query: str
    endpoint: str  # stash-box endpoint (to check if already linked)


class FindLinkedEntityRequest(BaseModel):
    """Request to find local entity linked to a stash-box endpoint+ID."""
    entity_type: str  # "performer", "tag", "studio"
    endpoint: str
    stashbox_id: str


class UpstreamScenePreviewRequest(BaseModel):
    """Request for a live stash-box scene preview (tagger-style comparison)."""
    endpoint: str
    stashbox_id: str


class LinkEntityRequest(BaseModel):
    """Request to link a local entity to a stash-box ID."""
    entity_type: str  # "performer", "tag", "studio"
    entity_id: str
    endpoint: str
    stashbox_id: str


class UpdateSceneRequest(BaseModel):
    """Request to apply upstream changes to a scene."""
    scene_id: str
    fields: dict = {}
    performer_ids: list[str] | None = None
    tag_ids: list[str] | None = None
    studio_id: str | None = None


async def _create_performer_from_stashbox(stash, stashbox_data: dict, endpoint: str, stashbox_id: str) -> dict:
    """Create (or link) a local performer from StashBox data with stash_id link."""
    performer_name = (stashbox_data.get("name") or "").strip()
    if not performer_name:
        raise HTTPException(status_code=422, detail="Performer name is required")

    existing = await _link_existing_performer_by_name_or_alias(
        stash,
        performer_name,
        endpoint,
        stashbox_id,
    )
    if existing:
        return existing

    fields = {"name": performer_name}
    if stashbox_data.get("aliases"):
        fields["alias_list"] = stashbox_data["aliases"]
    if stashbox_data.get("gender"):
        fields["gender"] = stashbox_data["gender"]
    fields["stash_ids"] = [{"endpoint": endpoint, "stash_id": stashbox_id}]
    try:
        created = await stash.create_performer(**fields)
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower():
            raise
        existing = await _link_existing_performer_by_name_or_alias(
            stash,
            performer_name,
            endpoint,
            stashbox_id,
        )
        if existing:
            return existing
        raise
    await _enrich_performer_after_create(stash, created["id"], stashbox_id, endpoint)
    return created


def _normalize_endpoint_for_compare(endpoint: Optional[str]) -> str:
    """Normalize endpoint for tolerant stash-id endpoint matching."""
    if not endpoint:
        return ""
    value = str(endpoint).strip().lower().rstrip("/")
    if value.startswith("https://"):
        value = value[len("https://"):]
    elif value.startswith("http://"):
        value = value[len("http://"):]
    if value.endswith("/graphql"):
        value = value[:-len("/graphql")]
    return value


def _stash_id_link_exists(stash_ids: list[dict], endpoint: str, stashbox_id: str) -> bool:
    """Return True if stash_ids already contain the endpoint + stashbox_id mapping."""
    endpoint_norm = _normalize_endpoint_for_compare(endpoint)
    stash_id_str = str(stashbox_id)
    for sid in stash_ids or []:
        sid_endpoint_norm = _normalize_endpoint_for_compare(sid.get("endpoint"))
        sid_stash_id = str(sid.get("stash_id", ""))
        if sid_endpoint_norm == endpoint_norm and sid_stash_id == stash_id_str:
            return True
    return False


def _performer_matches_name_or_alias(performer: dict, expected_name: str) -> bool:
    """Match performer by exact normalized name or alias."""
    name_norm = (expected_name or "").strip().lower()
    if not name_norm:
        return False

    if (performer.get("name") or "").strip().lower() == name_norm:
        return True

    aliases = performer.get("alias_list")
    if aliases is None:
        aliases = performer.get("aliases")

    alias_set = {str(a).strip().lower() for a in (aliases or []) if str(a).strip()}
    return name_norm in alias_set


async def _link_existing_performer_by_name_or_alias(
    stash,
    performer_name: str,
    endpoint: str,
    stashbox_id: str,
) -> Optional[dict]:
    """Find and link an existing local performer matching by name/alias."""
    local_matches = await stash.search_performers(performer_name, limit=25)

    # Some backends may not include alias hits in search results. Fall back to
    # a full performer scan for exact name/alias match before creating duplicates.
    if not any(_performer_matches_name_or_alias(match, performer_name) for match in local_matches):
        try:
            all_performers = await stash.get_all_performers()
            local_matches = [*local_matches, *all_performers]
        except Exception:
            # If fallback query fails, continue with search results only.
            pass

    for match in local_matches:
        if not _performer_matches_name_or_alias(match, performer_name):
            continue
        current_stash_ids = list(match.get("stash_ids") or [])
        if not _stash_id_link_exists(current_stash_ids, endpoint, stashbox_id):
            current_stash_ids.append({"endpoint": endpoint, "stash_id": stashbox_id})
            await stash.update_performer(match["id"], stash_ids=current_stash_ids)
            logger.warning(
                "Linked existing performer '%s' (ID: %s) to StashBox %s on %s",
                match.get("name"),
                match.get("id"),
                stashbox_id,
                endpoint,
            )
        return {"id": match["id"], "name": match.get("name")}

    return None


async def _find_linked_entity_by_stash_id(
    stash,
    entity_type: str,
    endpoint: str,
    stashbox_id: str,
) -> Optional[dict]:
    """Find a local entity already linked to endpoint+stashbox_id."""
    endpoint = str(endpoint or "").strip()
    stashbox_id = str(stashbox_id or "").strip()
    if not endpoint or not stashbox_id:
        return None

    stash_filter = {
        "stash_id_endpoint": {
            "endpoint": endpoint,
            "stash_id": stashbox_id,
            "modifier": "EQUALS",
        }
    }

    if entity_type == "performer":
        data = await stash._execute(
            """
            query FindLinkedPerformer($performer_filter: PerformerFilterType) {
              findPerformers(performer_filter: $performer_filter, filter: { per_page: 1 }) {
                performers {
                  id
                  name
                  disambiguation
                  alias_list
                  stash_ids { endpoint stash_id }
                }
              }
            }
            """,
            {"performer_filter": stash_filter},
        )
        rows = (data.get("findPerformers") or {}).get("performers") or []
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "disambiguation": row.get("disambiguation"),
            "aliases": row.get("alias_list") or [],
            "stash_ids": row.get("stash_ids") or [],
        }

    if entity_type == "tag":
        data = await stash._execute(
            """
            query FindLinkedTag($tag_filter: TagFilterType) {
              findTags(tag_filter: $tag_filter, filter: { per_page: 1 }) {
                tags {
                  id
                  name
                  aliases
                  stash_ids { endpoint stash_id }
                }
              }
            }
            """,
            {"tag_filter": stash_filter},
        )
        rows = (data.get("findTags") or {}).get("tags") or []
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "aliases": row.get("aliases") or [],
            "stash_ids": row.get("stash_ids") or [],
        }

    if entity_type == "studio":
        data = await stash._execute(
            """
            query FindLinkedStudio($studio_filter: StudioFilterType) {
              findStudios(studio_filter: $studio_filter, filter: { per_page: 1 }) {
                studios {
                  id
                  name
                  aliases
                  stash_ids { endpoint stash_id }
                }
              }
            }
            """,
            {"studio_filter": stash_filter},
        )
        rows = (data.get("findStudios") or {}).get("studios") or []
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "aliases": row.get("aliases") or [],
            "stash_ids": row.get("stash_ids") or [],
        }

    raise HTTPException(status_code=400, detail=f"Unknown entity type: {entity_type}")


def _tag_matches_name_or_alias(tag: dict, expected_name: str) -> bool:
    """Match tag by exact normalized name or alias."""
    name_norm = (expected_name or "").strip().lower()
    if not name_norm:
        return False
    if (tag.get("name") or "").strip().lower() == name_norm:
        return True
    aliases = {str(a).strip().lower() for a in (tag.get("aliases") or []) if str(a).strip()}
    return name_norm in aliases


async def _link_existing_tag_by_name_or_alias(
    stash,
    tag_name: str,
    endpoint: str,
    stashbox_id: str,
) -> Optional[dict]:
    """Find and link an existing local tag matching the StashBox tag by name/alias."""
    local_matches = await stash.search_tags(tag_name, limit=25)

    # Some backends may not include alias hits in search results. Fall back to
    # a full tag scan for exact name/alias match before creating duplicates.
    if not any(_tag_matches_name_or_alias(match, tag_name) for match in local_matches):
        try:
            try:
                all_tags = await stash.get_all_tags_with_aliases()
            except Exception:
                data = await stash._execute(
                    """
                    query AllTagsWithAliases {
                      findTags(filter: { per_page: -1 }) {
                        tags {
                          id
                          name
                          aliases
                          stash_ids { endpoint stash_id }
                        }
                      }
                    }
                    """
                )
                all_tags = data.get("findTags", {}).get("tags", [])
            local_matches = [*local_matches, *all_tags]
        except Exception:
            # If fallback query fails, continue with search results only.
            pass

    for match in local_matches:
        if not _tag_matches_name_or_alias(match, tag_name):
            continue
        current_stash_ids = list(match.get("stash_ids") or [])
        if not _stash_id_link_exists(current_stash_ids, endpoint, stashbox_id):
            current_stash_ids.append({"endpoint": endpoint, "stash_id": stashbox_id})
            await stash.update_tag(match["id"], stash_ids=current_stash_ids)
            logger.warning(
                "Linked existing tag '%s' (ID: %s) to StashBox %s on %s",
                match.get("name"),
                match.get("id"),
                stashbox_id,
                endpoint,
            )
        return {"id": match["id"], "name": match.get("name")}
    return None


async def _create_tag_from_stashbox(stash, stashbox_data: dict, endpoint: str, stashbox_id: str) -> dict:
    """Create (or link) a local tag from StashBox data with stash_id link."""
    tag_name = (stashbox_data.get("name") or "").strip()
    if not tag_name:
        raise HTTPException(status_code=422, detail="Tag name is required")

    # Prefer linking an existing local tag (exact name/alias) instead of failing create.
    existing = await _link_existing_tag_by_name_or_alias(stash, tag_name, endpoint, stashbox_id)
    if existing:
        return existing

    fields = {"name": tag_name}
    if stashbox_data.get("description"):
        fields["description"] = stashbox_data["description"]
    if stashbox_data.get("aliases"):
        fields["aliases"] = stashbox_data["aliases"]
    fields["stash_ids"] = [{"endpoint": endpoint, "stash_id": stashbox_id}]
    try:
        created = await stash.create_tag(**fields)
    except RuntimeError as exc:
        # Handle race/duplicate conflicts: tag was created/exists locally already.
        if "already exists" not in str(exc).lower():
            raise
        existing = await _link_existing_tag_by_name_or_alias(stash, tag_name, endpoint, stashbox_id)
        if existing:
            return existing
        raise
    await _enrich_tag_after_create(stash, created["id"], stashbox_id, endpoint)
    return created


def _studio_matches_name_or_alias(studio: dict, expected_name: str) -> bool:
    """Match studio by exact normalized name or alias."""
    name_norm = (expected_name or "").strip().lower()
    if not name_norm:
        return False
    if (studio.get("name") or "").strip().lower() == name_norm:
        return True
    aliases = {str(a).strip().lower() for a in (studio.get("aliases") or []) if str(a).strip()}
    return name_norm in aliases


async def _link_existing_studio_by_name_or_alias(
    stash,
    studio_name: str,
    endpoint: str,
    stashbox_id: str,
) -> Optional[dict]:
    """Find and link an existing local studio matching the StashBox studio by name/alias."""
    local_matches = await stash.search_studios(studio_name, limit=25)
    for match in local_matches:
        if not _studio_matches_name_or_alias(match, studio_name):
            continue
        current_stash_ids = list(match.get("stash_ids") or [])
        if not _stash_id_link_exists(current_stash_ids, endpoint, stashbox_id):
            current_stash_ids.append({"endpoint": endpoint, "stash_id": stashbox_id})
            await stash.update_studio(match["id"], stash_ids=current_stash_ids)
            logger.warning(
                "Linked existing studio '%s' (ID: %s) to StashBox %s on %s",
                match.get("name"),
                match.get("id"),
                stashbox_id,
                endpoint,
            )
        return {"id": match["id"], "name": match.get("name")}
    return None


async def _create_studio_from_stashbox(stash, stashbox_data: dict, endpoint: str, stashbox_id: str) -> dict:
    """Create (or link) a local studio from StashBox data with stash_id link."""
    studio_name = (stashbox_data.get("name") or "").strip()
    if not studio_name:
        raise HTTPException(status_code=422, detail="Studio name is required")

    existing = await _link_existing_studio_by_name_or_alias(stash, studio_name, endpoint, stashbox_id)
    if existing:
        return existing

    urls = [
        u.get("url") if isinstance(u, dict) else u
        for u in (stashbox_data.get("urls") or [])
    ]
    try:
        created = await stash.create_studio(
            name=studio_name,
            stash_ids=[{"endpoint": endpoint, "stash_id": stashbox_id}],
            urls=urls if urls else None,
        )
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower():
            raise
        existing = await _link_existing_studio_by_name_or_alias(stash, studio_name, endpoint, stashbox_id)
        if existing:
            return existing
        raise
    await _enrich_studio_after_create(stash, created["id"], stashbox_id, endpoint)
    return created


async def _apply_scene_update(
    stash, scene_id: str, fields: dict,
    performer_ids: list[str] | None = None,
    tag_ids: list[str] | None = None,
    studio_id: str | None = None,
):
    """Apply a scene update including simple fields and relational IDs."""
    update_fields = dict(fields)
    if performer_ids is not None:
        update_fields["performer_ids"] = performer_ids
    if tag_ids is not None:
        update_fields["tag_ids"] = tag_ids
    if studio_id is not None:
        update_fields["studio_id"] = studio_id
    return await stash.update_scene(scene_id, **update_fields)


@router.post("/actions/create-performer")
async def create_performer_action(request: CreatePerformerRequest):
    """Create a new local performer from StashBox data."""
    stash = get_stash_client()
    result = await _create_performer_from_stashbox(
        stash, request.stashbox_data, request.endpoint, request.stashbox_id
    )
    return {"success": True, "performer": result}


@router.post("/actions/create-tag")
async def create_tag_action(request: CreateTagRequest):
    """Create a new local tag from StashBox data."""
    stash = get_stash_client()
    result = await _create_tag_from_stashbox(
        stash, request.stashbox_data, request.endpoint, request.stashbox_id
    )
    return {"success": True, "tag": result}


@router.post("/actions/create-studio")
async def create_studio_action(request: CreateStudioRequest):
    """Create a new local studio from StashBox data."""
    stash = get_stash_client()
    result = await _create_studio_from_stashbox(
        stash, request.stashbox_data, request.endpoint, request.stashbox_id
    )
    return {"success": True, "studio": result}


@router.post("/actions/search-entities")
async def search_entities_action(request: SearchEntitiesRequest):
    """Search local entities by name for linking to stash-box IDs."""
    stash = get_stash_client()

    if request.entity_type == "performer":
        results = await stash.search_performers(request.query)
        return {
            "results": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "disambiguation": p.get("disambiguation"),
                    "aliases": p.get("alias_list") or p.get("aliases") or [],
                    "linked": any(
                        _normalize_endpoint_for_compare(s.get("endpoint"))
                        == _normalize_endpoint_for_compare(request.endpoint)
                        for s in (p.get("stash_ids") or [])
                    ),
                    "stash_ids": p.get("stash_ids") or [],
                }
                for p in results
            ]
        }
    elif request.entity_type == "tag":
        results = await stash.search_tags(request.query)
        return {
            "results": [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "aliases": t.get("aliases") or [],
                    "linked": any(
                        _normalize_endpoint_for_compare(s.get("endpoint"))
                        == _normalize_endpoint_for_compare(request.endpoint)
                        for s in (t.get("stash_ids") or [])
                    ),
                    "stash_ids": t.get("stash_ids") or [],
                }
                for t in results
            ]
        }
    elif request.entity_type == "studio":
        results = await stash.search_studios(request.query)
        return {
            "results": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "aliases": s.get("aliases") or [],
                    "linked": any(
                        _normalize_endpoint_for_compare(sid.get("endpoint"))
                        == _normalize_endpoint_for_compare(request.endpoint)
                        for sid in (s.get("stash_ids") or [])
                    ),
                    "stash_ids": s.get("stash_ids") or [],
                }
                for s in results
            ]
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown entity type: {request.entity_type}")


@router.post("/actions/find-linked-entity")
async def find_linked_entity_action(request: FindLinkedEntityRequest):
    """Find a local entity that is already linked to endpoint+stashbox_id."""
    stash = get_stash_client()
    result = await _find_linked_entity_by_stash_id(
        stash=stash,
        entity_type=request.entity_type,
        endpoint=request.endpoint,
        stashbox_id=request.stashbox_id,
    )
    return {"result": result}


@router.post("/actions/upstream-scene-preview")
async def upstream_scene_preview_action(request: UpstreamScenePreviewRequest):
    """Fetch a live stash-box scene preview (cover, studio, performers, duration)
    for tagger-style side-by-side comparison against the local scene."""
    sbc = _get_stashbox_client(request.endpoint)
    scene = await sbc.get_scene(request.stashbox_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found upstream")

    images = scene.get("images") or []
    studio = scene.get("studio")
    performers = [
        {
            "id": (pa.get("performer") or {}).get("id"),
            "name": (pa.get("performer") or {}).get("name"),
            "as": pa.get("as"),
        }
        for pa in (scene.get("performers") or [])
    ]

    return {
        "title": scene.get("title"),
        "date": scene.get("release_date") or scene.get("date"),
        "studio": {"id": studio.get("id"), "name": studio.get("name")} if studio else None,
        "performers": performers,
        "duration": scene.get("duration"),
        "cover_url": images[0]["url"] if images else None,
    }


@router.post("/actions/link-entity")
async def link_entity_action(request: LinkEntityRequest):
    """Link a local entity to a stash-box ID by adding a stash_id entry."""
    stash = get_stash_client()
    new_stash_id = {"endpoint": request.endpoint, "stash_id": request.stashbox_id}

    if request.entity_type == "performer":
        entity = await stash.get_performer(request.entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Performer not found")
        current_stash_ids = entity.get("stash_ids") or []
        # Skip if already linked
        if not any(s["endpoint"] == request.endpoint and s["stash_id"] == request.stashbox_id for s in current_stash_ids):
            current_stash_ids.append(new_stash_id)
        await stash.update_performer(request.entity_id, stash_ids=current_stash_ids)
        return {"success": True, "entity_id": request.entity_id, "entity_name": entity["name"]}

    elif request.entity_type == "tag":
        # Fetch tag with stash_ids
        tags = await stash.search_tags(request.entity_id, limit=1)
        # search_tags won't find by ID, use a direct query instead
        tag = await stash._execute("""
            query GetTag($id: ID!) {
              findTag(id: $id) { id name stash_ids { endpoint stash_id } }
            }
        """, {"id": request.entity_id})
        tag = tag["findTag"]
        if not tag:
            raise HTTPException(status_code=404, detail="Tag not found")
        current_stash_ids = tag.get("stash_ids") or []
        if not any(s["endpoint"] == request.endpoint and s["stash_id"] == request.stashbox_id for s in current_stash_ids):
            current_stash_ids.append(new_stash_id)
        await stash.update_tag(request.entity_id, stash_ids=current_stash_ids)
        return {"success": True, "entity_id": request.entity_id, "entity_name": tag["name"]}

    elif request.entity_type == "studio":
        studio = await stash._execute("""
            query GetStudio($id: ID!) {
              findStudio(id: $id) { id name stash_ids { endpoint stash_id } }
            }
        """, {"id": request.entity_id})
        studio = studio["findStudio"]
        if not studio:
            raise HTTPException(status_code=404, detail="Studio not found")
        current_stash_ids = studio.get("stash_ids") or []
        if not any(s["endpoint"] == request.endpoint and s["stash_id"] == request.stashbox_id for s in current_stash_ids):
            current_stash_ids.append(new_stash_id)
        await stash.update_studio(request.entity_id, stash_ids=current_stash_ids)
        return {"success": True, "entity_id": request.entity_id, "entity_name": studio["name"]}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown entity type: {request.entity_type}")


@router.post("/actions/update-scene")
async def update_scene_fields(request: UpdateSceneRequest):
    """Apply selected upstream changes to a scene."""
    logger.debug("Action: update-scene scene_id=%s fields=%s", request.scene_id, sorted((request.fields or {}).keys()))
    stash = get_stash_client()
    try:
        result = await _apply_scene_update(
            stash,
            scene_id=request.scene_id,
            fields=request.fields,
            performer_ids=request.performer_ids,
            tag_ids=request.tag_ids,
            studio_id=request.studio_id,
        )
    except RuntimeError as exc:
        msg = str(exc)
        lowered = msg.lower()
        if "scene with id" in lowered and "not found" in lowered:
            db = get_rec_db()
            rec = db.get_recommendation_by_target(
                "upstream_scene_changes",
                "scene",
                str(request.scene_id),
                status="pending",
            )
            if rec:
                db.delete_recommendation(rec.id)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Scene {request.scene_id} not found. "
                    "Removed stale upstream scene recommendation."
                ),
            ) from exc
        raise

    # Resolve recommendation
    db = get_rec_db()
    rec = db.get_recommendation_by_target("upstream_scene_changes", "scene", request.scene_id, status="pending")
    if rec:
        db.resolve_recommendation(rec.id, action="update_scene")

    return {"success": True, "scene": result}


class UpstreamDismissRequest(BaseModel):
    """Request to dismiss an upstream recommendation."""
    reason: Optional[str] = Field(None)
    permanent: bool = Field(False, description="If true, never show updates for this entity again")


@router.post("/{rec_id}/dismiss-upstream")
async def dismiss_upstream_recommendation(rec_id: int, request: UpstreamDismissRequest = None):
    """Dismiss an upstream recommendation with permanent option."""
    permanent = request.permanent if request else False
    logger.debug("Action: dismiss-upstream rec_id=%s permanent=%s", rec_id, permanent)
    db = get_rec_db()
    reason = request.reason if request else None
    success = db.dismiss_recommendation(rec_id, reason=reason, permanent=permanent)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True, "permanent": permanent}


@router.get("/upstream/field-config/{endpoint_b64}", response_model=FieldConfigResponse)
async def get_field_config(endpoint_b64: str, entity_type: str = "performer"):
    """Get field monitoring config for an endpoint.

    Args:
        endpoint_b64: Base64-encoded stash-box endpoint URL.
        entity_type: Entity type to get config for (default: "performer").
    """
    import base64
    from upstream_field_mapper import get_field_config as get_entity_field_config
    endpoint = base64.b64decode(endpoint_b64).decode()
    db = get_rec_db()

    try:
        field_cfg = get_entity_field_config(entity_type)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown entity type: {entity_type}")

    default_fields = field_cfg["default_fields"]
    labels = field_cfg["labels"]

    fields = db.get_enabled_fields(endpoint, entity_type)
    if fields is None:
        return {
            "endpoint": endpoint,
            "fields": {f: {"enabled": True, "label": labels.get(f, f)} for f in default_fields},
        }
    return {
        "endpoint": endpoint,
        "fields": {f: {"enabled": f in fields, "label": labels.get(f, f)} for f in default_fields},
    }


@router.post("/upstream/field-config/{endpoint_b64}", response_model=SuccessResponse)
async def set_field_config(endpoint_b64: str, field_configs: dict[str, bool], entity_type: str = "performer"):
    """Set field monitoring config for an endpoint.

    Args:
        endpoint_b64: Base64-encoded stash-box endpoint URL.
        field_configs: Dict mapping field name to enabled bool.
        entity_type: Entity type to set config for (default: "performer").
    """
    import base64
    endpoint = base64.b64decode(endpoint_b64).decode()
    db = get_rec_db()
    db.set_field_config(endpoint, entity_type, field_configs)
    return {"success": True}


# ==================== Scene Fingerprint Match Actions ====================


class AcceptFingerprintMatchRequest(BaseModel):
    recommendation_id: int
    scene_id: str
    endpoint: str
    stash_id: str


async def _accept_fingerprint_match(
    stash, db, rec_id: int, scene_id: str, endpoint: str, stash_id: str,
):
    """Accept a fingerprint match: add stash_id to local scene, resolve rec."""
    rec = db.get_recommendation(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    if rec.status != "pending":
        return

    scene = await stash.get_scene_by_id(scene_id)
    existing_stash_ids = scene.get("stash_ids") or []

    # Append new stash_id (avoid duplicates)
    already_linked = any(
        s["endpoint"] == endpoint and s["stash_id"] == stash_id
        for s in existing_stash_ids
    )
    if not already_linked:
        updated_stash_ids = existing_stash_ids + [{"endpoint": endpoint, "stash_id": stash_id}]
        await stash.update_scene(scene_id, stash_ids=updated_stash_ids)

    db.resolve_recommendation(rec_id, action="accepted")

    # Accepting one match for a local scene should dismiss all other pending
    # matches for that same scene (across endpoints).
    scene_id_str = str(scene_id)
    db.dismiss_pending_scene_fingerprint_for_scene(
        scene_id=scene_id_str,
        exclude_rec_id=rec_id,
        reason=f"Auto-dismissed after accepting scene fingerprint match for local scene {scene_id_str}",
    )


@router.post("/actions/accept-fingerprint-match", response_model=SuccessResponse)
async def accept_fingerprint_match(request: AcceptFingerprintMatchRequest):
    """Accept a scene fingerprint match — links the stash_id to the local scene."""
    logger.debug("Action: accept-fingerprint-match rec_id=%s scene_id=%s stash_id=%s", request.recommendation_id, request.scene_id, request.stash_id)
    stash = get_stash_client()
    db = get_rec_db()
    await _accept_fingerprint_match(
        stash, db,
        rec_id=request.recommendation_id,
        scene_id=request.scene_id,
        endpoint=request.endpoint,
        stash_id=request.stash_id,
    )
    return {"success": True}


class AcceptAllFingerprintMatchesRequest(BaseModel):
    endpoint: Optional[str] = None


class AcceptSceneTagOnlyChangeRequest(BaseModel):
    rec_id: int


async def _accept_all_fingerprint_matches(
    stash, db, endpoint: Optional[str] = None,
) -> int:
    """Accept all high-confidence fingerprint matches. Returns count accepted."""
    recs = db.get_recommendations(
        status="pending", type="scene_fingerprint_match", limit=10000,
    )

    accepted = 0
    for rec in recs:
        details = rec.details or {}
        if not details.get("high_confidence"):
            continue
        if endpoint and details.get("endpoint") != endpoint:
            continue
        current = db.get_recommendation(rec.id)
        if not current or current.status != "pending":
            continue

        try:
            await _accept_fingerprint_match(
                stash, db,
                rec_id=rec.id,
                scene_id=details["local_scene_id"],
                endpoint=details["endpoint"],
                stash_id=details["stashbox_scene_id"],
            )
            accepted += 1
        except Exception as e:
            logger.warning("Failed to accept rec %s: %s", rec.id, e)

    return accepted


# ==================== Scene Face Matches ====================


class SceneFaceMatchSelection(BaseModel):
    recommendation_id: int


class AcceptSceneFaceMatchesRequest(BaseModel):
    scene_id: str
    selections: list[SceneFaceMatchSelection]


async def _resolve_scene_face_match_performer_id(stash, rec: Recommendation) -> str:
    """Resolve one accepted scene_face_match candidate to a local Stash
    performer id: use the already-linked local performer directly, or
    find-or-create from the candidate's stashbox/catalogue identity (reusing
    the same helper the upstream-performer-review flow uses)."""
    details = rec.details or {}
    local_performer_id = details.get("local_performer_id")
    if local_performer_id:
        return str(local_performer_id)

    endpoint_domain = details.get("endpoint")
    stashdb_id = details.get("stashdb_id")
    name = details.get("name")
    if not endpoint_domain or not stashdb_id or not name:
        raise HTTPException(
            status_code=422,
            detail=f"Recommendation {rec.id} is missing performer identity data",
        )

    # details.endpoint is the short domain form (from universal_id's
    # "<domain>:<id>" split, see PerformerMatchResponse.endpoint) -- resolve
    # to this deployment's actual configured GraphQL URL before storing it
    # in stash_ids, matching the convention every other stash_ids write in
    # this codebase uses (see stashbox_router.py's own endpoint_url
    # resolution). Falls back to the bare domain if this Stash instance
    # has no configured connection for it (still usable as a stash_ids
    # value, just not matchable against a real configured endpoint later).
    from stashbox_connection_manager import get_connection_manager
    endpoint_url = get_connection_manager().get_endpoint_url(endpoint_domain) or endpoint_domain

    performer = await _create_performer_from_stashbox(stash, {"name": name}, endpoint_url, stashdb_id)
    return str(performer["id"])


@router.post("/actions/accept-scene-face-matches", response_model=SuccessResponse)
async def accept_scene_face_matches(request: AcceptSceneFaceMatchesRequest):
    """Accept one or more identified performers for a scene: adds them to
    the scene, resolves the selected recommendations, and dismisses the
    rest of that scene's still-pending candidates (the scene is now
    "handled" and should leave the pending queue)."""
    if not request.selections:
        raise HTTPException(status_code=422, detail="No selections provided")

    stash = get_stash_client()
    db = get_rec_db()

    scene = await stash.get_scene_by_id(request.scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    current_ids = [str(p["id"]) for p in (scene.get("performers") or [])]

    resolved_ids: list[str] = []
    accepted_rec_ids: list[int] = []
    for selection in request.selections:
        rec = db.get_recommendation(selection.recommendation_id)
        if not rec or rec.status != "pending":
            continue
        performer_id = await _resolve_scene_face_match_performer_id(stash, rec)
        resolved_ids.append(performer_id)
        accepted_rec_ids.append(rec.id)

    if not accepted_rec_ids:
        raise HTTPException(status_code=422, detail="No valid pending selections to accept")

    # SceneUpdateInput's performer_ids replaces the full list -- merge with
    # whatever's already on the scene rather than clobbering it.
    merged_ids = list(dict.fromkeys(current_ids + resolved_ids))
    await stash.update_scene_performers(request.scene_id, merged_ids)

    for rec_id in accepted_rec_ids:
        db.resolve_recommendation(rec_id, action="accepted")

    db.dismiss_pending_scene_face_match_for_scene(
        scene_id=request.scene_id,
        exclude_rec_ids=accepted_rec_ids,
        reason=f"Auto-dismissed after accepting scene face matches for scene {request.scene_id}",
    )

    return {"success": True}


class RejectAllSceneFaceMatchesRequest(BaseModel):
    scene_id: str


@router.post("/actions/reject-all-scene-face-matches", response_model=SuccessResponse)
async def reject_all_scene_face_matches(request: RejectAllSceneFaceMatchesRequest):
    """Dismiss every pending scene_face_match candidate for one scene."""
    db = get_rec_db()
    db.dismiss_pending_scene_face_match_for_scene(scene_id=request.scene_id)
    return {"success": True}


class DismissSceneFaceMatchRequest(BaseModel):
    rec_id: int


@router.post("/actions/dismiss-scene-face-match", response_model=SuccessResponse)
async def dismiss_scene_face_match(request: DismissSceneFaceMatchRequest):
    """Dismiss a single scene_face_match candidate (one person's one
    performer guess), leaving the rest of that scene's candidates pending."""
    db = get_rec_db()
    ok = db.dismiss_recommendation(request.rec_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True}


_SCENE_BULK_ALLOWED_SIMPLE_FIELDS = {"code", "urls"}


def _is_real_field_change(change: dict) -> bool:
    """Return True if this change represents an actual value difference.

    Mirrors JS filterRealChanges: skips entries where local and upstream
    are both empty-like or effectively equal, so the two filters agree.
    """
    local = change.get("local_value")
    upstream = change.get("upstream_value")

    def _empty_like(v) -> bool:
        return v is None or v == "" or v == 0 or v == "null" or (isinstance(v, list) and len(v) == 0)

    if _empty_like(local) and _empty_like(upstream):
        return False
    if isinstance(local, str) and isinstance(upstream, str):
        ln, un = local.strip().lower(), upstream.strip().lower()
        if ln == un:
            return False
        if (ln in ("", "null")) and (un in ("", "null")):
            return False
    if isinstance(local, list) and isinstance(upstream, list):
        def _norm(v: object) -> str:
            s = str(v).strip().lower().rstrip("/")
            return "" if s == "null" else s
        ls = {_norm(v) for v in local if _norm(v)}
        us = {_norm(v) for v in upstream if _norm(v)}
        if ls == us:
            return False
    if local == upstream:
        return False
    return True


def _is_tag_url_code_only_scene_change(details: dict) -> bool:
    """Return True when changes are limited to tags/URLs/code only."""
    if not isinstance(details, dict):
        return False

    all_changes = details.get("changes") or []
    simple_changes = [c for c in all_changes if _is_real_field_change(c)]
    for change in simple_changes:
        field = str(change.get("field") or "")
        if field not in _SCENE_BULK_ALLOWED_SIMPLE_FIELDS:
            return False

    if details.get("studio_change"):
        return False

    performer_changes = details.get("performer_changes") or {}
    if performer_changes.get("added") or performer_changes.get("removed"):
        return False

    tag_changes = details.get("tag_changes") or {}
    has_tag_changes = bool(tag_changes.get("added") or tag_changes.get("removed"))
    has_simple_changes = bool(simple_changes)
    return has_tag_changes or has_simple_changes


async def _get_scene_tags_with_stash_ids(stash, scene_id: str) -> list[dict]:
    """Fetch the latest scene tag assignments including stash_id links."""
    data = await stash._execute(
        """
        query SceneTags($id: ID!) {
          findScene(id: $id) {
            id
            tags {
              id
              name
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """,
        {"id": scene_id},
    )
    scene = data.get("findScene")
    if not scene:
        raise RuntimeError(f"Scene not found: {scene_id}")
    return scene.get("tags") or []


def _resolve_scene_tag_local_id(
    scene_tags: list[dict],
    endpoint: str,
    stashbox_id: str,
) -> Optional[str]:
    """Resolve local scene tag ID by upstream endpoint + stashbox tag ID."""
    endpoint_norm = _normalize_endpoint_for_compare(endpoint)
    stashbox_id_str = str(stashbox_id)
    for tag in scene_tags:
        local_tag_id = tag.get("id")
        if local_tag_id is None:
            continue
        for sid in (tag.get("stash_ids") or []):
            sid_endpoint_norm = _normalize_endpoint_for_compare(sid.get("endpoint"))
            sid_stash_id = str(sid.get("stash_id", ""))
            if sid_endpoint_norm == endpoint_norm and sid_stash_id == stashbox_id_str:
                return str(local_tag_id)
    return None


async def _apply_scene_tag_only_recommendation(stash, db, rec) -> dict:
    """Apply one pending upstream scene recommendation (tags/URLs/code only)."""
    current = db.get_recommendation(rec.id)
    if not current or current.status != "pending":
        raise RuntimeError(f"Recommendation {rec.id} is no longer pending")

    details = current.details or {}
    if current.type != "upstream_scene_changes" or not _is_tag_url_code_only_scene_change(details):
        if current.type != "upstream_scene_changes":
            _rejection_reason = f"wrong type: {current.type!r}"
        else:
            _blocking = []
            for _ch in (details.get("changes") or []):
                if not _is_real_field_change(_ch):
                    continue
                _f = str(_ch.get("field") or "")
                if _f not in _SCENE_BULK_ALLOWED_SIMPLE_FIELDS:
                    _blocking.append(f"field={_f!r}")
            if details.get("studio_change"):
                _blocking.append("studio_change")
            _pc = details.get("performer_changes") or {}
            if _pc.get("added") or _pc.get("removed"):
                _blocking.append(f"performer_changes(added={len(_pc.get('added') or [])},removed={len(_pc.get('removed') or [])})")
            _tc = details.get("tag_changes") or {}
            _sc = details.get("changes") or []
            if not _tc.get("added") and not _tc.get("removed") and not _sc:
                _blocking.append("no_tag_or_simple_changes")
            _rejection_reason = ", ".join(_blocking) if _blocking else "unknown"
        logger.debug(
            "Rec %s rejected as not tag/url/code-only: %s | detail_keys=%s",
            rec.id, _rejection_reason, sorted(details.keys()),
        )
        raise RuntimeError(
            f"Recommendation {rec.id} is not a tag/url/code-only upstream scene change"
        )

    scene_id = str(details.get("scene_id") or "").strip()
    endpoint = str(details.get("endpoint") or "").strip()
    logger.debug("Applying tag/url/code-only scene change rec_id=%s scene_id=%s endpoint=%s", rec.id, scene_id, endpoint)
    tag_changes = details.get("tag_changes") or {}
    simple_changes = details.get("changes") or []
    if not scene_id or not endpoint:
        raise RuntimeError(f"Recommendation {rec.id} missing scene_id/endpoint")

    ensured_tags = 0
    simple_fields: dict = {}
    scene_tags = await _get_scene_tags_with_stash_ids(stash, scene_id)
    current_tag_ids = {
        str(tag.get("id"))
        for tag in scene_tags
        if tag.get("id") is not None
    }
    next_tag_ids = set(current_tag_ids)

    for removed in (tag_changes.get("removed") or []):
        removed_stashbox_id = str(removed.get("id") or "").strip()
        if not removed_stashbox_id:
            continue
        local_tag_id = _resolve_scene_tag_local_id(
            scene_tags=scene_tags,
            endpoint=endpoint,
            stashbox_id=removed_stashbox_id,
        )
        if local_tag_id:
            next_tag_ids.discard(local_tag_id)

    for added in (tag_changes.get("added") or []):
        local_tag_id = str(((added.get("local_match") or {}).get("id") or "")).strip()
        if not local_tag_id:
            stashbox_id = str(added.get("id") or "").strip()
            if not stashbox_id:
                raise RuntimeError(
                    f"Missing upstream tag id in recommendation {current.id}"
                )
            tag_payload = {
                "name": added.get("name", ""),
                "aliases": added.get("aliases") or [],
            }
            created_or_linked = await _create_tag_from_stashbox(
                stash=stash,
                stashbox_data=tag_payload,
                endpoint=endpoint,
                stashbox_id=stashbox_id,
            )
            local_tag_id = str(created_or_linked.get("id"))
            ensured_tags += 1

        if local_tag_id:
            next_tag_ids.add(local_tag_id)

    def _normalize_nullish_text(value) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() == "null" else text

    for change in simple_changes:
        field = str(change.get("field") or "")
        if field not in _SCENE_BULK_ALLOWED_SIMPLE_FIELDS:
            raise RuntimeError(
                f"Unsupported simple field for bulk scene accept: {field}"
            )
        if field == "code":
            simple_fields["code"] = _normalize_nullish_text(change.get("upstream_value"))
        elif field == "urls":
            upstream_urls = change.get("upstream_value")
            if upstream_urls is None:
                simple_fields["urls"] = []
            elif isinstance(upstream_urls, list):
                simple_fields["urls"] = [
                    _normalize_nullish_text(url)
                    for url in upstream_urls
                    if _normalize_nullish_text(url)
                ]
            else:
                url = _normalize_nullish_text(upstream_urls)
                simple_fields["urls"] = [url] if url else []

    if next_tag_ids != current_tag_ids or simple_fields:
        await _apply_scene_update(
            stash=stash,
            scene_id=scene_id,
            fields=simple_fields,
            tag_ids=sorted(next_tag_ids),
        )
        action = "applied"
    else:
        action = "accepted_no_changes"

    db.resolve_recommendation(
        current.id,
        action=action,
        details={"bulk": "tag_url_code_only_scene_changes"},
    )
    logger.debug("Completed tag/url/code-only scene change rec_id=%s scene_id=%s action=%s ensured_tags=%d", current.id, scene_id, action, ensured_tags)
    return {
        "rec_id": current.id,
        "scene_id": scene_id,
        "action": action,
        "ensured_tags_count": ensured_tags,
    }


async def _accept_all_scene_tag_only_changes(stash, db) -> dict:
    """Accept pending scene recommendations with only tag/URL/code changes."""
    recs = db.get_recommendations(
        status="pending", type="upstream_scene_changes", limit=10000,
    )

    accepted = 0
    failed = 0
    skipped = 0
    ensured_tags = 0

    for rec in recs:
        current = db.get_recommendation(rec.id)
        if not current or current.status != "pending":
            continue

        details = current.details or {}
        if not _is_tag_url_code_only_scene_change(details):
            skipped += 1
            continue

        try:
            apply_result = await _apply_scene_tag_only_recommendation(stash, db, rec)
            ensured_tags += int(apply_result.get("ensured_tags_count") or 0)
            accepted += 1
        except Exception as e:
            failed += 1
            scene_id = str((details or {}).get("scene_id") or "")
            logger.warning(
                "Failed bulk tag/url/code-only apply for rec %s (scene %s): %s",
                current.id,
                scene_id,
                e,
            )

    return {
        "accepted_count": accepted,
        "failed_count": failed,
        "skipped_count": skipped,
        "ensured_tags_count": ensured_tags,
    }


def _is_url_only_performer_change(details: dict) -> bool:
    """Return True when the only change is to the urls field."""
    if not isinstance(details, dict):
        return False
    changes = details.get("changes") or []
    if not changes:
        return False
    for change in changes:
        if str(change.get("field") or "") != "urls":
            return False
    return True


async def _accept_all_performer_url_only_changes(stash, db) -> dict:
    """Accept pending performer recommendations with only URL changes."""
    recs = db.get_recommendations(
        status="pending", type="upstream_performer_changes", limit=10000,
    )

    accepted = 0
    failed = 0
    skipped = 0

    for rec in recs:
        current = db.get_recommendation(rec.id)
        if not current or current.status != "pending":
            continue

        details = current.details or {}
        if not _is_url_only_performer_change(details):
            skipped += 1
            continue

        performer_id = str(details.get("performer_id") or "").strip()
        if not performer_id:
            skipped += 1
            continue

        changes = details.get("changes") or []
        upstream_urls = None
        for change in changes:
            if change.get("field") == "urls":
                upstream_urls = change.get("upstream_value")
                break

        if upstream_urls is None:
            skipped += 1
            continue

        if isinstance(upstream_urls, list):
            url_list = [str(u) for u in upstream_urls if u]
        else:
            url_list = [str(upstream_urls)] if upstream_urls else []

        try:
            await stash.update_performer(performer_id, urls=url_list)
            db.resolve_recommendation(
                current.id,
                action="applied",
                details={"bulk": "url_only_performer_changes"},
            )
            accepted += 1
        except Exception as e:
            failed += 1
            logger.warning(
                "Failed bulk URL-only apply for rec %s (performer %s): %s",
                current.id,
                performer_id,
                e,
            )

    return {"accepted_count": accepted, "failed_count": failed, "skipped_count": skipped}


@router.post("/actions/accept-all-performer-url-only-changes")
async def accept_all_performer_url_only_changes():
    """Accept pending performer recommendations with only URL changes."""
    logger.debug("Action: accept-all-performer-url-only-changes started")
    stash = get_stash_client()
    db = get_rec_db()
    result = await _accept_all_performer_url_only_changes(stash, db)
    logger.debug("Action: accept-all-performer-url-only-changes done %s", result)
    return {"success": True, **result}


@router.post("/actions/accept-all-fingerprint-matches")
async def accept_all_fingerprint_matches(
    request: AcceptAllFingerprintMatchesRequest = AcceptAllFingerprintMatchesRequest(),
):
    """Accept all high-confidence scene fingerprint matches."""
    logger.debug("Action: accept-all-fingerprint-matches endpoint_filter=%s", request.endpoint)
    stash = get_stash_client()
    db = get_rec_db()
    accepted = await _accept_all_fingerprint_matches(stash, db, request.endpoint)
    logger.debug("Action: accept-all-fingerprint-matches done accepted=%d", accepted)
    return {"success": True, "accepted_count": accepted}


@router.post("/actions/scene-tag-only-stats")
async def scene_tag_only_stats():
    """Count pending upstream_scene_changes by tag/URL/code-only filter eligibility and log to debug."""
    db = get_rec_db()
    recs = db.get_recommendations(status="pending", type="upstream_scene_changes", limit=100000)
    total = len(recs)
    matching = sum(1 for r in recs if _is_tag_url_code_only_scene_change(r.details or {}))
    non_matching = total - matching
    logger.debug(
        "Tag/URL/code-only filter stats: %d pending upstream_scene_changes, %d match filter, %d do not match",
        total, matching, non_matching,
    )
    return {"total": total, "matching": matching, "non_matching": non_matching}


@router.post("/actions/performer-url-only-stats")
async def performer_url_only_stats():
    """Count pending upstream_performer_changes by URL-only filter eligibility and log to debug."""
    db = get_rec_db()
    recs = db.get_recommendations(status="pending", type="upstream_performer_changes", limit=100000)
    total = len(recs)
    matching = sum(1 for r in recs if _is_url_only_performer_change(r.details or {}))
    non_matching = total - matching
    logger.debug(
        "Performer URL-only filter stats: %d pending upstream_performer_changes, %d match filter, %d do not match",
        total, matching, non_matching,
    )
    return {"total": total, "matching": matching, "non_matching": non_matching}


@router.post("/actions/fingerprint-match-stats")
async def fingerprint_match_stats():
    """Count pending scene_fingerprint_match by high_confidence flag and log to debug."""
    db = get_rec_db()
    recs = db.get_recommendations(status="pending", type="scene_fingerprint_match", limit=100000)
    total = len(recs)
    high_conf = sum(1 for r in recs if (r.details or {}).get("high_confidence"))
    logger.debug(
        "Fingerprint match stats: %d pending scene_fingerprint_match, %d high_confidence, %d not",
        total, high_conf, total - high_conf,
    )
    return {"total": total, "high_confidence": high_conf, "non_high_confidence": total - high_conf}


class BulkAcceptStatsRequest(BaseModel):
    type: str


@router.post("/actions/bulk-accept-stats")
async def bulk_accept_stats(request: BulkAcceptStatsRequest):
    """Count pending recommendations of a type with any changes and log to debug.

    Used by the Accept All Changes button for upstream performer/tag/studio changes.
    Scenes are excluded from batch accept (require individual review).
    """
    if request.type == "upstream_scene_changes":
        logger.debug(
            "Bulk accept stats for upstream_scene_changes: scenes excluded from batch accept (require individual review)",
        )
        return {"total": 0, "with_changes": 0, "note": "scenes excluded from batch accept"}
    db = get_rec_db()
    recs = db.get_recommendations(status="pending", type=request.type, limit=100000)
    total = len(recs)
    with_changes = sum(1 for r in recs if (r.details or {}).get("changes"))
    logger.debug(
        "Bulk accept stats for %s: %d total pending, %d have changes",
        request.type, total, with_changes,
    )
    return {"total": total, "with_changes": with_changes}


@router.post("/actions/accept-all-scene-tag-only-changes")
async def accept_all_scene_tag_only_changes():
    """Accept pending scene recommendations with only tag/URL/code changes."""
    logger.debug("Action: accept-all-scene-tag-only-changes started")
    stash = get_stash_client()
    db = get_rec_db()
    result = await _accept_all_scene_tag_only_changes(stash, db)
    logger.debug("Action: accept-all-scene-tag-only-changes done %s", result)
    return {"success": True, **result}


@router.post("/actions/accept-scene-tag-only-change")
async def accept_scene_tag_only_change(request: AcceptSceneTagOnlyChangeRequest):
    """Accept one pending upstream scene recommendation (tags/URLs/code only)."""
    stash = get_stash_client()
    db = get_rec_db()
    rec = db.get_recommendation(request.rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    scene_id = str((rec.details or {}).get("scene_id") or "?")
    logger.debug("Action: accept-scene-tag-only-change rec_id=%s scene_id=%s", request.rec_id, scene_id)
    try:
        result = await _apply_scene_tag_only_recommendation(stash, db, rec)
        return {"success": True, **result}
    except RuntimeError as e:
        msg = str(e)
        if "scene not found" in msg.lower():
            db.delete_recommendation(rec.id)
            logger.debug("Deleted stale scene rec rec_id=%s scene_id=%s", rec.id, scene_id)
            return {"success": True, "action": "deleted_stale_scene", "rec_id": rec.id, "scene_id": scene_id}
        logger.debug("accept-scene-tag-only-change returned 400 rec_id=%s scene_id=%s: %s", rec.id, scene_id, msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.warning(
            "Unexpected error accepting scene tag/url/code change for rec %s (scene %s): %s",
            rec.id, scene_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Accept failed for rec {rec.id} (scene {scene_id}): {type(e).__name__}: {e}",
        )


async def _apply_full_scene_recommendation(stash, db, rec) -> dict:
    """Accept one pending upstream scene recommendation applying ALL change types.

    Applies simple field changes (upstream value as default), creates/links
    performers, tags, and studio as needed (with full enrichment), applies
    relational additions, and removes entities that were specifically removed
    upstream. Any local performers/tags not in the removed list are preserved.
    """
    current = db.get_recommendation(rec.id)
    if not current or current.status != "pending":
        raise RuntimeError(f"Recommendation {rec.id} is no longer pending")
    if current.type != "upstream_scene_changes":
        raise RuntimeError(f"Recommendation {rec.id} is not an upstream_scene_changes rec")

    details = current.details or {}
    scene_id = str(details.get("scene_id") or "").strip()
    endpoint = str(details.get("endpoint") or "").strip()
    if not scene_id or not endpoint:
        raise RuntimeError(f"Recommendation {rec.id} missing scene_id/endpoint")

    logger.debug("Applying full scene change rec_id=%s scene_id=%s endpoint=%s", rec.id, scene_id, endpoint)

    def _nullish_to_empty(v) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        return "" if s.lower() == "null" else s

    # 1. Simple field changes — take upstream value for all real changes
    simple_fields: dict = {}
    for change in [c for c in (details.get("changes") or []) if _is_real_field_change(c)]:
        field = str(change.get("field") or "")
        upstream_val = change.get("upstream_value")
        if not field:
            continue
        if field == "urls":
            if upstream_val is None:
                simple_fields["urls"] = []
            elif isinstance(upstream_val, list):
                simple_fields["urls"] = [u for u in [_nullish_to_empty(v) for v in upstream_val] if u]
            else:
                url = _nullish_to_empty(upstream_val)
                simple_fields["urls"] = [url] if url else []
        else:
            simple_fields[field] = _nullish_to_empty(upstream_val)

    # 2. Tags — build next_tag_ids from current set + add − remove
    scene_tags = await _get_scene_tags_with_stash_ids(stash, scene_id)
    current_tag_ids = {str(t["id"]) for t in scene_tags if t.get("id") is not None}
    next_tag_ids = set(current_tag_ids)
    ensured_tags = 0

    tag_changes = details.get("tag_changes") or {}
    for removed in (tag_changes.get("removed") or []):
        removed_stashbox_id = str(removed.get("id") or "").strip()
        if not removed_stashbox_id:
            continue
        local_tag_id = _resolve_scene_tag_local_id(scene_tags, endpoint, removed_stashbox_id)
        if local_tag_id:
            next_tag_ids.discard(local_tag_id)

    for added in (tag_changes.get("added") or []):
        local_tag_id = str(((added.get("local_match") or {}).get("id") or "")).strip()
        if not local_tag_id:
            stashbox_id = str(added.get("id") or "").strip()
            if not stashbox_id:
                continue
            # Look up existing by stash_id before creating
            linked = await _find_linked_entity_by_stash_id(stash, "tag", endpoint, stashbox_id)
            if linked and linked.get("id"):
                local_tag_id = str(linked["id"])
            else:
                tag_payload = {"name": added.get("name", ""), "aliases": added.get("aliases") or []}
                created_or_linked = await _create_tag_from_stashbox(stash, tag_payload, endpoint, stashbox_id)
                local_tag_id = str(created_or_linked.get("id"))
                ensured_tags += 1
        if local_tag_id:
            next_tag_ids.add(local_tag_id)

    # 3. Performers — build next_performer_ids from current set + add − remove
    current_performer_ids = set(str(p) for p in (details.get("current_performer_ids") or []))
    next_performer_ids = set(current_performer_ids)
    ensured_performers = 0

    performer_changes = details.get("performer_changes") or {}
    for removed in (performer_changes.get("removed") or []):
        removed_stashbox_id = str(removed.get("id") or "").strip()
        if not removed_stashbox_id:
            continue
        linked = await _find_linked_entity_by_stash_id(stash, "performer", endpoint, removed_stashbox_id)
        if linked and linked.get("id"):
            next_performer_ids.discard(str(linked["id"]))

    for added in (performer_changes.get("added") or []):
        local_match_id = str(((added.get("local_match") or {}).get("id") or "")).strip()
        if not local_match_id:
            stashbox_id = str(added.get("id") or "").strip()
            if not stashbox_id:
                continue
            linked = await _find_linked_entity_by_stash_id(stash, "performer", endpoint, stashbox_id)
            if linked and linked.get("id"):
                local_match_id = str(linked["id"])
            else:
                perf_payload = {
                    "name": added.get("name", ""),
                    "aliases": added.get("aliases") or [],
                    "gender": added.get("gender"),
                }
                created = await _create_performer_from_stashbox(stash, perf_payload, endpoint, stashbox_id)
                local_match_id = str(created.get("id"))
                ensured_performers += 1
        if local_match_id:
            next_performer_ids.add(local_match_id)

    # 4. Studio — look up or create upstream studio
    studio_id: Optional[str] = None
    studio_change = details.get("studio_change")
    if studio_change:
        upstream_studio = studio_change.get("upstream") or {}
        local_match_id = str(((upstream_studio.get("local_match") or {}).get("id") or "")).strip()
        if not local_match_id:
            stashbox_id = str(upstream_studio.get("id") or "").strip()
            if stashbox_id:
                linked = await _find_linked_entity_by_stash_id(stash, "studio", endpoint, stashbox_id)
                if linked and linked.get("id"):
                    local_match_id = str(linked["id"])
                else:
                    studio_payload = {"name": upstream_studio.get("name", "")}
                    created = await _create_studio_from_stashbox(stash, studio_payload, endpoint, stashbox_id)
                    local_match_id = str(created.get("id"))
        if local_match_id:
            studio_id = local_match_id

    performers_changed = next_performer_ids != current_performer_ids
    tags_changed = next_tag_ids != current_tag_ids
    has_changes = bool(simple_fields) or performers_changed or tags_changed or studio_id is not None

    if has_changes:
        await _apply_scene_update(
            stash=stash,
            scene_id=scene_id,
            fields=simple_fields,
            performer_ids=sorted(next_performer_ids) if performers_changed else None,
            tag_ids=sorted(next_tag_ids) if tags_changed else None,
            studio_id=studio_id,
        )
        action = "applied"
    else:
        action = "accepted_no_changes"

    db.resolve_recommendation(current.id, action=action, details={"bulk": "full_scene_changes"})
    logger.debug(
        "Completed full scene change rec_id=%s scene_id=%s action=%s ensured_performers=%d ensured_tags=%d",
        current.id, scene_id, action, ensured_performers, ensured_tags,
    )
    return {
        "rec_id": current.id,
        "scene_id": scene_id,
        "action": action,
        "ensured_performers_count": ensured_performers,
        "ensured_tags_count": ensured_tags,
    }


class AcceptSceneChangeRequest(BaseModel):
    rec_id: int


@router.post("/actions/accept-scene-change")
async def accept_scene_change(request: AcceptSceneChangeRequest):
    """Accept one pending upstream scene recommendation applying all change types.

    Creates/links performers, tags, and studio as needed (with full enrichment
    from the stashbox endpoint). Applies removals only for entities specifically
    removed upstream; local-only additions are preserved.
    """
    stash = get_stash_client()
    db = get_rec_db()
    rec = db.get_recommendation(request.rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    scene_id = str((rec.details or {}).get("scene_id") or "?")
    logger.debug("Action: accept-scene-change rec_id=%s scene_id=%s", request.rec_id, scene_id)
    try:
        result = await _apply_full_scene_recommendation(stash, db, rec)
        return {"success": True, **result}
    except RuntimeError as e:
        msg = str(e)
        if "scene not found" in msg.lower():
            db.delete_recommendation(rec.id)
            logger.debug("Deleted stale scene rec rec_id=%s scene_id=%s", rec.id, scene_id)
            return {"success": True, "action": "deleted_stale_scene", "rec_id": rec.id, "scene_id": scene_id}
        logger.debug("accept-scene-change returned 400 rec_id=%s scene_id=%s: %s", rec.id, scene_id, msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.warning(
            "Unexpected error accepting full scene change for rec %s (scene %s): %s",
            rec.id, scene_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Accept failed for rec {rec.id} (scene {scene_id}): {type(e).__name__}: {e}",
        )


class BatchDismissRequest(BaseModel):
    """Request to batch dismiss all pending recommendations of a type."""
    type: str = Field(..., description="Recommendation type to dismiss")
    permanent: bool = Field(False, description="If true, never show these again")


@router.post("/actions/batch-dismiss")
async def batch_dismiss(request: BatchDismissRequest):
    """Dismiss all pending recommendations of a given type."""
    logger.debug("Action: batch-dismiss type=%s permanent=%s", request.type, request.permanent)
    db = get_rec_db()
    dismissed_count = db.batch_dismiss_by_type(
        rec_type=request.type,
        permanent=request.permanent,
        reason="Batch dismissed by user",
    )
    return {"success": True, "dismissed_count": dismissed_count}

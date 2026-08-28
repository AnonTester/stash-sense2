"""Identification API endpoints for face recognition.

Provides routes for identifying performers in images, galleries, and scenes
using face recognition.
"""

import asyncio
import base64
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

import face_config
from recognizer import FaceRecognizer, PerformerMatch, RecognitionResult
from embeddings import load_image, DetectedFace, FaceEmbedding, gpu_compute_lock
from frame_extractor import (
    ExtractionResult,
    FrameExtractionConfig,
    extract_frames_from_stash_scene,
    check_ffmpeg_available,
)
from matching import MatchingConfig
from scene_matcher import (
    cluster_faces_by_person,
    merge_clusters_by_match,
    aggregate_matches,
    frequency_based_matching,
    clustered_frequency_matching,
    hybrid_matching,
)
from stashbox_utils import _get_stashbox_client, _extract_endpoint
from sprite_parser import fetch_sprite_from_stash
from recommendations_router import (
    save_scene_fingerprint,
    save_image_fingerprint,
    get_scene_signal_cache,
    is_scene_cache_compatible,
    save_scene_signal_cache,
    save_face_signal_cache,
    load_face_signal_cache,
)
from database_updater import UpdateStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["identification"])

# Image URL cache keyed by universal_id
_image_cache: dict[str, Optional[str]] = {}

# Best-effort, in-memory "what stage is this scene's /identify/scene call
# doing right now" tracker for the scene-page progress UI. Ephemeral and
# process-local (not persisted, not multi-worker safe) -- this is a coarse
# stage indicator for a single synchronous request, not a job/queue
# progress system. Keyed by scene_id since only one identify call per scene
# is expected at a time from the UI.
_scene_progress: dict[str, dict] = {}


def _set_stage(scene_id: str, stage: str) -> None:
    _scene_progress[scene_id] = {"stage": stage, "updated_at": time.time()}


def _scene_needs_vaapi(width: Optional[int], height: Optional[int]) -> bool:
    """True only for genuinely >=4K scenes (resolution-only trigger).

    FFMPEG_HWACCEL=vaapi exists specifically to keep 4K+ frame decode+resize
    off the CPU/RAM path -- large frames there were causing real OOM
    crashes. Below that threshold, CPU decode is fast enough on its own and
    skipping VAAPI also avoids driving GPU video-decode concurrently with
    ROCm/HIP compute on the same iGPU (the two were found to destabilize
    each other when run at once). FFMPEG_HWACCEL is the ceiling ("VAAPI
    allowed if set at all"); this decides whether a specific scene actually
    uses it.
    """
    return bool(width and width >= 3840) or bool(height and height >= 2160)

# Module-level globals set by init
_recognizer = None
_db_manifest = {}
_db_updater = None
_stash_url = ""
_stash_api_key = ""

# See embeddings.py's GPU_COMPUTE_LOCK/gpu_compute_lock() docstrings --
# shared process-wide, not local to this module, since
# local_performer_sync_job.py's own separate generator pool needs to
# serialize against the same GPU. Held only around actual detect calls,
# not extraction (ffmpeg, CPU-only), get_embeddings_batch (pure Python
# read-back, no GPU call -- see embeddings.py's own docstring), or matching
# (usearch query, cheap) -- see _process_sprite_frames/_identify_scene_compute.
_gpu_compute_lock = gpu_compute_lock


def init_identification_router(
    recognizer,
    db_manifest: dict,
    db_updater,
    stash_url: str,
    stash_api_key: str,
):
    """Initialize the identification router with runtime dependencies.

    Called from main.py lifespan after models are loaded.
    """
    global _recognizer, _db_manifest, _db_updater
    global _stash_url, _stash_api_key
    _recognizer = recognizer
    _db_manifest = db_manifest
    _db_updater = db_updater
    _stash_url = stash_url
    _stash_api_key = stash_api_key


_UNSET = object()  # sentinel to distinguish "not provided" from None


def update_identification_globals(
    recognizer=_UNSET,
    db_manifest=_UNSET,
):
    """Update globals after a database hot-swap or idle unload."""
    global _recognizer, _db_manifest
    if recognizer is not _UNSET:
        _recognizer = recognizer
    if db_manifest is not _UNSET:
        _db_manifest = db_manifest


# ==================== Pydantic Models ====================


class FaceBox(BaseModel):
    """Bounding box for a detected face."""
    x: int
    y: int
    width: int
    height: int
    confidence: float


class PerformerMatchResponse(BaseModel):
    """A potential performer match."""
    stashdb_id: str = Field(description="StashDB performer UUID")
    name: str = Field(description="Performer name")
    confidence: float = Field(description="Match confidence (0-1, higher is better)")
    distance: float = Field(description="Distance score (lower is better)")
    country: Optional[str] = None
    image_url: Optional[str] = Field(None, description="StashDB profile image URL")
    endpoint: Optional[str] = Field(None, description="StashBox endpoint domain (e.g. 'stashdb.org'), or 'local' for a local-library match")
    already_tagged: bool = Field(False, description="Whether this performer is already tagged on the scene")
    local_performer_id: Optional[str] = Field(None, description="Local Stash performer id, set only for local-index matches")
    source: Optional[str] = Field(None, description="Set only for catalogue (non-stash-box) matches, e.g. 'seekfans'")
    catalogue_url: Optional[str] = Field(None, description="Catalogue source's own profile page, set only for catalogue matches")
    profile_url: Optional[str] = Field(None, description="Link to the actual external content site (e.g. onlyfans.com), when the catalogue source has one")
    top_timestamps_sec: list[float] = Field(default_factory=list, description="Up to 4 timestamps (seconds) of this match's strongest frames, for scene-player jump buttons. Only populated by scene identification's live ffmpeg-extraction path (matching_mode='cluster'); empty otherwise.")


class FaceResult(BaseModel):
    """Recognition result for a single detected face."""
    box: FaceBox
    matches: list[PerformerMatchResponse]


class IdentifyRequest(BaseModel):
    """Request to identify performers in an image."""
    image_url: Optional[str] = Field(None, description="URL to fetch image from")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image data")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")


class IdentifyResponse(BaseModel):
    """Response with identification results."""
    faces: list[FaceResult]
    face_count: int


class ImageIdentifyRequest(BaseModel):
    """Request to identify performers in a Stash image by ID."""
    image_id: str = Field(description="Stash image ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")


class GalleryPerformerResult(BaseModel):
    """A performer identified across a gallery."""
    performer_id: str = Field(description="StashDB performer UUID")
    name: str
    best_distance: float
    avg_distance: float
    confidence: float = Field(description="Best match confidence (0-1)")
    image_count: int = Field(description="Number of images this performer appeared in")
    image_ids: list[str] = Field(description="Stash image IDs where performer was found")
    country: Optional[str] = None
    image_url: Optional[str] = Field(None, description="StashDB profile image URL")
    endpoint: Optional[str] = Field(None, description="StashBox endpoint domain")
    local_performer_id: Optional[str] = Field(None, description="Local Stash performer id, set only for local-index matches")


class GalleryIdentifyRequest(BaseModel):
    """Request to identify performers in a Stash gallery."""
    gallery_id: str = Field(description="Stash gallery ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")


class GalleryIdentifyResponse(BaseModel):
    """Response with gallery identification results."""
    gallery_id: str
    total_images: int
    images_processed: int
    faces_detected: int
    performers: list[GalleryPerformerResult]
    errors: list[str] = []


class SceneIdentifyRequest(BaseModel):
    """Request to identify performers in a scene using ffmpeg frame extraction."""
    stash_url: Optional[str] = Field(None, description="Base URL of Stash instance (or use STASH_URL env var)")
    scene_id: str = Field(description="Scene ID in Stash")
    api_key: Optional[str] = Field(None, description="Stash API key (or use STASH_API_KEY env var)")

    # Frame extraction settings (defaults from face_config.py)
    num_frames: int = Field(face_config.NUM_FRAMES, ge=5, le=120, description="Number of frames to extract")
    start_offset_pct: float = Field(face_config.START_OFFSET_PCT, ge=0.0, le=0.5, description="Skip first N% of video")
    end_offset_pct: float = Field(face_config.END_OFFSET_PCT, ge=0.5, le=1.0, description="Stop at N% of video")

    # Face detection settings
    min_face_size: int = Field(face_config.MIN_FACE_SIZE, ge=20, le=200, description="Minimum face size in pixels")
    min_face_confidence: float = Field(face_config.MIN_FACE_CONFIDENCE, ge=0.1, le=1.0, description="Minimum face detection confidence")

    # Matching settings
    top_k: int = Field(face_config.TOP_K, ge=1, le=10, description="Matches per person")
    max_distance: float = Field(face_config.MAX_DISTANCE, ge=0.0, le=2.0, description="Maximum match distance")

    # Clustering settings
    cluster_threshold: float = Field(face_config.CLUSTER_THRESHOLD, ge=0.2, le=3.0, description="Distance threshold for face clustering")

    # Matching mode: "cluster", "frequency", or "hybrid"
    matching_mode: str = Field("frequency", description="Matching mode: 'cluster' (cluster faces then match), 'frequency' (count performer appearances), or 'hybrid' (combine both)")

    # Already-tagged performers (StashDB IDs) for boosting
    scene_performer_stashdb_ids: list[str] = Field(default_factory=list, description="StashDB IDs of performers already tagged on this scene")

    # Cache settings
    use_cache: bool = Field(True, description="Reuse a cached prior extraction on this scene when detection params match, skipping ffmpeg/detection/embedding and only redoing matching")

    # Sprite-sheet settings
    use_sprite: bool = Field(False, description="Additionally detect and match faces from the scene's sprite/thumbnail sheet (Stash's scrubber-bar preview tiles), merged in alongside video-frame results")
    skip_frame_extraction: bool = Field(False, description="Skip ffmpeg video-frame extraction entirely -- identify from sprite tiles only. No fingerprint is saved when this is set, since no video frames were analyzed.")


class PersonResult(BaseModel):
    """A unique person detected across multiple frames."""
    person_id: int = Field(description="Unique ID for this person in the scene")
    frame_count: int = Field(description="Number of frames this person appeared in")
    best_match: Optional[PerformerMatchResponse] = Field(description="Best performer match")
    all_matches: list[PerformerMatchResponse] = Field(description="All potential matches")


class SceneIdentifyResponse(BaseModel):
    """Response with scene identification results."""
    scene_id: str
    frames_analyzed: int
    frames_requested: int = 0
    faces_detected: int
    faces_after_filter: int = 0
    persons: list[PersonResult]
    errors: list[str] = []
    fingerprint_saved: bool = False
    fingerprint_error: Optional[str] = None
    timing: Optional[dict] = None
    used_cache: bool = False


# ==================== Helper Functions ====================


def distance_to_confidence(distance: float) -> float:
    """Convert distance score to confidence (0-1, higher is better)."""
    # Cosine distance ranges from 0 (identical) to 2 (opposite)
    # Map to confidence: 0 distance -> 1.0 confidence, 1.0 distance -> 0.0 confidence
    return max(0.0, min(1.0, 1.0 - distance))


def _match_to_response(m, **overrides) -> PerformerMatchResponse:
    """Convert a PerformerMatch (or PerformerMatchResponse) to PerformerMatchResponse."""
    uid = getattr(m, "universal_id", None)
    score = getattr(m, "combined_score", getattr(m, "distance", 0))
    defaults = dict(
        stashdb_id=m.stashdb_id,
        name=m.name,
        confidence=distance_to_confidence(score),
        distance=score,
        country=m.country,
        image_url=m.image_url,
        endpoint=_extract_endpoint(uid) or getattr(m, "endpoint", None),
        local_performer_id=getattr(m, "local_performer_id", None),
        source=getattr(m, "source", None),
        catalogue_url=getattr(m, "catalogue_url", None),
        profile_url=getattr(m, "profile_url", None),
    )
    defaults.update(overrides)
    return PerformerMatchResponse(**defaults)


async def _fetch_image_for_match(match: PerformerMatch) -> None:
    """Fetch and cache image URL for a single match from StashBox."""
    uid = match.universal_id
    if uid in _image_cache:
        match.image_url = _image_cache[uid]
        return

    endpoint = _extract_endpoint(uid)
    if not endpoint:
        return

    client = _get_stashbox_client(endpoint)
    if not client:
        return

    try:
        performer = await client.get_performer(match.stashdb_id)
        if performer:
            images = performer.get("images") or []
            image_url = images[0].get("url") if images else None
            _image_cache[uid] = image_url
            match.image_url = image_url
        else:
            _image_cache[uid] = None
    except Exception as e:
        logger.debug(f"Failed to fetch image for {uid}: {e}")
        _image_cache[uid] = None


async def _fetch_missing_images(all_matches: list[PerformerMatch]) -> None:
    """Fetch missing image URLs from StashBox for matches that have None."""
    needs_fetch = [
        m for m in all_matches
        if m.image_url is None and m.universal_id not in _image_cache
    ]

    # Apply cache hits for already-cached entries
    for m in all_matches:
        if m.image_url is None and m.universal_id in _image_cache:
            m.image_url = _image_cache[m.universal_id]

    if not needs_fetch:
        return

    tasks = [_fetch_image_for_match(m) for m in needs_fetch]
    await asyncio.gather(*tasks, return_exceptions=True)


async def require_db_available():
    """Ensure face recognition is loaded, return 503 if unavailable.

    Normally already loaded by main.py's startup eager-load task by the
    time any real request arrives; this is the fallback path (still eager
    triggers ResourceManager's lazy-load) for a request that lands while
    that background load is still in flight, or if it failed and a later
    request is retrying. Returns 503 if a database update is in progress
    or if loading fails.
    """
    if _db_updater and _db_updater._state.status in (
        UpdateStatus.SWAPPING, UpdateStatus.RELOADING,
    ):
        raise HTTPException(
            status_code=503,
            detail="Database update in progress",
            headers={"Retry-After": "10"},
        )

    # Lazy-load face recognition via ResourceManager if not yet loaded,
    # and touch last_access on every request to prevent idle unloading
    # while the resource is actively in use (e.g. fingerprint generation).
    #
    # require() runs in a thread (not awaited inline) so this blocking,
    # multi-second model load doesn't stall the whole event loop -- a
    # concurrent /health poll needs to keep responding promptly (with
    # loading=True) so the frontend can show real feedback instead of the
    # request just hanging with no explanation.
    try:
        from resource_manager import get_resource_manager
        mgr = get_resource_manager()
        if _recognizer is None:
            await asyncio.to_thread(mgr.require, "face_recognition")
            # After require(), router globals are updated by the loader
        else:
            mgr.touch("face_recognition")
    except RuntimeError:
        # ResourceManager not initialized
        if _recognizer is None:
            raise HTTPException(status_code=503, detail="Database not loaded")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Face recognition unavailable: {e}",
        )

    # Double-check after lazy load attempt
    if _recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")


# ==================== Route Handlers ====================


@router.post("/identify", response_model=IdentifyResponse)
async def identify_performers(request: IdentifyRequest, _=Depends(require_db_available)):
    """
    Identify performers in an image.

    Provide either `image_url` or `image_base64`. Returns detected faces
    with potential performer matches sorted by confidence.
    """
    # Validate input
    if not request.image_url and not request.image_base64:
        raise HTTPException(
            status_code=400,
            detail="Must provide either image_url or image_base64"
        )

    # Fetch/decode image
    try:
        if request.image_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(request.image_url)
                response.raise_for_status()
                image_bytes = response.content
        else:
            image_bytes = base64.b64decode(request.image_base64)

        image = load_image(image_bytes)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Run recognition
    try:
        async with _gpu_compute_lock():
            results = _recognizer.recognize_image(
                image,
                top_k=request.top_k,
                max_distance=request.max_distance,
                min_face_confidence=request.min_face_confidence,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {e}")

    # Fetch missing images from StashBox
    all_matches = [m for r in results for m in r.matches]
    await _fetch_missing_images(all_matches)

    # Convert to response format
    faces = []
    for result in results:
        bbox = result.face.bbox  # dict with x, y, w, h in pixels
        face_box = FaceBox(
            x=int(bbox["x"]),
            y=int(bbox["y"]),
            width=int(bbox["w"]),
            height=int(bbox["h"]),
            confidence=result.face.confidence,
        )

        matches = [_match_to_response(m) for m in result.matches]

        faces.append(FaceResult(box=face_box, matches=matches))

    return IdentifyResponse(faces=faces, face_count=len(faces))


@router.post("/identify/url")
async def identify_from_url(
    url: str = Query(..., description="Image URL to analyze"),
    top_k: int = Query(5, ge=1, le=20),
    max_distance: float = Query(0.6, ge=0.0, le=2.0),
    _=Depends(require_db_available),
):
    """Convenience endpoint to identify from URL via query params."""
    return await identify_performers(
        IdentifyRequest(image_url=url, top_k=top_k, max_distance=max_distance)
    )


@router.post("/identify/image", response_model=IdentifyResponse)
async def identify_image(request: ImageIdentifyRequest, _=Depends(require_db_available)):
    """
    Identify performers in a Stash image by image ID.
    Fetches the image from Stash, runs face recognition, and stores fingerprint.
    """
    base_url = _stash_url.rstrip("/")
    api_key = _stash_api_key

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    # Fetch image info from Stash
    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(base_url, api_key)
    image_data = await stash_client.get_image_by_id(request.image_id)

    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = image_data.get("paths", {}).get("image")
    if not image_url:
        raise HTTPException(status_code=400, detail="Image has no image path")

    # Fetch the image
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"ApiKey": api_key} if api_key else {}
            response = await client.get(image_url, headers=headers)
            response.raise_for_status()
            image = load_image(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    # Run recognition
    try:
        async with _gpu_compute_lock():
            results = _recognizer.recognize_image(
                image,
                top_k=request.top_k,
                max_distance=request.max_distance,
                min_face_confidence=request.min_face_confidence,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {e}")

    # Fetch missing images from StashBox
    all_matches = [m for r in results for m in r.matches]
    await _fetch_missing_images(all_matches)

    faces = []
    img_h, img_w = image.shape[:2]

    for result in results:
        bbox = result.face.bbox
        face_box = FaceBox(
            x=int(bbox["x"]),
            y=int(bbox["y"]),
            width=int(bbox["w"]),
            height=int(bbox["h"]),
            confidence=result.face.confidence,
        )

        matches = [_match_to_response(m) for m in result.matches]

        faces.append(FaceResult(box=face_box, matches=matches))

    # Save fingerprint
    try:
        save_image_fingerprint(
            image_id=request.image_id,
            gallery_id=None,
            faces=results,
            image_shape=(img_h, img_w),
            db_version=_db_manifest.get("version"),
        )
    except Exception as e:
        print(f"[identify_image] Failed to save fingerprint: {e}")

    return IdentifyResponse(faces=faces, face_count=len(faces))


@router.post("/identify/gallery", response_model=GalleryIdentifyResponse)
async def identify_gallery(request: GalleryIdentifyRequest, _=Depends(require_db_available)):
    """
    Identify all performers across a gallery.
    Processes each image, aggregates results per-performer, and stores fingerprints.
    """
    t_start = time.time()

    base_url = _stash_url.rstrip("/")
    api_key = _stash_api_key

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    # Fetch gallery info from Stash
    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(base_url, api_key)
    gallery_data = await stash_client.get_gallery_by_id(request.gallery_id)

    if not gallery_data:
        raise HTTPException(status_code=404, detail="Gallery not found")

    images = gallery_data.get("images", [])
    if not images:
        return GalleryIdentifyResponse(
            gallery_id=request.gallery_id,
            total_images=0,
            images_processed=0,
            faces_detected=0,
            performers=[],
        )

    total_images = len(images)
    print(f"[identify_gallery] === START gallery_id={request.gallery_id}, {total_images} images ===")

    # Process each image
    performer_appearances: dict[str, list[dict]] = defaultdict(list)
    performer_info: dict[str, dict] = {}
    total_faces = 0
    images_processed = 0
    errors = []

    headers = {"ApiKey": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, img in enumerate(images):
            img_id = img["id"]
            img_url = img.get("paths", {}).get("image")

            if not img_url:
                errors.append(f"Image {img_id} has no URL")
                continue

            try:
                resp = await client.get(img_url, headers=headers)
                resp.raise_for_status()
                image = load_image(resp.content)

                async with _gpu_compute_lock():
                    results = _recognizer.recognize_image(
                        image,
                        top_k=request.top_k,
                        max_distance=request.max_distance,
                        min_face_confidence=request.min_face_confidence,
                    )

                img_h, img_w = image.shape[:2]
                total_faces += len(results)
                images_processed += 1

                # Save per-image fingerprint
                try:
                    save_image_fingerprint(
                        image_id=img_id,
                        gallery_id=request.gallery_id,
                        faces=results,
                        image_shape=(img_h, img_w),
                        db_version=_db_manifest.get("version"),
                    )
                except Exception as e:
                    print(f"[identify_gallery] Failed to save fingerprint for image {img_id}: {e}")

                # Collect per-performer data
                for result in results:
                    if result.matches:
                        best = result.matches[0]
                        pid = best.stashdb_id

                        performer_appearances[pid].append({
                            "image_id": img_id,
                            "distance": best.combined_score,
                        })

                        # Keep best info
                        if pid not in performer_info or best.combined_score < performer_info[pid]["distance"]:
                            performer_info[pid] = {
                                "name": best.name,
                                "distance": best.combined_score,
                                "country": best.country,
                                "image_url": best.image_url,
                                "endpoint": _extract_endpoint(best.universal_id),
                                "local_performer_id": getattr(best, "local_performer_id", None),
                            }

                if (i + 1) % 10 == 0:
                    print(f"[identify_gallery] [{time.time()-t_start:.1f}s] Processed {i+1}/{total_images} images")

            except Exception as e:
                errors.append(f"Image {img_id}: {str(e)[:100]}")
                print(f"[identify_gallery] Error processing image {img_id}: {e}")

    # Aggregate results
    performers = []
    for pid, appearances in performer_appearances.items():
        distances = [a["distance"] for a in appearances]
        image_ids = list(set(a["image_id"] for a in appearances))
        image_count = len(image_ids)
        best_distance = min(distances)
        avg_distance = sum(distances) / len(distances)

        # Filter: 2+ appearances OR single match with distance < 0.4
        if image_count < 2 and best_distance >= 0.4:
            continue

        info = performer_info[pid]
        performers.append(GalleryPerformerResult(
            performer_id=pid,
            name=info["name"],
            best_distance=best_distance,
            avg_distance=avg_distance,
            confidence=max(0.0, min(1.0, 1.0 - best_distance)),
            image_count=image_count,
            image_ids=image_ids,
            country=info.get("country"),
            image_url=info.get("image_url"),
            endpoint=info.get("endpoint"),
            local_performer_id=info.get("local_performer_id"),
        ))

    # Sort by image count desc, then best distance asc
    performers.sort(key=lambda p: (-p.image_count, p.best_distance))

    top_names = [p.name for p in performers[:3]]
    print(f"[identify_gallery] [{time.time()-t_start:.1f}s] === DONE === "
          f"{images_processed}/{total_images} images, {total_faces} faces, "
          f"{len(performers)} performers: {', '.join(top_names)}")

    return GalleryIdentifyResponse(
        gallery_id=request.gallery_id,
        total_images=total_images,
        images_processed=images_processed,
        faces_detected=total_faces,
        performers=performers,
        errors=errors[:10],
    )


def _cluster_and_match(
    all_results: list[tuple[int, RecognitionResult]], request: "SceneIdentifyRequest",
    frame_timestamps: Optional[dict[int, float]] = None,
) -> list["PersonResult"]:
    """Run the configured matching_mode (hybrid/frequency/cluster) over
    already-detected+embedded+matched results. Shared by the cache
    fast-path, the full ffmpeg pipeline, and sprite-only identification --
    factored out so all three stay in sync instead of duplicating this
    dispatch/dedup logic three times.

    frame_timestamps (frame_index -> timestamp_sec) is only available from
    the live ffmpeg-extraction path (identify_scene's main body) and is only
    threaded into the plain "cluster" branch's aggregate_matches() call
    below -- see aggregate_matches's own docstring for why."""
    if request.matching_mode == "hybrid":
        return hybrid_matching(
            all_results, _recognizer,
            cluster_threshold=request.cluster_threshold,
            top_k=request.top_k * 2, max_distance=request.max_distance,
        )
    elif request.matching_mode == "frequency":
        return clustered_frequency_matching(
            all_results, _recognizer,
            cluster_threshold=request.cluster_threshold,
            top_k=request.top_k, max_distance=request.max_distance,
            scene_performer_stashdb_ids=request.scene_performer_stashdb_ids,
        )

    clusters = cluster_faces_by_person(all_results, _recognizer, distance_threshold=request.cluster_threshold)
    clusters = merge_clusters_by_match(clusters)
    persons = []
    used_performers: set[str] = set()
    all_persons = []
    for person_id, cluster in enumerate(clusters):
        aggregated_matches = aggregate_matches(cluster, top_k=request.top_k, frame_timestamps=frame_timestamps)
        all_persons.append((len(cluster), PersonResult(
            person_id=person_id, frame_count=len(cluster),
            best_match=aggregated_matches[0] if aggregated_matches else None,
            all_matches=aggregated_matches,
        )))
    all_persons.sort(key=lambda x: x[0], reverse=True)
    for _, person in all_persons:
        if person.best_match:
            if person.best_match.stashdb_id in used_performers:
                for alt_match in person.all_matches[1:]:
                    if alt_match.stashdb_id not in used_performers:
                        person.best_match = alt_match
                        used_performers.add(alt_match.stashdb_id)
                        break
                else:
                    person.best_match = None
            else:
                used_performers.add(person.best_match.stashdb_id)
        person.all_matches = [m for m in person.all_matches if m.stashdb_id not in used_performers or m.stashdb_id == (person.best_match.stashdb_id if person.best_match else None)]
        persons.append(person)
    for i, person in enumerate(persons):
        person.person_id = i
    return persons


async def _process_sprite_frames(
    base_url: str, scene_id: str, api_key: str,
    min_face_size: int, min_face_confidence: float, match_config: "MatchingConfig",
    t_start: float,
) -> list[tuple[int, RecognitionResult]]:
    """Fetch the scene's sprite/VTT sheet, crop each tile, and run the same
    detect -> batch-embed -> match pipeline used for ffmpeg frames and the
    screenshot. Tagged with frame_index=-2 (mirroring the -1 sentinel used
    for screenshot faces), though -- unlike v1's implementation of this
    feature -- nothing here persists into recommendations_db.py's
    scene_face_embeddings cache: that table's write path is already
    disabled for the buffalo_l migration (see _identify_scene_from_cache's
    docstring), so sprite results are recomputed on every call rather than
    cached, consistent with video-frame results' current (also uncached)
    behavior in v2.

    Detection runs per-tile, not once on the whole composite sheet: the
    detector resizes its input to a fixed internal resolution, so stitching
    many small tiles into one large image would shrink each already-small
    face (a sprite tile is ~160x90px) well below what's reliably
    detectable.

    Returns an empty list if the sheet couldn't be fetched (e.g. Stash
    hasn't generated one for this scene yet) or no usable faces were found.
    """
    try:
        # Stash's REST route for sprite/vtt files is keyed by the file
        # checksum (e.g. /scene/<checksum>_sprite.jpg), not the numeric
        # scene ID -- these absolute URLs must come from the scene's own
        # paths, not be constructed from scene_id.
        async with httpx.AsyncClient(timeout=15.0) as client:
            gql_query = {
                "query": f'{{ findScene(id: "{scene_id}") {{ paths {{ sprite vtt }} }} }}'
            }
            headers = {"ApiKey": api_key, "Content-Type": "application/json"}
            response = await client.post(f"{base_url}/graphql", json=gql_query, headers=headers)
            response.raise_for_status()
            paths = response.json().get("data", {}).get("findScene", {}).get("paths") or {}
            sprite_url = paths.get("sprite")
            vtt_url = paths.get("vtt")

        if not sprite_url or not vtt_url:
            print(f"[identify_scene] [{time.time()-t_start:.1f}s] Sprite: scene has no sprite/vtt paths yet")
            return []

        sprite_frames = await fetch_sprite_from_stash(
            sprite_url, vtt_url, api_key, max_frames=face_config.SPRITE_MAX_FRAMES,
        )
    except Exception as e:
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Sprite fetch failed (scene may have no sprite generated yet): {e}")
        return []

    if not sprite_frames:
        return []

    detected: list[tuple[float, "DetectedFace", "np.ndarray"]] = []
    async with _gpu_compute_lock():
        per_tile_faces = await asyncio.to_thread(
            _recognizer.detect_faces_parallel,
            [sf.image for sf in sprite_frames],
            min_face_confidence,
        )
    for sf, faces in zip(sprite_frames, per_tile_faces):
        for face in faces:
            if face.bbox["w"] >= min_face_size and face.bbox["h"] >= min_face_size:
                detected.append((sf.timestamp, face, sf.image))

    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Sprite: {len(sprite_frames)} tiles, {len(detected)} usable faces")
    if not detected:
        return []

    # get_embeddings_batch is pure Python (reads embeddings buffalo_l
    # already computed during detect_faces_parallel above), no separate GPU
    # call -- doesn't need _gpu_compute_lock.
    embeddings = _recognizer.generator.get_embeddings_batch([face for _, face, _ in detected])

    extra_results: list[tuple[int, RecognitionResult]] = []
    for (_, face, tile_image), embedding in zip(detected, embeddings):
        matches, _, _ = _recognizer.recognize_face_v2(face, match_config, embedding=embedding, image=tile_image)
        extra_results.append((-2, RecognitionResult(face=face, matches=matches, embedding=embedding)))

    return extra_results


async def _identify_scene_from_cache(
    request: "SceneIdentifyRequest",
    cache_meta: dict,
    t_start: float,
    extra_results: Optional[list[tuple[int, RecognitionResult]]] = None,
) -> "SceneIdentifyResponse":
    """Reconstruct matching inputs from cached face signals and rerun only
    the DB-dependent steps (matching, clustering).

    Skips ffmpeg frame extraction and buffalo_l detection+embedding
    entirely -- reused verbatim from a prior full run on this scene. This is
    what makes a performer-database version bump cheap: is_scene_cache_compatible()
    deliberately doesn't check db_version, so a stale-version scene still
    hits this path and only redoes matching, not detection.

    `extra_results` (optional) are results computed by the caller outside
    the cache -- currently just freshly-fetched sprite results, since
    sprite detection isn't cached at all in v2 (see _process_sprite_frames)
    -- concatenated in before clustering runs.
    """
    scene_id_int = int(request.scene_id)

    match_config = MatchingConfig(
        query_k=100,
        max_results=request.top_k * 2,
        max_distance=request.max_distance,
    )

    _set_stage(request.scene_id, "cache_check")
    cached_faces = load_face_signal_cache(scene_id_int)
    # Empty is a legitimate cached result here (this scene genuinely has no
    # detectable faces), not a cache miss -- the caller already confirmed
    # cache_meta exists and is param-compatible before calling this
    # function, and scene_signal_cache/scene_face_embeddings are always
    # written together (see _identify_scene_compute), so a scene can't have
    # one without the other.
    all_results: list[tuple[int, RecognitionResult]] = []
    for row in cached_faces:
        bbox = json.loads(row["bbox_json"])
        face = DetectedFace(image=None, bbox=bbox, confidence=row["confidence"], yaw=row["yaw"], embedding=None)
        embedding = FaceEmbedding(
            embedding=np.frombuffer(row["embedding"], dtype=np.float32),
        )
        matches, _match_result, _ = _recognizer.recognize_face_v2(face, match_config, embedding=embedding)
        all_results.append((row["frame_index"], RecognitionResult(face=face, matches=matches, embedding=embedding)))
    if extra_results:
        all_results.extend(extra_results)

    scene_all_matches = [m for _, r in all_results for m in r.matches]
    await _fetch_missing_images(scene_all_matches)

    t_match = time.time()
    _set_stage(request.scene_id, "matching_performers")
    persons = _cluster_and_match(all_results, request)

    top_names = [p.best_match.name for p in persons[:3] if p.best_match]
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] === DONE (cache) === Top matches: {', '.join(top_names)}")

    # Save the fingerprint even when zero performers matched -- that's a
    # valid, complete result (this scene simply has nobody recognizable in
    # it), not an error. Gating the save on non-empty persons left the
    # pre-emptive "error" status row (written before matching ever runs)
    # permanently in place for any such scene, so it never flipped to
    # "complete" no matter how many times fingerprinting re-ran.
    fingerprint_saved = False
    fingerprint_error = None
    current_db_version = _db_manifest.get("version")
    _set_stage(request.scene_id, "saving_fingerprint")
    fp_id, fp_error = save_scene_fingerprint(
        scene_id=scene_id_int,
        frames_analyzed=cache_meta["frames_analyzed"],
        persons=persons,
        db_version=current_db_version,
        used_sprite=request.use_sprite,
    )
    if fp_id:
        fingerprint_saved = True
    else:
        fingerprint_error = fp_error

    timing_data = {
        "total_ms": round((time.time() - t_start) * 1000),
        "matching_ms": round((time.time() - t_match) * 1000),
    }

    _set_stage(request.scene_id, "done")
    return SceneIdentifyResponse(
        scene_id=request.scene_id,
        frames_analyzed=cache_meta["frames_analyzed"],
        frames_requested=request.num_frames,
        faces_detected=len(all_results),
        faces_after_filter=len(all_results),
        persons=persons,
        errors=[],
        fingerprint_saved=fingerprint_saved,
        fingerprint_error=fingerprint_error,
        timing=timing_data,
        used_cache=True,
    )


@router.post("/identify/scene", response_model=SceneIdentifyResponse)
async def identify_scene(request: SceneIdentifyRequest, _=Depends(require_db_available)):
    """
    Identify all performers in a scene using ffmpeg frame extraction.

    Extracts full-resolution frames from the video stream using ffmpeg,
    detects faces, clusters them by person, and returns matches.
    """
    return await _identify_scene_impl(request)


@dataclass
class SceneExtractionBundle:
    """Everything the compute half (_identify_scene_compute) needs from the
    decode half (_extract_scene_frames), so the two can run with a
    deliberate gap in between -- see scene_batch_orchestrator.py, which
    runs _extract_scene_frames for a whole batch of >=4K scenes to
    completion before starting compute for any of them, keeping VAAPI
    decode and ROCm/HIP compute from ever overlapping for those scenes."""
    extraction_result: ExtractionResult
    duration_sec: float
    screenshot_url: Optional[str]
    file_info: dict
    extraction_ms: float  # wall time of the ffmpeg extraction call itself


async def _extract_scene_frames(
    request: SceneIdentifyRequest, num_frames: int, t_start: float,
) -> SceneExtractionBundle:
    """Decode half of scene identification: resolves scene info from Stash
    (duration, resolution, screenshot path), decides per-scene hwaccel via
    _scene_needs_vaapi(), and extracts frames via ffmpeg. Pulled out of
    _identify_scene_impl (which still calls this directly for the live
    endpoint / non-batched callers) so scene_batch_orchestrator.py can also
    call it standalone for >=4K scenes -- see SceneExtractionBundle."""
    base_url = _stash_url.rstrip("/")
    api_key = _stash_api_key

    if not check_ffmpeg_available():
        raise HTTPException(status_code=503, detail="ffmpeg not available")

    print(f"[identify_scene] === START scene_id={request.scene_id} ===")
    print(f"[identify_scene] Settings: num_frames={request.num_frames}, min_face_size={request.min_face_size}, max_distance={request.max_distance}, mode={request.matching_mode}")

    # Get scene info from Stash
    screenshot_url = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            gql_query = {
                "query": f'''{{
                    findScene(id: "{request.scene_id}") {{
                        files {{
                            duration
                            width
                            height
                        }}
                        paths {{
                            screenshot
                        }}
                    }}
                }}'''
            }
            headers = {"ApiKey": api_key, "Content-Type": "application/json"}
            response = await client.post(f"{base_url}/graphql", json=gql_query, headers=headers)
            response.raise_for_status()
            data = response.json()

            scene_data = data.get("data", {}).get("findScene", {})
            if not scene_data or not scene_data.get("files"):
                raise HTTPException(status_code=404, detail="Scene not found or has no files")

            file_info = scene_data["files"][0]
            duration_sec = file_info.get("duration", 0)
            if not duration_sec:
                raise HTTPException(status_code=400, detail="Scene has no duration")

            # Get screenshot URL if available
            paths = scene_data.get("paths", {})
            screenshot_url = paths.get("screenshot") if paths else None

            print(f"[identify_scene] [{time.time()-t_start:.1f}s] Scene info: duration={duration_sec:.1f}s, resolution={file_info.get('width')}x{file_info.get('height')}, screenshot={'yes' if screenshot_url else 'no'}")

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to query scene: {e}")

    # Resolve frame extraction concurrency from settings
    max_concurrent = 8
    try:
        from settings import get_setting
        max_concurrent = int(get_setting("frame_extraction_concurrency"))
    except (RuntimeError, KeyError):
        pass

    # Configure frame extraction. FFMPEG_HWACCEL is a ceiling ("VAAPI
    # allowed if set at all") -- _scene_needs_vaapi() decides whether this
    # specific scene actually uses it, so below-4K scenes always decode on
    # CPU even when the env var permits VAAPI, keeping GPU video-decode from
    # ever running alongside ROCm/HIP compute for the common case.
    _hwaccel_env = os.environ.get("FFMPEG_HWACCEL", "none").lower().strip()
    _hwaccel_allowed = None if _hwaccel_env in ("none", "", "cpu") else _hwaccel_env
    hwaccel = (
        _hwaccel_allowed
        if _hwaccel_allowed and _scene_needs_vaapi(file_info.get("width"), file_info.get("height"))
        else None
    )

    # VAAPI requires a dynamically-linked ffmpeg to dlopen libva backend
    # plugins at runtime. The mwader static build (/usr/local/bin/ffmpeg)
    # is fully static and cannot load those plugins, so VAAPI silently fails
    # with "Device creation failed". Use the system package (/usr/bin/ffmpeg)
    # instead, which is dynamically linked and works correctly with libva +
    # mesa-va-drivers. CPU and CUDA modes use the static build as normal.
    ffmpeg_path = "/usr/bin/ffmpeg" if hwaccel == "vaapi" else "ffmpeg"

    config = FrameExtractionConfig(
        num_frames=num_frames,
        start_offset_pct=request.start_offset_pct,
        end_offset_pct=request.end_offset_pct,
        min_face_size=request.min_face_size,
        min_face_confidence=request.min_face_confidence,
        max_concurrent_extractions=max_concurrent,
        hwaccel=hwaccel,
        ffmpeg_path=ffmpeg_path,
    )

    # Extract frames using ffmpeg
    _set_stage(request.scene_id, "extracting_frames")
    t_extract = time.time()
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Extracting {request.num_frames} frames...")
    extraction_result = await extract_frames_from_stash_scene(
        stash_url=base_url,
        scene_id=request.scene_id,
        duration_sec=duration_sec,
        api_key=api_key,
        config=config,
    )

    extraction_ms = (time.time() - t_extract) * 1000
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Extracted {len(extraction_result.frames)} frames in {extraction_ms/1000:.1f}s")
    if extraction_result.errors:
        print(f"[identify_scene] Errors: {extraction_result.errors[:3]}")

    return SceneExtractionBundle(
        extraction_result=extraction_result,
        duration_sec=duration_sec,
        screenshot_url=screenshot_url,
        file_info=file_info,
        extraction_ms=extraction_ms,
    )


async def _identify_scene_compute(
    request: SceneIdentifyRequest,
    bundle: SceneExtractionBundle,
    num_frames: int,
    match_config: MatchingConfig,
    scene_id_int: int,
    sprite_extra_results: list[tuple[int, RecognitionResult]],
    t_start: float,
) -> "SceneIdentifyResponse":
    """Detect+embed+match+fingerprint half of scene identification, given
    an already-extracted bundle (see _extract_scene_frames). Pulled out of
    _identify_scene_impl (which still calls this directly right after
    _extract_scene_frames for the live endpoint / non-batched callers) so
    scene_batch_orchestrator.py can call it separately, after a deliberate
    gap where an entire batch of >=4K scenes finished decoding first."""
    api_key = _stash_api_key
    extraction_result = bundle.extraction_result
    duration_sec = bundle.duration_sec
    screenshot_url = bundle.screenshot_url
    file_info = bundle.file_info

    # Phase 1: Detect all faces from all frames -- spread across
    # _recognizer's detection pool (recognizer.py's DETECTION_POOL_SIZE
    # generators) instead of one shared generator processing frames
    # sequentially, off the event loop via asyncio.to_thread since
    # detect_faces_parallel() blocks until every frame is done.
    _set_stage(request.scene_id, "analyzing_frames")
    detected_faces: list[tuple[int, "DetectedFace", "np.ndarray"]] = []  # (frame_index, face, frame_image)
    total_faces = 0

    t_face_loop = time.time()
    t_det = time.time()
    async with _gpu_compute_lock():
        per_frame_faces = await asyncio.to_thread(
            _recognizer.detect_faces_parallel,
            [frame.image for frame in extraction_result.frames],
            request.min_face_confidence,
        )
    t_detect_total = time.time() - t_det

    for frame, faces in zip(extraction_result.frames, per_frame_faces):
        for face in faces:
            total_faces += 1
            if face.bbox["w"] >= request.min_face_size and face.bbox["h"] >= request.min_face_size:
                detected_faces.append((frame.frame_index, face, frame.image))

    filtered_faces = len(detected_faces)
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Detection: {t_detect_total:.1f}s | {total_faces} detected, {filtered_faces} after filter")

    # Phase 2: read back embeddings buffalo_l already computed during detection
    t_embed = time.time()
    if detected_faces:
        embeddings = _recognizer.generator.get_embeddings_batch([face for _, face, _ in detected_faces])
    else:
        embeddings = []
    t_embed_total = time.time() - t_embed
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Batch embedding: {t_embed_total:.1f}s for {len(embeddings)} faces ({t_embed_total*1000/max(1,len(embeddings)):.1f}ms/face)")

    # Phase 3: Match each face against the index using pre-computed embeddings
    all_results: list[tuple[int, RecognitionResult]] = []
    t_recognize_total = 0.0

    # Collected alongside matching and persisted via save_face_signal_cache
    # below -- this is what lets a later re-identify (e.g. after a
    # performer-database version bump) skip straight to
    # _identify_scene_from_cache instead of redoing detection+embedding.
    # Video-frame faces only, matching _identify_scene_from_cache's read
    # side -- screenshot/sprite faces are always recomputed fresh (see that
    # function's docstring).
    face_cache_rows: list[dict] = []

    for (frame_idx, face, frame_image), embedding in zip(detected_faces, embeddings):
        t_rec = time.time()
        matches, _match_result, _ = _recognizer.recognize_face_v2(face, match_config, embedding=embedding, image=frame_image)
        t_recognize_total += time.time() - t_rec

        result = RecognitionResult(face=face, matches=matches, embedding=embedding)
        all_results.append((frame_idx, result))
        face_cache_rows.append({
            "frame_index": frame_idx,
            "bbox": face.bbox,
            "confidence": face.confidence,
            "yaw": face.yaw,
            "embedding": np.asarray(embedding.embedding, dtype=np.float32).tobytes(),
        })

    t_face_loop_total = time.time() - t_face_loop
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Matching: {t_recognize_total:.1f}s | Total face pipeline: {t_face_loop_total:.1f}s")

    # Process screenshot if available (high-quality cover image often has clear faces)
    # Stash serves thumbnails via paths.screenshot, so we scale up if needed
    _set_stage(request.scene_id, "analyzing_screenshot")
    screenshot_faces = 0
    if screenshot_url:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"ApiKey": api_key}
                screenshot_resp = await client.get(screenshot_url, headers=headers)
                if screenshot_resp.status_code == 200:
                    screenshot_image = load_image(screenshot_resp.content)
                    img_h, img_w = screenshot_image.shape[:2]

                    # Scale up thumbnail if significantly smaller than video resolution
                    video_width = file_info.get("width", 1920)
                    if img_w < video_width * 0.8:
                        import cv2
                        scale_factor = video_width / img_w
                        new_w = int(img_w * scale_factor)
                        new_h = int(img_h * scale_factor)
                        screenshot_image = cv2.resize(screenshot_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot upscaled: {img_w}x{img_h} -> {new_w}x{new_h}")
                        img_w, img_h = new_w, new_h

                    async with _gpu_compute_lock():
                        screenshot_detected = _recognizer.generator.detect_faces(
                            screenshot_image,
                            min_confidence=request.min_face_confidence,
                        )
                    # Filter and batch-embed screenshot faces
                    ss_faces = [f for f in screenshot_detected
                                if f.bbox["w"] >= request.min_face_size and f.bbox["h"] >= request.min_face_size]
                    if ss_faces:
                        # Pure Python read-back (see the Phase 2 comment
                        # above) -- no GPU call, no lock needed.
                        ss_embeddings = _recognizer.generator.get_embeddings_batch(ss_faces)
                        for face, emb in zip(ss_faces, ss_embeddings):
                            matches, _, _ = _recognizer.recognize_face_v2(face, match_config, embedding=emb, image=screenshot_image)
                            result = RecognitionResult(face=face, matches=matches, embedding=emb)
                            all_results.append((-1, result))
                            screenshot_faces += 1
                            top_match = matches[0].name if matches else "no match"
                            print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot face: {face.bbox['w']}x{face.bbox['h']}px -> {top_match}")
                    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot ({img_w}x{img_h}): {len(screenshot_detected)} faces, {screenshot_faces} usable")
        except Exception as e:
            print(f"[identify_scene] Screenshot processing failed: {e}")

    # Merge in sprite results resolved up front (see the comment where
    # sprite_extra_results was computed, near the top of this function).
    if sprite_extra_results:
        all_results.extend(sprite_extra_results)

    # Fetch missing images from StashBox for all detected matches
    scene_all_matches = [m for _, r in all_results for m in r.matches]
    await _fetch_missing_images(scene_all_matches)

    # Written together, unconditionally (including when face_cache_rows is
    # empty -- a scene with genuinely zero detectable faces is still a
    # valid, fast-path-cacheable result) so the two tables can never
    # diverge -- see _identify_scene_from_cache's cache_check comment.
    save_scene_signal_cache(
        scene_id_int,
        num_frames=num_frames,
        min_face_size=request.min_face_size,
        min_face_confidence=request.min_face_confidence,
        start_offset_pct=request.start_offset_pct,
        end_offset_pct=request.end_offset_pct,
        frames_analyzed=len(extraction_result.frames),
    )
    save_face_signal_cache(scene_id_int, face_cache_rows)

    # Choose matching mode
    t_match_end = 0.0
    t_match = time.time()
    _set_stage(request.scene_id, "matching_performers")
    frame_timestamps = {f.frame_index: f.timestamp_sec for f in extraction_result.frames}
    persons = _cluster_and_match(all_results, request, frame_timestamps=frame_timestamps)
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Matching ({request.matching_mode}): {len(persons)} persons in {time.time()-t_match:.1f}s")

    t_match_end = time.time()
    top_names = [p.best_match.name for p in persons[:3] if p.best_match]
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] === DONE === Top matches: {', '.join(top_names)}")

    # Persist fingerprint to stash_sense.db for duplicate detection.
    #
    # Save even when zero performers matched -- that's a valid, complete
    # result (nobody recognizable in this scene), not an error. Gating the
    # save on non-empty persons left the pre-emptive "error" status row
    # (written before matching ever runs, see the fingerprint generator's
    # create_scene_fingerprint call) permanently in place for any such
    # scene: it never flipped to "complete" no matter how many times
    # fingerprinting re-ran, silently inflating the "missing" count forever.
    fingerprint_saved = False
    fingerprint_error = None

    current_db_version = _db_manifest.get("version")
    _set_stage(request.scene_id, "saving_fingerprint")
    fp_id, fp_error = save_scene_fingerprint(
        scene_id=int(request.scene_id),
        frames_analyzed=len(extraction_result.frames),
        persons=persons,
        db_version=current_db_version,
        used_sprite=request.use_sprite,
    )
    if fp_id:
        fingerprint_saved = True
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Saved fingerprint #{fp_id} with {len(persons)} persons")
    else:
        fingerprint_error = fp_error
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Failed to save fingerprint: {fp_error}")

    timing_data = {
        "total_ms": round((time.time() - t_start) * 1000),
        "extraction_ms": round(bundle.extraction_ms),
        "face_loop_ms": round(t_face_loop_total * 1000),
        "detection_ms": round(t_detect_total * 1000),
        "embedding_ms": round(t_embed_total * 1000),
        "recognition_ms": round(t_recognize_total * 1000),
        "matching_ms": round((t_match_end - t_match) * 1000),
    }
    print(f"[identify_scene] Timing: {timing_data}")

    response_errors = list(extraction_result.errors[:5]) if extraction_result.errors else []

    _set_stage(request.scene_id, "done")
    return SceneIdentifyResponse(
        scene_id=request.scene_id,
        frames_analyzed=len(extraction_result.frames),
        frames_requested=request.num_frames,
        faces_detected=total_faces,
        faces_after_filter=filtered_faces,
        persons=persons,
        errors=response_errors[:5],
        fingerprint_saved=fingerprint_saved,
        fingerprint_error=fingerprint_error,
        timing=timing_data,
    )


@dataclass
class PreparedSceneIdentify:
    """Output of _prepare_scene_identify() when the scene needs the full
    decode+compute pipeline (no skip_frame_extraction / cache-hit short-
    circuit applied) -- everything _extract_scene_frames() and
    _identify_scene_compute() need, resolved once so a batch caller
    (scene_batch_orchestrator.py) doesn't redo sprite processing or the
    cache check for each scene it batches."""
    num_frames: int
    match_config: MatchingConfig
    scene_id_int: int
    sprite_extra_results: list[tuple[int, RecognitionResult]]
    t_start: float


async def _prepare_scene_identify(
    request: SceneIdentifyRequest,
) -> "PreparedSceneIdentify | SceneIdentifyResponse":
    """Everything common to every /identify/scene call that happens before
    any ffmpeg decode: resolves num_frames/match_config, runs sprite-tile
    processing (independent of video decode -- sprites come from Stash's
    own pre-generated JPEG, not this scene's video file), and applies the
    skip_frame_extraction / cache-hit short-circuits.

    Returns a finished SceneIdentifyResponse directly for either short-
    circuit, or a PreparedSceneIdentify when the full decode+compute
    pipeline (_extract_scene_frames + _identify_scene_compute) is still
    needed -- callers must check which type they got back before deciding
    whether to proceed to decode."""
    t_start = time.time()

    base_url = _stash_url.rstrip("/")
    api_key = _stash_api_key

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    # Resolve num_frames: use settings value when caller didn't override.
    # Done up front so it's available for the cache-compatibility check below.
    num_frames = request.num_frames
    try:
        from settings import get_setting
        settings_num_frames = int(get_setting("num_frames"))
        if num_frames == face_config.NUM_FRAMES:
            num_frames = settings_num_frames
    except (RuntimeError, KeyError):
        pass

    scene_id_int = int(request.scene_id)

    match_config = MatchingConfig(
        query_k=100,  # Get more candidates before threshold-filtering
        max_results=request.top_k * 2,
        max_distance=request.max_distance,
    )

    # Sprite results are resolved once, up front, so they're available
    # regardless of which video-frame path runs after this (cache fast-path /
    # full pipeline / skipped entirely via skip_frame_extraction). Unlike
    # v1's implementation of this feature, nothing here is cached (see
    # _process_sprite_frames's docstring) -- every use_sprite=True call
    # redoes sprite fetch+detect+embed, consistent with video-frame
    # results' current (also uncached) behavior in v2.
    sprite_extra_results: list[tuple[int, RecognitionResult]] = []
    if request.use_sprite:
        _set_stage(request.scene_id, "analyzing_sprite")
        sprite_extra_results = await _process_sprite_frames(
            base_url, request.scene_id, api_key,
            min_face_size=request.min_face_size,
            min_face_confidence=request.min_face_confidence,
            match_config=match_config, t_start=t_start,
        )

    if request.skip_frame_extraction:
        # User declined to fingerprint an unfingerprinted scene -- identify
        # from sprite tiles only. No screenshot, no ffmpeg, no fingerprint
        # write (that would misrepresent this scene as fingerprinted).
        scene_all_matches = [m for _, r in sprite_extra_results for m in r.matches]
        await _fetch_missing_images(scene_all_matches)
        _set_stage(request.scene_id, "matching_performers")
        persons = _cluster_and_match(sprite_extra_results, request)
        _set_stage(request.scene_id, "done")
        return SceneIdentifyResponse(
            scene_id=request.scene_id,
            frames_analyzed=0,
            frames_requested=0,
            faces_detected=len(sprite_extra_results),
            faces_after_filter=len(sprite_extra_results),
            persons=persons,
            errors=[],
            fingerprint_saved=False,
            fingerprint_error=None,
            timing={"total_ms": round((time.time() - t_start) * 1000)},
            used_cache=False,
        )

    if request.use_cache:
        cache_meta = get_scene_signal_cache(scene_id_int)
        if cache_meta is not None and is_scene_cache_compatible(
            cache_meta,
            num_frames=num_frames,
            min_face_size=request.min_face_size,
            min_face_confidence=request.min_face_confidence,
            start_offset_pct=request.start_offset_pct,
            end_offset_pct=request.end_offset_pct,
        ):
            print(f"[identify_scene] === START scene_id={request.scene_id} (cache hit) ===")
            try:
                return await _identify_scene_from_cache(
                    request, cache_meta, t_start, extra_results=sprite_extra_results,
                )
            except Exception as e:
                logger.warning(f"[identify_scene] Cache fast-path failed for scene {request.scene_id}, falling back to full pipeline: {e}")

    return PreparedSceneIdentify(
        num_frames=num_frames,
        match_config=match_config,
        scene_id_int=scene_id_int,
        sprite_extra_results=sprite_extra_results,
        t_start=t_start,
    )


async def _identify_scene_impl(request: SceneIdentifyRequest) -> "SceneIdentifyResponse":
    """Actual scene-identify implementation, split out from the identify_scene
    route so callers other than the HTTP route (e.g. the scene_face_match
    analyzer's batch job) can invoke it directly without going through
    FastAPI's Depends machinery. Callers must ensure face recognition is
    available themselves first (see require_db_available).

    Internally this is a thin wrapper over _prepare_scene_identify (settings/
    sprite/cache/skip short-circuits), _extract_scene_frames (decode), and
    _identify_scene_compute (detect+embed+match+fingerprint) -- split out so
    scene_batch_orchestrator.py can run decode for a whole batch of >=4K
    scenes before compute starts for any of them."""
    prepared = await _prepare_scene_identify(request)
    if isinstance(prepared, SceneIdentifyResponse):
        return prepared

    bundle = await _extract_scene_frames(request, prepared.num_frames, prepared.t_start)
    return await _identify_scene_compute(
        request, bundle, prepared.num_frames, prepared.match_config,
        prepared.scene_id_int, prepared.sprite_extra_results, prepared.t_start,
    )


class SceneIdentifyProgressResponse(BaseModel):
    """Best-effort, coarse stage indicator for an in-flight /identify/scene
    call on a given scene -- not a percentage, just "which phase is this
    running right now" for the scene page's step-list progress UI."""
    stage: Optional[str] = None
    updated_at: Optional[float] = None


@router.get("/identify/scene/{scene_id}/progress", response_model=SceneIdentifyProgressResponse)
async def get_scene_identify_progress(scene_id: str):
    """Poll the current stage of an in-flight /identify/scene call for this
    scene. Ephemeral/process-local -- nothing is returned once the call
    finishes and enough time has passed, or if no call has run yet."""
    entry = _scene_progress.get(scene_id)
    if not entry:
        return SceneIdentifyProgressResponse()
    return SceneIdentifyProgressResponse(stage=entry["stage"], updated_at=entry["updated_at"])

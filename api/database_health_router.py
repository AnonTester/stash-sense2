"""Database health and update API endpoints.

Provides routes for health checks, database info, rate limiter status,
ffmpeg availability, and database self-update management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from frame_extractor import check_ffmpeg_available

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Module-level globals set by init
_recognizer = None
_db_manifest = {}
_db_updater = None


def init_database_health_router(recognizer, db_manifest: dict, db_updater):
    """Initialize the database health router with runtime dependencies."""
    global _recognizer, _db_manifest, _db_updater
    _recognizer = recognizer
    _db_manifest = db_manifest
    _db_updater = db_updater


_UNSET = object()  # sentinel to distinguish "not provided" from None


def update_database_health_globals(recognizer=_UNSET, db_manifest=_UNSET):
    """Update globals after a database hot-swap or idle unload."""
    global _recognizer, _db_manifest
    if recognizer is not _UNSET:
        _recognizer = recognizer
    if db_manifest is not _UNSET:
        _db_manifest = db_manifest


# ==================== Pydantic Models ====================


class DatabaseInfo(BaseModel):
    """Information about the loaded database."""
    version: str
    performer_count: int
    face_count: int
    sources: list[str]
    created_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_loaded: bool
    performer_count: int = 0
    face_count: int = 0
    version: Optional[str] = None
    face_recognition_loading: bool = False


class CheckUpdateResponse(BaseModel):
    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    update_available: bool
    release_name: Optional[str] = None
    download_url: Optional[str] = None
    download_size_mb: Optional[int] = None
    published_at: Optional[str] = None
    error: Optional[str] = None
    # Delta path: a small download covering only what changed, when a
    # complete chain from current_version to latest_version exists.
    delta_available: bool = False
    delta_chain_length: Optional[int] = None
    delta_download_size_mb: Optional[float] = None


class StartUpdateResponse(BaseModel):
    job_id: str
    status: str


class UpdateStatusResponse(BaseModel):
    status: str
    progress_pct: int = 0
    current_version: Optional[str] = None
    target_version: Optional[str] = None
    error: Optional[str] = None


# ==================== Route Handlers ====================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and database status.

    Does NOT trigger lazy loading -- reports current state without
    side effects. The face recognition database loads on first
    /identify request.
    """
    from main import app as _app
    _version = getattr(_app, "version", None)

    face_recognition_loading = False
    try:
        from resource_manager import get_resource_manager
        status = get_resource_manager().get_status().get("face_recognition")
        if status:
            face_recognition_loading = status["loading"]
    except RuntimeError:
        pass  # ResourceManager not initialized yet (very early startup)

    if _recognizer is None:
        return HealthResponse(
            status="degraded",
            database_loaded=False,
            version=_version,
            face_recognition_loading=face_recognition_loading,
        )

    return HealthResponse(
        status="healthy",
        database_loaded=True,
        performer_count=len(_recognizer.performers),
        face_count=len(_recognizer.faces),
        version=_version,
        face_recognition_loading=face_recognition_loading,
    )


@router.get("/health/rate-limiter")
async def rate_limiter_status():
    """Get rate limiter metrics."""
    from rate_limiter import RateLimiter
    limiter = await RateLimiter.get_instance()
    return limiter.get_metrics()


@router.get("/health/ffmpeg")
async def ffmpeg_health():
    """Check if ffmpeg is available for V2 scene identification."""
    available = check_ffmpeg_available()
    return {
        "ffmpeg_available": available,
        "v2_endpoint_ready": available,
    }


@router.get("/database/info", response_model=DatabaseInfo)
async def database_info():
    """Get information about the loaded database.

    Returns manifest values when recognizer is lazily loaded but not yet
    initialized, so the UI always has database info available.
    """
    # Use live counts from recognizer if loaded, otherwise fall back to manifest
    if _recognizer is not None:
        performer_count = len(_recognizer.performers)
        face_count = len(_recognizer.faces)
    else:
        performer_count = _db_manifest.get("performer_count", 0)
        face_count = _db_manifest.get("face_count", 0)

    return DatabaseInfo(
        version=_db_manifest.get("version", "unknown"),
        performer_count=performer_count,
        face_count=face_count,
        sources=_db_manifest.get("sources", ["stashdb.org"]),
        created_at=_db_manifest.get("created_at"),
    )


@router.get("/database/check-update", response_model=CheckUpdateResponse)
async def check_database_update():
    """Check GitHub for a newer database release.

    Always performs a live GitHub API call (bypasses the 12-hour persistent
    cache) so the user always sees up-to-date information when they click
    "check for updates" manually.
    """
    if _db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    result = await _db_updater.check_update(force=True)
    return CheckUpdateResponse(**result)


@router.post("/database/update", response_model=StartUpdateResponse)
async def start_database_update(method: str = "auto"):
    """Trigger a database update.

    `method`: "auto" (default, prefers the small delta chain when a
    complete one exists) | "delta" (same as auto, explicit) | "full"
    (force the full-zip download even if a delta chain is available —
    useful for a clean re-sync).
    """
    if _db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    if _db_updater._update_task and not _db_updater._update_task.done():
        raise HTTPException(status_code=409, detail="Update already in progress")
    # Force a fresh check so we always start from accurate release info.
    check = await _db_updater.check_update(force=True)
    if not check.get("update_available"):
        raise HTTPException(status_code=400, detail="Already on latest version")

    use_delta = method != "full" and check.get("delta_available")
    if use_delta:
        job_id = await _db_updater.start_delta_update(_db_updater._last_delta_chain)
    else:
        job_id = await _db_updater.start_update(
            download_url=check["download_url"],
            target_version=check["latest_version"],
        )
    return StartUpdateResponse(job_id=job_id, status="started")


@router.get("/database/update/status", response_model=UpdateStatusResponse)
async def get_update_status():
    """Get current status of a database update."""
    if _db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    return UpdateStatusResponse(**_db_updater.get_status())

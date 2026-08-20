"""Settings API Router.

Endpoints for reading, updating, and resetting sidecar settings.
Also provides system info (hardware profile, version, uptime).
"""

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hardware import get_hardware_profile
from recommendations_router import get_rec_db
from settings import get_settings_manager, SETTING_DEFS
from stashbox_connection_manager import get_connection_manager

router = APIRouter(tags=["settings"])

# Set at startup
_start_time: Optional[float] = None
_version: str = "0.13.0"


def init_settings_router():
    """Record startup time. Called once during lifespan."""
    global _start_time
    _start_time = time.monotonic()


def _apply_setting_side_effects(key: str, value: Any, plugin_version: Optional[str] = None) -> None:
    """Apply runtime side-effects for settings that need them."""
    if key in ("debug_logging_enabled", "debug_logging_anonymize"):
        from debug_logging import configure_debug_logging
        from main import DATA_DIR
        from pathlib import Path
        mgr = get_settings_manager()
        enabled = bool(value) if key == "debug_logging_enabled" else bool(mgr.get("debug_logging_enabled"))
        anonymize = bool(value) if key == "debug_logging_anonymize" else bool(mgr.get("debug_logging_anonymize"))
        from main import app
        configure_debug_logging(enabled, Path(DATA_DIR), anonymize=anonymize, version=app.version, plugin_version=plugin_version)


# ==================== Request/Response models ====================

class UpdateSettingRequest(BaseModel):
    value: Any
    plugin_version: Optional[str] = None


class BulkUpdateRequest(BaseModel):
    settings: dict[str, Any]


# ==================== Settings endpoints ====================

@router.get("/settings")
async def get_all_settings():
    """Get all settings grouped by category with metadata for UI rendering."""
    mgr = get_settings_manager()
    return mgr.get_all_with_metadata()


# ==================== Endpoint Priority ====================
# NOTE: These must be registered BEFORE /settings/{key} to avoid
# the path parameter route catching "endpoint-priorities" as a key.

class EndpointPriorityRequest(BaseModel):
    endpoints: list[str]


@router.get("/settings/endpoint-priorities")
async def get_endpoint_priorities():
    """Get stash-box endpoints in priority order, plus disabled list.

    Returns all configured endpoints with their names, ordered by priority.
    Endpoints without explicit priority are appended in their default order.
    Disabled endpoints are returned separately.
    """
    mgr = get_connection_manager()
    connections = {c["endpoint"]: c for c in mgr.get_connections()}

    db = get_rec_db()
    priority_order = db.get_endpoint_priorities()
    disabled_list = set(db.get_disabled_endpoints())

    # Build ordered result: prioritized endpoints first, then any remaining (exclude disabled)
    result = []
    seen = set()
    for ep in priority_order:
        if ep in connections and ep not in disabled_list:
            result.append(connections[ep])
            seen.add(ep)
    for ep, conn in connections.items():
        if ep not in seen and ep not in disabled_list:
            result.append(conn)

    # Build disabled list with connection info
    disabled_result = [connections[ep] for ep in disabled_list if ep in connections]

    return {"endpoints": result, "disabled": disabled_result}


@router.post("/settings/endpoint-priorities")
async def set_endpoint_priorities(request: EndpointPriorityRequest):
    """Set the priority order for stash-box endpoints.

    Endpoints listed first have highest priority. When an entity is linked
    to multiple endpoints, only the highest-priority endpoint generates
    upstream change recommendations.
    """
    db = get_rec_db()
    db.set_endpoint_priorities(request.endpoints)
    return {"success": True}


class EndpointDisableRequest(BaseModel):
    endpoint: str
    clear_recommendations: bool = False


class EndpointEnableRequest(BaseModel):
    endpoint: str


@router.post("/settings/endpoint-disable")
async def disable_endpoint(request: EndpointDisableRequest):
    """Disable a stash-box endpoint from upstream analysis."""
    db = get_rec_db()

    # Add to disabled list
    disabled = db.get_disabled_endpoints()
    if request.endpoint not in disabled:
        disabled.append(request.endpoint)
        db.set_disabled_endpoints(disabled)

    # Remove from priority list
    priorities = db.get_endpoint_priorities()
    if request.endpoint in priorities:
        priorities.remove(request.endpoint)
        db.set_endpoint_priorities(priorities)

    # Optionally clear recommendations and snapshots for this endpoint
    cleared_count = 0
    if request.clear_recommendations:
        with db._connection() as conn:
            # Dismiss pending recs that came from this endpoint
            rows = conn.execute(
                "SELECT id, details FROM recommendations WHERE status = 'pending'"
            ).fetchall()
            for row in rows:
                try:
                    details = json.loads(row['details']) if isinstance(row['details'], str) else row['details']
                except (json.JSONDecodeError, TypeError):
                    continue
                if details.get('endpoint') == request.endpoint:
                    conn.execute(
                        "UPDATE recommendations SET status = 'dismissed', updated_at = datetime('now') WHERE id = ?",
                        (row['id'],)
                    )
                    cleared_count += 1

            # Clear snapshots for this endpoint
            conn.execute(
                "DELETE FROM upstream_snapshots WHERE endpoint = ?",
                (request.endpoint,)
            )

            # Clear watermarks for this endpoint
            # Watermark keys are stored as "{analyzer_type}:{endpoint}"
            conn.execute(
                "DELETE FROM analysis_watermarks WHERE type LIKE ?",
                (f"%:{request.endpoint}",)
            )

    return {"success": True, "cleared_count": cleared_count}


@router.post("/settings/endpoint-enable")
async def enable_endpoint(request: EndpointEnableRequest):
    """Re-enable a disabled stash-box endpoint."""
    db = get_rec_db()

    # Remove from disabled list
    disabled = db.get_disabled_endpoints()
    if request.endpoint in disabled:
        disabled.remove(request.endpoint)
        db.set_disabled_endpoints(disabled)

    # Append to end of priority list
    priorities = db.get_endpoint_priorities()
    if request.endpoint not in priorities:
        priorities.append(request.endpoint)
        db.set_endpoint_priorities(priorities)

    return {"success": True}


# ==================== Debug log file management ====================
# Must be registered BEFORE /settings/{key} so fixed paths win over the wildcard.

_LOG_FILENAME_RE = re.compile(r'^stash_sense_debug\.log(\.\d+)?$')


def _get_log_dir() -> Path:
    from main import DATA_DIR
    return Path(DATA_DIR) / "logs"


def _safe_log_path(filename: str) -> Path:
    """Resolve a log filename to an absolute path, raising 404 if invalid."""
    if not _LOG_FILENAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="Log file not found")
    path = _get_log_dir() / filename
    if not str(path.resolve()).startswith(str(_get_log_dir().resolve())):
        raise HTTPException(status_code=404, detail="Log file not found")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    return path


@router.get("/settings/logs")
async def list_log_files():
    """List debug log files with name, size, and modification time."""
    log_dir = _get_log_dir()
    files = []
    if log_dir.exists():
        for p in sorted(log_dir.iterdir()):
            if _LOG_FILENAME_RE.match(p.name):
                stat = p.stat()
                files.append({
                    "filename": p.name,
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                })
    files.sort(key=lambda f: (0 if f["filename"] == "stash_sense_debug.log" else 1, f["filename"]))
    return {"files": files}


@router.get("/settings/logs/download-all")
async def download_all_log_files():
    """Return all debug log files as a base64-encoded zip archive."""
    import io
    import zipfile

    log_dir = _get_log_dir()
    buf = io.BytesIO()
    added = []
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if log_dir.exists():
            for p in sorted(log_dir.iterdir()):
                if _LOG_FILENAME_RE.match(p.name):
                    zf.write(p, arcname=p.name)
                    added.append(p.name)
    content = buf.getvalue()
    return {
        "filename": "stash_sense_logs.zip",
        "content_b64": base64.b64encode(content).decode("ascii"),
        "size_bytes": len(content),
        "file_count": len(added),
    }


@router.get("/settings/logs/download/{filename}")
async def download_log_file(filename: str):
    """Return log file content as base64 for browser download."""
    path = _safe_log_path(filename)
    content = path.read_bytes()
    return {
        "filename": filename,
        "content_b64": base64.b64encode(content).decode("ascii"),
        "size_bytes": len(content),
    }


@router.delete("/settings/logs/{filename}")
async def delete_log_file(filename: str):
    """Delete a specific debug log file."""
    path = _safe_log_path(filename)
    path.unlink()
    return {"deleted": filename}


@router.delete("/settings/logs")
async def delete_all_log_files():
    """Delete all debug log files."""
    log_dir = _get_log_dir()
    deleted = []
    if log_dir.exists():
        for p in log_dir.iterdir():
            if _LOG_FILENAME_RE.match(p.name):
                p.unlink()
                deleted.append(p.name)
    return {"deleted_count": len(deleted), "deleted": deleted}


# ==================== Individual settings ====================

@router.get("/settings/{key}")
async def get_setting(key: str):
    """Get a single setting with metadata."""
    if key not in SETTING_DEFS:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")

    mgr = get_settings_manager()
    defn = SETTING_DEFS[key]
    default = mgr.get_default(key)
    is_override = mgr.has_override(key)
    value = mgr.get(key)

    result = {
        "key": key,
        "value": value,
        "default": default,
        "is_override": is_override,
        "type": defn.type.value,
        "label": defn.label,
        "description": defn.description,
    }
    if defn.min_val is not None:
        result["min"] = defn.min_val
    if defn.max_val is not None:
        result["max"] = defn.max_val

    return result


@router.put("/settings/{key}")
async def update_setting(key: str, request: UpdateSettingRequest):
    """Set a single setting override."""
    if key not in SETTING_DEFS:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")

    mgr = get_settings_manager()
    try:
        stored = mgr.set(key, request.value)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    _apply_setting_side_effects(key, stored, plugin_version=request.plugin_version)
    return {"key": key, "value": stored, "is_override": True}


@router.put("/settings")
async def bulk_update_settings(request: BulkUpdateRequest):
    """Update multiple settings at once. Only keys present are updated."""
    mgr = get_settings_manager()
    errors = {}
    stored = {}

    for key, value in request.settings.items():
        if key not in SETTING_DEFS:
            errors[key] = f"Unknown setting: {key}"
            continue
        try:
            stored[key] = mgr.set(key, value)
        except ValueError as e:
            errors[key] = str(e)

    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors, "stored": stored})

    for key, value in stored.items():
        _apply_setting_side_effects(key, value)

    return {"stored": stored}


@router.delete("/settings/{key}")
async def reset_setting(key: str):
    """Reset a setting to its tier default."""
    if key not in SETTING_DEFS:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")

    mgr = get_settings_manager()
    mgr.delete(key)
    default = mgr.get_default(key)

    _apply_setting_side_effects(key, default)
    return {"key": key, "value": default, "is_override": False}


# ==================== System info ====================

@router.get("/system/info")
async def get_system_info():
    """Get hardware profile, version, and uptime."""
    profile = get_hardware_profile()
    uptime_seconds = time.monotonic() - _start_time if _start_time else 0

    return {
        "version": _version,
        "uptime_seconds": round(uptime_seconds),
        "hardware": {
            "gpu_available": profile.gpu_available,
            "gpu_name": profile.gpu_name,
            "gpu_vram_mb": profile.gpu_vram_mb,
            "cpu_cores": profile.cpu_cores,
            "memory_total_mb": profile.memory_total_mb,
            "memory_available_mb": profile.memory_available_mb,
            "storage_free_mb": profile.storage_free_mb,
            "tier": profile.tier,
            "summary": profile.summary(),
        },
    }


# ==================== StashBox connections ====================

@router.get("/system/stashbox-connections")
async def get_stashbox_connections():
    """List all stash-box endpoints discovered from Stash's configuration."""
    mgr = get_connection_manager()
    return {"connections": mgr.get_connections()}


@router.post("/system/refresh-stashbox-config")
async def refresh_stashbox_config():
    """Re-read stash-box endpoint config from Stash without restarting."""
    mgr = get_connection_manager()
    count = await mgr.refresh()
    return {"endpoints_loaded": count, "connections": mgr.get_connections()}

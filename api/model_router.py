"""Model management API router.

Endpoints for checking model installation status, downloading models
from GitHub Releases, monitoring download progress, and reporting
available capabilities.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from capabilities import detect_capabilities
from model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# Set at startup
_manager: Optional[ModelManager] = None


def init_model_router(manager: ModelManager) -> None:
    """Store the ModelManager reference. Called once during lifespan."""
    global _manager
    _manager = manager


def _get_manager() -> ModelManager:
    """Get the model manager, raising if not initialized."""
    if _manager is None:
        raise RuntimeError("Model router not initialized")
    return _manager


@router.get("/models/status")
async def get_model_status():
    """Get installation status of all models."""
    mgr = _get_manager()
    return {"models": mgr.get_status()}


@router.post("/models/download/{model_name}")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """Start downloading a model in the background.

    Returns immediately with a confirmation. Use GET /models/download-progress
    to monitor the download.
    """
    mgr = _get_manager()

    # Validate model name exists in manifest
    status = mgr.get_status()
    if model_name not in status:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_name}",
        )

    if status[model_name].get("deprecated"):
        raise HTTPException(
            status_code=410,
            detail=f"Model {model_name} is deprecated and no longer available for download",
        )

    async def _do_download():
        try:
            await mgr.download_model(model_name)
        except Exception as e:
            logger.warning(f"Background download failed for {model_name}: {e}")

    background_tasks.add_task(_do_download)
    return {"status": "download_started", "model": model_name}


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete an installed model file that is no longer used (deprecated).

    Models still in use can't be deleted (409).
    """
    mgr = _get_manager()
    try:
        deleted = mgr.delete_model(model_name)
    except ValueError as e:
        status_code = 404 if str(e).startswith("Unknown model") else 409
        raise HTTPException(status_code=status_code, detail=str(e))
    return {"status": "deleted" if deleted else "not_installed", "model": model_name}


@router.post("/models/download-all")
async def download_all_models(background_tasks: BackgroundTasks):
    """Start downloading all missing models in the background.

    Returns immediately with a confirmation. Use GET /models/download-progress
    to monitor progress.
    """
    mgr = _get_manager()

    async def _do_download_all():
        try:
            await mgr.download_all()
        except Exception as e:
            logger.warning(f"Background download-all failed: {e}")

    background_tasks.add_task(_do_download_all)
    return {"status": "download_started"}


@router.get("/models/download-progress")
async def get_download_progress():
    """Get progress of active and recent downloads."""
    mgr = _get_manager()
    return {"progress": mgr.get_progress()}


@router.get("/capabilities")
async def get_capabilities():
    """Get available capabilities based on installed models and data."""
    mgr = _get_manager()
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    models_dir = data_dir / "models"
    caps = detect_capabilities(data_dir=data_dir, models_dir=models_dir)
    return {
        "capabilities": caps,
        "models": mgr.get_status(),
    }

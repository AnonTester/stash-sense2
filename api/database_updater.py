"""Database updater – checks GitHub releases and hot-swaps data files.

The sidecar ships with a pre-baked face recognition database.  This module
lets it self-update by downloading newer releases from the public
``AnonTester/stash-sense2-data`` repository on GitHub.

Lifecycle
---------
1. ``check_update()`` – hit the GitHub API (cached 10 min) and compare
   the latest release tag against the local ``manifest.json`` version.
2. ``start_update(download_url, target_version)`` – kick off a background
   ``asyncio.Task`` that runs the pipeline:
   download → extract → verify checksums → swap files → reload → cleanup.
3. ``get_status()`` – poll progress at any time.

Safety guarantees
-----------------
* ``stash_sense.db`` (local recommendations DB) is **never** touched.
* A timestamped backup of the old data is created before swapping.
* On any failure the old files are restored from backup (rollback).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from delta_applier import apply_delta_chain, find_delta_chain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Overridable so forks can point at their own data repo (e.g. a fork whose
# own dataset-update pipeline has picked up where a stale/abandoned upstream
# data repo left off) without patching this file again on every pull.
GITHUB_REPO = os.environ.get("DATABASE_UPDATE_REPO", "AnonTester/stash-sense2-data")
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CACHE_TTL_SECONDS = 600          # 10 minutes — in-memory short-term cache
PERSISTENT_CACHE_TTL = 43_200   # 12 hours — survives Docker restarts

# Files that may appear in a release zip.
RELEASE_FILES = {
    "performers.db",
    "face_embeddings.usearch",
    "faces.json",
    "performers.json",
    "manifest.json",
}

# Subset of RELEASE_FILES that *must* be present for a valid release.
REQUIRED_FILES = {
    "performers.db",
    "face_embeddings.usearch",
    "faces.json",
    "performers.json",
    "manifest.json",
}

# Download buffer size (64 KiB).
_CHUNK_SIZE = 65_536


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------

class UpdateStatus(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    SWAPPING = "swapping"
    RELOADING = "reloading"
    COMPLETE = "complete"
    FAILED = "failed"


class UpdateState:
    """Mutable state bag tracked by :class:`DatabaseUpdater`."""

    __slots__ = ("status", "progress_pct", "current_version", "target_version", "error")

    def __init__(self) -> None:
        self.status: UpdateStatus = UpdateStatus.IDLE
        self.progress_pct: int = 0
        self.current_version: Optional[str] = None
        self.target_version: Optional[str] = None
        self.error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "progress_pct": self.progress_pct,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# DatabaseUpdater
# ---------------------------------------------------------------------------

class DatabaseUpdater:
    """Manages self-updating the face-recognition data directory."""

    def __init__(self, data_dir: Path, reload_fn: Callable[[Path], bool]) -> None:
        self._data_dir = data_dir
        self._reload_fn = reload_fn

        self._state = UpdateState()
        self._state.current_version = self._get_current_version()

        # GitHub API response cache — in-memory (lost on restart)
        self._cache: Optional[dict[str, Any]] = None
        self._cache_time: float = 0.0

        # Persistent cache file — survives Docker restarts
        self._persistent_cache_file = data_dir / ".update_check_cache.json"

        # Background task handle
        self._update_task: Optional[asyncio.Task] = None

        # Full hop details (incl. download URLs) for the delta chain found
        # by the most recent check_update() — consumed by start_update()
        # when choosing delta vs. full. Not part of the cached/returned
        # result dict (that only carries summary fields for the API).
        self._last_delta_chain: Optional[list[dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _get_current_version(self) -> Optional[str]:
        """Read the version string from ``manifest.json`` in the data dir."""
        manifest_path = self._data_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path) as f:
                return json.load(f).get("version")
        except (json.JSONDecodeError, OSError):
            return None

    def get_status(self) -> dict[str, Any]:
        """Return the current update state as a plain dict."""
        self._state.current_version = self._get_current_version()
        return self._state.to_dict()

    # ------------------------------------------------------------------
    # Check for updates (GitHub API)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Persistent cache helpers
    # ------------------------------------------------------------------

    def _load_persistent_cache(self) -> Optional[dict[str, Any]]:
        """Return the cached check result if it is still within 12 hours."""
        try:
            with open(self._persistent_cache_file) as f:
                data = json.load(f)
            checked_at: float = data.get("checked_at", 0.0)
            if (time.time() - checked_at) < PERSISTENT_CACHE_TTL:
                return data.get("result")
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        return None

    def _save_persistent_cache(self, result: dict[str, Any]) -> None:
        """Write the check result to disk with the current timestamp."""
        try:
            payload = {"checked_at": time.time(), "result": result}
            self._persistent_cache_file.write_text(json.dumps(payload))
        except OSError as exc:
            logger.warning("Could not write update check cache: %s", exc)

    # ------------------------------------------------------------------
    # Check for updates (GitHub API)
    # ------------------------------------------------------------------

    async def check_update(self, force: bool = False) -> dict[str, Any]:
        """Query the latest GitHub release tag and compare to local version,
        and separately check whether a delta chain (small download) can get
        there instead of the full zip.

        Caching strategy (to avoid GitHub API rate limits on crash-loops):
        - Persistent file cache (12 h) — survives Docker restarts.
        - In-memory cache (10 min) — avoids redundant disk reads.
        - ``force=True`` — bypasses both caches (used by the manual UI action).

        The delta chain's full hop details (download URLs) are cached too
        (as an internal ``_delta_chain`` key, stripped before returning) so
        a cache hit doesn't cost an extra GitHub API call, and so
        ``start_update()`` can reuse whatever chain the most recent check
        found via ``self._last_delta_chain`` without re-querying.
        """
        if not force:
            # Fast path: in-memory cache
            now = time.monotonic()
            if self._cache is not None and (now - self._cache_time) < CACHE_TTL_SECONDS:
                self._last_delta_chain = self._cache.get("_delta_chain")
                return self._public_result(self._cache)

            # Persistent cache — survives container restarts
            persistent = self._load_persistent_cache()
            if persistent is not None:
                # Refresh the in-memory cache from disk so subsequent calls
                # within the same process don't hit the file again.
                self._cache = persistent
                self._cache_time = time.monotonic()
                self._last_delta_chain = persistent.get("_delta_chain")
                return self._public_result(persistent)

        async with httpx.AsyncClient() as client:
            resp = await client.get(GITHUB_API_URL, follow_redirects=True)
            resp.raise_for_status()
            release = resp.json()

        latest_tag: str = release["tag_name"].lstrip("v")
        current = self._get_current_version()

        # Find the *full bundle* zip asset — must match the naming
        # convention exactly (not just "any .zip"), now that delta zips
        # (stash-sense2-delta-*.zip) live on the same release and asset
        # order isn't a contract worth relying on.
        download_url: Optional[str] = None
        download_size_mb: Optional[int] = None
        for asset in release.get("assets", []):
            if asset["name"].startswith("stash-sense2-data-") and asset["name"].endswith(".zip"):
                download_url = asset["browser_download_url"]
                size_bytes = asset.get("size")
                if size_bytes:
                    download_size_mb = round(size_bytes / 1_000_000)
                break

        update_available = current is None or latest_tag > current

        delta_chain: Optional[list[dict[str, Any]]] = None
        if current is not None:
            try:
                delta_chain = await find_delta_chain(GITHUB_REPO, current)
            except Exception as exc:
                logger.warning("Delta chain lookup failed, will offer full download only: %s", exc)

        result: dict[str, Any] = {
            "update_available": update_available,
            "current_version": current,
            "latest_version": latest_tag,
            "release_name": release.get("name"),
            "download_url": download_url,
            "download_size_mb": download_size_mb,
            "published_at": release.get("published_at"),
            "delta_available": bool(delta_chain),
            "delta_chain_length": len(delta_chain) if delta_chain else None,
            "delta_download_size_mb": round(sum(h["size_mb"] for h in delta_chain), 2) if delta_chain else None,
            "_delta_chain": delta_chain,
        }

        # Update both caches (with the internal chain details included —
        # see docstring)
        self._cache = result
        self._cache_time = time.monotonic()
        self._save_persistent_cache(result)
        self._last_delta_chain = delta_chain
        return self._public_result(result)

    @staticmethod
    def _public_result(result: dict[str, Any]) -> dict[str, Any]:
        """Strip internal-only keys (full delta hop details) before this
        goes anywhere near the API response model."""
        return {k: v for k, v in result.items() if not k.startswith("_")}

    # ------------------------------------------------------------------
    # Start / run update
    # ------------------------------------------------------------------

    async def start_update(self, download_url: str, target_version: str) -> str:
        """Kick off the update pipeline in a background ``asyncio.Task``.

        Returns a job ID string.  Raises ``RuntimeError`` if an update
        is already in progress.
        """
        if self._update_task is not None and not self._update_task.done():
            raise RuntimeError("Update already running")

        job_id = uuid.uuid4().hex[:12]
        self._state.target_version = target_version
        self._update_task = asyncio.create_task(
            self._run_update(download_url, target_version),
        )
        return job_id

    async def _run_update(self, download_url: str, target_version: str) -> None:
        """Execute the full update pipeline.

        On any exception the state is set to FAILED and a rollback is
        attempted (if a backup exists).
        """
        work_dir = self._data_dir / f"update_{uuid.uuid4().hex[:8]}"
        zip_path = work_dir / "release.zip"
        extract_dir = work_dir / "extracted"
        backup_dir: Optional[Path] = None

        try:
            work_dir.mkdir(parents=True, exist_ok=True)

            # 1. Download
            self._state.status = UpdateStatus.DOWNLOADING
            self._state.progress_pct = 0
            await self._download(download_url, zip_path)

            # 2. Extract
            self._state.status = UpdateStatus.EXTRACTING
            self._state.progress_pct = 30
            self._extract(zip_path, extract_dir)

            # 3. Verify
            self._state.status = UpdateStatus.VERIFYING
            self._state.progress_pct = 50
            self._load_and_verify(extract_dir)

            # 4. Swap
            self._state.status = UpdateStatus.SWAPPING
            self._state.progress_pct = 70
            backup_dir = self._swap_files(extract_dir)

            # 5. Reload
            self._state.status = UpdateStatus.RELOADING
            self._state.progress_pct = 90
            self._reload_fn(self._data_dir)

            # 6. Complete
            self._state.status = UpdateStatus.COMPLETE
            self._state.progress_pct = 100
            self._state.current_version = target_version

            # Invalidate both caches so the next poll reflects the new version.
            self._cache = None
            try:
                self._persistent_cache_file.unlink(missing_ok=True)
            except OSError:
                pass

            logger.warning("Database update to %s complete", target_version)

        except Exception as exc:
            logger.warning("Database update failed: %s", exc)
            self._state.status = UpdateStatus.FAILED
            self._state.error = str(exc)

            # Attempt rollback if we created a backup
            if backup_dir is not None and backup_dir.exists():
                try:
                    self._rollback(backup_dir)
                    logger.warning("Rolled back to backup %s", backup_dir.name)
                except Exception as rb_exc:
                    logger.error("Rollback also failed: %s", rb_exc)

        finally:
            # Clean up temp working directory
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Delta update (small download, when a chain to latest exists)
    # ------------------------------------------------------------------

    async def start_delta_update(self, chain: list[dict[str, Any]]) -> str:
        """Kick off delta-chain application in a background task. Mirrors
        start_update()'s single-task guard; returns a job ID."""
        if self._update_task is not None and not self._update_task.done():
            raise RuntimeError("Update already running")

        job_id = uuid.uuid4().hex[:12]
        self._state.target_version = chain[-1]["to_version"] if chain else self._get_current_version()
        self._update_task = asyncio.create_task(self._run_delta_update(chain))
        return job_id

    async def _run_delta_update(self, chain: list[dict[str, Any]]) -> None:
        """Run apply_delta_chain() and reflect its progress/outcome in
        self._state, same shape as _run_update(). Unlike _run_update, no
        separate backup/rollback here — apply_delta_chain() already
        guarantees the data dir is fully applied or fully rolled back
        before it returns/raises, covering the whole chain as one unit.
        """
        try:
            self._state.status = UpdateStatus.DOWNLOADING
            self._state.progress_pct = 0

            def _progress(pct: int) -> None:
                self._state.status = UpdateStatus.SWAPPING
                self._state.progress_pct = pct

            result = await apply_delta_chain(chain, self._data_dir, progress_cb=_progress)

            self._state.status = UpdateStatus.RELOADING
            self._state.progress_pct = 95
            self._reload_fn(self._data_dir)

            self._state.status = UpdateStatus.COMPLETE
            self._state.progress_pct = 100
            self._state.current_version = result.get("new_version", self._state.current_version)

            self._cache = None
            self._last_delta_chain = None
            try:
                self._persistent_cache_file.unlink(missing_ok=True)
            except OSError:
                pass

            logger.warning("Delta update to %s complete: %s", self._state.current_version, result)

        except Exception as exc:
            logger.warning("Delta update failed (rolled back automatically): %s", exc)
            self._state.status = UpdateStatus.FAILED
            self._state.error = str(exc)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _download(self, url: str, dest: Path) -> None:
        """Stream-download *url* to *dest* with progress tracking."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0

                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=_CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            self._state.progress_pct = int(30 * downloaded / total)

    def _extract(self, zip_path: Path, extract_dir: Path) -> None:
        """Unzip *zip_path* into *extract_dir*."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    def _load_and_verify(self, extract_dir: Path) -> dict[str, Any]:
        """Load the manifest from the extract dir, check required files,
        and verify checksums.  Returns the parsed manifest dict.
        """
        manifest_path = extract_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Missing required file: manifest.json")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Check that all required files are present
        missing = [
            fname for fname in REQUIRED_FILES
            if not (extract_dir / fname).exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing required files: {', '.join(sorted(missing))}")

        self._verify_checksums(extract_dir, manifest)
        return manifest

    def _verify_checksums(self, extract_dir: Path, manifest: dict[str, Any]) -> None:
        """SHA-256 verify every file listed in the manifest's checksums."""
        checksums: dict[str, str] = manifest.get("checksums", {})
        for filename, expected_raw in checksums.items():
            file_path = extract_dir / filename
            if not file_path.exists():
                # File listed in checksums but not present – skip
                # (required-file check is done separately).
                continue

            # Expected format: "sha256:<hex>"
            if ":" in expected_raw:
                algo, expected_hash = expected_raw.split(":", 1)
            else:
                algo, expected_hash = "sha256", expected_raw

            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Checksum mismatch for {filename}: "
                    f"expected {expected_hash[:16]}…, got {actual_hash[:16]}…"
                )

    def _swap_files(self, extract_dir: Path) -> Path:
        """Move old data files into a backup dir, then move new files in.

        Only backs up files that have a replacement in the new release.
        Files in ``RELEASE_FILES`` that exist locally but are absent from
        the new release are left untouched. ``stash_sense.db`` and any
        other non-release artefacts are never touched.

        Returns the backup directory path.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = self._data_dir / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True)

        # Determine which release files are in the new release
        new_files = {
            item.name for item in extract_dir.iterdir()
            if item.name in RELEASE_FILES
        }

        # Phase 1: back up only files that will be replaced
        for fname in new_files:
            src = self._data_dir / fname
            if src.exists():
                shutil.move(str(src), str(backup_dir / fname))

        # Phase 2: move new files into data dir
        for fname in new_files:
            src = extract_dir / fname
            shutil.move(str(src), str(self._data_dir / fname))

        return backup_dir

    def _rollback(self, backup_dir: Path) -> None:
        """Restore files from *backup_dir* back into the data dir.

        Only restores files in ``RELEASE_FILES`` — ``stash_sense.db``
        is never touched.
        """
        for item in backup_dir.iterdir():
            if item.name in RELEASE_FILES:
                dest = self._data_dir / item.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(item), str(dest))

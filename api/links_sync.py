"""Keeps performer_links.json and aliases.json in step with the database release they belong to.

Why this exists: these two files are NOT part of what a delta carries into the sidecar's data directory, and the full-zip
swap used to ignore them too (they were missing from database_updater.RELEASE_FILES). So an install that updated
through releases never received new performer link groups (the matcher then treated two catalogue records of one real
person as two people) nor new aliases (used by "Prefer Western Names"), and a fresh install had neither file at all.

Each release now also carries them as a small asset, ``stash-sense2-links-<version>.zip`` (performer_links.json +
aliases.json), and states their sha256 in the release notes::

    links-sha256: <hex>
    aliases-sha256: <hex>

The notes are already fetched for the update check, so comparing the local files with the release costs nothing extra;
only a mismatch downloads the asset (a couple of MB). The files are state, not a diff: whatever path the install took to
this version (full zip, one delta, a chain, a stale copy), replacing them with the release's own copy gives the right
result, and doing it again is a no-op.

A release without the hash lines or the asset (published before this existed) is left alone.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

LINK_FILES = ("performer_links.json", "aliases.json")
_LINKS_ASSET_RE = re.compile(r"^stash-sense2-links-(?P<version>.+)\.zip$")
_HASH_LINE_RES = {
    "performer_links.json": re.compile(r"^links-sha256:\s*([0-9a-fA-F]{64})\s*$", re.M),
    "aliases.json": re.compile(r"^aliases-sha256:\s*([0-9a-fA-F]{64})\s*$", re.M),
}
_CHUNK_SIZE = 65_536


def parse_link_hashes(body: Optional[str]) -> Optional[dict[str, str]]:
    """{file name: sha256} from a release's notes, or None unless BOTH lines are present."""
    if not body:
        return None
    out: dict[str, str] = {}
    for name, rx in _HASH_LINE_RES.items():
        m = rx.search(body)
        if not m:
            return None
        out[name] = m.group(1).lower()
    return out


def find_links_asset(release: dict[str, Any]) -> Optional[dict[str, Any]]:
    return next((a for a in release.get("assets", []) if _LINKS_ASSET_RE.match(a.get("name", ""))), None)


def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def stale_files(data_dir: Path, hashes: dict[str, str]) -> list[str]:
    """Which of the link files differ from (or are missing against) the release's hashes."""
    return [name for name in LINK_FILES if sha256_file(data_dir / name) != hashes[name]]


async def _download(url: str, dest: Path) -> None:
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=_CHUNK_SIZE):
                    f.write(chunk)


def _install(zip_path: Path, data_dir: Path, hashes: dict[str, str]) -> list[str]:
    """Verify then atomically replace. Nothing is touched unless BOTH files in the zip match the release's hashes."""
    staging = data_dir / ".links_sync_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in LINK_FILES:
                if name not in zf.namelist():
                    raise ValueError(f"{name} missing from the links asset")
                zf.extract(name, staging)
        for name in LINK_FILES:
            if sha256_file(staging / name) != hashes[name]:
                raise ValueError(f"checksum mismatch for {name} in the links asset")
            json.loads((staging / name).read_text())      # must at least be valid JSON before it replaces anything
        changed = []
        for name in LINK_FILES:
            target = data_dir / name
            if sha256_file(target) == hashes[name]:
                continue
            if target.exists():
                shutil.copy2(target, data_dir / f"{name}.prev")   # one generation back, in case it needs undoing
            os.replace(staging / name, target)
            changed.append(name)
        _refresh_manifest_checksums(data_dir, hashes, changed)
        return changed
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _refresh_manifest_checksums(data_dir: Path, hashes: dict[str, str], changed: list[str]) -> None:
    """If manifest.json records checksums for these files, keep them true."""
    manifest_path = data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict):
        return
    touched = False
    for name in changed:
        if name in checksums:
            checksums[name] = f"sha256:{hashes[name]}"
            touched = True
    if touched:
        manifest_path.write_text(json.dumps(manifest, indent=2))


def changed_link_uids(previous: Path, current: Path) -> list[str]:
    """Every universal_id in a link group that was added, removed or changed between two performer_links.json files
    (groups compared as sets). Empty when there is no previous file (a fresh install has no stored matches to redo)."""
    try:
        old = {frozenset(g) for g in json.loads(previous.read_text())}
        new = {frozenset(g) for g in json.loads(current.read_text())}
    except (OSError, ValueError):
        return []
    return sorted({uid for group in (old ^ new) for uid in group})


async def sync_link_files(data_dir: Path, release: dict[str, Any], local_version: Optional[str]) -> dict[str, Any]:
    """Bring performer_links.json / aliases.json in line with `release` (a GitHub release object).

    Returns {"status": "in_sync" | "updated" | "skipped" | "failed", "reason": str, "changed": [file names],
    "link_uids": [universal_ids in link groups that changed -- only when performer_links.json was replaced]}.
    Never raises: a problem here must not break an update or startup (the old files keep working)."""
    result: dict[str, Any] = {"status": "skipped", "reason": "", "changed": []}
    try:
        tag = (release.get("tag_name") or "").lstrip("v")
        if not local_version or local_version != tag:
            result["reason"] = f"local database is {local_version}, release is {tag}"
            return result
        hashes = parse_link_hashes(release.get("body"))
        asset = find_links_asset(release)
        if hashes is None or asset is None:
            result["reason"] = "release carries no links asset"
            return result
        if not stale_files(data_dir, hashes):
            result["status"] = "in_sync"
            return result

        zip_path = data_dir / ".links_sync_download.zip"
        try:
            await _download(asset["browser_download_url"], zip_path)
            result["changed"] = _install(zip_path, data_dir, hashes)
        finally:
            zip_path.unlink(missing_ok=True)
        if "performer_links.json" in result["changed"]:
            result["link_uids"] = changed_link_uids(data_dir / "performer_links.json.prev", data_dir / "performer_links.json")
        result["status"] = "updated" if result["changed"] else "in_sync"
        logger.warning("Link/alias files synced to release %s: %s", tag, ", ".join(result["changed"]) or "nothing to change")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Link/alias file sync failed (existing files left as they were): %s", exc)
        result.update(status="failed", reason=str(exc))
    return result

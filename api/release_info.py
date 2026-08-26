"""Sidecar/plugin release tracking -- lets the plugin learn "is a newer
sidecar/plugin available" and "does the currently-connected sidecar need
a newer plugin than I am" without ever hitting GitHub itself.

The browser polling GitHub directly for this (once per connected device,
every health check) is exactly the 60-requests/hour unauthenticated
rate-limit trap fixed elsewhere in this project today for the database-
version check -- except worse here, since it would multiply per browser/
device rather than being one shared sidecar-side check. This module is
the sidecar doing that check itself, on a slow background interval, and
caching the result in memory; `/health` (see database_health_router.py)
just reads that cache on every poll -- no I/O on the request path, so
`/health`'s own "no side effects" contract still holds.

Two independent lookups:
- Latest sidecar version: GitHub REST API (`api.github.com`, subject to
  the same unauthenticated rate limit as database_updater.py's own
  check -- but this only calls it once an hour, not once per browser).
- Latest plugin version: `stash-plugin-repo`'s `index.yml`, fetched from
  `raw.githubusercontent.com` -- a different, unrelated CDN path with no
  shared rate-limit bucket with the REST API at all.

A failed background check logs a warning and keeps the last known-good
value rather than clearing it -- a single flaky refresh should never
flicker the UI from "here's the real state" to "no info available."
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Overridable so a fork checks its own releases/plugin listing, not
# upstream's -- mirrors database_updater.py's DATABASE_UPDATE_REPO.
SIDECAR_REPO = os.environ.get("SIDECAR_UPDATE_REPO", "AnonTester/stash-sense2")
PLUGIN_INDEX_REPO = os.environ.get("PLUGIN_INDEX_REPO", "AnonTester/stash-plugin-repo")
PLUGIN_ID = "stash-sense2"

_SIDECAR_RELEASES_URL = f"https://api.github.com/repos/{SIDECAR_REPO}/releases/latest"
_PLUGIN_INDEX_URL = f"https://raw.githubusercontent.com/{PLUGIN_INDEX_REPO}/main/index.yml"

# Lowest plugin version this sidecar actually works correctly against --
# same discipline as the plugin's own MIN_SIDECAR_VERSION
# (plugin/stash-sense-core.js): bump only in the specific commit where a
# sidecar-side change starts requiring plugin-side behavior an older
# build doesn't have (a new field it doesn't send, a response shape it
# doesn't parse), not reflexively on every sidecar release.
MIN_PLUGIN_VERSION = "0.14.19"

REFRESH_INTERVAL_SECONDS = 3600  # 1 hour -- see module docstring

# Deployed (Docker) layout flattens api/*.py directly into /app (see
# Dockerfile*'s `COPY api/ ./`), so changelog.txt (COPY'd alongside it)
# ends up a *sibling* of this file there -- but local dev (`cd api &&
# make sidecar`, per this repo's own CLAUDE.md) keeps api/ as a real
# subdirectory, where changelog.txt is one level *up* at the repo root
# instead. Try the deployed layout first (the common/production case),
# fall back to the dev layout; changelog_since() already handles "file
# doesn't exist at all" gracefully either way.
_CHANGELOG_CANDIDATES = (
    Path(__file__).resolve().parent / "changelog.txt",
    Path(__file__).resolve().parent.parent / "changelog.txt",
)
_CHANGELOG_PATH = next((p for p in _CHANGELOG_CANDIDATES if p.exists()), _CHANGELOG_CANDIDATES[0])
_VERSION_HEADER_RE = re.compile(r"^### (\d+(?:\.\d+)*) \((sidecar|plugin) only\)\s*$")
_DATE_HEADER_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$")
_INDEX_ENTRY_ID_RE = re.compile(r"^-\s*id:\s*(.+?)\s*$")
_INDEX_ENTRY_VERSION_RE = re.compile(r"^\s*version:\s*(.+?)\s*$")


def compare_versions(a: Optional[str], b: Optional[str]) -> int:
    """Same plain dot-segment comparison as the plugin's own
    compareVersions() in stash-sense-core.js -- keep these two in sync if
    either changes; neither handles pre-release/build metadata."""
    def _parts(v):
        return [int(p) if p.isdigit() else 0 for p in (v or "0").split(".")]
    pa, pb = _parts(a), _parts(b)
    for x, y in zip(pa + [0] * max(0, len(pb) - len(pa)), pb + [0] * max(0, len(pa) - len(pb))):
        if x != y:
            return -1 if x < y else 1
    return 0


def _parse_plugin_version_from_index(text: str, plugin_id: str) -> Optional[str]:
    """Pull `version:` out of the entry matching `- id: <plugin_id>` in
    stash-plugin-repo's index.yml. Deliberately not a real YAML parser --
    avoids adding PyYAML as a new dependency (and the image-rebuild cost
    of touching requirements.docker.txt) just for this one flat, stable
    list-of-dicts structure. Brittle if that file's shape changes
    significantly; it hasn't so far and the format is simple."""
    in_entry = False
    for line in text.splitlines():
        id_match = _INDEX_ENTRY_ID_RE.match(line)
        if id_match:
            in_entry = id_match.group(1) == plugin_id
            continue
        if in_entry:
            version_match = _INDEX_ENTRY_VERSION_RE.match(line)
            if version_match:
                return version_match.group(1).strip('"').strip("'")
    return None


def changelog_since(component: str, since_version: Optional[str]) -> list[dict]:
    """Every changelog.txt entry for `component` ("sidecar" or "plugin")
    newer than `since_version`, as [{"version", "date", "bullets"}, ...].
    Returns [] if changelog.txt isn't present (e.g. an older image built
    before Dockerfile* started COPYing it in) or `since_version` is None
    (caller has nothing to diff against)."""
    if since_version is None or not _CHANGELOG_PATH.exists():
        return []

    entries: list[dict] = []
    current_date: Optional[str] = None
    current_version: Optional[str] = None
    current_component: Optional[str] = None
    current_bullets: list[str] = []

    def _flush():
        if (current_version and current_component == component
                and compare_versions(current_version, since_version) > 0):
            entries.append({"version": current_version, "date": current_date, "bullets": current_bullets})

    for line in _CHANGELOG_PATH.read_text().splitlines():
        date_match = _DATE_HEADER_RE.match(line)
        if date_match:
            _flush()
            current_date, current_version, current_component, current_bullets = date_match.group(1), None, None, []
            continue
        version_match = _VERSION_HEADER_RE.match(line)
        if version_match:
            _flush()
            current_version, current_component, current_bullets = version_match.group(1), version_match.group(2), []
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            current_bullets.append(stripped[2:])
        elif stripped and current_bullets:
            current_bullets[-1] += " " + stripped  # wrapped continuation of the previous bullet
    _flush()
    return entries


class ReleaseInfoCache:
    """In-memory cache refreshed by the background loop below; `/health`
    reads it synchronously via get_info(). One instance, module-level
    singleton (see `_cache` at bottom) -- there's only ever one sidecar
    process, no per-request state needed."""

    def __init__(self):
        self.latest_sidecar_version: Optional[str] = None
        self.latest_plugin_version: Optional[str] = None
        self.last_refreshed_at: Optional[float] = None

    async def refresh(self) -> None:
        await self._refresh_sidecar_version()
        await self._refresh_plugin_version()
        self.last_refreshed_at = time.monotonic()

    async def _refresh_sidecar_version(self) -> None:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(_SIDECAR_RELEASES_URL, follow_redirects=True, timeout=15.0)
                resp.raise_for_status()
                self.latest_sidecar_version = resp.json()["tag_name"].lstrip("v")
        except Exception as exc:
            logger.warning("Could not check latest sidecar release (keeping last known value %s): %s",
                            self.latest_sidecar_version, exc)

    async def _refresh_plugin_version(self) -> None:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(_PLUGIN_INDEX_URL, follow_redirects=True, timeout=15.0)
                resp.raise_for_status()
                version = _parse_plugin_version_from_index(resp.text, PLUGIN_ID)
                if version is None:
                    logger.warning("Could not find plugin id %r in %s (keeping last known value %s)",
                                    PLUGIN_ID, _PLUGIN_INDEX_URL, self.latest_plugin_version)
                else:
                    self.latest_plugin_version = version
        except Exception as exc:
            logger.warning("Could not check latest plugin release (keeping last known value %s): %s",
                            self.latest_plugin_version, exc)

    def get_info(self, sidecar_version: Optional[str], reported_plugin_version: Optional[str]) -> dict:
        """Synchronous, no I/O -- safe to call on every /health request."""
        return {
            "latest_sidecar_version": self.latest_sidecar_version,
            "min_plugin_version": MIN_PLUGIN_VERSION,
            "latest_plugin_version": self.latest_plugin_version,
            "sidecar_changelog": changelog_since("sidecar", sidecar_version)
            if self.latest_sidecar_version and compare_versions(self.latest_sidecar_version, sidecar_version) > 0
            else [],
            "plugin_changelog": changelog_since("plugin", reported_plugin_version)
            if self.latest_plugin_version and reported_plugin_version
            and compare_versions(self.latest_plugin_version, reported_plugin_version) > 0
            else [],
        }


_cache = ReleaseInfoCache()


def get_cache() -> ReleaseInfoCache:
    return _cache


async def refresh_loop(interval: float = REFRESH_INTERVAL_SECONDS) -> None:
    """Background task -- same shape as main.py's existing _idle_checker
    (asyncio.create_task at startup in lifespan(), cancelled+awaited at
    shutdown). Refreshes once immediately (so /health has real data
    within seconds of startup, not only after the first hour), then on
    `interval`."""
    await _cache.refresh()
    while True:
        await asyncio.sleep(interval)
        await _cache.refresh()

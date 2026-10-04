"""Disambiguation lookup for matched performers, resolved at response time.

Several performers can share one name (single-name performers especially), so
the disambiguation ("Linda (GGG 2002)") is what tells them apart in a match
list. performers.json carries no disambiguation, but performers.db does, so a
match's disambiguation is read from there when a response is built. That also
covers results stored before this existed, and a database update takes effect
immediately, with no rescan or migration.

Takes a universal_id in any of the shapes stashbox_utils.classify_universal_id
knows ("<endpoint domain>:<stashbox uuid>" for stash-box, "<source>:<performers.id>"
for catalogue). "local:<id>" is not resolved here: a local-library match's own
disambiguation lives in Stash, not in performers.db.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from export_db_to_json import make_universal_id
from stashbox_utils import classify_universal_id

logger = logging.getLogger(__name__)

_MAX_CACHE = 20000

# universal_id -> disambiguation ("" = looked up, none set). Dropped whenever
# performers.db changes on disk (a database update rewrites it).
_cache: dict[str, str] = {}
_cache_token: Optional[tuple[int, int]] = None


def _db_path() -> Path:
    return Path(os.environ.get("DATA_DIR", "./data")) / "performers.db"


def _lookup(conn: sqlite3.Connection, universal_id: str) -> str:
    domain, _, ident = universal_id.partition(":")
    category = classify_universal_id(universal_id)
    if category == "stashbox":
        # performers.db stores the short endpoint name ("stashdb"), a universal_id the
        # domain ("stashdb.org"): narrow by the domain's stem (hits the primary key), then
        # confirm with make_universal_id so this can never disagree with how ids are built.
        row = None
        for endpoint, disambiguation in conn.execute(
            "SELECT s.endpoint, p.disambiguation FROM stashbox_ids s JOIN performers p ON p.id = s.performer_id "
            "WHERE s.endpoint = ? AND s.stashbox_performer_id = ?",
            (domain.split(".")[0], ident),
        ):
            if make_universal_id(endpoint, ident) == universal_id:
                row = (disambiguation,)
                break
    elif category == "catalogue" and ident.isdigit():
        row = conn.execute("SELECT disambiguation FROM performers WHERE id = ?", (int(ident),)).fetchone()
    else:
        return ""
    return (row[0] or "").strip() if row else ""


def get_disambiguations(universal_ids) -> dict[str, str]:
    """universal_id -> disambiguation, only for ids that have one. Never raises:
    a missing/locked database just means no disambiguation is shown."""
    global _cache_token
    ids = {u for u in universal_ids if u and classify_universal_id(u) in ("stashbox", "catalogue")}
    if not ids:
        return {}
    path = _db_path()
    try:
        st = path.stat()
        token = (st.st_mtime_ns, st.st_size)
        if token != _cache_token or len(_cache) > _MAX_CACHE:
            _cache.clear()
            _cache_token = token
        missing = [u for u in ids if u not in _cache]
        if missing:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
            try:
                for uid in missing:
                    _cache[uid] = _lookup(conn, uid)
            finally:
                conn.close()
    except (OSError, sqlite3.Error) as e:
        logger.debug("disambiguation lookup unavailable: %s", e)
        return {u: _cache[u] for u in ids if _cache.get(u)}
    return {u: _cache[u] for u in ids if _cache.get(u)}


def get_disambiguation(universal_id: Optional[str]) -> Optional[str]:
    if not universal_id:
        return None
    return get_disambiguations([universal_id]).get(universal_id) or None

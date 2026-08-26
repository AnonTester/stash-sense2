"""Delta update application — the small-download alternative to
database_updater.py's full-zip swap.

A delta package (built by stash-sense2-data-gen's build/export_delta.py) is
a lean SQLite file (`delta.db`) listing exactly what changed since some
prior release: performer upserts/removals, new face vectors (already
tagged with the *server-assigned* usearch index — we reuse it verbatim,
never recompute), and a `removed_faces` table of exactly which indices to
drop. Because removals are precomputed server-side, applying a delta here
is purely mechanical replay — no "diff the image list" business logic
needed client-side, that already happened when the delta was built.

Two independent halves, both applied by `apply_delta_db` below: the
`performers`/`faces`/`removed_faces` tables above (stashbox-sourced,
identity is `(endpoint, stashbox_id)` via the `stashbox_ids` table -- a
locally-inserted performer's own autoincrement id is never assumed to
match the server's, since identity doesn't depend on it), and
`catalogue_performers`/`catalogue_performer_urls`/`catalogue_faces`
(pornbox.com/seekfans.com/legacy-site performers with no stashbox linkage
at all -- identity there IS the raw `performers.id`, portable specifically
because the server's own performers.db is one continuously-evolving file,
never regenerated from scratch, so a given performer's id is permanent).
The catalogue tables are strictly additive and may not exist in an
older delta.db -- `_has_table` guards every read of them, so applying a
pre-catalogue-support delta through this code is a no-op for that half,
not an error.

Lifecycle
---------
1. ``find_delta_chain(current_version)`` — walk release history back from
   latest, following each release's delta asset filename
   (``stash-sense-delta-{from}-to-{to}.zip``) until reaching the version
   currently installed. Returns ``None`` on any gap (missing delta asset,
   broken chain) — the caller should fall back to the existing full-zip
   path in that case, never error out.
2. ``apply_delta_chain(chain, data_dir, download_fn)`` — single backup of
   the current data files, apply each delta in the chain in order, verify
   checksums as each is downloaded, regenerate faces.json/performers.json
   (via export_db_to_json.py's own functions — no new export logic here),
   write a new manifest.json. Rolls back the *entire* chain on any failure,
   mirroring database_updater.py's backup/rollback pattern exactly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import numpy as np
from usearch.index import Index

from export_db_to_json import export_faces_json, export_performers_json

logger = logging.getLogger(__name__)

_DELTA_ASSET_RE = re.compile(r"^stash-sense2-delta-(?P<from_version>.+)-to-(?P<to_version>.+)\.zip$")
_CHUNK_SIZE = 65_536

# Same set database_updater.py backs up/restores — kept identical so a
# rollback here is indistinguishable from a rollback of the full-zip path.
BACKED_UP_FILES = ("performers.db", "face_embeddings.usearch", "faces.json", "performers.json", "manifest.json")


# ---------------------------------------------------------------------------
# Chain discovery
# ---------------------------------------------------------------------------

async def _fetch_releases(github_repo: str, per_page: int = 30) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{github_repo}/releases?per_page={per_page}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()  # newest first, per GitHub API default


async def find_delta_chain(github_repo: str, current_version: Optional[str]) -> Optional[list[dict[str, Any]]]:
    """Return an ordered (oldest-to-newest) list of delta hops to apply to
    get from `current_version` to latest, or None if no complete chain
    exists (caller should fall back to a full download).
    """
    if current_version is None:
        return None  # fresh install — full download is simpler and correct

    releases = await _fetch_releases(github_repo)
    if not releases:
        return None

    latest_version = releases[0]["tag_name"].lstrip("v")
    if latest_version == current_version:
        return []  # already up to date

    by_to_version: dict[str, dict[str, Any]] = {}
    for rel in releases:
        to_version = rel["tag_name"].lstrip("v")
        delta_asset = next(
            (a for a in rel.get("assets", []) if _DELTA_ASSET_RE.match(a["name"])), None,
        )
        if delta_asset is None:
            continue
        m = _DELTA_ASSET_RE.match(delta_asset["name"])
        by_to_version[to_version] = {
            "to_version": to_version,
            "from_version": m.group("from_version"),
            "download_url": delta_asset["browser_download_url"],
            "size_mb": round(delta_asset.get("size", 0) / 1_000_000, 2),
        }

    chain: list[dict[str, Any]] = []
    cursor = latest_version
    seen: set[str] = set()
    while cursor != current_version:
        if cursor in seen:
            logger.warning("Delta chain has a cycle at version %s — falling back to full download", cursor)
            return None
        seen.add(cursor)
        hop = by_to_version.get(cursor)
        if hop is None:
            logger.info("No delta chain from %s to %s (gap at %s) — falling back to full download",
                        current_version, latest_version, cursor)
            return None
        chain.append(hop)
        cursor = hop["from_version"]

    chain.reverse()
    return chain


# ---------------------------------------------------------------------------
# Applying a single delta.db onto the live data files
# ---------------------------------------------------------------------------

def _get_performer_id(conn: sqlite3.Connection, endpoint: str, stashbox_id: str) -> Optional[int]:
    row = conn.execute(
        """
        SELECT s.performer_id FROM stashbox_ids s
        JOIN performers p ON p.id = s.performer_id
        WHERE s.endpoint = ? AND s.stashbox_performer_id = ?
        """,
        (endpoint, stashbox_id),
    ).fetchone()
    return row[0] if row else None


def _upsert_performer(conn: sqlite3.Connection, p: sqlite3.Row) -> int:
    endpoint, stashbox_id = p["endpoint"], p["stashbox_id"]
    performer_id = _get_performer_id(conn, endpoint, stashbox_id)
    images = json.loads(p["images_json"] or "[]")
    fields = dict(
        canonical_name=p["name"], disambiguation=p["disambiguation"], gender=p["gender"],
        country=p["country"], ethnicity=p["ethnicity"], birth_date=p["birth_date"],
        death_date=p["death_date"], height_cm=p["height"], eye_color=p["eye_color"],
        hair_color=p["hair_color"], career_start_year=p["career_start_year"],
        career_end_year=p["career_end_year"],
        image_url=images[0]["url"] if images else None,
    )

    if performer_id is None:
        cursor = conn.execute(
            """
            INSERT INTO performers (
                canonical_name, disambiguation, gender, country, ethnicity, birth_date,
                death_date, height_cm, eye_color, hair_color, career_start_year,
                career_end_year, image_url, face_count, updated_at, stashdb_updated_at
            ) VALUES (
                :canonical_name, :disambiguation, :gender, :country, :ethnicity, :birth_date,
                :death_date, :height_cm, :eye_color, :hair_color, :career_start_year,
                :career_end_year, :image_url, 0, datetime('now'), :stashdb_updated_at
            )
            """,
            {**fields, "stashdb_updated_at": p["updated"] if endpoint == "stashdb" else None},
        )
        performer_id = cursor.lastrowid
    else:
        set_clause = ", ".join(f"{k} = :{k}" for k in fields)
        params = {**fields, "performer_id": performer_id}
        if endpoint == "stashdb":
            set_clause += ", stashdb_updated_at = :stashdb_updated_at"
            params["stashdb_updated_at"] = p["updated"]
        conn.execute(
            f"UPDATE performers SET {set_clause}, updated_at = datetime('now') WHERE id = :performer_id",
            params,
        )

    conn.execute(
        """
        INSERT INTO stashbox_ids (performer_id, endpoint, stashbox_performer_id)
        VALUES (?, ?, ?)
        ON CONFLICT(endpoint, stashbox_performer_id) DO UPDATE SET performer_id = excluded.performer_id
        """,
        (performer_id, endpoint, stashbox_id),
    )

    conn.execute("DELETE FROM aliases WHERE performer_id = ? AND source_endpoint = ?", (performer_id, endpoint))
    aliases = json.loads(p["aliases_json"] or "[]")
    conn.executemany(
        "INSERT OR IGNORE INTO aliases (performer_id, alias, source_endpoint) VALUES (?, ?, ?)",
        [(performer_id, alias, endpoint) for alias in aliases],
    )
    return performer_id


def _remove_performer(conn: sqlite3.Connection, p: sqlite3.Row) -> Optional[int]:
    endpoint, stashbox_id = p["endpoint"], p["stashbox_id"]
    performer_id = _get_performer_id(conn, endpoint, stashbox_id)
    if performer_id is None:
        return None

    if p["merged_into_id"]:
        logger.info("Performer %s:%s removed (merged into %s)", endpoint, stashbox_id, p["merged_into_id"])

    conn.execute("DELETE FROM stashbox_ids WHERE endpoint = ? AND stashbox_performer_id = ?", (endpoint, stashbox_id))
    remaining = conn.execute(
        "SELECT COUNT(*) FROM stashbox_ids WHERE performer_id = ?", (performer_id,)
    ).fetchone()[0]
    if remaining == 0:
        conn.execute("DELETE FROM aliases WHERE performer_id = ?", (performer_id,))
        conn.execute("DELETE FROM performers WHERE id = ?", (performer_id,))
        return None  # fully gone, nothing left to face-count-sync
    return performer_id


def _sync_face_count(conn: sqlite3.Connection, performer_id: int) -> None:
    conn.execute(
        "UPDATE performers SET face_count = (SELECT COUNT(*) FROM faces WHERE performer_id = ?) WHERE id = ?",
        (performer_id, performer_id),
    )


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    """Whether `name` exists in `conn`'s schema -- used to check the
    `catalogue_*` tables before querying them, since a delta.db built
    before stash-sense2-data-gen added catalogue-performer support won't
    have them. An older, unpatched build of this file querying a *newer*
    delta.db is the mirror case, and needs no code here at all: it simply
    never runs the new SELECTs below, so the new tables are silently
    ignored rather than erroring -- exactly why they were added as
    strictly additive tables server-side (see stash-sense2-data-gen's
    build/export_delta.py) rather than folded into the existing
    performers/faces shape."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
    ).fetchone() is not None


def _upsert_catalogue_performer(conn: sqlite3.Connection, p: sqlite3.Row) -> None:
    """Unlike _upsert_performer (stashbox path), this INSERTs with the
    server-assigned `id` explicitly rather than letting SQLite
    autoincrement pick one. That's required, not just tidy: a catalogue
    performer's universal_id is `f"{source_endpoint}:{performer_id}"`
    (see export_db_to_json.py's make_catalogue_id) -- it has no
    stashbox_ids row to resolve identity through the way stashbox
    performers do, so the raw id itself IS the portable identity, and
    only stays portable if this client never invents its own."""
    conn.execute(
        """
        INSERT INTO performers (id, canonical_name, gender, country, image_url, face_count, updated_at)
        VALUES (?, ?, ?, ?, ?, 0, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            canonical_name = excluded.canonical_name, gender = excluded.gender,
            country = excluded.country, image_url = excluded.image_url,
            updated_at = datetime('now')
        """,
        (p["id"], p["canonical_name"], p["gender"], p["country"], p["image_url"]),
    )


def _upsert_catalogue_performer_url(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    exists = conn.execute(
        "SELECT 1 FROM performer_urls WHERE performer_id = ? AND url = ?",
        (row["performer_id"], row["url"]),
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO performer_urls (performer_id, url, url_type, source_endpoint) VALUES (?, ?, ?, ?)",
            (row["performer_id"], row["url"], row["url_type"], row["source_endpoint"]),
        )


def apply_delta_db(delta_db_path: Path, data_dir: Path) -> dict[str, int]:
    """Apply one delta.db onto the live performers.db + usearch index in
    `data_dir`. Does not touch faces.json/performers.json/manifest.json —
    the caller regenerates those once after the whole chain is applied.
    """
    conn = sqlite3.connect(data_dir / "performers.db")
    conn.execute("PRAGMA foreign_keys = ON")

    index = Index(ndim=512, metric="cos")
    index_path = data_dir / "face_embeddings.usearch"
    if index_path.exists():
        index.load(str(index_path))

    delta_conn = sqlite3.connect(f"file:{delta_db_path}?mode=ro", uri=True)
    delta_conn.row_factory = sqlite3.Row

    touched_performers: set[int] = set()
    upserted = removed = faces_added = faces_removed = 0

    for p in delta_conn.execute("SELECT * FROM performers"):
        if p["action"] == "removed":
            pid = _remove_performer(conn, p)
            removed += 1
            if pid is not None:
                touched_performers.add(pid)
        else:
            pid = _upsert_performer(conn, p)
            upserted += 1
            touched_performers.add(pid)

    for f in delta_conn.execute("SELECT * FROM faces"):
        performer_id = _get_performer_id(conn, f["endpoint"], f["stashbox_id"])
        if performer_id is None:
            logger.warning("Delta face for unknown performer %s:%s — skipping", f["endpoint"], f["stashbox_id"])
            continue
        vec = np.frombuffer(f["embedding"], dtype=np.float32)
        index.add(f["embedding_index"], vec)
        conn.execute(
            """
            INSERT INTO faces (performer_id, embedding_index, image_url,
                                source_endpoint, quality_score, yaw)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (performer_id, f["embedding_index"], f["image_url"],
             f["endpoint"], f["quality_score"], f["yaw"]),
        )
        touched_performers.add(performer_id)
        faces_added += 1

    for r in delta_conn.execute("SELECT * FROM removed_faces"):
        idx = r["embedding_index"]
        row = conn.execute("SELECT performer_id FROM faces WHERE embedding_index = ?", (idx,)).fetchone()
        if idx in index:
            index.remove(idx)
        conn.execute("DELETE FROM faces WHERE embedding_index = ?", (idx,))
        if row is not None:
            touched_performers.add(row[0])
        faces_removed += 1

    # Catalogue-sourced (pornbox/seekfans/legacy-site) performers/faces --
    # see _has_table's docstring and stash-sense2-data-gen's
    # build/export_delta.py for why these are separate, additive tables
    # rather than folded into the loops above.
    catalogue_upserted = catalogue_faces_added = 0
    if _has_table(delta_conn, "catalogue_performers"):
        for p in delta_conn.execute("SELECT * FROM catalogue_performers"):
            _upsert_catalogue_performer(conn, p)
            touched_performers.add(p["id"])
            catalogue_upserted += 1

        for row in delta_conn.execute("SELECT * FROM catalogue_performer_urls"):
            _upsert_catalogue_performer_url(conn, row)

        for f in delta_conn.execute("SELECT * FROM catalogue_faces"):
            vec = np.frombuffer(f["embedding"], dtype=np.float32)
            index.add(f["embedding_index"], vec)
            conn.execute(
                """
                INSERT INTO faces (performer_id, embedding_index, image_url,
                                    source_endpoint, quality_score, yaw)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (f["performer_id"], f["embedding_index"], f["image_url"],
                 f["source_endpoint"], f["quality_score"], f["yaw"]),
            )
            touched_performers.add(f["performer_id"])
            catalogue_faces_added += 1

    for pid in touched_performers:
        _sync_face_count(conn, pid)

    index.save(str(index_path))

    conn.commit()
    conn.close()
    delta_conn.close()

    return {
        "performers_upserted": upserted, "performers_removed": removed,
        "faces_added": faces_added, "faces_removed": faces_removed,
        "catalogue_performers_upserted": catalogue_upserted,
        "catalogue_faces_added": catalogue_faces_added,
    }


# ---------------------------------------------------------------------------
# Chain orchestration — download, verify, apply, regenerate, backup/rollback
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=_CHUNK_SIZE):
                    f.write(chunk)


def _backup(data_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / f"backup_delta_{timestamp}"
    backup_dir.mkdir(parents=True)
    for fname in BACKED_UP_FILES:
        src = data_dir / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
    return backup_dir


def _rollback(data_dir: Path, backup_dir: Path) -> None:
    for fname in BACKED_UP_FILES:
        src = backup_dir / fname
        if src.exists():
            shutil.copy2(src, data_dir / fname)


async def apply_delta_chain(
    chain: list[dict[str, Any]], data_dir: Path, progress_cb: Optional[Callable[[int], None]] = None,
) -> dict[str, Any]:
    """Download, verify, and apply every hop in `chain` (oldest-to-newest),
    then regenerate faces.json/performers.json and write a new manifest.
    Rolls back to the pre-chain state on any failure — nothing is left
    half-applied.
    """
    if not chain:
        return {"applied_hops": 0}

    backup_dir = _backup(data_dir)
    work_dir = data_dir / f"delta_work_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    totals = {"performers_upserted": 0, "performers_removed": 0, "faces_added": 0, "faces_removed": 0,
              "catalogue_performers_upserted": 0, "catalogue_faces_added": 0}

    try:
        for i, hop in enumerate(chain):
            zip_path = work_dir / f"delta_{i}.zip"
            extract_dir = work_dir / f"extracted_{i}"
            await _download(hop["download_url"], zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            hop_manifest = json.loads((extract_dir / "delta_manifest.json").read_text())
            expected = hop_manifest["checksums"]["delta.db"].split(":", 1)[1]
            actual = _sha256(extract_dir / "delta.db")
            if actual != expected:
                raise ValueError(f"Checksum mismatch for delta {hop['from_version']}->{hop['to_version']}: "
                                  f"expected {expected[:16]}…, got {actual[:16]}…")

            hop_result = apply_delta_db(extract_dir / "delta.db", data_dir)
            for k in totals:
                totals[k] += hop_result[k]

            if progress_cb:
                progress_cb(int(100 * (i + 1) / len(chain)))

        conn = sqlite3.connect(data_dir / "performers.db")
        try:
            face_count = export_faces_json(conn, data_dir / "faces.json")
            performer_count = export_performers_json(conn, data_dir / "performers.json")
        finally:
            conn.close()

        old_manifest = json.loads((backup_dir / "manifest.json").read_text())
        new_manifest = {
            **old_manifest,
            "version": chain[-1]["to_version"],
            "performer_count": performer_count,
            "face_count": face_count,
            "checksums": {
                fname: f"sha256:{_sha256(data_dir / fname)}"
                for fname in ("performers.db", "face_embeddings.usearch", "faces.json", "performers.json")
            },
        }
        (data_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2))

        return {"applied_hops": len(chain), **totals, "new_version": chain[-1]["to_version"]}

    except Exception:
        logger.warning("Delta chain application failed — rolling back to pre-chain state", exc_info=True)
        _rollback(data_dir, backup_dir)
        raise

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

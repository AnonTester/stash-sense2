"""Local performer face index -- built from this Stash instance's own
performer cover images, kept alongside the main StashDB-derived index.

Unlike the main index (many faces per performer, ids allocated sequentially
and tracked separately in performers.db), each local performer contributes
at most one vector, keyed directly by their Stash performer ID -- no
separate id-allocation bookkeeping needed. Mirrors the usearch usage
pattern in stash-sense2-data-gen's build/usearch_index.py (same library,
same cosine/512-dim setup), just invoked in-process instead of offline.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx
import numpy as np
from usearch.index import Index

logger = logging.getLogger(__name__)

DIMENSIONS = 512
STASHDB_ENDPOINT = "https://stashdb.org/graphql"

# Performer fields copied into the index entry besides the name/urls/ids; kept as a list so the sync job and the
# single-performer hook stay in step. `aliases` is Stash's alias_list.
DETAIL_FIELDS = ("disambiguation", "gender", "country", "birthdate", "favorite")


def endpoint_domain(endpoint: Optional[str]) -> Optional[str]:
    """Domain of a stash-box endpoint URL ("https://javstash.org/graphql" -> "javstash.org") -- the same prefix a
    universal_id carries ("javstash.org:<uuid>")."""
    if not endpoint:
        return None
    host = urlsplit(endpoint if "//" in endpoint else f"//{endpoint}").hostname
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def extract_stash_ids(performer: dict) -> dict[str, str]:
    """{endpoint domain: stash_id} for EVERY stash-box a Stash performer is linked to (first id per endpoint)."""
    out: dict[str, str] = {}
    for sid in performer.get("stash_ids") or []:
        domain = endpoint_domain(sid.get("endpoint"))
        if domain and sid.get("stash_id") and domain not in out:
            out[domain] = sid["stash_id"]
    return out


def performer_metadata(performer: dict) -> dict:
    """Everything the index keeps about a performer except its embedding/cover: what a metadata-only refresh writes
    and what an embed writes alongside the vector."""
    stash_ids = extract_stash_ids(performer)
    meta = {
        "name": performer.get("name"),
        "stashdb_id": stash_ids.get(endpoint_domain(STASHDB_ENDPOINT)),
        "stash_ids": stash_ids,
        "urls": performer.get("urls") or [],
        "aliases": performer.get("alias_list") or [],
    }
    for field in DETAIL_FIELDS:
        meta[field] = performer.get(field)
    return meta


def linked_stash_uids(entry: dict) -> list[str]:
    """Every "<domain>:<stash_id>" universal_id a local performer is linked to, StashDB first. Entries written before
    all stash-boxes were stored only carry `stashdb_id`."""
    ids = dict(entry.get("stash_ids") or {})
    if entry.get("stashdb_id"):
        ids.setdefault(endpoint_domain(STASHDB_ENDPOINT), entry["stashdb_id"])
    stashdb_domain = endpoint_domain(STASHDB_ENDPOINT)
    ordered = sorted(ids, key=lambda d: (d != stashdb_domain, d))
    return [f"{d}:{ids[d]}" for d in ordered]


class LocalPerformerIndex:
    """Wraps a usearch index for local Stash performers, plus a JSON
    sidecar mapping performer_id -> metadata.
    """

    def __init__(self, index_path: Path, mapping_path: Path):
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.index = self._load_or_create(self.index_path)
        self.mapping: dict[str, dict] = self._load_mapping()

    @staticmethod
    def _load_or_create(path: Path) -> Index:
        index = Index(ndim=DIMENSIONS, metric="cos", connectivity=16)
        if path.exists():
            index.load(str(path))
        return index

    def _load_mapping(self) -> dict:
        if self.mapping_path.exists():
            with open(self.mapping_path) as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(self.index_path))
        # keys starting with "_" are runtime annotations (recognizer.py resolves each entry's dataset identity once per
        # load) -- derived from the main database, so never persisted
        persisted = {pid: {k: v for k, v in entry.items() if not k.startswith("_")} for pid, entry in self.mapping.items()}
        with open(self.mapping_path, "w") as f:
            json.dump(persisted, f)

    def upsert(
        self, performer_id: int, name: str, stashdb_id: Optional[str],
        image_hash: str, image_url: Optional[str], embedding: np.ndarray,
        urls: Optional[list[str]] = None, bbox: Optional[dict] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Add or replace this performer's embedding (delete-then-add for
        clarity/consistency with the main build pipeline's convention,
        though usearch's own `add()` would also happily overwrite a key
        in place).

        `urls` -- this performer's own Stash `urls` field (plain strings,
        unlike a stash-box performer's `{url, type}` shape) -- lets
        matching.py's merge_local_candidates() cross-check a catalogue
        (seekfans/pornbox) main-index candidate's profile/catalogue url
        against this local performer, catching a same-person duplicate
        that a stash_id-only comparison can't (catalogue sources have no
        stash_id to link).

        `bbox` -- the detected face's own region within the cover image
        (the same {x, y, w, h, rotation_applied} shape DetectedFace.bbox
        always carries), added 2026-10-01. Unlike the main pipeline's own
        faces, this was never persisted before -- only the embedding
        vector and the whole cover image survived sync_one_performer(),
        so a local match's own crop slot had no tight face region to show,
        only the full (often non-square, often multi-subject) cover photo.
        Optional and may be None for a performer indexed before this
        existed, or if a future caller genuinely has no bbox to give --
        consumers (the new local-performer-crop route) fall back to the
        whole cover image in that case, same as before this existed.

        `details` -- the rest of performer_metadata()'s output (stash_ids for every stash-box, aliases,
        disambiguation, gender, ...); name/stashdb_id/urls given explicitly above win over the same keys in it."""
        self.remove(performer_id)
        self.index.add(performer_id, embedding.astype(np.float32))
        entry = {k: v for k, v in (details or {}).items() if k not in ("name", "stashdb_id", "urls")}
        entry.update({
            "name": name, "stashdb_id": stashdb_id,
            "image_hash": image_hash, "image_url": image_url,
            "urls": urls or [], "bbox": bbox,
        })
        self.mapping[str(performer_id)] = entry

    def remove(self, performer_id: int) -> None:
        if performer_id in self.index:
            self.index.remove(performer_id)
        self.mapping.pop(str(performer_id), None)

    def get_image_hash(self, performer_id: int) -> Optional[str]:
        entry = self.mapping.get(str(performer_id))
        return entry["image_hash"] if entry else None

    def get_bbox(self, performer_id: int) -> Optional[dict]:
        entry = self.mapping.get(str(performer_id))
        return entry.get("bbox") if entry else None

    def get_image_url(self, performer_id: int) -> Optional[str]:
        entry = self.mapping.get(str(performer_id))
        return entry.get("image_url") if entry else None

    def update_metadata(self, performer_id: int, meta: dict) -> bool:
        """Refreshes every stored detail of an already-indexed performer (name, all stash-box ids, urls, aliases,
        ...) without touching the embedding -- used when the cover is unchanged, so a rename or a newly added
        stash-box link in Stash reaches the index. Only keys present in `meta` are written; returns whether
        anything actually changed."""
        entry = self.mapping.get(str(performer_id))
        if entry is None:
            return False
        changed = False
        for key, value in meta.items():
            if key.startswith("_") or key in ("image_hash", "image_url", "bbox"):
                continue
            if entry.get(key) != value:
                entry[key] = value
                changed = True
        return changed

    def __contains__(self, performer_id: int) -> bool:
        return str(performer_id) in self.mapping

    def __len__(self) -> int:
        return len(self.mapping)


def _image_fingerprint(data: bytes) -> str:
    """Short, stable fingerprint of image content.

    Originally this hashed the image_path URL instead of its bytes, on the
    assumption that Stash's cache-busting query param only changes when the
    underlying image is replaced -- cheap, no download needed. Empirically
    that's wrong: the param tracks the performer's updated_at, which
    changes on *any* field edit (e.g. toggling favorite), not just an
    image swap. That made the hook path re-embed on every unrelated
    metadata edit. Hashing the actual bytes (still cheap: sha256 of an
    already-fetched avatar-sized image) is the only signal that actually
    tracks image content, matching the plan's original warning against
    using updated_at as a proxy for "did the image change"."""
    return hashlib.sha256(data).hexdigest()[:16]


def _relative_image_url(image_path: str) -> str:
    """Strip scheme+host from Stash's image_path, keeping only path+query.

    image_path comes back absolute (e.g. "http://192.168.1.100:9997/performer/5/image?t=...")
    because that's the host the sidecar itself uses to reach Stash. The
    browser rendering match results may be reaching this same Stash
    instance through a completely different address (a domain name, a
    reverse proxy, a VPN/tailnet IP) -- shipping the sidecar's own
    internal address to the browser would load (or fail to load) the
    image from the wrong place. A root-relative URL resolves against
    whatever origin the browser is actually using, same as how Stash's
    own web UI references its images."""
    parsed = urlsplit(image_path)
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


async def sync_one_performer(
    stash, generator, index: "LocalPerformerIndex", performer_id: int, event_type: str,
) -> str:
    """Sync a single performer into the local index. Shared by the full
    sync job and the fast single-performer endpoint (used by the Stash
    hook handler), so both stay behind one code path.

    event_type is one of "create"/"update"/"destroy" (case-insensitive,
    matches the Stash hook operation names). Returns a short status for
    logging/response: "removed", "added", "updated", "metadata_updated"
    (cover unchanged, but name/ids/urls/aliases/... differed and were
    refreshed in place -- see update_metadata()), "skipped_no_image", or
    "unchanged".
    """
    if event_type.lower() == "destroy":
        was_present = performer_id in index
        index.remove(performer_id)
        return "removed" if was_present else "unchanged"

    performer = await stash.get_performer(str(performer_id))
    image_path = performer.get("image_path") if performer else None
    # Stash marks its own placeholder avatar with a "default=true" query
    # param on the URL -- that image is an SVG/icon, not a decodable
    # photo, so skip the fetch entirely rather than let it fail as a
    # decode error further down.
    has_custom_image = bool(image_path) and "default=true" not in image_path
    if not performer or not has_custom_image:
        was_present = performer_id in index
        index.remove(performer_id)
        return "removed" if was_present else "skipped_no_image"

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(image_path, headers={"ApiKey": stash.api_key})
        resp.raise_for_status()
        image_bytes = resp.content

    fingerprint = _image_fingerprint(image_bytes)
    meta = performer_metadata(performer)
    if index.get_image_hash(performer_id) == fingerprint:
        if index.update_metadata(performer_id, meta):
            return "metadata_updated"
        return "unchanged"

    from embeddings import detect_local_performer_faces, load_image, gpu_compute_lock, select_local_performer_face
    image = load_image(image_bytes)
    # See embeddings.py's gpu_compute_lock() docstring -- this runs as a
    # real coroutine on the event loop (the Performer hook handler calls it
    # directly), so it needs the async wrapper, not the raw
    # threading.Lock local_performer_sync_job.py's _embed_worker uses from
    # its own plain OS thread.
    async with gpu_compute_lock():
        faces = detect_local_performer_faces(generator, image, min_confidence=0.5)

    if not faces:
        # No detectable face -- either a default placeholder (no custom
        # image set yet) or a genuinely faceless cover. Drop any stale
        # embedding rather than leave it wrong; picked up automatically
        # once a real image is set.
        was_present = performer_id in index
        index.remove(performer_id)
        return "removed" if was_present else "skipped_no_image"

    best_face = select_local_performer_face(faces, generator)
    embedding = generator.get_embedding(best_face)

    was_present = performer_id in index
    index.upsert(
        performer_id=performer_id,
        name=meta["name"],
        stashdb_id=meta["stashdb_id"],
        image_hash=fingerprint,
        image_url=_relative_image_url(image_path),
        embedding=embedding.embedding,
        urls=meta["urls"],
        bbox=best_face.bbox,
        details=meta,
    )
    return "updated" if was_present else "added"

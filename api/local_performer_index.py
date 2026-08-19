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
        with open(self.mapping_path, "w") as f:
            json.dump(self.mapping, f)

    def upsert(
        self, performer_id: int, name: str, stashdb_id: Optional[str],
        image_hash: str, image_url: Optional[str], embedding: np.ndarray,
    ) -> None:
        """Add or replace this performer's embedding (delete-then-add for
        clarity/consistency with the main build pipeline's convention,
        though usearch's own `add()` would also happily overwrite a key
        in place)."""
        self.remove(performer_id)
        self.index.add(performer_id, embedding.astype(np.float32))
        self.mapping[str(performer_id)] = {
            "name": name, "stashdb_id": stashdb_id,
            "image_hash": image_hash, "image_url": image_url,
        }

    def remove(self, performer_id: int) -> None:
        if performer_id in self.index:
            self.index.remove(performer_id)
        self.mapping.pop(str(performer_id), None)

    def get_image_hash(self, performer_id: int) -> Optional[str]:
        entry = self.mapping.get(str(performer_id))
        return entry["image_hash"] if entry else None

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
    logging/response: "removed", "added", "updated", "skipped_no_image",
    or "unchanged".
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
    if index.get_image_hash(performer_id) == fingerprint:
        return "unchanged"

    from embeddings import load_image
    image = load_image(image_bytes)
    faces = generator.detect_faces(image, min_confidence=0.5)

    if not faces:
        # No detectable face -- either a default placeholder (no custom
        # image set yet) or a genuinely faceless cover. Drop any stale
        # embedding rather than leave it wrong; picked up automatically
        # once a real image is set.
        was_present = performer_id in index
        index.remove(performer_id)
        return "removed" if was_present else "skipped_no_image"

    best_face = max(faces, key=lambda f: f.bbox["w"] * f.bbox["h"])
    embedding = generator.get_embedding(best_face)

    stashdb_id = next(
        (sid["stash_id"] for sid in performer.get("stash_ids", [])
         if sid.get("endpoint") == STASHDB_ENDPOINT),
        None,
    )
    was_present = performer_id in index
    index.upsert(
        performer_id=performer_id,
        name=performer["name"],
        stashdb_id=stashdb_id,
        image_hash=fingerprint,
        image_url=_relative_image_url(image_path),
        embedding=embedding.embedding,
    )
    return "updated" if was_present else "added"

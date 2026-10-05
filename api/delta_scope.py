"""Delta-scoped Refresh Outdated / Face Recommendations.

Decides which already-fingerprinted, db_version-outdated scenes a bulk
rematch pass actually needs to touch, instead of rematching every single
one of them just because the main database's version moved -- which is
what both fingerprint_generator.py's Refresh Outdated and
analyzers/scene_face_match.py's full scan used to do unconditionally, even
though a typical delta only ever touches a small fraction of the whole
performer/face corpus.

See recommendations_db.py's DELTA_SCOPE_SCHEMA for the tables this reads,
and delta_applier.apply_delta_chain's _record_dirty_state for how they get
populated as a side effect of applying a delta.

Two tiers of provably-safe invalidation:

- Tier 2 (identity): a scene's OWN already-stored scene_fingerprint_matches
  rows reference a universal_id the delta touched (a face changed/removed
  for that identity, or the identity itself was removed) -- a cheap SQL
  join (get_scene_ids_with_dirty_matches), no vector math.

- Tier 3 (vector): a delta added/changed a face vector that comes within
  `max_distance` of one of the scene's own already-cached face embeddings,
  clustered into per-person centroids exactly the way the real hybrid
  matcher does (scene_matcher.cluster_faces_by_person) -- meaning a
  genuinely NEW candidate could now surface for a scene that didn't
  reference any touched performer at all before. Matching filters every
  candidate at `combined_distance <= max_distance` everywhere else in this
  codebase (see matching.py), so "nothing comes within max_distance (plus
  MATCH_MARGIN's own slack)" is a sound bound, not a heuristic, for "this
  delta cannot change what a real hybrid match would return for this
  scene" -- see MATCH_MARGIN below for the one deliberate extra slack this
  still adds.

A scene is only safe to bump (db_version updated in place, no rematch) if
it clears BOTH tiers AND its own stored db_version is not older than dirty
tracking's own coverage floor (get_dirty_tracking_marker) -- anything else
falls back to a real rematch, exactly reproducing the pre-this-feature
behavior. This is what makes the whole thing safe to ship: every fallback
direction is "do more work than strictly necessary," never "skip work
that was actually needed."
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING

import numpy as np

import face_config

if TYPE_CHECKING:
    from recommendations_db import RecommendationsDB

logger = logging.getLogger(__name__)

# Deliberate slack above the real cutoff used everywhere else in matching
# (face_config.MAX_DISTANCE). Defense in depth against this module's own
# clustering call landing on very slightly different centroids than the
# live matcher's own run of the SAME function on the SAME cached input
# would (floating-point/ordering differences, not a different algorithm).
# Costs nothing -- Tier 3 is already the cheap side of this -- and only
# ever makes this module MORE likely to (correctly, if unnecessarily)
# flag a scene for a real rematch, never less.
MATCH_MARGIN = 0.02

# Must match scene_matcher.cluster_faces_by_person's own default --
# this module's centroids need to correspond to what a real rematch would
# actually cluster and compare, not some other grouping.
CLUSTER_DISTANCE_THRESHOLD = 0.6


# Tier 3 compares every scene person's centroid against every changed vector (44k for a catalogue-heavy delta).
# Done one scene at a time that re-reads the whole vector matrix per scene (memory-bound, ~600 GB of traffic for 6,600
# scenes) and lets BLAS fan out over every core -- measured: the system's fans spun up and the sidecar was
# unresponsive for the first 10-15 s of a Refresh Outdated run. So: centroids are gathered across scenes and compared in
# batches (each vector block read once per batch), and BLAS is held to a few threads.
TIER3_BATCH_ROWS = 512        # centroids per matrix product
TIER3_VECTOR_BLOCK = 8192     # changed vectors per block (bounds the temporary similarity matrix)
BLAS_THREADS = 4


def _blas_limit():
    """Cap BLAS threads for the duration of the comparison (no-op if threadpoolctl is unavailable)."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=BLAS_THREADS, user_api="blas")
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _lightweight_results_from_cache(rows: list[dict]) -> list:
    """Build minimal (frame_index, RecognitionResult) tuples straight from
    scene_face_embeddings rows -- deliberately NOT
    identification_router._reconstruct_from_cached_faces, which also calls
    FaceRecognizer.recognize_face_v2() per row: a REAL match against the
    full main+local usearch index. Reusing that helper here would silently
    reintroduce, for every outdated scene, the exact per-scene full-index
    query cost this whole module exists to let most scenes skip. This only
    needs embeddings clustered by person -- cluster_faces_by_person never
    reads `.matches`, only `.embedding`."""
    from embeddings import DetectedFace, FaceEmbedding
    from recognizer import RecognitionResult

    results = []
    for row in rows:
        bbox = json.loads(row["bbox_json"]) if row.get("bbox_json") else None
        face = DetectedFace(image=None, bbox=bbox, confidence=row["confidence"], embedding=None, yaw=row.get("yaw"))
        embedding = FaceEmbedding(embedding=np.frombuffer(row["embedding"], dtype=np.float32))
        results.append((row["frame_index"], RecognitionResult(face=face, matches=[], embedding=embedding)))
    return results


def _scene_person_centroids(rec_db: "RecommendationsDB", stash_scene_id: int) -> list[np.ndarray]:
    """One centroid vector per distinct person detected in a scene
    (video-frame + sprite-tile faces together), clustered exactly like a
    real hybrid match would -- so Tier 3's vector check compares against
    the SAME units the live matcher actually scores, not raw per-frame
    embeddings (which would be an unsound approximation of what a
    real rematch could find -- see module docstring)."""
    from scene_matcher import cluster_faces_by_person

    rows = rec_db.get_face_embeddings(stash_scene_id, is_sprite=None)
    if not rows:
        return []
    all_results = _lightweight_results_from_cache(rows)
    # recognizer=None is safe here: cluster_faces_by_person only falls
    # back to it when a result has no precomputed embedding, which never
    # happens for these cache-reconstructed rows (every one carries its
    # stored embedding already).
    clusters = cluster_faces_by_person(all_results, recognizer=None, distance_threshold=CLUSTER_DISTANCE_THRESHOLD)
    return [np.mean([r.embedding.embedding for _, r in cluster], axis=0) for cluster in clusters]


def scenes_needing_rematch(rec_db: "RecommendationsDB", candidates: list[dict]) -> tuple[set[int], set[int]]:
    """Splits `candidates` -- scene_fingerprints rows (dicts with at least
    stash_scene_id, db_version, total_faces) for scenes that are already
    known to be fingerprint_status='complete' and db_version-outdated --
    into (must_rematch, safe_to_bump). Every candidate's stash_scene_id
    lands in exactly one of the two returned sets.

    Callers: fingerprint_generator.generate_all() (Refresh Outdated) and
    analyzers/scene_face_match.py's full-scan branch (Face
    Recommendations). Neither this function nor its callers ever act on
    'metadata_only'-reason dirty rows -- those are consumed separately
    (see delta_applier._record_dirty_state's own comment on why a pure
    metadata edit needs a cheap denormalized-field patch, not a rematch).
    """
    marker = rec_db.get_dirty_tracking_marker()
    if not isinstance(marker, str):
        # Anything other than a real version string -- None is the normal
        # cold-start case (dirty tracking has never run under this feature
        # at all yet, see module docstring) -- can't be compared against a
        # scene's own db_version, so nothing here can be trusted. Exactly
        # reproduces the pre-this-feature always-rematch behavior until the
        # first delta lands under the new capture code -- after that one
        # pass, every scene's db_version is current (>= whatever marker
        # gets set), so this branch is naturally never hit again except on
        # a fresh/never-updated install.
        return {c["stash_scene_id"] for c in candidates}, set()

    eligible_ids = {c["stash_scene_id"] for c in candidates if (c.get("db_version") or "") >= marker}
    eligible = [c for c in candidates if c["stash_scene_id"] in eligible_ids]
    must_rematch: set[int] = {c["stash_scene_id"] for c in candidates} - eligible_ids
    if not eligible:
        return must_rematch, set()

    # ---- Tier 2: identity -----------------------------------------------
    dirty_uids = rec_db.get_dirty_universal_ids(reasons=["needs_rematch"])
    tier2_hit_ids = rec_db.get_scene_ids_with_dirty_matches(dirty_uids) if dirty_uids else set()
    must_rematch |= {c["stash_scene_id"] for c in eligible if c["stash_scene_id"] in tier2_hit_ids}

    # ---- Tier 3: vector ---------------------------------------------------
    remaining = [c for c in eligible if c["stash_scene_id"] not in tier2_hit_ids]
    vector_rows = rec_db.get_dirty_face_vectors() if remaining else []
    if vector_rows:
        V = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in vector_rows]).astype(np.float32)
        V = np.ascontiguousarray(V / np.linalg.norm(V, axis=1, keepdims=True))
        threshold = face_config.MAX_DISTANCE + MATCH_MARGIN

        row_vectors: list[np.ndarray] = []
        row_scene_ids: list[int] = []

        def _flush_rows() -> None:
            if not row_vectors:
                return
            C = np.stack(row_vectors).astype(np.float32)
            C = C / np.linalg.norm(C, axis=1, keepdims=True)
            best = np.full(len(C), -np.inf, dtype=np.float32)   # highest similarity to any changed vector
            for start in range(0, len(V), TIER3_VECTOR_BLOCK):
                best = np.maximum(best, (C @ V[start:start + TIER3_VECTOR_BLOCK].T).max(axis=1))
            # same test as before: 1 - similarity <= threshold for the closest changed vector
            for scene_id, dist in zip(row_scene_ids, 1.0 - best):
                if dist <= threshold:
                    must_rematch.add(scene_id)
            row_vectors.clear()
            row_scene_ids.clear()

        with _blas_limit():
            for c in remaining:
                if not c.get("total_faces"):
                    # Nothing was ever detected in this scene at all -- no
                    # embedding exists for a new/changed vector to be close
                    # to, so no delta can possibly introduce a new candidate.
                    continue
                if c["stash_scene_id"] in must_rematch:
                    continue   # already decided; its centroids cannot change that
                for centroid in _scene_person_centroids(rec_db, c["stash_scene_id"]):
                    row_vectors.append(centroid)
                    row_scene_ids.append(c["stash_scene_id"])
                if len(row_vectors) >= TIER3_BATCH_ROWS:
                    _flush_rows()
            _flush_rows()

    safe_to_bump = {c["stash_scene_id"] for c in eligible} - must_rematch
    return must_rematch, safe_to_bump


def finalize_dirty_state(rec_db: "RecommendationsDB", db_version: str) -> None:
    """Drain dirty tracking up to `db_version` -- call only after a full
    Refresh Outdated pass completes with nothing skipped/errored (mirrors
    this codebase's existing "only advance a watermark on a clean sweep"
    convention, e.g. analyzers/scene_face_match.py's own watermark guard).
    A partial/interrupted run must NOT call this: the dirty rows it hasn't
    consumed yet need to survive for the next run to pick up."""
    rec_db.clear_dirty_state(db_version)

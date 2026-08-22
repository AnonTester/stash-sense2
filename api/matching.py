"""Face matching against the performer database.

buffalo_l migration: the legacy pipeline queried two separate models
(FaceNet512 + ArcFace) and needed adaptive health detection to handle
ArcFace's occasional degenerate output (near-identical distances across
unrelated performers) -- see git history for that logic if it's ever
needed again. buffalo_l produces one embedding per face, so there's only
one index to query and nothing to fuse or arbitrate between; this module
is correspondingly a plain single-index nearest-neighbor lookup +
threshold filter.
"""
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

from usearch.index import Index

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Tune these values in face_config.py
# =============================================================================

@dataclass
class MatchingConfig:
    """Configuration for face matching. All values are tunable."""

    # Query parameters
    query_k: int = 100  # Number of candidates to fetch from the index

    # Output filtering
    max_results: int = 10
    max_distance: float = 0.8  # Maximum distance to return


DEFAULT_CONFIG = MatchingConfig()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IndexQueryResult:
    """Result from querying the embedding index."""
    neighbors: np.ndarray  # Face indices
    distances: np.ndarray  # Distances to neighbors


@dataclass
class CandidateMatch:
    """A candidate match."""
    face_index: int
    universal_id: str
    name: str

    distance: float = 0.0
    combined_distance: float = 0.0  # kept as the field name every caller (recognizer.py,
                                     # local_performer_index dedup, plugin JS) already reads
    confidence: float = 0.0  # 0-1, higher is better
    rank: Optional[int] = None


@dataclass
class MatchingResult:
    """Complete result from the matching process."""
    matches: list[CandidateMatch]
    candidate_count: int = 0


# =============================================================================
# MATCHING LOGIC
# =============================================================================

def query_index(
    embedding: np.ndarray,
    index: Index,
    config: MatchingConfig = DEFAULT_CONFIG,
) -> IndexQueryResult:
    """Query the embedding index for nearest neighbors."""
    matches = index.search(embedding, config.query_k)
    return IndexQueryResult(neighbors=matches.keys, distances=matches.distances)


def build_matches(
    query_result: IndexQueryResult,
    faces_mapping: list[str],  # index -> universal_id
    performers: dict[str, dict],  # universal_id -> performer info
    config: MatchingConfig = DEFAULT_CONFIG,
) -> MatchingResult:
    """Build sorted, threshold-filtered CandidateMatch objects from one
    index query's results."""
    candidates: dict[int, CandidateMatch] = {}
    faces_count = len(faces_mapping)

    for rank, (idx, dist) in enumerate(zip(query_result.neighbors, query_result.distances)):
        idx = int(idx)
        # Skip if index is out of bounds (can happen with index/metadata mismatch)
        if idx < 0 or idx >= faces_count:
            continue
        uid = faces_mapping[idx]
        # Skip null entries (gaps from deleted faces or missing stashbox IDs)
        if uid is None:
            continue
        info = performers.get(uid, {})
        candidate = CandidateMatch(
            face_index=idx, universal_id=uid, name=info.get("name", "Unknown"),
            distance=float(dist), combined_distance=float(dist), rank=rank + 1,
        )
        candidate.confidence = max(0.0, min(1.0, 1.0 - candidate.combined_distance))
        # Defensive: keep only the closer entry if the ANN index ever
        # returns the same idx twice within one query's neighbor list
        # (shouldn't happen in practice, but cheap to guard).
        existing = candidates.get(idx)
        if existing is None or candidate.combined_distance < existing.combined_distance:
            candidates[idx] = candidate

    # The index can hold multiple embedding entries for the same performer
    # (e.g. several training crops), each landing as its own idx-keyed
    # candidate above -- left alone, the same person can surface twice (or
    # more) in the returned list at similar scores. Collapse to one
    # candidate per resolved identity, keeping the best-scoring entry,
    # before sorting/truncating to max_results.
    best_by_uid: dict[str, CandidateMatch] = {}
    for candidate in candidates.values():
        existing = best_by_uid.get(candidate.universal_id)
        if existing is None or candidate.combined_distance < existing.combined_distance:
            best_by_uid[candidate.universal_id] = candidate

    sorted_candidates = sorted(best_by_uid.values(), key=lambda c: c.combined_distance)
    filtered = [c for c in sorted_candidates if c.combined_distance <= config.max_distance]

    return MatchingResult(matches=filtered[:config.max_results], candidate_count=len(candidates))


# Score multiplier applied to local-index matches' combined_distance before
# they're merged into the main results (lower distance = better, so < 1.0
# is a boost). A face appearing in the user's own library is more likely to
# be a performer they've already added locally than a random main-DB
# entry. Deliberately a single tunable constant for now -- calibration is
# expected to happen later against real results, not during this build.
LOCAL_MATCH_BOOST = 0.85


def fuse_local_results(
    query_result: IndexQueryResult,
    local_performers_mapping: dict[str, dict],  # str(performer_id) -> {name, stashdb_id, image_url, ...}
    config: MatchingConfig = DEFAULT_CONFIG,
) -> list[CandidateMatch]:
    """Build CandidateMatch objects from a local-performer-index query.

    Against a completely separate id space (Stash performer id, not a
    shared face index with the main database) -- can't be merged into the
    same candidates dict build_matches() uses, so it's built separately
    and combined by merge_local_candidates() below. Every candidate gets
    tagged with a "local:" universal_id prefix (mirroring the existing
    "stashdb.org:" convention -- see stashbox_utils._extract_endpoint,
    which will naturally read this as endpoint "local") and scored down by
    LOCAL_MATCH_BOOST.
    """
    candidates: dict[str, CandidateMatch] = {}
    for rank, (pid, dist) in enumerate(zip(query_result.neighbors, query_result.distances)):
        pid_str = str(int(pid))
        info = local_performers_mapping.get(pid_str)
        if info is None:
            continue  # stale entry (deleted since the index was last saved)
        uid = f"local:{pid_str}"
        candidate = CandidateMatch(
            face_index=int(pid), universal_id=uid, name=info.get("name", "Unknown"),
            distance=float(dist), rank=rank + 1,
        )
        candidate.combined_distance = candidate.distance * LOCAL_MATCH_BOOST
        candidate.confidence = max(0.0, min(1.0, 1.0 - candidate.combined_distance))
        existing = candidates.get(uid)
        if existing is None or candidate.combined_distance < existing.combined_distance:
            candidates[uid] = candidate

    return list(candidates.values())


# Endpoint short-name matching universal_id's own "<endpoint_domain>:<uuid>"
# convention for StashDB entries (see stashbox_utils._extract_endpoint) --
# local_performer_index.py only ever links against StashDB itself (its own
# STASHDB_ENDPOINT is the full GraphQL URL, used solely to filter a
# performer's stash_ids down to the StashDB one), so this is the only
# endpoint a local candidate's tracked stashdb_id could ever correspond to.
_STASHDB_ENDPOINT_SHORT_NAME = "stashdb.org"


def merge_local_candidates(
    main_matches: list[CandidateMatch],
    local_candidates: list[CandidateMatch],
    local_performers_mapping: dict[str, dict],
) -> list[CandidateMatch]:
    """Merges local-index candidates into the main-index results,
    deduplicating a performer present in *both* databases instead of
    returning two separate, weaker entries for the same real person.

    local_performer_index.py already tracks each local performer's linked
    StashDB id -- sync_one_performer() reads it straight off the
    performer's own `stash_ids` in Stash -- but until now that linkage was
    only ever used for display (recognizer.py's own id-selection logic for
    the response), never to detect a duplicate against the main index's
    own candidates. A performer who is both locally added *and* already in
    the main database is a real, plausibly common case (anyone reasonably
    well-known would be in both) -- without this, they come back as two
    separate candidates for the same person: diluted evidence instead of
    reinforced, and confusing to present as two different "top matches".

    Only merges a genuine duplicate -- a local candidate whose linked
    stashdb_id matches a `universal_id` already present in `main_matches`
    *for this specific call*. A performer who exists in only one database
    is returned unchanged: never dropped, never penalized, and never
    summed/boosted into a double-counted score -- of the two duplicate
    entries, only the single better-scoring one survives.
    """
    main_by_universal_id = {c.universal_id: c for c in main_matches}
    merged: list[CandidateMatch] = list(main_matches)

    for local_candidate in local_candidates:
        local_id = local_candidate.universal_id.split(":", 1)[1]  # "local:<pid>" -> "<pid>"
        stashdb_id = (local_performers_mapping.get(local_id) or {}).get("stashdb_id")
        linked_universal_id = f"{_STASHDB_ENDPOINT_SHORT_NAME}:{stashdb_id}" if stashdb_id else None

        main_entry = main_by_universal_id.get(linked_universal_id) if linked_universal_id else None
        if main_entry is not None:
            if local_candidate.combined_distance < main_entry.combined_distance:
                merged[merged.index(main_entry)] = local_candidate
            # else: main_entry already scored better -- keep it, drop the local duplicate silently
        else:
            merged.append(local_candidate)

    return merged


def match_face(
    embedding: np.ndarray,
    index: Index,
    faces_mapping: list[str],
    performers: dict[str, dict],
    config: MatchingConfig = DEFAULT_CONFIG,
    local_index: Optional[Index] = None,
    local_performers_mapping: Optional[dict[str, dict]] = None,
) -> MatchingResult:
    """
    Match a face against the database.

    This is the main entry point for face matching.

    Args:
        embedding: buffalo_l embedding vector (512-dim)
        index: usearch index for the main database
        faces_mapping: List mapping face index to universal_id
        performers: Dict mapping universal_id to performer info
        config: Matching configuration
        local_index: Optional secondary index built from this Stash
            instance's own performer cover images (local_performer_index.py)
        local_performers_mapping: str(performer_id) -> {name, stashdb_id,
            image_url, ...} for the local index. Required alongside
            local_index above for local matching to run.

    Returns:
        MatchingResult with candidates
    """
    query_result = query_index(embedding, index, config)
    result = build_matches(query_result, faces_mapping, performers, config)

    # Optionally merge in local-performer-index matches (see fuse_local_results).
    # A handful of local performers is common (especially right after the
    # first sync), so index.search()'s k can exceed the index size -- guard
    # with try/except rather than requiring every caller to pre-check size.
    if local_index is not None and local_performers_mapping and len(local_index) > 0:
        try:
            local_query_result = query_index(embedding, local_index, config)
            local_candidates = fuse_local_results(local_query_result, local_performers_mapping, config)
            merged = merge_local_candidates(result.matches, local_candidates, local_performers_mapping)
            merged.sort(key=lambda c: c.combined_distance)
            result.matches = [c for c in merged if c.combined_distance <= config.max_distance][:config.max_results]
        except Exception as e:
            logger.warning(f"Local performer index query failed, skipping local matches: {e}")

    return result


# =============================================================================
# DEBUGGING UTILITIES
# =============================================================================

def format_matching_result(result: MatchingResult, expected_names: list[str] = None) -> str:
    """Format a matching result for debugging output."""
    lines = []

    lines.append(f"{result.candidate_count} candidates")
    lines.append("")

    expected_names = expected_names or []

    for i, match in enumerate(result.matches[:10]):
        marker = "★" if match.name in expected_names else " "
        lines.append(f"{marker} {i+1}. {match.name[:30]:<30} distance={match.combined_distance:.3f}@{match.rank}")

    return "\n".join(lines)

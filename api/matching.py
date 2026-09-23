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

from stashbox_utils import classify_universal_id, normalize_url_for_compare

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

    # Steep-angle soft penalty. buffalo_l's per-face yaw estimate (see
    # stash-sense2-data-gen's embed/embeddings.py) is a rough heuristic,
    # not a calibrated pose estimator, and a face crop shot from a steep
    # angle carries less reliable identity information than a frontal
    # one -- confirmed live: a performer whose only reference images were
    # both >30 degrees off-frontal kept surfacing as a false-positive
    # match across unrelated queries. Deliberately a soft multiplier,
    # not a hard exclusion -- yaw is a rough estimate, and a genuinely
    # steep-angle match can still be correct. Scales linearly from 1.0
    # (no penalty) at yaw_penalty_threshold up to yaw_penalty_at_90 at a
    # full 90-degree profile. Applied per-candidate-face (the specific
    # matched vector's own yaw), not per-performer -- a performer can
    # have both great frontal shots and steep-angle ones, so this can't
    # be collapsed to one number per performer.
    yaw_penalty_threshold: float = 45.0
    yaw_penalty_at_90: float = 0.5


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


def _apply_yaw_penalty(candidate: CandidateMatch, face_yaw: list, config: MatchingConfig) -> None:
    """Multiplies `candidate.confidence` down (never excludes) when the
    specific reference face that matched was shot at a steep angle --
    looked up by its own usearch vector id (`candidate.face_index`), not
    aggregated across the performer. See MatchingConfig's own comment for
    the rationale and the linear-scaling formula.

    Also pushes `combined_distance` back out to match -- every
    downstream ranking/filtering step reads `combined_distance`, not
    `confidence`."""
    if not face_yaw or candidate.face_index >= len(face_yaw):
        return
    yaw = face_yaw[candidate.face_index]
    if yaw is None:
        return
    abs_yaw = min(abs(yaw), 90.0)
    if abs_yaw <= config.yaw_penalty_threshold:
        return
    span = 90.0 - config.yaw_penalty_threshold
    frac = (abs_yaw - config.yaw_penalty_threshold) / span if span > 0 else 1.0
    multiplier = 1.0 - frac * (1.0 - config.yaw_penalty_at_90)
    candidate.confidence *= multiplier
    candidate.combined_distance = 1.0 - candidate.confidence


def build_matches(
    query_result: IndexQueryResult,
    faces_mapping: list[str],  # index -> universal_id
    performers: dict[str, dict],  # universal_id -> performer info
    config: MatchingConfig = DEFAULT_CONFIG,
    face_yaw: Optional[list] = None,  # index -> yaw degrees, or None
) -> MatchingResult:
    """Build sorted, threshold-filtered CandidateMatch objects from one
    index query's results.

    `face_yaw`: index -> yaw degrees (same shape as faces_mapping), for
    the steep-angle soft penalty -- optional, and a no-op when omitted or
    shorter than the matched index (an older dataset published before
    this feature has no face_yaw.json at all)."""
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
        _apply_yaw_penalty(candidate, face_yaw, config)
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
    performers: Optional[dict[str, dict]] = None,
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

    A second pass (only when `performers` -- the universal_id -> info
    mapping build_matches() reads -- is supplied) catches what stash_id
    linkage can't: a local performer's own stored `urls` matching a
    *catalogue*-sourced main candidate's `profile_url`/`catalogue_url`
    (e.g. a seekfans match whose OnlyFans link is already on file for an
    already-added local performer -- reported live: the seekfans face
    embedding surfaced instead of the local entry even though the local
    performer's own urls already had the same OnlyFans link).
    Real-stashbox main candidates aren't compared this way -- their own
    urls aren't in this dataset at all (see export_json.py, which only
    exports profile_url/catalogue_url for catalogue-shaped ids), and
    fetching them live per candidate on every match query would be too
    expensive for a hot path; that case still relies on the stash_id
    linkage above (kept in sync automatically today by the local
    performer sync hook, when enabled). Optional purely so an existing
    caller that doesn't pass `performers` keeps today's behavior.
    """
    main_by_universal_id = {c.universal_id: c for c in main_matches}
    merged: list[CandidateMatch] = list(main_matches)
    matched_local_ids: set[str] = set()

    for local_candidate in local_candidates:
        local_id = local_candidate.universal_id.split(":", 1)[1]  # "local:<pid>" -> "<pid>"
        stashdb_id = (local_performers_mapping.get(local_id) or {}).get("stashdb_id")
        linked_universal_id = f"{_STASHDB_ENDPOINT_SHORT_NAME}:{stashdb_id}" if stashdb_id else None

        main_entry = main_by_universal_id.get(linked_universal_id) if linked_universal_id else None
        if main_entry is not None:
            matched_local_ids.add(local_id)
            if local_candidate.combined_distance < main_entry.combined_distance:
                merged[merged.index(main_entry)] = local_candidate
            # else: main_entry already scored better -- keep it, drop the local duplicate silently
        else:
            merged.append(local_candidate)

    if performers:
        for local_candidate in local_candidates:
            local_id = local_candidate.universal_id.split(":", 1)[1]
            if local_id in matched_local_ids:
                continue  # already resolved via stash_id above
            local_urls = {
                normalize_url_for_compare(u)
                for u in (local_performers_mapping.get(local_id) or {}).get("urls") or []
                if u
            }
            if not local_urls:
                continue
            for candidate in main_matches:
                if classify_universal_id(candidate.universal_id) != "catalogue":
                    continue
                if not any(m is candidate for m in merged):
                    continue  # already resolved by an earlier local candidate this pass
                info = performers.get(candidate.universal_id) or {}
                candidate_urls = {
                    normalize_url_for_compare(u)
                    for u in (info.get("profile_url"), info.get("catalogue_url"))
                    if u
                }
                if not (local_urls & candidate_urls):
                    continue
                # Same real person surfaced under both identities -- keep
                # whichever scored better, same convention as the
                # stash_id-based merge above.
                loser = candidate if local_candidate.combined_distance < candidate.combined_distance else local_candidate
                merged = [m for m in merged if m is not loser]
                break

    return merged


def _resolve_local_link_uid(uid: str, local_performers_mapping: Optional[dict[str, dict]]) -> str:
    """A local-index candidate's own `universal_id` stays "local:<id>"
    even when that local performer has been stash_id-linked to a real
    StashDB entry (local_performer_index.py's own sync tracks this in
    `local_performers_mapping[<id>]["stashdb_id"]`). For GROUPING
    purposes only -- never changes what gets displayed -- resolve it to
    the linked "stashdb.org:<uuid>" id instead, so collapse_linked_
    candidates recognizes it as part of a performer_link_index group that
    only ever lists real dataset ids, never "local:" ones. Mirrors
    stash-sense2-data-gen's own scene_matcher.py _resolve_local_link,
    same problem one level up (a single face's own candidate list here,
    cross-frame/cross-cluster tallying there)."""
    if not uid or not uid.startswith("local:") or not local_performers_mapping:
        return uid
    local_id = uid.split(":", 1)[1]
    stashdb_id = (local_performers_mapping.get(local_id) or {}).get("stashdb_id")
    if stashdb_id and stashdb_id != local_id:
        return f"{_STASHDB_ENDPOINT_SHORT_NAME}:{stashdb_id}"
    return uid


def collapse_linked_candidates(
    matches: list[CandidateMatch], performer_link_index: dict[str, list[str]],
    endpoint_priority_domains: list[str],
    local_performers_mapping: Optional[dict[str, dict]] = None,
) -> list[CandidateMatch]:
    """Collapses multiple CandidateMatch entries in `matches` that
    represent the SAME real person, per stash-sense2-data-gen's own
    non-destructive performer_link_groups/members (see that project's
    build/link_duplicate_performers.py -- name+face-cosine-confirmed,
    never a merge/delete on that side either). Same class of problem
    merge_local_candidates() above already solves for local-vs-main
    duplicates (stash_id linkage, matching URLs) -- this is a third such
    signal, sourced from stash-sense2-data-gen rather than derived here.

    Unlike merge_local_candidates' "keep whichever scored better" rule,
    the winner here is picked by `endpoint_priority_domains` (the user's
    own configured stash-box endpoint order, Settings > ... > Endpoint
    priority -- see recognizer.py's _endpoint_priority_domains()) when at
    least one present group member has a stashbox-shaped universal_id --
    the actual goal is which entity gets CREATED in the user's Stash
    instance, not just which one has a marginally tighter distance score
    today (confirmed live: this is what motivated the feature -- an iafd/
    pornpics catalogue match was surfacing instead of an already-known
    StashDB entry for the same person, risking a duplicate performer
    getting created in Stash). Falls back to match-score, same convention
    as merge_local_candidates, only when NO present group member has a
    stashbox endpoint at all (an all-catalogue group, e.g. two catalogue-
    only performers linked to each other with no stashbox member present
    in this particular match's results).

    `performer_link_index` empty (no dataset support yet, or nothing
    linked) is a no-op returning `matches` unchanged -- same "optional,
    absent means skip" tolerance as face_yaw.

    `local_performers_mapping`, when given, resolves a local-index
    candidate to its linked stashdb.org id (_resolve_local_link_uid) for
    GROUP LOOKUP purposes only -- so a local candidate that's itself
    linked to a StashDB entry which is ALSO in a performer_link_index
    group (e.g. with a pornbox record) is recognized as belonging to that
    group, instead of surviving alongside its own linked group's other
    member as if the two were unrelated candidates. Confirmed live:
    without this, match_face()'s caller-side ordering (merge_local_
    candidates must run BEFORE this function now, precisely so a local
    candidate is even present here to resolve in the first place) still
    left a linked local candidate and a linked catalogue candidate both
    surviving as separate "possible matches" for the same real person."""
    if not performer_link_index:
        return matches

    # Keyed by each candidate's EFFECTIVE (resolved) identity throughout --
    # a local candidate contributes under its linked stashdb.org key here,
    # not its literal "local:<id>" universal_id, so a performer_link_index
    # group listing that stashdb.org id (never a "local:" one -- the
    # dataset that produces performer_link_index has no concept of this
    # sidecar's own local Stash performers) finds it as present. Mixing
    # raw and resolved keys in this lookup was the actual bug the first
    # version of this fix shipped with: checking `other in <raw-keyed
    # dict>` for a group member id that only ever exists in resolved form
    # never matched, silently leaving the local candidate ungrouped.
    by_resolved: dict[str, CandidateMatch] = {
        _resolve_local_link_uid(c.universal_id, local_performers_mapping): c for c in matches
    }
    seen: set[str] = set()
    result: list[CandidateMatch] = []

    def endpoint_rank(resolved_uid: str) -> int:
        domain = _extract_endpoint_domain(resolved_uid)
        if domain in endpoint_priority_domains:
            return endpoint_priority_domains.index(domain)
        return len(endpoint_priority_domains)  # no configured stashbox endpoint -- lowest priority

    for candidate in matches:
        resolved_uid = _resolve_local_link_uid(candidate.universal_id, local_performers_mapping)
        if resolved_uid in seen:
            continue
        present_group = [resolved_uid] + [
            other for other in performer_link_index.get(resolved_uid, [])
            if other in by_resolved and other != resolved_uid
        ]
        seen.update(present_group)
        if len(present_group) == 1:
            result.append(candidate)
            continue

        ranked = sorted(present_group, key=endpoint_rank)
        if endpoint_rank(ranked[0]) < len(endpoint_priority_domains):
            winner_uid = ranked[0]
        else:
            winner_uid = min(present_group, key=lambda u: by_resolved[u].combined_distance)
        result.append(by_resolved[winner_uid])

    return result


def _extract_endpoint_domain(universal_id: str) -> Optional[str]:
    return universal_id.split(":", 1)[0] if ":" in universal_id else None


def match_face(
    embedding: np.ndarray,
    index: Index,
    faces_mapping: list[str],
    performers: dict[str, dict],
    config: MatchingConfig = DEFAULT_CONFIG,
    local_index: Optional[Index] = None,
    local_performers_mapping: Optional[dict[str, dict]] = None,
    face_yaw: Optional[list] = None,
    performer_link_index: Optional[dict[str, list[str]]] = None,
    endpoint_priority_domains: Optional[list[str]] = None,
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
        face_yaw: index -> yaw degrees for the main index (see
            build_matches's own docstring). Optional; omitting it skips
            the steep-angle penalty. Not applicable to local-index
            candidates (each local performer contributes exactly one
            vector from their own cover image, no per-vector yaw tracked).
        performer_link_index: universal_id -> every OTHER universal_id
            stash-sense2-data-gen determined is the same real person (see
            recognizer.py's own loading of performer_links.json). Optional;
            omitting it (or an empty dict) skips collapse_linked_candidates
            entirely -- see that function's own docstring.
        endpoint_priority_domains: The user's current configured stash-box
            endpoint priority, as domains in priority order (see
            recognizer.py's _endpoint_priority_domains()) -- used only to
            pick a winner among a linked group's present candidates.

    Returns:
        MatchingResult with candidates
    """
    query_result = query_index(embedding, index, config)
    result = build_matches(
        query_result, faces_mapping, performers, config,
        face_yaw=face_yaw,
    )

    # Optionally merge in local-performer-index matches (see fuse_local_results).
    # Runs BEFORE collapse_linked_candidates below now (fixed 2026-09-14) --
    # see that function's own docstring for why the old order (collapse
    # first, merge local candidates in afterward) could leave a local
    # candidate and its own linked catalogue duplicate both surviving as
    # separate "possible matches" for the same real person: collapse had
    # already picked a winner from the main-index-only candidates before
    # the local candidate even existed to be compared against it.
    # A handful of local performers is common (especially right after the
    # first sync), so index.search()'s k can exceed the index size -- guard
    # with try/except rather than requiring every caller to pre-check size.
    if local_index is not None and local_performers_mapping and len(local_index) > 0:
        try:
            local_query_result = query_index(embedding, local_index, config)
            local_candidates = fuse_local_results(local_query_result, local_performers_mapping, config)
            result.matches = merge_local_candidates(result.matches, local_candidates, local_performers_mapping, performers)
        except Exception as e:
            logger.warning(f"Local performer index query failed, skipping local matches: {e}")

    # Filter to max_distance BEFORE collapse, not just at the very end --
    # fuse_local_results' LOCAL_MATCH_BOOST can still leave a local
    # candidate's own combined_distance above max_distance even after the
    # boost (a weak match is still weak). Confirmed live 2026-09-14: with
    # this filter only applied at the end (as it was immediately after
    # the collapse/merge reorder above), collapse_linked_candidates could
    # pick that out-of-threshold local candidate as the group's
    # endpoint-priority "winner" over a genuinely in-threshold linked
    # catalogue candidate -- discarding the valid candidate for an
    # invalid one that then ALSO got filtered out by the final max_distance
    # cut, losing the match entirely. Confirmed against the real reported
    # scene: frame 57 had a valid pornbox candidate (distance 0.495, well
    # within the 0.5 threshold) discarded in favor of a linked local
    # candidate at a boosted 0.511 -- itself then filtered out -- taking a
    # 26-frame "Sylwia" cluster down to 4 and dropping her out of the
    # scene's top-ranked match entirely.
    result.matches = [c for c in result.matches if c.combined_distance <= config.max_distance]

    if performer_link_index:
        result.matches = collapse_linked_candidates(
            result.matches, performer_link_index, endpoint_priority_domains or [],
            local_performers_mapping=local_performers_mapping,
        )

    # Re-sort/truncate to max_results -- merge_local_candidates/
    # collapse_linked_candidates above can both change membership (add
    # local candidates; remove collapsed losers), and the max_distance
    # filter already ran above, so this is just sort+truncate now.
    result.matches.sort(key=lambda c: c.combined_distance)
    result.matches = result.matches[:config.max_results]

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

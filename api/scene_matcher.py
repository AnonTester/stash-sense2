"""Scene-level face matching: clustering, frequency, and hybrid strategies.

Provides algorithms for identifying performers across multiple video frames:
- Embedding-based face clustering (cluster_faces_by_person)
- Frequency-based matching (frequency_based_matching)
- Clustered frequency matching (clustered_frequency_matching)
- Hybrid matching combining cluster and frequency (hybrid_matching)
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from matching import _extract_endpoint_domain, _STASHDB_ENDPOINT_SHORT_NAME
from recognizer import FaceRecognizer, PerformerMatch, RecognitionResult

logger = logging.getLogger(__name__)


def _link_key(universal_id: Optional[str], performer_link_index: Optional[dict[str, list[str]]]) -> Optional[str]:
    """Canonical identity key for `universal_id`, collapsing stash-sense2-
    data-gen's linked same-real-person/different-catalog-record groups
    (performer_links.json, see recognizer.py's own loading of it and
    matching.py's collapse_linked_candidates -- that function solves the
    identical problem for a single face's own ranked candidate list; this
    is the same fix applied to the tallying/merging below, which spans
    frames and clusters instead).

    Without this, two frames/clusters that each happen to match a
    *different* (but linked) database record for the same real person --
    e.g. one frame closer to "Elma"'s reference photo, another closer to
    "Sonya Chrystal"'s, when both are the same performer catalogued twice
    -- get tallied as two separate identities instead of one, splitting a
    single real person into two separate low-frame-count "persons" in the
    scene face match UI.

    Returns `universal_id` unchanged when unlinked, or when
    `performer_link_index` is empty/absent (no dataset support yet) --
    same "optional, absent means skip" tolerance as collapse_linked_candidates.
    The specific member chosen as the key (lexicographically lowest) is
    only for a stable grouping key, not a display choice -- callers still
    show whichever actual match object won the tally.
    """
    if not universal_id or not performer_link_index:
        return universal_id
    group = performer_link_index.get(universal_id)
    if not group:
        return universal_id
    return min([universal_id, *group])


def _canonical_identity(
    match_like, performer_link_index: Optional[dict[str, list[str]]],
) -> Optional[str]:
    """Full canonical identity key for a match-shaped object (a raw
    PerformerMatch, or a PerformerMatchResponse -- both carry
    universal_id-equivalent fields), resolving TWO separate "this is
    actually the same real person" signals: local_performer_index's own
    local-performer-linked-to-StashDB link (_resolve_local_link), then
    stash-sense2-data-gen's own performer_link_index on top (_link_key)
    -- a stash_id-linked local performer could ALSO belong to a data-gen
    link group that never saw the local record.

    Without this, a local-index match and the main index's own StashDB
    entry for the exact same real person (same stashdb_id) show up as two
    separate low-frame-count "persons" instead of one -- confirmed live:
    "Marley Brinx" via the local index (universal_id "local:2519") and via
    stashdb.org directly (universal_id "stashdb.org:23d3aa04-...", same
    stashdb_id) never merged.
    """
    return _link_key(_resolve_local_link(match_like), performer_link_index)


def _resolve_local_link(match_like) -> Optional[str]:
    """A local-index match's `universal_id` stays "local:<id>", but
    recognizer.py already resolves its `stashdb_id` field to the real
    linked stashdb.org uuid when that local performer has been
    stash_id-linked (see recognizer.py's PerformerMatch construction,
    category == "local" branch) -- matching.py's merge_local_candidates()
    already treats this as one person within a single face's own
    candidate list. Returns the resolved "stashdb.org:<uuid>" id for a
    linked local match, or `universal_id` unchanged otherwise (including
    for an *unlinked* local match, whose stashdb_id just falls back to
    its own local id, per recognizer.py -- detected by stashdb_id
    disagreeing with local_performer_id).

    Used both for grouping (_canonical_identity, so a local match and the
    main index's own entry for the same real person collapse into one
    identity) and for display-priority ranking (_endpoint_rank/
    _pick_priority_match, so that linked local match is correctly judged
    by its real stashdb.org endpoint rather than the "local" pseudo-
    endpoint, which never matches a configured priority domain and would
    otherwise leave the winner decided by raw match score alone).
    """
    uid = getattr(match_like, "universal_id", None)
    if uid is None:
        uid = match_universal_id(match_like)

    local_performer_id = getattr(match_like, "local_performer_id", None)
    stashdb_id = getattr(match_like, "stashdb_id", None)
    if (
        uid and uid.startswith("local:") and local_performer_id and stashdb_id
        and str(stashdb_id) != str(local_performer_id)
    ):
        return f"{_STASHDB_ENDPOINT_SHORT_NAME}:{stashdb_id}"
    return uid


def _endpoint_rank(universal_id: Optional[str], endpoint_priority_domains: list[str]) -> int:
    """Same ranking matching.py's collapse_linked_candidates uses: a lower
    number is higher priority. A universal_id whose endpoint domain is in
    the user's configured stash-box endpoint priority order (Settings >
    ... > Endpoint priority) ranks by its position there; anything else
    (an unconfigured stashbox endpoint, a catalogue source like pornbox/
    iafd, or a local match) ranks last, tied at len(endpoint_priority_domains)."""
    domain = _extract_endpoint_domain(universal_id) if universal_id else None
    if domain in endpoint_priority_domains:
        return endpoint_priority_domains.index(domain)
    return len(endpoint_priority_domains)


def _pick_priority_match(
    matches: list[PerformerMatch], endpoint_priority_domains: list[str],
) -> PerformerMatch:
    """Pick which of several PerformerMatch objects for the same (possibly
    linked) identity to actually show, by the same rule
    collapse_linked_candidates already applies to a single face's own
    candidate list: prefer the user's configured stash-box endpoint
    priority order over a marginally better match score -- e.g. a linked
    group's stashdb.org member should win over its pornbox member even if
    a given frame happened to score the pornbox photo a hair closer.
    Ranks each candidate by its EFFECTIVE endpoint (_resolve_local_link),
    not its literal universal_id -- a local-index match linked to a
    stashdb.org entry must be judged as stashdb.org here too, or it ties
    with an unconfigured catalogue source (both fall back to "no
    configured priority") and the winner ends up decided by raw match
    score alone, which flips unpredictably run to run. Falls back to best
    (lowest) combined_score only when NONE of the candidates resolves to
    a configured priority endpoint at all."""
    if len(matches) == 1:
        return matches[0]
    ranked = sorted(matches, key=lambda m: _endpoint_rank(_resolve_local_link(m), endpoint_priority_domains))
    if _endpoint_rank(_resolve_local_link(ranked[0]), endpoint_priority_domains) < len(endpoint_priority_domains):
        return ranked[0]
    return min(matches, key=lambda m: m.combined_score)


def match_universal_id(match) -> Optional[str]:
    """Reconstruct the universal_id a PerformerMatchResponse was built from.

    The response model only carries the decomposed parts (endpoint,
    stashdb_id, local_performer_id), not universal_id itself. For a
    local-index match, `stashdb_id` may hold that performer's *linked*
    StashDB uuid rather than their local id, so local_performer_id (always
    the local id for a local match) must be checked first -- endpoint+
    stashdb_id alone would silently reconstruct the wrong key. For
    stashbox/catalogue matches, endpoint+stashdb_id already exactly
    reproduces the original "<endpoint>:<id>".

    Shared by scene_face_match.py (recommendation creation) and
    identification_router.py's save_scene_fingerprint (persisted match
    storage) -- both need the same dedup/join key for the same match.
    Duck-typed on `.local_performer_id`/`.endpoint`/`.stashdb_id` rather
    than importing PerformerMatchResponse, to avoid a circular import with
    identification_router.py.
    """
    if match.local_performer_id:
        return f"local:{match.local_performer_id}"
    if match.endpoint and match.stashdb_id:
        return f"{match.endpoint}:{match.stashdb_id}"
    return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def cluster_faces_by_person(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: FaceRecognizer,
    distance_threshold: float = 0.6,
) -> list[list[tuple[int, RecognitionResult]]]:
    """
    Cluster detected faces by person using embedding cosine similarity.

    Uses greedy clustering: assign each face to the nearest existing
    cluster or create a new cluster if too far from all existing ones.

    Uses cosine distance (consistent with Voyager indices) instead of L2,
    and reuses pre-computed embeddings from recognition when available.

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        recognizer: FaceRecognizer instance (fallback for embedding generation)
        distance_threshold: Max cosine distance to consider same person (default 0.6)

    Returns:
        List of clusters, each containing faces of the same person
    """
    if not all_results:
        return []

    clusters: list[list[tuple[int, RecognitionResult, np.ndarray]]] = []

    for frame_idx, result in all_results:
        # Use stored embedding if available, otherwise recompute
        if result.embedding is not None:
            face_vector = result.embedding.embedding
        else:
            embedding = recognizer.generator.get_embedding(result.face.image)
            face_vector = embedding.embedding

        # Find nearest cluster by cosine distance
        best_cluster_idx = -1
        best_distance = float("inf")

        for cluster_idx, cluster in enumerate(clusters):
            # Compare to cluster centroid (average of all faces in cluster)
            cluster_vectors = [c[2] for c in cluster]
            centroid = np.mean(cluster_vectors, axis=0)
            distance = _cosine_distance(face_vector, centroid)

            if distance < best_distance:
                best_distance = distance
                best_cluster_idx = cluster_idx

        # Add to existing cluster or create new one
        if best_distance < distance_threshold and best_cluster_idx >= 0:
            clusters[best_cluster_idx].append((frame_idx, result, face_vector))
        else:
            clusters.append([(frame_idx, result, face_vector)])

    # Remove embedding vectors from output
    return [[(f, r) for f, r, _ in cluster] for cluster in clusters]


def merge_clusters_by_match(
    clusters: list[list[tuple[int, RecognitionResult]]],
    performer_link_index: Optional[dict[str, list[str]]] = None,
) -> list[list[tuple[int, RecognitionResult]]]:
    """
    Merge clusters that have the same best performer match.

    If multiple clusters all have "Xander Corvus" as their top match,
    they're probably the same person and should be merged. Also merges
    across two clusters whose respective top matches are DIFFERENT
    database records that are nonetheless linked as the same real person
    (see _canonical_identity) -- otherwise a real person catalogued under
    two records (e.g. a name change, or a second site's own profile)
    whose embeddings happen to sit slightly closer to one record in some
    frames and the other record in other frames gets split into two
    separate face clusters/"persons" instead of merged into one.

    Each cluster's own DOMINANT identity decides merging -- the canonical
    identity with the best weighted (min-distance + frame-count-bonus)
    score across THAT cluster's own frames alone, same ranking philosophy
    aggregate_matches itself uses. Two clusters merge only when their own
    dominant identities agree (a plain group-by, not transitive
    union-find across every frame-level candidate seen anywhere).

    A prior version of this function (2026-09-14) unioned clusters that
    shared ANY canonical identity across ANY single frame, transitively.
    Confirmed live this over-merges badly: cluster_faces_by_person's own
    greedy clustering already produces a messy raw cluster in a crowded
    scene (many different real people, each contributing one frame whose
    own weak top-1 candidate happens to be some other performer entirely)
    -- and if even ONE of those frames' weak top-1 pick matched a
    genuinely-linked identity another cluster was keyed on, that whole
    messy cluster got pulled in as a "bridge", silently absorbing
    unrelated people into one blob (confirmed: a real person's own
    cluster grew from a correct 4 frames to an incorrect 26-frame blob
    mixing in "Harley Quinn", "Anna Huse", "Daisy Lee", and half a dozen
    other unrelated performers, none of whom were actually the linked
    real person). This is the exact same class of bug stash-sense2-data-
    gen's own build/link_duplicate_performers.py already hit and
    deliberately fixed once before (see that project's
    _build_name_clusters docstring -- a single bridging alias performer
    collapsing half its dataset into one cluster via transitive
    union-find). Dominant-identity grouping avoids it the same way that
    fix did: a cluster's OWN aggregate vote decides its identity, not any
    single incidental frame-level candidate anywhere in it.
    """
    if len(clusters) <= 1:
        return clusters

    def dominant_identity(cluster: list[tuple[int, RecognitionResult]]) -> Optional[str]:
        scores: dict[str, list[float]] = defaultdict(list)
        for _, result in cluster:
            if not result.matches:
                continue
            key = _canonical_identity(result.matches[0], performer_link_index)
            if key:
                scores[key].append(result.matches[0].combined_score)
        if not scores:
            return None
        # Same min-distance-plus-frame-bonus ranking aggregate_matches
        # itself uses, so the cluster-level "winner" here agrees with
        # what aggregate_matches would independently pick for this same
        # cluster's own frames.
        def weighted_score(key: str) -> float:
            appearances = scores[key]
            confidence = max(0.0, 1.0 - min(appearances))
            return confidence * (1 + 0.1 * (len(appearances) - 1))
        return max(scores, key=weighted_score)

    groups: dict[str, list[tuple[int, RecognitionResult]]] = defaultdict(list)
    unidentified: list[list[tuple[int, RecognitionResult]]] = []
    for cluster in clusters:
        identity = dominant_identity(cluster)
        if identity is None:
            unidentified.append(cluster)
        else:
            groups[identity].extend(cluster)

    return list(groups.values()) + unidentified


def aggregate_matches(
    cluster: list[tuple[int, RecognitionResult]],
    top_k: int = 3,
    _match_to_response=None,
    _distance_to_confidence=None,
    frame_timestamps: Optional[dict[int, float]] = None,
    performer_link_index: Optional[dict[str, list[str]]] = None,
    endpoint_priority_domains: Optional[list[str]] = None,
) -> list:
    """
    Aggregate matches across multiple frames for a person.

    Combines match scores across frames, preferring performers that appear
    consistently with low distances.

    Args:
        cluster: List of (frame_index, RecognitionResult) tuples
        top_k: Maximum number of matches to return
        _match_to_response: Callback to convert PerformerMatch to response model
        _distance_to_confidence: Callback to convert distance to confidence score
        frame_timestamps: Optional frame_index -> timestamp_sec mapping (from
            the live ffmpeg extraction pass). When provided, each aggregated
            match's 4 lowest-distance frames are resolved to timestamps and
            attached as top_timestamps_sec, for "jump to this frame" UI.
            Frame indices with no mapping entry (e.g. the -1 screenshot
            sentinel, or no mapping at all) are silently skipped. Sprite-tile
            faces each get their own unique negative frame_index (-2, -3,
            ...) rather than sharing one sentinel, specifically so they can
            resolve real per-face timestamps here too -- see
            identification_router.py's _process_sprite_frames.
        performer_link_index: see _link_key -- tallies linked (same real
            person, different catalog record) matches together instead of
            as separate candidates.
        endpoint_priority_domains: see _pick_priority_match -- decides
            which linked member's match object gets shown (e.g. the
            stashdb.org entry over a pornbox entry for the same real
            person) when a link group has more than one candidate present
            across this cluster's frames.
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from identification_router import _match_to_response
    if _distance_to_confidence is None:
        from identification_router import distance_to_confidence as _distance_to_confidence

    # Collect all matches across frames, keyed by each match's canonical
    # linked-group id (see _link_key) rather than its raw stashdb_id, so a
    # linked duplicate doesn't fork into a second entry here.
    match_scores: dict[str, list[float]] = defaultdict(list)
    match_frames: dict[str, list[tuple[int, float]]] = defaultdict(list)
    match_candidates: dict[str, list[PerformerMatch]] = defaultdict(list)

    for frame_idx, result in cluster:
        for match in result.matches:
            key = _canonical_identity(match, performer_link_index)
            match_scores[key].append(match.combined_score)
            match_frames[key].append((frame_idx, match.combined_score))
            match_candidates[key].append(match)

    # Rank (and display) by: the best single frame's distance, with a
    # modest bonus for appearing in more frames -- same
    # min_distance-plus-frame_bonus philosophy clustered_frequency_matching
    # and frequency_based_matching already use, applied here too so all
    # three matching modes agree on both *which* candidate wins and *what
    # confidence* gets shown for it.
    #
    # This used to rank by avg_score / sqrt(appearance_ratio) instead --
    # averaging in every weaker/borderline frame a tracked person-cluster
    # happened to include, then dividing by how large a fraction of the
    # cluster's frames this performer appeared in at all. Confirmed live,
    # two distinct failures from that:
    #   1. A correctly-identified 30+ frame "Cristina Agave" cluster whose
    #      best frame matched well still displayed 0% confidence, because
    #      dividing by sqrt(appearance_ratio) pushed the shown score above
    #      the distance-1.0 floor purely for not appearing in *every*
    #      frame.
    #   2. Once (1) was fixed by displaying min_score without also fixing
    #      ranking, the *displayed*, sorted-by-confidence candidate list
    #      and the *pre-selected* (is_best_match) candidate could
    #      disagree: a scene showed a 60%-confidence, first-listed pornbox
    #      candidate left unchecked while a 56%-confidence stashdb/local
    #      candidate sorted lower was the one pre-selected -- ranking was
    #      still using the old averaged metric while display and list
    #      order had already switched to min_score.
    # No hard confidence threshold is applied here (unlike the other two
    # matching modes' `if (1 - min_distance) < min_confidence: continue`)
    # -- this mode's whole purpose is to still surface single-frame/weak
    # matches rather than drop them outright (see TOP_K_PER_PERSON's
    # comment in scene_face_match.py); a weak match's low confidence and
    # zero frame_bonus already rank it near the bottom on its own.
    aggregated: list[tuple[object, float]] = []
    for link_key, scores in match_scores.items():
        min_score = min(scores)
        appearances = len(scores)
        confidence = max(0.0, 1.0 - min_score)
        frame_bonus = 0.1 * (appearances - 1)
        weighted_score = confidence * (1 + frame_bonus)

        match = _pick_priority_match(match_candidates[link_key], endpoint_priority_domains or [])

        top_timestamps_sec: list[float] = []
        if frame_timestamps:
            best_frames = sorted(match_frames[link_key], key=lambda fs: fs[1])[:4]
            timestamps = {
                frame_timestamps[frame_idx]
                for frame_idx, _ in best_frames
                if frame_idx in frame_timestamps
            }
            top_timestamps_sec = sorted(timestamps)

        response = _match_to_response(
            match,
            confidence=_distance_to_confidence(min_score),
            distance=min_score,
            top_timestamps_sec=top_timestamps_sec,
        )
        aggregated.append((response, weighted_score))

    # Sort by weighted score (higher is better) -- same ordering
    # principle now used for ranking, list order, and display.
    aggregated.sort(key=lambda pair: pair[1], reverse=True)
    return [response for response, _ in aggregated[:top_k]]


def frequency_based_matching(
    all_results: list[tuple[int, RecognitionResult]],
    top_k: int = 5,
    min_appearances: int = 2,
    min_unique_frames: int = 2,
    max_distance: float = 0.5,
    min_confidence: float = 0.35,
    _match_to_response=None,
    _distance_to_confidence=None,
    performer_link_index: Optional[dict[str, list[str]]] = None,
    endpoint_priority_domains: Optional[list[str]] = None,
) -> list:
    """
    Identify performers by counting appearances across all face matches.

    Instead of clustering faces first (which can fail when embeddings vary too much),
    this approach:
    1. Collects all matches from all detected faces
    2. Counts how many times each performer appears in the top matches
    3. Weights by match distance (closer matches count more)
    4. Returns performers sorted by weighted frequency

    This is more robust when clustering fails but may include false positives
    if the same wrong performer happens to match multiple faces.

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        top_k: Number of performers to return
        min_appearances: Minimum number of face matches to include a performer
        min_unique_frames: Minimum unique frames a performer must appear in
        max_distance: Only count matches below this distance
        min_confidence: Minimum confidence threshold (filters low-quality matches)
        _match_to_response: Callback to convert PerformerMatch to response model
        _distance_to_confidence: Callback to convert distance to confidence score
        performer_link_index: see _link_key -- tallies linked (same real
            person, different catalog record) matches together instead of
            as separate candidates.
        endpoint_priority_domains: see _pick_priority_match -- decides
            which linked member's match object gets shown.

    Returns:
        List of PersonResult objects, one per identified performer
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from identification_router import _match_to_response
    if _distance_to_confidence is None:
        from identification_router import distance_to_confidence as _distance_to_confidence
    from identification_router import PersonResult

    # Collect all matches across all faces, keyed by canonical linked-group
    # id (see _link_key) rather than raw stashdb_id.
    performer_matches: dict[str, list[tuple[float, PerformerMatch, int]]] = defaultdict(list)

    for frame_idx, result in all_results:
        for match in result.matches:
            if match.combined_score <= max_distance:
                key = _canonical_identity(match, performer_link_index)
                performer_matches[key].append((match.combined_score, match, frame_idx))

    # Calculate weighted frequency score for each performer
    # Score = appearances * (1 / avg_distance) - rewards frequent, close matches
    performer_scores = []
    for link_key, matches in performer_matches.items():
        if len(matches) < min_appearances:
            continue

        distances = [m[0] for m in matches]
        avg_distance = np.mean(distances)
        min_distance = min(distances)
        appearances = len(matches)
        unique_frames = len(set(m[2] for m in matches))

        # Require appearance in multiple unique frames to reduce false positives
        if unique_frames < min_unique_frames:
            continue

        # Filter by minimum confidence (1 - distance)
        if (1 - min_distance) < min_confidence:
            continue

        # Weighted score: primarily based on match quality, with modest bonus for frame count
        # Formula: confidence * (1 + small_frame_bonus)
        # This ensures a single excellent match beats multiple mediocre matches
        confidence = 1 - min_distance  # Use best match, not average
        frame_bonus = 0.1 * (unique_frames - 1)  # Small bonus: +10% per additional frame
        weighted_score = confidence * (1 + frame_bonus)

        # For display, prefer the endpoint-priority winner (e.g. stashdb.org
        # over pornbox for the same linked real person); falls back to the
        # best (lowest) distance match when no candidate has a configured
        # priority endpoint -- see _pick_priority_match.
        best_match = _pick_priority_match([m[1] for m in matches], endpoint_priority_domains or [])

        performer_scores.append({
            "stashdb_id": link_key,
            "appearances": appearances,
            "unique_frames": unique_frames,
            "avg_distance": avg_distance,
            "min_distance": min_distance,
            "weighted_score": weighted_score,
            "best_match": best_match,
        })

    # Sort by weighted score (higher is better)
    performer_scores.sort(key=lambda p: p["weighted_score"], reverse=True)

    # Convert to PersonResult format
    #
    # min_distance, not avg_distance -- see clustered_frequency_matching's
    # _to_match_response for the full explanation; same bug, same fix:
    # selection/gating above is min_distance-based (this function's own
    # comment at its definition already says "For display, use the match
    # with the best (lowest) distance"), so displaying avg_distance here
    # contradicted that and could show 0% for a correctly-identified,
    # well-matched performer just because they were tracked across enough
    # frames for weak ones to drag the mean past 1.0.
    persons = []
    for i, p in enumerate(performer_scores[:top_k]):
        match = p["best_match"]
        resp = _match_to_response(
            match,
            confidence=_distance_to_confidence(p["min_distance"]),
            distance=p["min_distance"],
        )
        persons.append(PersonResult(
            person_id=i,
            frame_count=p["unique_frames"],
            best_match=resp,
            all_matches=[resp],
        ))

    return persons


def clustered_frequency_matching(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: "FaceRecognizer",
    cluster_threshold: float = 0.6,
    top_k: int = 5,
    max_distance: float = 0.5,
    min_confidence: float = 0.35,
    scene_performer_stashdb_ids: list[str] | None = None,
    tagged_boost: float = 0.03,
    _match_to_response=None,
    _distance_to_confidence=None,
) -> list:
    """
    Cluster faces first to determine person count, then use frequency matching
    within each cluster to identify who each person is.

    This combines the strengths of both approaches:
    - Clustering answers "how many distinct people are there?"
    - Frequency matching answers "who is each person?"

    The result is one PersonResult per face cluster, with alternative matches
    for each cluster shown as all_matches.

    This is the scene player's own default matching_mode (identification_
    router.py's SceneIdentifyRequest.matching_mode default is "frequency",
    which dispatches HERE, not to hybrid_matching -- confirmed live 2026-
    09-14 as a real gap: the scene player's "Re-identify"/"Identify Full
    Video" buttons never pass an explicit matching_mode, so every fix
    aimed at hybrid_matching's own linked-candidate handling had no effect
    on what that UI actually shows). Tallies each cluster's own candidates
    by canonical linked-group identity (_canonical_identity), not raw
    stashdb_id, and picks the display winner via endpoint priority
    (_pick_priority_match) -- same treatment aggregate_matches/
    frequency_based_matching/hybrid_matching already got, applied here too
    so a linked duplicate (e.g. a local-index match and its own linked
    pornbox record) doesn't show up as a spurious "other possible match"
    for a person the winner already represents.
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from identification_router import _match_to_response
    if _distance_to_confidence is None:
        from identification_router import distance_to_confidence as _distance_to_confidence
    from identification_router import PersonResult

    if not all_results:
        return []

    tagged_ids = set(scene_performer_stashdb_ids or [])
    performer_link_index = getattr(recognizer, "performer_link_index", None) or {}
    endpoint_priority_domains = (
        recognizer._endpoint_priority_domains() if hasattr(recognizer, "_endpoint_priority_domains") else []
    )

    # Step 1: Cluster faces by embedding similarity
    clusters = cluster_faces_by_person(
        all_results, recognizer, distance_threshold=cluster_threshold
    )
    # Merge clusters that have the same (or linked) best match
    clusters = merge_clusters_by_match(clusters, performer_link_index=performer_link_index)

    print(f"[clustered_freq] {len(all_results)} face detections -> {len(clusters)} person clusters")

    # Step 2: For each cluster, run frequency matching to find the best performer
    persons = []
    used_performers: set[str] = set()

    # Sort clusters by size (most prominent person first)
    sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)

    for cluster_idx, cluster in sorted_clusters:
        cluster_size = len(cluster)
        unique_frames = len(set(frame_idx for frame_idx, _ in cluster))

        # Collect all matches from faces in this cluster, keyed by each
        # match's canonical linked-group identity (see _canonical_identity)
        # rather than raw stashdb_id -- otherwise a linked local/catalogue
        # duplicate (e.g. a local-index match and its own linked pornbox
        # record) tallies as two separate candidates within the same
        # cluster, and the loser shows up as a spurious "other possible
        # match" for a person the winner already correctly represents.
        performer_matches: dict[str, list[tuple[float, PerformerMatch, int]]] = defaultdict(list)

        for frame_idx, result in cluster:
            for match in result.matches:
                if match.combined_score <= max_distance:
                    key = _canonical_identity(match, performer_link_index)
                    performer_matches[key].append(
                        (match.combined_score, match, frame_idx)
                    )

        if not performer_matches:
            # No matches in this cluster - unknown person
            persons.append(PersonResult(
                person_id=len(persons),
                frame_count=unique_frames,
                best_match=None,
                all_matches=[],
            ))
            continue

        # Score each performer within this cluster
        candidates = []
        for link_key, matches in performer_matches.items():
            distances = [m[0] for m in matches]
            min_distance = min(distances)
            match_unique_frames = len(set(m[2] for m in matches))

            if (1 - min_distance) < min_confidence:
                continue

            confidence = 1 - min_distance
            frame_bonus = 0.1 * (match_unique_frames - 1)
            weighted_score = confidence * (1 + frame_bonus)

            # For display, prefer the endpoint-priority winner among this
            # identity's own linked candidates (e.g. stashdb.org over
            # pornbox for the same real person) -- falls back to best
            # (lowest) distance when none has a configured priority
            # endpoint. See _pick_priority_match.
            raw_matches = [m[1] for m in matches]
            best_match = _pick_priority_match(raw_matches, endpoint_priority_domains)
            is_tagged = any(m.stashdb_id in tagged_ids for m in raw_matches)

            # Apply small boost for already-tagged performers
            if is_tagged:
                weighted_score += tagged_boost

            candidates.append({
                "stashdb_id": link_key,
                "appearances": len(matches),
                "unique_frames": match_unique_frames,
                "min_distance": min_distance,
                "avg_distance": float(np.mean(distances)),
                "weighted_score": weighted_score,
                "best_match": best_match,
                "is_tagged": is_tagged,
            })

        # Sort by weighted score (higher is better)
        candidates.sort(key=lambda c: c["weighted_score"], reverse=True)

        # Pick best performer not yet used by a higher-priority cluster
        best_candidate = None
        alt_candidates = []
        for c in candidates:
            if c["stashdb_id"] not in used_performers:
                if best_candidate is None:
                    best_candidate = c
                    used_performers.add(c["stashdb_id"])
                else:
                    alt_candidates.append(c)
            else:
                alt_candidates.append(c)

        if best_candidate is None:
            # All candidates already used
            persons.append(PersonResult(
                person_id=len(persons),
                frame_count=unique_frames,
                best_match=None,
                all_matches=[],
            ))
            continue

        # Build all_matches list: best first, then alternatives (up to top_k)
        #
        # Shows min_distance (the single best-matching frame), not
        # avg_distance -- candidate selection above (min_confidence gate,
        # weighted_score, frame_bonus) is entirely min_distance-based, so
        # avg_distance here was displaying a number that had nothing to do
        # with why this candidate won. It also actively fights frame_bonus:
        # a person tracked across many frames (more chances for a weak/
        # blurry/off-angle frame to drag the mean up) could show a WORSE
        # confidence than the same identity tracked in only one or two
        # clean frames, even though the ranking logic rewards more frames.
        # Confirmed live: several correctly-identified 30+ frame clusters
        # (same performer's own db entry, name matched) displayed 0%
        # confidence via avg_distance > 1.0, while other scenes/frames of
        # the identical performer/db-entry pairing showed 0.35-0.65
        # confidence -- a correct identification made to look like a
        # non-match purely because of how many frames happened to be
        # tracked alongside the good one(s).
        def _to_match_response(c: dict):
            m = c["best_match"]
            return _match_to_response(
                m,
                confidence=_distance_to_confidence(c["min_distance"]),
                distance=c["min_distance"],
                already_tagged=c["is_tagged"],
            )

        best_response = _to_match_response(best_candidate)
        all_matches = [best_response]
        for alt in alt_candidates[:top_k - 1]:
            all_matches.append(_to_match_response(alt))

        persons.append(PersonResult(
            person_id=len(persons),
            frame_count=unique_frames,
            best_match=best_response,
            all_matches=all_matches,
        ))

    # Sort: persons with matches first (by frame count desc), then unknowns
    persons.sort(key=lambda p: (p.best_match is not None, p.frame_count), reverse=True)

    # Re-assign person IDs after sorting
    for i, p in enumerate(persons):
        p.person_id = i

    return persons


def hybrid_matching(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: "FaceRecognizer",
    cluster_threshold: float = 0.6,
    top_k: int = 5,
    max_distance: float = 0.5,
    min_appearances: int = 2,
    min_unique_frames: int = 2,
    min_confidence: float = 0.35,
    frame_timestamps: Optional[dict[int, float]] = None,
    _match_to_response=None,
    _distance_to_confidence=None,
) -> list:
    """
    Hybrid matching combining cluster and frequency approaches.

    Runs both methods and combines results:
    - Performers found by BOTH methods get a significant boost
    - Uses the best (lowest) distance from either method
    - Sorts by combined score

    This helps when:
    - Clustering works well (cluster mode catches it)
    - Clustering fails but frequency catches appearances (frequency mode catches it)

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        recognizer: FaceRecognizer instance for clustering
        cluster_threshold: Distance threshold for face clustering
        top_k: Maximum number of performers to return
        max_distance: Maximum distance threshold for matches
        min_appearances: Minimum face matches required per performer
        min_unique_frames: Minimum unique frames a performer must appear in
        min_confidence: Minimum confidence threshold (1 - distance)
        frame_timestamps: Optional frame_index -> timestamp_sec mapping (see
            aggregate_matches's own docstring). Threaded only into this
            function's internal cluster-mode component -- frequency_based_
            matching has no per-frame-cluster concept to resolve timestamps
            from, so a performer found only by frequency never gets
            top_timestamps_sec here, same as calling frequency mode
            directly. When a performer is found by both methods, the
            cluster component's timestamps (if any) are kept even though
            the frequency component's match object wins for display
            fields -- see the "found_by" branch below.
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from identification_router import _match_to_response
    if _distance_to_confidence is None:
        from identification_router import distance_to_confidence as _distance_to_confidence
    from identification_router import PersonResult

    # recognizer=None is a real, supported call shape (see
    # tests/test_scene_matcher_logic.py's TestHybridMatchingFrameTimestamps),
    # not just a defensive fallback.
    performer_link_index = getattr(recognizer, "performer_link_index", None) or {}
    endpoint_priority_domains = (
        recognizer._endpoint_priority_domains() if hasattr(recognizer, "_endpoint_priority_domains") else []
    )

    # Get frequency results (as a dict for lookup)
    freq_persons = frequency_based_matching(
        all_results,
        top_k=top_k * 3,
        min_appearances=min_appearances,
        min_unique_frames=min_unique_frames,
        max_distance=max_distance,
        min_confidence=min_confidence,
        _match_to_response=_match_to_response,
        _distance_to_confidence=_distance_to_confidence,
        performer_link_index=performer_link_index,
        endpoint_priority_domains=endpoint_priority_domains,
    )
    # Keyed by canonical identity (see _canonical_identity), not each
    # PersonResult's own raw stashdb_id -- otherwise two linked-but-
    # distinct records (or a local-index match and its own linked
    # stashdb.org entry) combine correctly inside frequency_based_matching
    # itself but split right back apart here when merged against the
    # cluster-mode results below.
    freq_by_id = {
        _canonical_identity(p.best_match, performer_link_index): p
        for p in freq_persons if p.best_match
    }

    # Get cluster results
    clusters = cluster_faces_by_person(all_results, recognizer, cluster_threshold)
    clusters = merge_clusters_by_match(clusters, performer_link_index=performer_link_index)

    cluster_persons = []
    for cluster in clusters:
        aggregated = aggregate_matches(
            cluster, top_k=3, frame_timestamps=frame_timestamps,
            _match_to_response=_match_to_response,
            _distance_to_confidence=_distance_to_confidence,
            performer_link_index=performer_link_index,
            endpoint_priority_domains=endpoint_priority_domains,
        )
        if aggregated:
            cluster_persons.append({
                "stashdb_id": _canonical_identity(aggregated[0], performer_link_index),
                "name": aggregated[0].name,
                "frame_count": len(cluster),
                "distance": aggregated[0].distance,
                "match": aggregated[0],
                "top_timestamps_sec": aggregated[0].top_timestamps_sec,
            })

    cluster_by_id = {p["stashdb_id"]: p for p in cluster_persons}

    # Combine results
    all_performers = set(freq_by_id.keys()) | set(cluster_by_id.keys())

    combined_scores = []
    for stashdb_id in all_performers:
        freq_result = freq_by_id.get(stashdb_id)
        cluster_result = cluster_by_id.get(stashdb_id)

        # Determine best distance and frame count
        if freq_result and cluster_result:
            # Found by both - use best distance, combine frame counts
            best_distance = min(freq_result.best_match.distance, cluster_result["distance"])
            frame_count = max(freq_result.frame_count, cluster_result["frame_count"])
            found_by = "both"
            # Strong boost for being found by both methods (high confidence signal)
            confidence_boost = 0.25
        elif freq_result:
            best_distance = freq_result.best_match.distance
            frame_count = freq_result.frame_count
            found_by = "frequency"
            confidence_boost = 0.0
        else:
            best_distance = cluster_result["distance"]
            frame_count = cluster_result["frame_count"]
            found_by = "cluster"
            confidence_boost = 0.0

        # Calculate hybrid score: confidence with boost, plus small frame bonus
        base_confidence = 1 - best_distance
        boosted_confidence = min(1.0, base_confidence + confidence_boost)
        frame_bonus = 0.05 * (frame_count - 1)  # Small bonus per additional frame
        hybrid_score = boosted_confidence * (1 + frame_bonus)

        # Get the match object
        if freq_result:
            match_obj = freq_result.best_match
        else:
            match_obj = cluster_result["match"]

        combined_scores.append({
            "stashdb_id": stashdb_id,
            "name": match_obj.name,
            "frame_count": frame_count,
            "distance": best_distance,
            "hybrid_score": hybrid_score,
            "found_by": found_by,
            "match": match_obj,
            # Only the cluster component ever resolves real timestamps
            # (see aggregate_matches) -- kept independently of which
            # match_obj won for display fields, so a performer found by
            # both methods doesn't lose jump-to-frame data just because
            # frequency's match object was preferred above.
            "top_timestamps_sec": cluster_result["top_timestamps_sec"] if cluster_result else [],
        })

    # Sort by hybrid score (higher is better)
    combined_scores.sort(key=lambda p: p["hybrid_score"], reverse=True)

    # Filter by minimum confidence and frame requirements
    filtered_scores = [
        p for p in combined_scores
        if (1 - p["distance"]) >= min_confidence and p["frame_count"] >= min_unique_frames
    ]

    # Heuristic: limit max performers based on number of clusters
    # A scene with 3 face clusters probably has 2-3 performers, not 10
    max_performers = max(2, min(top_k, len(clusters)))
    filtered_scores = filtered_scores[:max_performers]

    # Convert to PersonResult format
    persons = []
    for i, p in enumerate(filtered_scores):
        match = p["match"]
        resp = _match_to_response(
            match,
            confidence=_distance_to_confidence(p["distance"]),
            distance=p["distance"],
            top_timestamps_sec=p["top_timestamps_sec"],
        )
        persons.append(PersonResult(
            person_id=i,
            frame_count=p["frame_count"],
            best_match=resp,
            all_matches=[resp],
        ))

    return persons

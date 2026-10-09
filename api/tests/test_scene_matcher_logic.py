"""Tests for scene_matcher.py pure functions - cosine distance and cluster merging."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

# scene_matcher does `from recognizer import PerformerMatch` at import time, so recognizer is mocked for exactly
# that import and then put back. Leaving the Mock in sys.modules (as this file used to) made every test that imports
# the real recognizer AFTER this file is collected get a Mock PerformerMatch -- order-dependent failures.
_previous_recognizer = sys.modules.get('recognizer')
sys.modules['recognizer'] = Mock()
try:
    from scene_matcher import _cosine_distance, merge_clusters_by_match, hybrid_matching, clustered_frequency_matching
finally:
    if _previous_recognizer is None:
        del sys.modules['recognizer']
    else:
        sys.modules['recognizer'] = _previous_recognizer


class TestCosineDistance:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(_cosine_distance(v, v), abs=1e-6) == 0.0

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert pytest.approx(_cosine_distance(a, b), abs=1e-6) == 1.0

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert pytest.approx(_cosine_distance(a, b), abs=1e-6) == 2.0

    def test_zero_vector_returns_one(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _cosine_distance(a, b) == 1.0

    def test_both_zero_vectors(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert _cosine_distance(a, b) == 1.0

    def test_similar_vectors_small_distance(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1.0, 1.1])
        dist = _cosine_distance(a, b)
        assert dist < 0.01  # Very close vectors


def _make_match(stashdb_id, combined_score, universal_id=None, local_performer_id=None):
    """Create a mock match object. universal_id defaults to a stashdb.org-
    shaped id derived from stashdb_id (same convention real PerformerMatch
    objects use) so two _make_match calls with the same stashdb_id keep
    comparing equal on universal_id too -- pass an explicit universal_id to
    simulate two different database records (e.g. a linked duplicate).
    local_performer_id defaults to None (matching PerformerMatch's own
    dataclass default) -- a bare Mock() would otherwise auto-vivify a
    truthy attribute here, which _canonical_identity's local-index-link
    check reads as "this looks like a stash_id-linked local match" even
    for a match never meant to simulate one. matched_embedding_index
    defaults to None the same way -- aggregate_matches reads it straight
    off each match into top_timestamp_embedding_indices, which _resp
    below validates through a real PerformerMatchResponse (a Pydantic
    list[Optional[int]] field); an auto-vivified Mock there fails that
    validation instead of a plain AttributeError, same risk as
    local_performer_id already guards against."""
    match = Mock()
    match.stashdb_id = stashdb_id
    match.combined_score = combined_score
    match.universal_id = universal_id or f"stashdb.org:{stashdb_id}"
    match.local_performer_id = local_performer_id
    match.matched_embedding_index = None
    return match


def _make_result(matches):
    """Create a mock RecognitionResult with given matches."""
    result = Mock()
    result.matches = matches
    return result


class TestMergeClusters:
    def test_single_cluster_unchanged(self):
        match = _make_match("perf-1", 0.2)
        result = _make_result([match])
        clusters = [[(0, result)]]

        merged = merge_clusters_by_match(clusters)

        assert len(merged) == 1
        assert len(merged[0]) == 1

    def test_two_clusters_same_match_merged(self):
        match1 = _make_match("perf-1", 0.2)
        result1 = _make_result([match1])
        match2 = _make_match("perf-1", 0.3)
        result2 = _make_result([match2])

        clusters = [[(0, result1)], [(1, result2)]]

        merged = merge_clusters_by_match(clusters)

        # Should merge into one cluster with 2 entries
        assert len(merged) == 1
        assert len(merged[0]) == 2

    def test_two_clusters_different_matches_unchanged(self):
        match1 = _make_match("perf-1", 0.2)
        result1 = _make_result([match1])
        match2 = _make_match("perf-2", 0.3)
        result2 = _make_result([match2])

        clusters = [[(0, result1)], [(1, result2)]]

        merged = merge_clusters_by_match(clusters)

        assert len(merged) == 2

    def test_clusters_with_no_matches_preserved(self):
        result_no_match = _make_result([])
        match = _make_match("perf-1", 0.2)
        result_with_match = _make_result([match])

        clusters = [[(0, result_no_match)], [(1, result_with_match)]]

        merged = merge_clusters_by_match(clusters)

        # Both should be preserved (no-match cluster is kept separately)
        assert len(merged) == 2

    def test_empty_list_returns_empty(self):
        assert merge_clusters_by_match([]) == []

    def test_three_clusters_two_same_one_different(self):
        match_a1 = _make_match("perf-1", 0.2)
        result_a1 = _make_result([match_a1])
        match_a2 = _make_match("perf-1", 0.25)
        result_a2 = _make_result([match_a2])
        match_b = _make_match("perf-2", 0.3)
        result_b = _make_result([match_b])

        clusters = [[(0, result_a1)], [(1, result_a2)], [(2, result_b)]]

        merged = merge_clusters_by_match(clusters)

        # perf-1 clusters merge, perf-2 stays separate
        assert len(merged) == 2
        sizes = sorted(len(c) for c in merged)
        assert sizes == [1, 2]

    def test_merge_picks_best_score_across_cluster(self):
        # Both results in a cluster have different best matches, but
        # the cluster's best is whichever has lowest combined_score
        match1 = _make_match("perf-1", 0.4)
        match2 = _make_match("perf-2", 0.1)  # Better score
        result1 = _make_result([match1])
        result2 = _make_result([match2])

        clusters = [[(0, result1), (1, result2)]]

        merged = merge_clusters_by_match(clusters)

        assert len(merged) == 1

    def test_linked_but_different_records_unmerged_without_link_index(self):
        # perf-1 and perf-2 are different database records; with no
        # performer_link_index at all, they're just two different people.
        match1 = _make_match("perf-1", 0.2, universal_id="stashdb.org:perf-1")
        match2 = _make_match("perf-2", 0.3, universal_id="local:perf-2")
        result1 = _make_result([match1])
        result2 = _make_result([match2])

        merged = merge_clusters_by_match([[(0, result1)], [(1, result2)]])

        assert len(merged) == 2

    def test_linked_records_merged_via_performer_link_index(self):
        # Same two clusters as above, but perf-1/perf-2 are now known (via
        # stash-sense2-data-gen's own performer_links.json) to be the same
        # real person catalogued under two different records -- e.g.
        # "Elma" and "Sonya Chrystal" being the same performer. They must
        # merge into one cluster/person instead of showing as two.
        match1 = _make_match("perf-1", 0.2, universal_id="stashdb.org:perf-1")
        match2 = _make_match("perf-2", 0.3, universal_id="local:perf-2")
        result1 = _make_result([match1])
        result2 = _make_result([match2])
        link_index = {
            "stashdb.org:perf-1": ["local:perf-2"],
            "local:perf-2": ["stashdb.org:perf-1"],
        }

        merged = merge_clusters_by_match(
            [[(0, result1)], [(1, result2)]], performer_link_index=link_index,
        )

        assert len(merged) == 1
        assert len(merged[0]) == 2

    def test_local_index_match_merges_with_its_own_linked_stashdb_entry(self):
        # A local-index match (universal_id "local:<id>") whose stashdb_id
        # has been resolved to a real linked StashDB uuid (recognizer.py's
        # own PerformerMatch construction for a stash_id-linked local
        # performer) must merge with the main index's own entry for that
        # same stashdb_id -- no performer_link_index needed at all, this
        # is the separate local_performer_index-linkage signal
        # matching.py's merge_local_candidates() already handles within a
        # single face's own candidate list. Confirmed live: "Marley Brinx"
        # via the local index and via stashdb.org directly never merged.
        match_local = _make_match(
            "23d3aa04-uuid", 0.38, universal_id="local:2519", local_performer_id="2519",
        )
        match_stashdb = _make_match("23d3aa04-uuid", 0.47, universal_id="stashdb.org:23d3aa04-uuid")
        result1 = _make_result([match_local])
        result2 = _make_result([match_stashdb])

        merged = merge_clusters_by_match([[(0, result1)], [(1, result2)]])

        assert len(merged) == 1
        assert len(merged[0]) == 2

    def test_unlinked_local_match_with_same_bare_id_stays_separate(self):
        # stashdb_id falling back to the bare local id (recognizer.py's
        # "not actually linked" convention) must NOT be treated as a link
        # just because it happens to equal a real stashdb.org match's own
        # id string.
        match_local = _make_match("2519", 0.38, universal_id="local:2519", local_performer_id="2519")
        match_stashdb = _make_match("2519", 0.47, universal_id="stashdb.org:2519")
        result1 = _make_result([match_local])
        result2 = _make_result([match_stashdb])

        merged = merge_clusters_by_match([[(0, result1)], [(1, result2)]])

        assert len(merged) == 2


def _resp(m, **overrides):
    """Minimal stand-in for identification_router._match_to_response.
    Returns a real PerformerMatchResponse (not a dict/SimpleNamespace):
    hybrid_matching's cluster component reads attributes straight off
    aggregate_matches's returned list (never routed through pydantic
    coercion), while frequency_based_matching's PersonResult(best_match=...)
    needs an actual PerformerMatchResponse instance -- only a real instance
    satisfies both."""
    from identification_router import PerformerMatchResponse
    # Derive the default endpoint from m's own universal_id (falling back
    # to stashdb.org for a bare Mock with no real one set) so
    # match_universal_id() reconstructs the SAME id hybrid_matching's own
    # keying expects -- a hardcoded default here previously made every
    # response reconstruct to "stashdb.org:<stashdb_id>" regardless of
    # which source (e.g. a linked pornbox record) actually won, which
    # silently broke both the dict-keying (every performer collapsed onto
    # the same key) and endpoint-priority tests (the winning match's own
    # source domain must survive into the response to be checked at all).
    # Mirrors identification_router._match_to_response's own
    # `endpoint=_extract_endpoint(uid) or getattr(m, "endpoint", None)`
    # fallback -- needed for hybrid_matching's "found by both" combine
    # path, which re-wraps an ALREADY-BUILT PerformerMatchResponse (from
    # freq_by_id/cluster_by_id) through this same function a second time;
    # that response has no real universal_id (see below), so without this
    # fallback its own already-correct .endpoint would silently be
    # discarded and replaced with the "stashdb.org" default -- confirmed
    # live, 2026-09-28: this exact gap made
    # test_unlinked_local_wins_over_catalogue_even_with_a_priority_list_configured
    # fail even though every earlier stage (frequency_based_matching,
    # each cluster) already correctly resolved "local".
    default_endpoint = "stashdb.org"
    uid = getattr(m, "universal_id", None)
    if isinstance(uid, str) and ":" in uid:
        default_endpoint = uid.split(":", 1)[0]
    else:
        default_endpoint = getattr(m, "endpoint", None) or default_endpoint
    return PerformerMatchResponse(
        stashdb_id=m.stashdb_id, name=m.name,
        endpoint=overrides.get("endpoint", default_endpoint),
        confidence=overrides.get("confidence", 0.0),
        distance=overrides.get("distance", 0.0),
        top_timestamps_sec=overrides.get("top_timestamps_sec", []),
        # Also mirrors _match_to_response's own `universal_id=uid` --
        # preserved through a double-wrap the same way `endpoint` is now
        # (see above), so a THIRD wrap (not currently exercised, but no
        # reason to leave the same trap for later) would work correctly
        # too. isinstance-guarded the same way default_endpoint's own uid
        # check is: a bare Mock's auto-vivified non-string uid must never
        # reach a str-typed pydantic field.
        universal_id=uid if isinstance(uid, str) else overrides.get("universal_id"),
        # Must round-trip for _canonical_identity's local-index-link check
        # (match_universal_id() reads this off the response, prioritized
        # over endpoint+stashdb_id) -- a local match's own universal_id
        # ("local:<id>") could otherwise never be reconstructed correctly.
        local_performer_id=overrides.get("local_performer_id", getattr(m, "local_performer_id", None)),
    )


def _conf(distance):
    return max(0.0, 1.0 - distance)


def _embedded_result(matches, vector):
    """Like _make_result, but with a real embedding vector so
    cluster_faces_by_person's cosine-distance clustering has something
    real to compare (a bare Mock() would fail the np.mean/dot math).

    Also carries a minimal `.face` (aggregate_matches reads result.face
    unconditionally, not just when frame_timestamps is given, to pair
    each frame with its own bbox/matched_embedding_index) -- a bare
    SimpleNamespace with no such attribute raises AttributeError the
    instant any test here exercises more than one frame, which every
    multi-result test in this file does. rotation_applied=0.0 keeps the
    bbox on the plain (non-rotated) path; no test in this file exercises
    _derotate_bbox itself, that's test_scene_matcher_logic's own
    rotation-specific coverage if/when added."""
    face = SimpleNamespace(
        bbox={"x": 0, "y": 0, "w": 10, "h": 10, "rotation_applied": 0.0},
        confidence=0.9,
    )
    return SimpleNamespace(matches=matches, face=face, embedding=SimpleNamespace(embedding=np.array(vector)))


class TestHybridMatchingFrameTimestamps:
    """hybrid_matching's cluster component can resolve real
    top_timestamps_sec (via aggregate_matches, given frame_timestamps) --
    regression coverage for the "jump to frame" buttons silently going
    empty once Face Recommendations/Face Identification switched to
    matching_mode="hybrid" without ever threading frame_timestamps through
    this function at all."""

    def test_cluster_only_match_gets_real_timestamps(self):
        # A single performer, two frames close together in embedding
        # space (so they cluster into one person). Whether
        # frequency_based_matching also finds them (found_by="both" vs
        # "cluster") doesn't matter here -- the fix preserves the cluster
        # component's timestamps in both cases (see combined_scores'
        # "top_timestamps_sec" key), so this only asserts the observable
        # result, not which internal branch produced it.
        match_a = _make_match("uuid-1", 0.10)
        match_a.name = "Renee Rose"
        match_b = _make_match("uuid-1", 0.20)
        match_b.name = "Renee Rose"
        result_a = _embedded_result([match_a], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_b], [0.99, 0.01, 0.0])
        all_results = [(0, result_a), (1, result_b)]
        frame_timestamps = {0: 12.5, 1: 18.0}

        persons = hybrid_matching(
            all_results, recognizer=None,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            frame_timestamps=frame_timestamps,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.stashdb_id == "uuid-1"
        assert persons[0].best_match.top_timestamps_sec == [12.5, 18.0]

    def test_omitting_frame_timestamps_keeps_prior_behavior(self):
        match_a = _make_match("uuid-1", 0.10)
        match_a.name = "Renee Rose"
        result_a = _embedded_result([match_a], [1.0, 0.0, 0.0])

        persons = hybrid_matching(
            [(0, result_a)], recognizer=None,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.top_timestamps_sec == []


class TestHybridMatchingLinkedPerformers:
    """Regression coverage for the reported bug: a real person catalogued
    under two different database records (stash-sense2-data-gen's own
    non-destructive performer_links.json, e.g. "Elma" and "Sonya
    Chrystal" turning out to be the same performer) split into two
    separate low-frame-count "persons" in the Face Recommendations UI
    because neither cluster-merging nor the frequency/cluster combine
    step knew the two records were linked -- only a single face's own
    ranked candidate list ever got that treatment (matching.py's
    collapse_linked_candidates)."""

    def _two_far_apart_clusters(self):
        # Orthogonal vectors -> cosine distance 1.0, well past the 0.6
        # clustering threshold, so these are always two separate clusters
        # before any linking-aware merge is applied.
        match_a = _make_match("perf-1", 0.10, universal_id="stashdb.org:perf-1")
        match_a.name = "Elma"
        match_b = _make_match("perf-2", 0.15, universal_id="stashdb.org:perf-2")
        match_b.name = "Sonya Chrystal"
        result_a = _embedded_result([match_a], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_b], [0.0, 1.0, 0.0])
        return [(0, result_a), (1, result_b)]

    def test_unlinked_records_stay_separate_persons(self):
        persons = hybrid_matching(
            self._two_far_apart_clusters(), recognizer=None,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 2

    def test_linked_records_merge_into_one_person(self):
        recognizer = SimpleNamespace(performer_link_index={
            "stashdb.org:perf-1": ["stashdb.org:perf-2"],
            "stashdb.org:perf-2": ["stashdb.org:perf-1"],
        })

        persons = hybrid_matching(
            self._two_far_apart_clusters(), recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].frame_count == 2


class TestLinkedGroupEndpointPriority:
    """Within a merged linked group, the display candidate must follow the
    user's configured stash-box endpoint priority order (Settings > ... >
    Endpoint priority) -- e.g. a stashdb.org record should win over a
    pornbox record for the same real person -- the same rule
    matching.py's collapse_linked_candidates already applies to a single
    face's own candidate list, now also applied to the cross-frame/
    cross-cluster tallying in this module."""

    def _elma_and_pornbox_sonya(self):
        # Sonya's (pornbox) match scores closer (lower combined_score) in
        # its one frame than Elma's (stashdb.org) does in its own frame --
        # priority must still prefer Elma once a priority list is given.
        match_a = _make_match("elma-id", 0.30, universal_id="stashdb.org:elma-id")
        match_a.name = "Elma"
        match_b = _make_match("sonya-id", 0.10, universal_id="pornbox:sonya-id")
        match_b.name = "Sonya Chrystal"
        result_a = _embedded_result([match_a], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_b], [0.0, 1.0, 0.0])
        return [(0, result_a), (1, result_b)]

    def _link_index(self):
        return {
            "stashdb.org:elma-id": ["pornbox:sonya-id"],
            "pornbox:sonya-id": ["stashdb.org:elma-id"],
        }

    def test_falls_back_to_best_score_without_a_priority_list(self):
        recognizer = SimpleNamespace(performer_link_index=self._link_index())

        persons = hybrid_matching(
            self._elma_and_pornbox_sonya(), recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Sonya Chrystal"

    def test_stashdb_wins_over_pornbox_via_endpoint_priority(self):
        recognizer = SimpleNamespace(
            performer_link_index=self._link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org", "theporndb.net"],
        )

        persons = hybrid_matching(
            self._elma_and_pornbox_sonya(), recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Elma"

    def test_unlinked_local_wins_over_catalogue_even_with_a_priority_list_configured(self):
        # The Dianaholiday report (2026-09-28), via hybrid_matching (the
        # mode every bulk/cached identify call actually uses) rather than
        # clustered_frequency_matching (already covered in
        # test_matching_logic.py/TestClusteredFrequencyMatchingLocalCatalogueLink)
        # -- an UNLINKED local performer (no stash_id, created directly
        # from a catalogue profile) grouped with her own catalogue entry
        # via local_catalogue_link_index (matching.build_local_catalogue_
        # link_index), NOT performer_link_index. Must always win, even
        # over a much better-scoring catalogue candidate and even with a
        # stashbox priority list configured (which doesn't include
        # "local" at all -- the point of this test is that local wins
        # unconditionally, not by ranking into that list).
        match_local = _make_match(
            "2915", 0.40, universal_id="local:2915", local_performer_id="2915",
        )
        match_local.name = "Dianaholiday"
        match_catalogue = _make_match("465878", 0.34, universal_id="babepedia:465878")
        match_catalogue.name = "Dianaholiday"
        result_a = _embedded_result([match_local], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_catalogue], [0.0, 1.0, 0.0])

        # match_local/match_catalogue's embedding vectors are deliberately
        # orthogonal (won't cosine-cluster together), so hybrid_matching's
        # own cluster-mode component sees them as two SEPARATE clusters --
        # each independently canonicalizes to the same linked-group key
        # (_canonical_identity), and each independently runs its own
        # _substitute_linked_priority_winner. A real FaceRecognizer always
        # has build_linked_substitute_match (added as a bound method), so
        # both clusters converge on the same "local wins" answer; a mock
        # missing it would leave one cluster un-substituted and expose a
        # real but separate ordering quirk in how cluster_by_id collapses
        # same-key clusters -- not what this test is about, so provide it.
        def build_substitute(universal_id, template):
            assert universal_id == "local:2915"
            return SimpleNamespace(
                universal_id=universal_id, stashdb_id="2915", name="Dianaholiday",
                country=None, image_url=None, local_performer_id="2915",
                source=None, catalogue_url=None, profile_url=None,
                # template is a Mock (from _make_match) that never had .distance
                # explicitly set -- reading it back would just auto-vivify
                # another Mock, not a real number, so use a literal instead
                # (matches the catalogue candidate's own known score, 0.34).
                distance=0.34, combined_score=0.34, matched_embedding_index=None, original_name=None,
            )

        recognizer = SimpleNamespace(
            performer_link_index={},
            local_catalogue_link_index={"local:2915": ["babepedia:465878"], "babepedia:465878": ["local:2915"]},
            _endpoint_priority_domains=lambda: ["stashdb.org"],
            build_linked_substitute_match=build_substitute,
        )

        persons = hybrid_matching(
            [(0, result_a), (1, result_b)], recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        # PerformerMatchResponse (_resp) has no real universal_id field --
        # same convention as TestHybridMatchingLocalIndexStashdbLink's own
        # test_stashdb_entry_wins_display_over_the_local_entry below,
        # checking .endpoint/.local_performer_id instead.
        assert persons[0].best_match.endpoint == "local"
        assert persons[0].best_match.local_performer_id == "2915"
        assert persons[0].best_match.name == "Dianaholiday"


class TestLinkedGroupSubstituteWinnerNeverFound:
    """The "Eden Petty / Lindsey" report (2026-09-28): a matched pornbox
    candidate's OWN performer_link_index group lists a stashdb.org entry
    for the same real person, but that stashdb.org entry's own reference
    photo never independently ranked as a candidate in any frame of this
    scene at all -- so there's nothing for _pick_priority_match to choose
    between (only one side was ever found). Without a substitute, a
    pornbox-only performer would get created in Stash for someone who may
    already exist there (or get added separately later) as Eden Petty,
    exactly the duplicate the dataset's own link already knows to avoid.
    See recognizer.build_linked_substitute_match / scene_matcher.
    _substitute_linked_priority_winner for the fix."""

    def _lindsey_only(self):
        match = _make_match("180730", 0.485, universal_id="pornbox:180730")
        match.name = "Lindsey"
        result = _embedded_result([match], [1.0, 0.0, 0.0])
        return [(0, result)]

    def _link_index(self):
        return {
            "pornbox:180730": ["stashdb.org:13b6304a-uuid"],
            "stashdb.org:13b6304a-uuid": ["pornbox:180730"],
        }

    def _recognizer(self):
        def build_substitute(universal_id, template):
            # A plain duck-typed stand-in, not a real recognizer.
            # PerformerMatch -- this file replaces the whole `recognizer`
            # module with a Mock() at import time (see the top of this
            # file) specifically so scene_matcher.py's own module-level
            # `from recognizer import PerformerMatch` doesn't pull in the
            # real ML-heavy recognizer.py; importing the real class here
            # would resolve to that same Mock module and silently return
            # a MagicMock instead of a real instance (confirmed live,
            # 2026-09-28: exactly this mistake shipped in this test's
            # first version, caught by CI, not by local verification).
            assert universal_id == "stashdb.org:13b6304a-uuid"
            return SimpleNamespace(
                universal_id=universal_id, stashdb_id="13b6304a-uuid", name="Eden Petty",
                country="US", image_url="https://stashdb.org/images/xyz.jpg",
                local_performer_id=None, source=None, catalogue_url=None, profile_url=None,
                # template is the ORIGINAL (pornbox) match's own Mock --
                # carrying over its measured score the same way the real
                # recognizer.build_linked_substitute_match does, but as a
                # literal (the Mock never had .distance/.combined_score
                # explicitly set, so reading them back would just yield
                # more auto-vivified Mocks, not template's real 0.485).
                distance=0.485, combined_score=0.485, matched_embedding_index=None, original_name=None,
            )

        return SimpleNamespace(
            performer_link_index=self._link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org", "theporndb.net"],
            build_linked_substitute_match=build_substitute,
        )

    def test_hybrid_matching_substitutes_the_never_found_stashdb_entry(self):
        persons = hybrid_matching(
            self._lindsey_only(), recognizer=self._recognizer(),
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Eden Petty"
        assert persons[0].best_match.endpoint == "stashdb.org"
        # Original detection's own distance is carried over, not fabricated.
        assert persons[0].best_match.distance == 0.485

    def test_clustered_frequency_matching_substitutes_the_never_found_stashdb_entry(self):
        persons = clustered_frequency_matching(
            self._lindsey_only(), self._recognizer(),
            max_distance=0.5, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Eden Petty"

    def test_no_substitution_without_a_recognizer_with_the_builder(self):
        # A bare recognizer (e.g. no build_linked_substitute_match, same
        # shape every OTHER test in this file's SimpleNamespace mocks
        # already use) must not crash -- falls back to showing whatever
        # was actually found.
        recognizer = SimpleNamespace(
            performer_link_index=self._link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org"],
        )

        persons = hybrid_matching(
            self._lindsey_only(), recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Lindsey"

    def test_no_substitution_when_the_better_member_was_already_found(self):
        # Both sides present -- _pick_priority_match alone already picks
        # the right one; the substitute builder must never even be
        # consulted (would raise if it were, proving this).
        lindsey = _make_match("180730", 0.10, universal_id="pornbox:180730")
        lindsey.name = "Lindsey"
        eden = _make_match("13b6304a-uuid", 0.50, universal_id="stashdb.org:13b6304a-uuid")
        eden.name = "Eden Petty"
        result_a = _embedded_result([lindsey], [1.0, 0.0, 0.0])
        result_b = _embedded_result([eden], [0.0, 1.0, 0.0])

        def build_substitute(universal_id, template):
            raise AssertionError("should never be called -- both sides were already found")

        recognizer = SimpleNamespace(
            performer_link_index=self._link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org"],
            build_linked_substitute_match=build_substitute,
        )

        persons = hybrid_matching(
            [(0, result_a), (1, result_b)], recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Eden Petty"


class TestHybridMatchingLocalIndexStashdbLink:
    """End-to-end regression coverage (via hybrid_matching itself, not
    just merge_clusters_by_match in isolation) for the reported "Marley
    Brinx" case: a local-index match and the main index's own StashDB
    entry for the exact same real person, requiring no
    performer_link_index at all -- purely local_performer_index's own
    stash_id link (recognizer.py resolves a linked local match's
    stashdb_id to the real StashDB uuid, while its universal_id stays
    "local:<id>")."""

    def _local_and_stashdb_clusters(self):
        match_local = _make_match(
            "23d3aa04-uuid", 0.38, universal_id="local:2519", local_performer_id="2519",
        )
        match_local.name = "Marley Brinx"
        match_stashdb = _make_match("23d3aa04-uuid", 0.47, universal_id="stashdb.org:23d3aa04-uuid")
        match_stashdb.name = "Marley Brinx"
        result_a = _embedded_result([match_local], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_stashdb], [0.0, 1.0, 0.0])
        return [(0, result_a), (1, result_b)]

    def test_local_and_stashdb_entries_merge_into_one_person(self):
        persons = hybrid_matching(
            self._local_and_stashdb_clusters(), recognizer=None,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].frame_count == 2

    def test_stashdb_entry_wins_display_over_the_local_entry(self):
        # Once merged, the real stashdb.org record should be shown (has a
        # real image_url/link) rather than the local-index stand-in --
        # same endpoint-priority preference as a data-gen-linked group.
        recognizer = SimpleNamespace(
            performer_link_index={}, _endpoint_priority_domains=lambda: ["stashdb.org"],
        )

        persons = hybrid_matching(
            self._local_and_stashdb_clusters(), recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.endpoint == "stashdb.org"

    def test_unlinked_local_match_with_same_bare_id_stays_separate(self):
        match_local = _make_match("2519", 0.38, universal_id="local:2519", local_performer_id="2519")
        match_local.name = "Some Local Performer"
        match_stashdb = _make_match("2519", 0.47, universal_id="stashdb.org:2519")
        match_stashdb.name = "Unrelated StashDB Performer"
        result_a = _embedded_result([match_local], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_stashdb], [0.0, 1.0, 0.0])

        persons = hybrid_matching(
            [(0, result_a), (1, result_b)], recognizer=None,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 2


class TestLocalLinkChainedWithDataGenLink:
    """Regression coverage for the "Sylwia/Zdenka" report: a local-index
    match linked to a StashDB entry (local_performer_index's own signal)
    that ALSO belongs to a stash-sense2-data-gen performer_link_index
    group with a catalogue (pornbox) record -- the two link mechanisms
    chained together, not just each in isolation. Confirmed live: this
    combination merged into one person, but display picked whichever of
    the local/pornbox candidates scored better on a given run, because
    _pick_priority_match ranked the local candidate by its literal
    "local" endpoint (never a configured priority domain) instead of its
    true linked stashdb.org endpoint -- so the winner was effectively
    decided by raw score, not priority, flipping unpredictably."""

    def _sylwia_local_and_zdenka_pornbox(self, local_score=0.10, pornbox_score=0.50):
        # Scores deliberately swapped between the two tests below -- the
        # point is priority must win regardless of which one scores
        # better.
        match_local = _make_match(
            "e317d8ea-uuid", local_score, universal_id="local:2846", local_performer_id="2846",
        )
        match_local.name = "Sylwia"
        match_pornbox = _make_match("232515", pornbox_score, universal_id="pornbox:232515")
        match_pornbox.name = "Zdenka"
        result_a = _embedded_result([match_local], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_pornbox], [0.0, 1.0, 0.0])
        return [(0, result_a), (1, result_b)]

    def _link_index(self):
        return {
            "stashdb.org:e317d8ea-uuid": ["pornbox:232515"],
            "pornbox:232515": ["stashdb.org:e317d8ea-uuid"],
        }

    def test_merges_into_one_person_regardless_of_which_scores_better(self):
        recognizer = SimpleNamespace(performer_link_index=self._link_index())

        persons = hybrid_matching(
            self._sylwia_local_and_zdenka_pornbox(), recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].frame_count == 2

    def test_stashdb_linked_local_wins_even_when_pornbox_scores_better(self):
        recognizer = SimpleNamespace(
            performer_link_index=self._link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org"],
        )
        # pornbox scores much better (lower distance) than the local
        # candidate here -- priority must still win.
        clusters = self._sylwia_local_and_zdenka_pornbox(local_score=0.50, pornbox_score=0.10)

        persons = hybrid_matching(
            clusters, recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Sylwia"

    def test_falls_back_to_best_score_without_a_priority_list(self):
        recognizer = SimpleNamespace(performer_link_index=self._link_index())
        clusters = self._sylwia_local_and_zdenka_pornbox(local_score=0.50, pornbox_score=0.10)

        persons = hybrid_matching(
            clusters, recognizer=recognizer,
            min_appearances=1, min_unique_frames=1, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Zdenka"


class TestMergeClustersDominantIdentity:
    """Regression coverage for merge_clusters_by_match's dominant-identity
    grouping (fixed 2026-09-14, replacing a same-day transitive
    union-find version that shipped and then had to be reverted --
    confirmed live to over-merge). Each cluster's own aggregate winner
    (best weighted min-distance-plus-frame-bonus score across ITS OWN
    frames, same ranking aggregate_matches itself uses) decides which
    group it joins -- not any single incidental frame-level candidate
    anywhere in it, and never transitively through a third cluster."""

    def test_merges_when_dominant_identities_agree(self):
        # Cluster A's own best-weighted identity is the linked
        # "stashdb.org:perf-1"; cluster B's is "pornbox:perf-2". Linked
        # -> must merge (this is the ORIGINAL "Elma/Sonya Chrystal" bug
        # this whole feature exists to fix).
        match_a = _make_match("perf-1", 0.20, universal_id="stashdb.org:perf-1")
        match_a.name = "Elma"
        match_b = _make_match("perf-2", 0.25, universal_id="pornbox:perf-2")
        match_b.name = "Sonya Chrystal"

        clusters = [[(0, _make_result([match_a]))], [(1, _make_result([match_b]))]]
        link_index = {
            "stashdb.org:perf-1": ["pornbox:perf-2"],
            "pornbox:perf-2": ["stashdb.org:perf-1"],
        }

        merged = merge_clusters_by_match(clusters, performer_link_index=link_index)

        assert len(merged) == 1
        assert len(merged[0]) == 2

    def test_does_not_merge_on_a_weak_non_dominant_frame_alone(self):
        # Regression guard for the actual live incident: cluster A's own
        # best-scoring frame is a strong, unrelated match; a single OTHER,
        # much weaker frame in the SAME cluster happens to match a linked
        # identity cluster B is keyed on. Cluster A's own DOMINANT vote
        # is still the unrelated performer (one strong frame outweighs
        # one weak one) -- must NOT merge into cluster B's group. Confirmed
        # live: an earlier "compare every frame, merge on ANY shared
        # identity" version of this function let cluster A act as a
        # bridge, silently absorbing it into an unrelated cluster and
        # inflating a correct 4-frame person into a wrong 26-frame blob
        # mixing in half a dozen other real people.
        match_a_best = _make_match("other-id", 0.05, universal_id="stashdb.org:other-id")
        match_a_best.name = "Unrelated Performer"
        match_a_weak = _make_match("perf-1", 0.45, universal_id="stashdb.org:perf-1")
        match_a_weak.name = "Elma"
        result_a1 = _make_result([match_a_best])
        result_a2 = _make_result([match_a_weak])

        match_b = _make_match("perf-2", 0.20, universal_id="pornbox:perf-2")
        match_b.name = "Sonya Chrystal"
        result_b = _make_result([match_b])

        clusters = [[(0, result_a1), (1, result_a2)], [(2, result_b)]]
        link_index = {
            "stashdb.org:perf-1": ["pornbox:perf-2"],
            "pornbox:perf-2": ["stashdb.org:perf-1"],
        }

        merged = merge_clusters_by_match(clusters, performer_link_index=link_index)

        assert len(merged) == 2
        sizes = sorted(len(c) for c in merged)
        assert sizes == [1, 2]

    def test_does_not_transitively_bridge_through_a_third_cluster(self):
        # A's dominant identity is X. B is a mixed cluster whose OWN
        # dominant identity is Y (Y's frame scores much better than X's
        # weaker one in the same cluster), not X. C's dominant identity
        # is Y. A must NOT get pulled into the B/C group just because B
        # happens to also contain one weak frame matching X somewhere --
        # A and C share nothing, and B's own vote is for Y, not X.
        match_a = _make_match("x", 0.10, universal_id="stashdb.org:x")
        match_a.name = "X Performer"

        match_b_weak_x = _make_match("x", 0.45, universal_id="stashdb.org:x")
        match_b_weak_x.name = "X Performer"
        match_b_strong_y = _make_match("y", 0.10, universal_id="pornbox:y")
        match_b_strong_y.name = "Y Performer"

        match_c = _make_match("y", 0.15, universal_id="pornbox:y")
        match_c.name = "Y Performer"

        clusters = [
            [(0, _make_result([match_a]))],
            [(1, _make_result([match_b_weak_x])), (2, _make_result([match_b_strong_y]))],
            [(3, _make_result([match_c]))],
        ]

        merged = merge_clusters_by_match(clusters)

        assert len(merged) == 2
        sizes = sorted(len(c) for c in merged)
        assert sizes == [1, 3]  # A alone; B+C merged (both keyed on Y)

    def test_cluster_with_no_matches_has_no_dominant_identity(self):
        result_no_match = _make_result([])
        match = _make_match("perf-1", 0.2)
        result_with_match = _make_result([match])

        merged = merge_clusters_by_match([[(0, result_no_match)], [(1, result_with_match)]])

        assert len(merged) == 2


class TestClusteredFrequencyMatchingLinkedCandidates:
    """Regression coverage for the scene player's OWN default matching_mode
    ("frequency" -> clustered_frequency_matching, NOT hybrid_matching --
    confirmed live 2026-09-14 as the actual gap behind a real report: the
    "Re-identify"/"Identify Full Video" buttons never pass an explicit
    matching_mode, so every earlier fix aimed at hybrid_matching's own
    linked-candidate handling had zero effect on what those buttons
    actually show). A linked local candidate and its own linked catalogue
    duplicate, within the same merged cluster, must collapse to one
    candidate -- not show the loser as a spurious "other possible match"
    (all_matches[1:]) for a person the winner already represents."""

    def _sylwia_and_zdenka_same_cluster(self, local_score=0.10, pornbox_score=0.50):
        match_local = _make_match(
            "e317d8ea-uuid", local_score, universal_id="local:2846", local_performer_id="2846",
        )
        match_local.name = "Sylwia"
        match_pornbox = _make_match("232515", pornbox_score, universal_id="pornbox:232515")
        match_pornbox.name = "Zdenka"
        result_a = _embedded_result([match_local], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_pornbox], [0.0, 1.0, 0.0])
        return [(0, result_a), (1, result_b)]

    def _link_index(self):
        return {
            "stashdb.org:e317d8ea-uuid": ["pornbox:232515"],
            "pornbox:232515": ["stashdb.org:e317d8ea-uuid"],
        }

    def test_linked_duplicate_does_not_appear_as_an_other_possible_match(self):
        recognizer = SimpleNamespace(
            performer_link_index=self._link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org"],
        )

        persons = clustered_frequency_matching(
            self._sylwia_and_zdenka_same_cluster(), recognizer,
            max_distance=0.5, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Sylwia"
        # The whole point: Zdenka must NOT show up as an alternate --
        # she's the same real person as the winner, already represented.
        assert [m.name for m in persons[0].all_matches] == ["Sylwia"]

    def test_priority_winner_chosen_even_when_catalogue_scores_better(self):
        recognizer = SimpleNamespace(
            performer_link_index=self._link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org"],
        )
        clusters = self._sylwia_and_zdenka_same_cluster(local_score=0.45, pornbox_score=0.10)

        persons = clustered_frequency_matching(
            clusters, recognizer,
            max_distance=0.5, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        assert persons[0].best_match.name == "Sylwia"
        assert [m.name for m in persons[0].all_matches] == ["Sylwia"]

    def test_unrelated_performer_still_shows_as_an_alternate(self):
        # Sanity check the fix didn't disable "other possible matches"
        # entirely -- a genuinely different (unlinked) performer sharing
        # the same cluster must still show up as an alternate.
        match_a = _make_match("perf-1", 0.10, universal_id="stashdb.org:perf-1")
        match_a.name = "Elma"
        match_b = _make_match("perf-2", 0.20, universal_id="stashdb.org:perf-2")
        match_b.name = "Unrelated Performer"
        result = _embedded_result([match_a, match_b], [1.0, 0.0, 0.0])
        recognizer = SimpleNamespace(performer_link_index={}, _endpoint_priority_domains=lambda: [])

        persons = clustered_frequency_matching(
            [(0, result)], recognizer,
            max_distance=0.5, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        names = [m.name for m in persons[0].all_matches]
        assert names[0] == "Elma"
        assert "Unrelated Performer" in names


class TestClusteredFrequencyMatchingLocalCatalogueLink:
    """Same collapsing behavior as TestClusteredFrequencyMatchingLinkedCandidates
    above, but for recognizer.local_catalogue_link_index (a local performer
    created directly from a catalogue profile, e.g. babepedia -- no
    stash_id to link with, so performer_link_index alone can't catch it;
    see matching.py's build_local_catalogue_link_index). Confirmed live: a
    local "Dianaholiday" performer and the babepedia "Dianaholiday" she was
    created from showed as two separate Face Recommendations candidates
    despite matching name and URL exactly."""

    def _local_and_catalogue_same_cluster(self, local_score=0.10, catalogue_score=0.50):
        # stashdb_id == local_performer_id simulates an UNLINKED local
        # match (recognizer.py's own convention -- see PerformerMatch's
        # docstring: a stash_id-linked local match has them differ).
        # local_catalogue_link_index is precisely the signal for an
        # unlinked local performer, so the test must simulate one, not a
        # stash_id-linked local match (a different, already-covered case
        # -- see TestClusteredFrequencyMatchingLinkedCandidates).
        match_local = _make_match(
            "99", local_score, universal_id="local:99", local_performer_id="99",
        )
        match_local.name = "Dianaholiday"
        match_catalogue = _make_match("42", catalogue_score, universal_id="babepedia:42")
        match_catalogue.name = "Dianaholiday"
        result_a = _embedded_result([match_local], [1.0, 0.0, 0.0])
        result_b = _embedded_result([match_catalogue], [0.0, 1.0, 0.0])
        return [(0, result_a), (1, result_b)]

    def _local_catalogue_link_index(self):
        return {"local:99": ["babepedia:42"], "babepedia:42": ["local:99"]}

    def test_local_and_catalogue_duplicate_collapse_to_one_person(self):
        recognizer = SimpleNamespace(
            performer_link_index={},
            local_catalogue_link_index=self._local_catalogue_link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org"],
        )

        persons = clustered_frequency_matching(
            self._local_and_catalogue_same_cluster(), recognizer,
            max_distance=0.5, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1
        # The whole point: the babepedia duplicate must NOT show up as a
        # separate "other possible match" for the local performer.
        assert [m.name for m in persons[0].all_matches] == ["Dianaholiday"]

    def test_stronger_catalogue_score_still_collapses_to_one_person(self):
        recognizer = SimpleNamespace(
            performer_link_index={},
            local_catalogue_link_index=self._local_catalogue_link_index(),
            _endpoint_priority_domains=lambda: ["stashdb.org"],
        )
        clusters = self._local_and_catalogue_same_cluster(local_score=0.45, catalogue_score=0.10)

        persons = clustered_frequency_matching(
            clusters, recognizer,
            max_distance=0.5, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 1

    def test_absent_local_catalogue_link_index_keeps_them_separate(self):
        # Sanity check the fix is what does the collapsing -- without it
        # (a recognizer with no such attribute, matching every real
        # instance before this fix), the two must still show as two
        # separate persons, proving the test actually exercises the fix.
        recognizer = SimpleNamespace(performer_link_index={}, _endpoint_priority_domains=lambda: ["stashdb.org"])

        persons = clustered_frequency_matching(
            self._local_and_catalogue_same_cluster(), recognizer,
            max_distance=0.5, min_confidence=0.0,
            _match_to_response=_resp, _distance_to_confidence=_conf,
        )

        assert len(persons) == 2

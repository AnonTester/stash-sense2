"""Tests for scene_matcher.py pure functions - cosine distance and cluster merging."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

# Mock recognizer before importing scene_matcher
sys.modules['recognizer'] = Mock()

import numpy as np
import pytest

from scene_matcher import _cosine_distance, merge_clusters_by_match, hybrid_matching, clustered_frequency_matching


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
    for a match never meant to simulate one."""
    match = Mock()
    match.stashdb_id = stashdb_id
    match.combined_score = combined_score
    match.universal_id = universal_id or f"stashdb.org:{stashdb_id}"
    match.local_performer_id = local_performer_id
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
    default_endpoint = "stashdb.org"
    uid = getattr(m, "universal_id", None)
    if isinstance(uid, str) and ":" in uid:
        default_endpoint = uid.split(":", 1)[0]
    return PerformerMatchResponse(
        stashdb_id=m.stashdb_id, name=m.name,
        endpoint=overrides.get("endpoint", default_endpoint),
        confidence=overrides.get("confidence", 0.0),
        distance=overrides.get("distance", 0.0),
        top_timestamps_sec=overrides.get("top_timestamps_sec", []),
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
    real to compare (a bare Mock() would fail the np.mean/dot math)."""
    return SimpleNamespace(matches=matches, embedding=SimpleNamespace(embedding=np.array(vector)))


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

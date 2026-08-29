"""Tests for scene_matcher.py pure functions - cosine distance and cluster merging."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

# Mock recognizer before importing scene_matcher
sys.modules['recognizer'] = Mock()

import numpy as np
import pytest

from scene_matcher import _cosine_distance, merge_clusters_by_match, hybrid_matching


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


def _make_match(stashdb_id, combined_score):
    """Create a mock match object."""
    match = Mock()
    match.stashdb_id = stashdb_id
    match.combined_score = combined_score
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


def _resp(m, **overrides):
    """Minimal stand-in for identification_router._match_to_response.
    Returns a real PerformerMatchResponse (not a dict/SimpleNamespace):
    hybrid_matching's cluster component reads attributes straight off
    aggregate_matches's returned list (never routed through pydantic
    coercion), while frequency_based_matching's PersonResult(best_match=...)
    needs an actual PerformerMatchResponse instance -- only a real instance
    satisfies both."""
    from identification_router import PerformerMatchResponse
    return PerformerMatchResponse(
        stashdb_id=m.stashdb_id, name=m.name,
        confidence=overrides.get("confidence", 0.0),
        distance=overrides.get("distance", 0.0),
        top_timestamps_sec=overrides.get("top_timestamps_sec", []),
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

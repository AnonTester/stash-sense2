"""Tests for matching.py -- single usearch-index nearest-neighbor matching.

buffalo_l produces one embedding per face (see matching.py's own module
docstring), so there's no dual-model fusion/health-arbitration logic left
to test here -- just query_index()'s passthrough, build_matches()'s
candidate construction/filtering/dedup, and the local-performer-index
merge path (fuse_local_results()/merge_local_candidates()).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from matching import (
    CandidateMatch,
    IndexQueryResult,
    LOCAL_MATCH_BOOST,
    MatchingConfig,
    build_matches,
    fuse_local_results,
    match_face,
    merge_local_candidates,
    query_index,
)


class _MockIndex:
    """A stand-in for a usearch.index.Index -- only .search() and len()
    are exercised by matching.py. (SimpleNamespace can't stand in for
    len() itself: Python's len() looks up __len__ on the type, not the
    instance, so a plain attribute assignment doesn't satisfy it.)"""

    def __init__(self, keys, distances):
        self._keys = keys
        self._distances = distances

    def __len__(self):
        return len(self._keys)

    def search(self, embedding, k):
        return SimpleNamespace(
            keys=np.array(self._keys, dtype=np.int64),
            distances=np.array(self._distances, dtype=np.float32),
        )


def _mock_index(keys, distances):
    return _MockIndex(keys, distances)


class TestQueryIndex:
    def test_passes_through_search_results(self):
        index = _mock_index(keys=[3, 1, 2], distances=[0.1, 0.2, 0.3])

        result = query_index(np.zeros(512, dtype=np.float32), index)

        assert isinstance(result, IndexQueryResult)
        assert list(result.neighbors) == [3, 1, 2]
        assert list(result.distances) == pytest.approx([0.1, 0.2, 0.3])

    def test_respects_query_k(self):
        captured = {}
        index = SimpleNamespace(
            search=lambda embedding, k: captured.update(k=k) or SimpleNamespace(
                keys=np.array([], dtype=np.int64), distances=np.array([], dtype=np.float32),
            ),
        )
        config = MatchingConfig(query_k=42)

        query_index(np.zeros(512, dtype=np.float32), index, config)

        assert captured["k"] == 42


class TestBuildMatches:
    def _query_result(self, neighbors, distances):
        return IndexQueryResult(
            neighbors=np.array(neighbors, dtype=np.int64),
            distances=np.array(distances, dtype=np.float32),
        )

    def _faces_mapping(self, n):
        return [f"stashdb.org:uuid-{i}" for i in range(n)]

    def _performers(self, n):
        return {f"stashdb.org:uuid-{i}": {"name": f"Performer {i}"} for i in range(n)}

    def test_builds_sorted_candidates_with_confidence(self):
        qr = self._query_result([1, 0], [0.4, 0.2])
        result = build_matches(qr, self._faces_mapping(2), self._performers(2))

        assert result.candidate_count == 2
        assert [m.face_index for m in result.matches] == [0, 1]  # sorted by distance
        assert result.matches[0].combined_distance == pytest.approx(0.2)
        assert result.matches[0].confidence == pytest.approx(0.8)
        assert result.matches[0].name == "Performer 0"
        assert result.matches[0].rank == 2  # rank reflects original query order (1-indexed)

    def test_out_of_bounds_index_skipped(self):
        qr = self._query_result([0, 99], [0.2, 0.3])
        result = build_matches(qr, self._faces_mapping(1), self._performers(1))

        assert [m.face_index for m in result.matches] == [0]

    def test_null_face_mapping_entry_skipped(self):
        qr = self._query_result([0, 1], [0.2, 0.3])
        faces = ["stashdb.org:uuid-0", None]
        result = build_matches(qr, faces, self._performers(1))

        assert len(result.matches) == 1
        assert result.matches[0].universal_id == "stashdb.org:uuid-0"

    def test_max_distance_filter(self):
        config = MatchingConfig(max_distance=0.3)
        qr = self._query_result([0, 1], [0.2, 0.9])
        result = build_matches(qr, self._faces_mapping(2), self._performers(2), config)

        assert all(m.combined_distance <= 0.3 for m in result.matches)
        assert len(result.matches) == 1

    def test_max_results_truncation(self):
        config = MatchingConfig(max_results=1)
        qr = self._query_result([0, 1], [0.2, 0.3])
        result = build_matches(qr, self._faces_mapping(2), self._performers(2), config)

        assert len(result.matches) == 1
        assert result.candidate_count == 2  # count reflects pre-truncation candidates

    def test_same_performer_multiple_embeddings_collapsed_to_best(self):
        # Both face indices belong to the same performer (e.g. two training
        # crops) -- only the closer-scoring one should survive.
        faces = ["stashdb.org:uuid-0", "stashdb.org:uuid-0"]
        qr = self._query_result([0, 1], [0.5, 0.2])

        result = build_matches(qr, faces, self._performers(1))

        assert len(result.matches) == 1
        assert result.matches[0].combined_distance == pytest.approx(0.2)

    def test_unknown_performer_falls_back_to_name(self):
        qr = self._query_result([0], [0.2])
        result = build_matches(qr, self._faces_mapping(1), performers={})

        assert result.matches[0].name == "Unknown"


class TestGenderMismatchPenalty:
    """build_matches()'s soft gender-mismatch confidence penalty -- see
    matching.py's own _effective_gender/_apply_gender_penalty."""

    def _query_result(self, neighbors, distances):
        return IndexQueryResult(
            neighbors=np.array(neighbors, dtype=np.int64),
            distances=np.array(distances, dtype=np.float32),
        )

    def test_confident_mismatch_penalizes_confidence_and_distance(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {"name": "Performer 0", "gender": "MALE"}}
        config = MatchingConfig(gender_confidence_floor=0.65, gender_mismatch_penalty=0.5)

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers, config,
            query_gender="FEMALE", query_gender_confidence=0.9,
        )

        match = result.matches[0]
        assert match.confidence == pytest.approx(0.8 * 0.5)
        # combined_distance must stay consistent with the penalized
        # confidence -- ranking/threshold-filtering read combined_distance,
        # not confidence, so a penalty that only touched confidence would
        # be invisible to them.
        assert match.combined_distance == pytest.approx(1.0 - match.confidence)

    def test_agreement_not_penalized(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {"name": "Performer 0", "gender": "FEMALE"}}

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers,
            query_gender="FEMALE", query_gender_confidence=0.9,
        )

        assert result.matches[0].confidence == pytest.approx(0.8)

    def test_low_confidence_query_gender_not_penalized(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {"name": "Performer 0", "gender": "MALE"}}

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers,
            query_gender="FEMALE", query_gender_confidence=0.3,  # below default floor
        )

        assert result.matches[0].confidence == pytest.approx(0.8)

    def test_low_confidence_inferred_gender_not_penalized(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {
            "name": "Performer 0", "inferred_gender": "MALE", "inferred_gender_confidence": 0.5,
        }}

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers,
            query_gender="FEMALE", query_gender_confidence=0.9,
        )

        assert result.matches[0].confidence == pytest.approx(0.8)

    def test_confident_inferred_gender_is_used_when_source_missing(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {
            "name": "Performer 0", "inferred_gender": "MALE", "inferred_gender_confidence": 0.9,
        }}

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers,
            query_gender="FEMALE", query_gender_confidence=0.9,
        )

        assert result.matches[0].confidence == pytest.approx(0.8 * 0.5)

    def test_source_gender_takes_precedence_over_inferred(self):
        # Source says FEMALE (agrees with query) even though inferred says
        # MALE -- source must win, no penalty.
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {
            "name": "Performer 0", "gender": "FEMALE",
            "inferred_gender": "MALE", "inferred_gender_confidence": 0.99,
        }}

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers,
            query_gender="FEMALE", query_gender_confidence=0.9,
        )

        assert result.matches[0].confidence == pytest.approx(0.8)

    def test_non_binary_source_gender_not_penalized(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {"name": "Performer 0", "gender": "NON_BINARY"}}

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers,
            query_gender="FEMALE", query_gender_confidence=0.9,
        )

        assert result.matches[0].confidence == pytest.approx(0.8)

    def test_transgender_source_gender_compared_by_presentation_bucket(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {"name": "Performer 0", "gender": "TRANSGENDER_FEMALE"}}

        result = build_matches(
            qr, ["stashdb.org:uuid-0"], performers,
            query_gender="MALE", query_gender_confidence=0.9,
        )

        assert result.matches[0].confidence == pytest.approx(0.8 * 0.5)

    def test_no_query_gender_is_a_no_op(self):
        qr = self._query_result([0], [0.2])
        performers = {"stashdb.org:uuid-0": {"name": "Performer 0", "gender": "MALE"}}

        result = build_matches(qr, ["stashdb.org:uuid-0"], performers)

        assert result.matches[0].confidence == pytest.approx(0.8)


class TestFuseLocalResults:
    def _query_result(self, neighbors, distances):
        return IndexQueryResult(
            neighbors=np.array(neighbors, dtype=np.int64),
            distances=np.array(distances, dtype=np.float32),
        )

    def test_applies_local_match_boost(self):
        qr = self._query_result([7], [0.4])
        mapping = {"7": {"name": "Local Performer", "stashdb_id": None}}

        candidates = fuse_local_results(qr, mapping)

        assert len(candidates) == 1
        assert candidates[0].universal_id == "local:7"
        assert candidates[0].combined_distance == pytest.approx(0.4 * LOCAL_MATCH_BOOST)
        assert candidates[0].distance == pytest.approx(0.4)

    def test_stale_entry_not_in_mapping_skipped(self):
        qr = self._query_result([7, 8], [0.4, 0.5])
        mapping = {"7": {"name": "Still Exists", "stashdb_id": None}}

        candidates = fuse_local_results(qr, mapping)

        assert len(candidates) == 1
        assert candidates[0].universal_id == "local:7"

    def test_duplicate_performer_id_keeps_best_score(self):
        qr = self._query_result([7, 7], [0.5, 0.2])
        mapping = {"7": {"name": "Local Performer", "stashdb_id": None}}

        candidates = fuse_local_results(qr, mapping)

        assert len(candidates) == 1
        assert candidates[0].distance == pytest.approx(0.2)


class TestMergeLocalCandidates:
    """A local performer with a linked StashDB id who also shows up as a
    main-index candidate must be merged into one entry, not returned as two
    separate (weaker) candidates for the same real person -- see
    merge_local_candidates()'s own docstring for the full rationale."""

    def _main_match(self, universal_id, distance, name="Main"):
        return CandidateMatch(
            face_index=1, universal_id=universal_id, name=name, combined_distance=distance,
        )

    def _local_match(self, local_id, distance, name="Local"):
        return CandidateMatch(
            face_index=1, universal_id=f"local:{local_id}", name=name, combined_distance=distance,
        )

    def test_duplicate_merged_local_wins(self):
        main = [self._main_match("stashdb.org:uuid-1", distance=0.45)]
        local = [self._local_match("7", distance=0.30)]
        mapping = {"7": {"name": "Local", "stashdb_id": "uuid-1"}}

        merged = merge_local_candidates(main, local, mapping)

        assert len(merged) == 1
        assert merged[0].combined_distance == pytest.approx(0.30)
        assert merged[0].universal_id == "local:7"

    def test_duplicate_merged_main_wins(self):
        main = [self._main_match("stashdb.org:uuid-1", distance=0.20)]
        local = [self._local_match("7", distance=0.40)]
        mapping = {"7": {"name": "Local", "stashdb_id": "uuid-1"}}

        merged = merge_local_candidates(main, local, mapping)

        assert len(merged) == 1
        assert merged[0].combined_distance == pytest.approx(0.20)
        assert merged[0].universal_id == "stashdb.org:uuid-1"

    def test_local_only_performer_not_dropped_or_penalized(self):
        main = [self._main_match("stashdb.org:uuid-1", distance=0.30)]
        local = [self._local_match("9", distance=0.35)]
        mapping = {"9": {"name": "Local Only", "stashdb_id": None}}

        merged = merge_local_candidates(main, local, mapping)

        assert len(merged) == 2
        assert {m.universal_id for m in merged} == {"stashdb.org:uuid-1", "local:9"}

    def test_main_only_performer_not_dropped_or_penalized(self):
        main = [self._main_match("stashdb.org:uuid-1", distance=0.30)]
        local: list[CandidateMatch] = []
        mapping: dict = {}

        merged = merge_local_candidates(main, local, mapping)

        assert merged == main

    def test_linked_performer_not_in_this_calls_main_results_kept_separate(self):
        main = [self._main_match("stashdb.org:uuid-OTHER", distance=0.30)]
        local = [self._local_match("7", distance=0.35)]
        mapping = {"7": {"name": "Local", "stashdb_id": "uuid-1"}}

        merged = merge_local_candidates(main, local, mapping)

        assert len(merged) == 2
        assert {m.universal_id for m in merged} == {"stashdb.org:uuid-OTHER", "local:7"}

    def test_multiple_local_candidates_mixed(self):
        main = [self._main_match("stashdb.org:uuid-1", distance=0.40)]
        local = [
            self._local_match("7", distance=0.25),   # duplicate of uuid-1, local wins
            self._local_match("9", distance=0.50),   # local-only, no link
        ]
        mapping = {
            "7": {"name": "Duplicate", "stashdb_id": "uuid-1"},
            "9": {"name": "Local Only", "stashdb_id": None},
        }

        merged = merge_local_candidates(main, local, mapping)

        assert len(merged) == 2
        by_id = {m.universal_id: m for m in merged}
        assert by_id["local:7"].combined_distance == pytest.approx(0.25)
        assert by_id["local:9"].combined_distance == pytest.approx(0.50)


class TestMatchFace:
    def test_main_index_only(self):
        index = _mock_index(keys=[0, 1], distances=[0.2, 0.5])
        faces = ["stashdb.org:uuid-0", "stashdb.org:uuid-1"]
        performers = {
            "stashdb.org:uuid-0": {"name": "A"},
            "stashdb.org:uuid-1": {"name": "B"},
        }

        result = match_face(np.zeros(512, dtype=np.float32), index, faces, performers)

        assert [m.name for m in result.matches] == ["A", "B"]

    def test_merges_local_index_when_provided(self):
        main_index = _mock_index(keys=[0], distances=[0.5])
        local_index = _mock_index(keys=[7], distances=[0.1])
        faces = ["stashdb.org:uuid-1"]
        performers = {"stashdb.org:uuid-1": {"name": "Main Only"}}
        local_mapping = {"7": {"name": "Local Only", "stashdb_id": None}}

        result = match_face(
            np.zeros(512, dtype=np.float32), main_index, faces, performers,
            local_index=local_index, local_performers_mapping=local_mapping,
        )

        universal_ids = {m.universal_id for m in result.matches}
        assert universal_ids == {"stashdb.org:uuid-1", "local:7"}

    def test_local_index_query_failure_does_not_break_main_results(self):
        main_index = _mock_index(keys=[0], distances=[0.2])
        faces = ["stashdb.org:uuid-1"]
        performers = {"stashdb.org:uuid-1": {"name": "Main Only"}}

        class _BrokenIndex:
            def __len__(self):
                return 5

            def search(self, embedding, k):
                raise RuntimeError("local index corrupt")

        broken_local_index = _BrokenIndex()

        result = match_face(
            np.zeros(512, dtype=np.float32), main_index, faces, performers,
            local_index=broken_local_index, local_performers_mapping={"1": {"name": "x", "stashdb_id": None}},
        )

        assert [m.universal_id for m in result.matches] == ["stashdb.org:uuid-1"]

    def test_empty_local_index_skipped(self):
        main_index = _mock_index(keys=[0], distances=[0.2])
        faces = ["stashdb.org:uuid-1"]
        performers = {"stashdb.org:uuid-1": {"name": "Main Only"}}
        empty_local_index = _mock_index(keys=[], distances=[])

        result = match_face(
            np.zeros(512, dtype=np.float32), main_index, faces, performers,
            local_index=empty_local_index, local_performers_mapping={},
        )

        assert [m.universal_id for m in result.matches] == ["stashdb.org:uuid-1"]

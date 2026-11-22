"""Tests for scene fingerprint match quality scoring."""

import pytest
from scene_fingerprint_scoring import score_match, is_high_confidence


class TestScoreMatch:
    """Test quality score computation for a candidate match."""

    def test_single_md5_match(self):
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "abc123", "duration": 1834.5, "submissions": 3}
            ],
            total_local_fingerprints=3,
            local_duration=1835.0,
        )
        assert result["match_count"] == 1
        assert result["match_percentage"] == pytest.approx(33.3, abs=0.1)
        assert result["has_exact_hash"] is True

    def test_multiple_fingerprints_all_match(self):
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "abc", "duration": 1800.0, "submissions": 5},
                {"algorithm": "OSHASH", "hash": "def", "duration": 1800.0, "submissions": 3},
                {"algorithm": "PHASH", "hash": "ghi", "duration": 1800.0, "submissions": 1},
            ],
            total_local_fingerprints=3,
            local_duration=1800.0,
        )
        assert result["match_count"] == 3
        assert result["match_percentage"] == pytest.approx(100.0)
        assert result["has_exact_hash"] is True

    def test_phash_only_match(self):
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "PHASH", "hash": "abc", "duration": 1800.0, "submissions": 1}
            ],
            total_local_fingerprints=2,
            local_duration=1800.0,
        )
        assert result["has_exact_hash"] is False

    def test_duration_agreement(self):
        close = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1834.0, "submissions": 1}
            ],
            total_local_fingerprints=1,
            local_duration=1835.0,
        )
        assert close["duration_agreement"] is True

        far = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1700.0, "submissions": 1}
            ],
            total_local_fingerprints=1,
            local_duration=1835.0,
        )
        assert far["duration_agreement"] is False

    def test_empty_fingerprints(self):
        result = score_match(
            matching_fingerprints=[],
            total_local_fingerprints=3,
            local_duration=1800.0,
        )
        assert result["match_count"] == 0
        assert result["match_percentage"] == 0.0


class TestCanonicalDurationPenalty:
    """Confirmed live rec=42642 (2026-11-22): a fingerprint hit that only
    matched one of a scene's several submitted fingerprint variants shouldn't
    be treated as a 100% match when the scene's own dominant (most-submitted)
    duration is wildly different from the local file -- the previous
    duration_agreement/duration_diff was computed only from the fingerprints
    that already matched, which is self-referentially always close to the
    local file and never catches this."""

    def test_small_diff_no_penalty(self):
        """Within the existing 5s tolerance: unchanged, no penalty."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1548.0, "submissions": 10}
            ],
            total_local_fingerprints=1,
            local_duration=1548.0,
            all_remote_fingerprints=[{"duration": 1548.0, "submissions": 10}],
        )
        assert result["duration_mismatch"] is False
        assert result["match_percentage"] == pytest.approx(100.0)

    def test_moderate_duration_gap_reduces_confidence(self):
        """Real example: local file 25:48 (1548s) vs. a scene whose dominant
        submitted duration is 22:59 (1379s) -- a fingerprint hit here is not
        a 100% match."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "PHASH", "hash": "a", "duration": 1548.0, "submissions": 3}
            ],
            total_local_fingerprints=1,
            local_duration=1548.0,
            all_remote_fingerprints=[
                {"duration": 1548.0, "submissions": 3},
                {"duration": 1379.0, "submissions": 50},
            ],
        )
        assert result["canonical_duration_remote"] == 1379.0
        assert result["canonical_duration_submissions"] == 50
        assert result["duration_mismatch"] is True
        assert result["match_percentage"] < 100.0
        assert result["match_percentage"] == pytest.approx(78.2, abs=0.5)

    def test_large_duration_gap_different_cut_entirely(self):
        """Real example: rec=42642 -- local file 24:28 (1468s) vs. a scene
        whose dominant submitted duration (265+151 submissions) is 57:35
        (3455s). Treated as effectively no match on duration, regardless of
        the fingerprint hit."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "PHASH", "hash": "a", "duration": 1468.0, "submissions": 3}
            ],
            total_local_fingerprints=1,
            local_duration=1468.0,
            all_remote_fingerprints=[
                {"duration": 1468.0, "submissions": 3},
                {"duration": 3455.0, "submissions": 265},
                {"duration": 3456.0, "submissions": 151},
            ],
        )
        assert result["canonical_duration_remote"] == 3455.0
        assert result["canonical_duration_submissions"] == 265
        assert result["duration_mismatch"] is True
        assert result["match_percentage"] == pytest.approx(0.0, abs=0.1)

    def test_no_remote_fingerprints_falls_back_to_scene_duration_field(self):
        """When the candidate scene has no fingerprints at all to inspect
        (shouldn't normally happen, but defensive), fall back to the
        scene-level duration field rather than skipping the check."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1548.0, "submissions": 3}
            ],
            total_local_fingerprints=1,
            local_duration=1548.0,
            all_remote_fingerprints=[],
            remote_scene_duration=1379.0,
        )
        assert result["canonical_duration_remote"] == 1379.0
        assert result["canonical_duration_submissions"] == 0
        assert result["duration_mismatch"] is True

    def test_no_canonical_data_at_all_skips_penalty(self):
        """No fingerprints and no scene-level duration either: nothing to
        compare against, so no penalty (never a false mismatch)."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1548.0, "submissions": 3}
            ],
            total_local_fingerprints=1,
            local_duration=1548.0,
        )
        assert result["canonical_duration_remote"] is None
        assert result["duration_mismatch"] is False
        assert result["match_percentage"] == pytest.approx(100.0)


class TestIsHighConfidence:
    def test_meets_both_thresholds(self):
        assert is_high_confidence(match_count=2, match_percentage=66.7) is True

    def test_below_count_threshold(self):
        assert is_high_confidence(match_count=1, match_percentage=100.0) is False

    def test_below_percentage_threshold(self):
        assert is_high_confidence(match_count=2, match_percentage=50.0) is False

    def test_custom_thresholds(self):
        assert is_high_confidence(match_count=3, match_percentage=80.0, min_count=3, min_percentage=80) is True

    def test_custom_thresholds_fail(self):
        assert is_high_confidence(match_count=2, match_percentage=80.0, min_count=3, min_percentage=80) is False

    def test_duration_mismatch_forces_not_high_confidence(self):
        """Confirmed live rec=42642: a fingerprint hit must never imply high
        confidence when the canonical scene duration disagrees this much,
        regardless of match_count/match_percentage."""
        assert is_high_confidence(
            match_count=3, match_percentage=100.0, duration_mismatch=True,
        ) is False

    def test_no_mismatch_defaults_to_false(self):
        """duration_mismatch defaults to False -- existing callers that don't
        pass it keep their prior behavior."""
        assert is_high_confidence(match_count=2, match_percentage=66.7) is True

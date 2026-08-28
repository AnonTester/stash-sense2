"""Tests for POST /recommendations/cleanup-entity -- the generic
performer/studio/tag destroy-hook cleanup endpoint, mirroring
cleanup-scene but for the simpler (mostly scalar-target_id) recommendation
types performer/studio/tag actually use.
"""
from unittest.mock import MagicMock, patch

import pytest

from recommendations_db import Recommendation
from recommendations_router import (
    EntityCleanupRequest,
    _extract_performer_ids_from_recommendation,
    _extract_simple_target_id,
    cleanup_entity_recommendations,
)


def _rec(id, type, target_id, target_type, details=None, status="pending"):
    return Recommendation(
        id=id, type=type, status=status, target_type=target_type, target_id=target_id,
        details=details or {}, resolution_action=None, resolution_details=None,
        resolved_at=None, confidence=None, source_analysis_id=None,
        created_at="2026-01-01", updated_at="2026-01-01",
    )


class TestExtractPerformerIds:
    def test_upstream_performer_changes_uses_target_id(self):
        rec = _rec(1, "upstream_performer_changes", "42", "performer")
        assert _extract_performer_ids_from_recommendation(rec) == ["42"]

    def test_duplicate_performer_includes_non_anchor_group_members(self):
        # target_id is only the suggested-keeper anchor (7) -- the other
        # group members (8, 9) live in details.performers and must also
        # be extracted, or a deleted non-anchor performer would leave this
        # recommendation stale.
        rec = _rec(
            1, "duplicate_performer", "7", "performer",
            details={"performers": [{"id": 7}, {"id": 8}, {"id": 9}]},
        )
        assert set(_extract_performer_ids_from_recommendation(rec)) == {"7", "8", "9"}

    def test_unrelated_target_type_returns_empty(self):
        rec = _rec(1, "upstream_studio_changes", "42", "studio")
        assert _extract_performer_ids_from_recommendation(rec) == []


class TestExtractSimpleTargetId:
    def test_matches_target_type(self):
        rec = _rec(1, "upstream_studio_changes", "42", "studio")
        assert _extract_simple_target_id(rec, "studio") == ["42"]

    def test_mismatched_target_type_returns_empty(self):
        rec = _rec(1, "upstream_tag_changes", "42", "tag")
        assert _extract_simple_target_id(rec, "studio") == []

    def test_empty_target_id_returns_empty(self):
        rec = _rec(1, "upstream_studio_changes", "", "studio")
        assert _extract_simple_target_id(rec, "studio") == []


class TestCleanupEntityRecommendations:
    async def test_performer_deletes_group_and_direct_matches(self):
        recs = [
            _rec(1, "upstream_performer_changes", "42", "performer"),
            _rec(2, "duplicate_performer", "7", "performer", details={"performers": [{"id": 7}, {"id": 42}]}),
            _rec(3, "upstream_performer_changes", "99", "performer"),
        ]
        db = MagicMock()

        with patch("recommendations_router.get_rec_db", return_value=db), \
             patch("recommendations_router._load_all_recommendations", return_value=recs):
            result = await cleanup_entity_recommendations(
                EntityCleanupRequest(target_type="performer", entity_id="42")
            )

        assert result == {"target_type": "performer", "entity_id": "42", "deleted": 2}
        deleted_ids = {call.args[0] for call in db.delete_recommendation.call_args_list}
        assert deleted_ids == {1, 2}

    async def test_studio_deletes_only_matching(self):
        recs = [
            _rec(1, "upstream_studio_changes", "5", "studio"),
            _rec(2, "upstream_studio_changes", "6", "studio"),
        ]
        db = MagicMock()

        with patch("recommendations_router.get_rec_db", return_value=db), \
             patch("recommendations_router._load_all_recommendations", return_value=recs):
            result = await cleanup_entity_recommendations(
                EntityCleanupRequest(target_type="studio", entity_id="5")
            )

        assert result == {"target_type": "studio", "entity_id": "5", "deleted": 1}
        db.delete_recommendation.assert_called_once_with(1)

    async def test_tag_deletes_only_matching(self):
        recs = [_rec(1, "upstream_tag_changes", "11", "tag")]
        db = MagicMock()

        with patch("recommendations_router.get_rec_db", return_value=db), \
             patch("recommendations_router._load_all_recommendations", return_value=recs):
            result = await cleanup_entity_recommendations(
                EntityCleanupRequest(target_type="tag", entity_id="11")
            )

        assert result == {"target_type": "tag", "entity_id": "11", "deleted": 1}

    async def test_unsupported_target_type_rejected(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            await cleanup_entity_recommendations(
                EntityCleanupRequest(target_type="scene", entity_id="1")
            )

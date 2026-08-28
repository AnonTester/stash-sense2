"""Tests for POST /recommendations/cleanup-scene -- the Scene.Destroy.Post
hook's cleanup endpoint. Deletes every recommendation (any status)
referencing a given scene id, reusing the same
_extract_scene_ids_from_recommendation logic the lazy per-item prune in
get_recommendation() already uses.
"""
from unittest.mock import MagicMock, patch

from recommendations_db import Recommendation
from recommendations_router import SceneCleanupRequest, cleanup_scene_recommendations


def _rec(id, type, target_id, details=None, target_type="scene"):
    return Recommendation(
        id=id, type=type, status="pending", target_type=target_type, target_id=target_id,
        details=details or {}, resolution_action=None, resolution_details=None,
        resolved_at=None, confidence=None, source_analysis_id=None,
        created_at="2026-01-01", updated_at="2026-01-01",
    )


class TestCleanupSceneRecommendations:
    async def test_deletes_only_recommendations_referencing_the_scene(self):
        recs = [
            _rec(1, "scene_face_match", "42|local:7"),
            _rec(2, "scene_face_match", "99|local:8"),
            _rec(3, "upstream_scene_changes", "42", details={"scene_id": "42"}),
            _rec(4, "duplicate_scenes", "42:99", details={"scene_a_id": "42", "scene_b_id": "99"}),
        ]
        db = MagicMock()

        with patch("recommendations_router.get_rec_db", return_value=db), \
             patch("recommendations_router._load_all_recommendations", return_value=recs):
            result = await cleanup_scene_recommendations(SceneCleanupRequest(scene_id="42"))

        assert result == {"scene_id": "42", "deleted": 3}
        deleted_ids = {call.args[0] for call in db.delete_recommendation.call_args_list}
        assert deleted_ids == {1, 3, 4}

    async def test_no_matches_deletes_nothing(self):
        recs = [_rec(1, "scene_face_match", "99|local:8")]
        db = MagicMock()

        with patch("recommendations_router.get_rec_db", return_value=db), \
             patch("recommendations_router._load_all_recommendations", return_value=recs):
            result = await cleanup_scene_recommendations(SceneCleanupRequest(scene_id="42"))

        assert result == {"scene_id": "42", "deleted": 0}
        db.delete_recommendation.assert_not_called()

    async def test_empty_recommendation_list(self):
        db = MagicMock()

        with patch("recommendations_router.get_rec_db", return_value=db), \
             patch("recommendations_router._load_all_recommendations", return_value=[]):
            result = await cleanup_scene_recommendations(SceneCleanupRequest(scene_id="42"))

        assert result == {"scene_id": "42", "deleted": 0}

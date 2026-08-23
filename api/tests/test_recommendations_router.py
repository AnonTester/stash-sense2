"""Tests for the recommendations API router."""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from recommendations_db import RecommendationsDB
import recommendations_router as rec_mod


@pytest.fixture
def db(tmp_path):
    """Create a fresh RecommendationsDB."""
    return RecommendationsDB(str(tmp_path / "test.db"))


@pytest.fixture
def client(db):
    """Create a test client with real DB and mocked stash_client."""
    original_db = rec_mod.rec_db
    original_stash = rec_mod.stash_client

    rec_mod.rec_db = db
    rec_mod.stash_client = Mock()

    app = FastAPI()
    app.include_router(rec_mod.router)
    test_client = TestClient(app)

    yield test_client

    rec_mod.rec_db = original_db
    rec_mod.stash_client = original_stash


def _seed_recommendations(db, count=3, rec_type="duplicate_performer", target_type="performer", status="pending"):
    """Helper to seed recommendations into the database."""
    ids = []
    for i in range(count):
        rec_id = db.create_recommendation(
            type=rec_type,
            target_type=target_type,
            target_id=str(100 + i),
            details={"name": f"Test Performer {i}"},
            confidence=0.9 - i * 0.1,
        )
        ids.append(rec_id)
    return ids


def _seed_analysis_run(db, run_type="duplicate_performer", items_total=10, complete=True):
    """Helper to seed an analysis run."""
    run_id = db.start_analysis_run(run_type, items_total=items_total)
    if complete:
        db.complete_analysis_run(run_id, recommendations_created=3)
    return run_id


# ==================== GET /recommendations ====================


class TestListRecommendations:
    """Test GET /recommendations."""

    def test_returns_empty_list(self, client):
        resp = client.get("/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] == []
        assert data["total"] == 0

    def test_returns_recommendations(self, client, db):
        _seed_recommendations(db, count=3)
        resp = client.get("/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) == 3
        assert data["total"] == 3

    def test_filter_by_status(self, client, db):
        ids = _seed_recommendations(db, count=3)
        # Dismiss one
        db.dismiss_recommendation(ids[0], reason="test")
        resp = client.get("/recommendations", params={"status": "pending"})
        data = resp.json()
        assert len(data["recommendations"]) == 2
        assert data["total"] == 2

    def test_filter_by_type(self, client, db):
        _seed_recommendations(db, count=2, rec_type="duplicate_performer")
        _seed_recommendations(db, count=1, rec_type="upstream_performer_changes")
        resp = client.get("/recommendations", params={"type": "upstream_performer_changes"})
        data = resp.json()
        assert len(data["recommendations"]) == 1
        assert data["total"] == 1
        assert data["recommendations"][0]["type"] == "upstream_performer_changes"

    def test_filter_by_target_type(self, client, db):
        _seed_recommendations(db, count=2, target_type="performer")
        _seed_recommendations(db, count=1, target_type="scene")
        resp = client.get("/recommendations", params={"target_type": "scene"})
        data = resp.json()
        assert len(data["recommendations"]) == 1
        assert data["total"] == 1

    def test_pagination_limit(self, client, db):
        _seed_recommendations(db, count=5)
        resp = client.get("/recommendations", params={"limit": 2})
        data = resp.json()
        assert len(data["recommendations"]) == 2
        assert data["total"] == 5

    def test_pagination_offset(self, client, db):
        _seed_recommendations(db, count=5)
        resp = client.get("/recommendations", params={"limit": 2, "offset": 3})
        data = resp.json()
        assert len(data["recommendations"]) == 2
        assert data["total"] == 5

    def test_recommendation_response_shape(self, client, db):
        _seed_recommendations(db, count=1)
        resp = client.get("/recommendations")
        rec = resp.json()["recommendations"][0]
        assert "id" in rec
        assert "type" in rec
        assert "status" in rec
        assert "target_type" in rec
        assert "target_id" in rec
        assert "details" in rec
        assert "confidence" in rec
        assert "created_at" in rec
        assert "updated_at" in rec

    def test_scene_fingerprint_pending_removed_when_scene_already_linked(self, client, db):
        rec_id = db.create_recommendation(
            type="scene_fingerprint_match",
            target_type="scene",
            target_id="42|https://theporndb.net/graphql|remote-123",
            details={},  # legacy rows may not include local_scene_id
            confidence=0.66,
        )
        rec_mod.stash_client.get_scene_by_id = AsyncMock(return_value={
            "id": "42",
            "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "already-linked"}],
        })

        resp = client.get(
            "/recommendations",
            params={"status": "pending", "type": "scene_fingerprint_match"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["recommendations"] == []
        assert db.get_recommendation(rec_id) is None

    def test_scene_fingerprint_pending_removed_when_scene_deleted(self, client, db):
        rec_id = db.create_recommendation(
            type="scene_fingerprint_match",
            target_type="scene",
            target_id="99|https://stashdb.org/graphql|remote-999",
            details={"local_scene_id": "99"},
            confidence=0.66,
        )
        rec_mod.stash_client.get_scene_by_id = AsyncMock(return_value=None)

        resp = client.get(
            "/recommendations",
            params={"status": "pending", "type": "scene_fingerprint_match"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["recommendations"] == []
        assert db.get_recommendation(rec_id) is None

    def test_duplicate_scenes_are_grouped_and_sorted_by_top_confidence(self, client, db):
        db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={
                "scene_a_id": 42,
                "scene_b_id": 77,
                "confidence": 80,
                "scene_a_summary": {"title": "Source 42"},
                "scene_b_summary": {"title": "Match 77"},
            },
            confidence=0.80,
        )
        db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={
                "scene_a_id": 42,
                "scene_b_id": 88,
                "confidence": 95,
                "scene_a_summary": {"title": "Source 42"},
                "scene_b_summary": {"title": "Match 88"},
            },
            confidence=0.95,
        )
        db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="13:17",
            details={
                "scene_a_id": 13,
                "scene_b_id": 17,
                "confidence": 90,
                "scene_a_summary": {"title": "Source 13"},
                "scene_b_summary": {"title": "Match 17"},
            },
            confidence=0.90,
        )

        resp = client.get("/recommendations", params={"type": "duplicate_scenes", "status": "pending"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["recommendations"]) == 2
        assert data["recommendations"][0]["details"]["source_scene_id"] == "42"
        assert data["recommendations"][0]["details"]["match_count"] == 2
        assert data["recommendations"][0]["confidence"] == pytest.approx(0.95)
        assert data["recommendations"][1]["details"]["source_scene_id"] == "13"


# ==================== GET /recommendations/counts ====================


class TestRecommendationCounts:
    """Test GET /recommendations/counts."""

    def test_empty_counts(self, client):
        resp = client.get("/recommendations/counts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"] == {}
        assert data["total_pending"] == 0

    def test_counts_by_type_and_status(self, client, db):
        _seed_recommendations(db, count=3, rec_type="duplicate_performer")
        _seed_recommendations(db, count=2, rec_type="upstream_performer_changes")
        resp = client.get("/recommendations/counts")
        data = resp.json()
        assert data["counts"]["duplicate_performer"]["pending"] == 3
        assert data["counts"]["upstream_performer_changes"]["pending"] == 2
        assert data["total_pending"] == 5

    def test_counts_include_dismissed(self, client, db):
        ids = _seed_recommendations(db, count=3, rec_type="duplicate_performer")
        db.dismiss_recommendation(ids[0], reason="test")
        resp = client.get("/recommendations/counts")
        data = resp.json()
        assert data["counts"]["duplicate_performer"]["pending"] == 2
        assert data["counts"]["duplicate_performer"]["dismissed"] == 1
        assert data["total_pending"] == 2

    def test_duplicate_scene_counts_use_grouped_sources(self, client, db):
        db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77, "confidence": 80},
            confidence=0.80,
        )
        db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={"scene_a_id": 42, "scene_b_id": 88, "confidence": 95},
            confidence=0.95,
        )
        db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="13:17",
            details={"scene_a_id": 13, "scene_b_id": 17, "confidence": 90},
            confidence=0.90,
        )

        resp = client.get("/recommendations/counts")
        data = resp.json()
        assert data["counts"]["duplicate_scenes"]["pending"] == 2
        assert data["total_pending"] == 2


# ==================== GET /recommendations/{rec_id} ====================


class TestGetRecommendation:
    """Test GET /recommendations/{rec_id}."""

    def test_returns_recommendation(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.get(f"/recommendations/{ids[0]}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == ids[0]
        assert data["type"] == "duplicate_performer"
        assert data["status"] == "pending"

    def test_404_for_missing(self, client):
        resp = client.get("/recommendations/99999")
        assert resp.status_code == 404

    def test_scene_based_rec_deleted_when_referenced_scene_missing(self, client, db):
        rec_id = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="101:202",
            details={"scene_a_id": 101, "scene_b_id": 202},
            confidence=0.9,
        )
        rec_mod.stash_client.get_scene_by_id = AsyncMock(side_effect=[None, {"id": "202"}])

        resp = client.get(f"/recommendations/{rec_id}")
        assert resp.status_code == 404
        assert "referenced scene no longer exists" in str(resp.json().get("detail", "")).lower()
        assert db.get_recommendation(rec_id) is None

    def test_duplicate_scene_get_returns_grouped_details(self, client, db):
        first_id = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={
                "scene_a_id": 42,
                "scene_b_id": 77,
                "confidence": 80,
                "scene_a_summary": {"title": "Source 42"},
                "scene_b_summary": {"title": "Match 77"},
                "reasoning": ["Likely duplicate"],
            },
            confidence=0.80,
        )
        db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={
                "scene_a_id": 42,
                "scene_b_id": 88,
                "confidence": 95,
                "scene_a_summary": {"title": "Source 42"},
                "scene_b_summary": {"title": "Match 88"},
                "reasoning": ["High confidence duplicate"],
            },
            confidence=0.95,
        )
        rec_mod.stash_client.get_scene_by_id = AsyncMock(side_effect=lambda scene_id: {"id": str(scene_id)})

        resp = client.get(f"/recommendations/{first_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["details"]["grouped"] is True
        assert data["details"]["source_scene_id"] == "42"
        assert len(data["details"]["duplicate_matches"]) == 2
        assert data["details"]["duplicate_matches"][0]["match_scene_id"] == "88"
        assert data["confidence"] == pytest.approx(0.95)

    def test_duplicate_scene_get_prunes_stale_sibling_matches(self, client, db):
        valid_id = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={
                "scene_a_id": 42,
                "scene_b_id": 88,
                "confidence": 95,
                "scene_a_summary": {"title": "Source 42"},
                "scene_b_summary": {"title": "Match 88"},
            },
            confidence=0.95,
        )
        stale_id = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={
                "scene_a_id": 42,
                "scene_b_id": 77,
                "confidence": 80,
                "scene_a_summary": {"title": "Source 42"},
                "scene_b_summary": {"title": "Match 77"},
            },
            confidence=0.80,
        )
        rec_mod.stash_client.get_scene_by_id = AsyncMock(
            side_effect=lambda scene_id: None if str(scene_id) == "77" else {"id": str(scene_id)}
        )

        resp = client.get(f"/recommendations/{valid_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["details"]["source_scene_id"] == "42"
        assert len(data["details"]["duplicate_matches"]) == 1
        assert data["details"]["duplicate_matches"][0]["match_scene_id"] == "88"
        assert db.get_recommendation(valid_id) is not None
        assert db.get_recommendation(stale_id) is None


# ==================== POST /recommendations/{rec_id}/resolve ====================


class TestResolveRecommendation:
    """Test POST /recommendations/{rec_id}/resolve."""

    def test_resolve_success(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.post(
            f"/recommendations/{ids[0]}/resolve",
            json={"action": "merged", "details": {"merged_into": "42"}},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Verify resolved in DB
        rec = db.get_recommendation(ids[0])
        assert rec.status == "resolved"
        assert rec.resolution_action == "merged"

    def test_resolve_404(self, client):
        resp = client.post(
            "/recommendations/99999/resolve",
            json={"action": "merged"},
        )
        assert resp.status_code == 404


# ==================== POST /recommendations/{rec_id}/dismiss ====================


class TestDismissRecommendation:
    """Test POST /recommendations/{rec_id}/dismiss."""

    def test_dismiss_success(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.post(
            f"/recommendations/{ids[0]}/dismiss",
            json={"reason": "Not relevant"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        rec = db.get_recommendation(ids[0])
        assert rec.status == "dismissed"

    def test_dismiss_without_reason(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.post(f"/recommendations/{ids[0]}/dismiss", json={})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_dismiss_404(self, client):
        resp = client.post(
            "/recommendations/99999/dismiss",
            json={"reason": "test"},
        )
        assert resp.status_code == 404


# ==================== GET /recommendations/analysis/runs ====================


class TestListAnalysisRuns:
    """Test GET /recommendations/analysis/runs."""

    def test_empty_runs(self, client):
        resp = client.get("/recommendations/analysis/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_runs(self, client, db):
        _seed_analysis_run(db, run_type="duplicate_performer")
        _seed_analysis_run(db, run_type="upstream_performer_changes")
        resp = client.get("/recommendations/analysis/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_filter_by_type(self, client, db):
        _seed_analysis_run(db, run_type="duplicate_performer")
        _seed_analysis_run(db, run_type="upstream_performer_changes")
        resp = client.get("/recommendations/analysis/runs", params={"type": "duplicate_performer"})
        data = resp.json()
        assert len(data) == 1
        assert data[0]["type"] == "duplicate_performer"

    def test_run_response_shape(self, client, db):
        _seed_analysis_run(db, run_type="duplicate_performer")
        resp = client.get("/recommendations/analysis/runs")
        run = resp.json()[0]
        assert "id" in run
        assert "type" in run
        assert "status" in run
        assert "started_at" in run
        assert "recommendations_created" in run


# ==================== GET /recommendations/analysis/runs/{run_id} ====================


class TestGetAnalysisRun:
    """Test GET /recommendations/analysis/runs/{run_id}."""

    def test_returns_run(self, client, db):
        run_id = _seed_analysis_run(db, run_type="duplicate_performer")
        resp = client.get(f"/recommendations/analysis/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == run_id
        assert data["type"] == "duplicate_performer"
        assert data["status"] == "completed"

    def test_404_for_missing(self, client):
        resp = client.get("/recommendations/analysis/runs/99999")
        assert resp.status_code == 404


# ==================== POST /recommendations/actions/batch-dismiss ====================


class TestBatchDismiss:
    """Test POST /recommendations/actions/batch-dismiss."""

    def test_batch_dismiss_by_type(self, client, db):
        _seed_recommendations(db, count=3, rec_type="duplicate_performer")
        _seed_recommendations(db, count=2, rec_type="upstream_performer_changes")

        resp = client.post(
            "/recommendations/actions/batch-dismiss",
            json={"type": "duplicate_performer"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["dismissed_count"] == 3

        # Verify only the correct type was dismissed
        remaining = db.get_recommendations(status="pending")
        assert len(remaining) == 2
        assert all(r.type == "upstream_performer_changes" for r in remaining)

    def test_batch_dismiss_returns_zero_for_none(self, client):
        resp = client.post(
            "/recommendations/actions/batch-dismiss",
            json={"type": "nonexistent_type"},
        )
        assert resp.status_code == 200
        assert resp.json()["dismissed_count"] == 0

    def test_batch_dismiss_permanent_flag(self, client, db):
        _seed_recommendations(db, count=2, rec_type="duplicate_performer")
        resp = client.post(
            "/recommendations/actions/batch-dismiss",
            json={"type": "duplicate_performer", "permanent": True},
        )
        assert resp.status_code == 200
        assert resp.json()["dismissed_count"] == 2


class TestDeleteSceneAction:
    """Test POST /recommendations/actions/delete-scene."""

    def test_delete_scene_cleans_pending_scene_fingerprint_recommendations(self, client, db):
        rec_id = db.create_recommendation(
            type="scene_fingerprint_match",
            target_type="scene",
            target_id="42|https://stashdb.org/graphql|sb-uuid-1",
            details={"local_scene_id": "42"},
            confidence=0.75,
        )
        rec_mod.stash_client.destroy_scene = AsyncMock(return_value=True)

        resp = client.post(
            "/recommendations/actions/delete-scene",
            json={"scene_id": "42", "delete_file": False},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert db.get_recommendation(rec_id) is None

    def test_delete_scene_cleans_pending_duplicate_scene_recommendations(self, client, db):
        dup_pair = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77},
            confidence=0.9,
        )
        dup_files = db.create_recommendation(
            type="duplicate_scene_files",
            target_type="scene",
            target_id="42",
            details={"scene_title": "Scene 42"},
            confidence=1.0,
        )
        keep_other = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="77:88",
            details={"scene_a_id": 77, "scene_b_id": 88},
            confidence=0.8,
        )
        rec_mod.stash_client.destroy_scene = AsyncMock(return_value=True)

        resp = client.post(
            "/recommendations/actions/delete-scene",
            json={"scene_id": "42", "delete_file": False},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert db.get_recommendation(dup_pair) is None
        assert db.get_recommendation(dup_files) is None
        assert db.get_recommendation(keep_other) is not None


class TestMergeScenesAction:
    """Test POST /recommendations/actions/merge-scenes."""

    def test_merge_scenes_cleans_pending_duplicate_scene_recommendations(self, client, db):
        dup_pair_a = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77},
            confidence=0.9,
        )
        dup_pair_b = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="21:42",
            details={"scene_a_id": 21, "scene_b_id": 42},
            confidence=0.8,
        )
        dup_files_dest = db.create_recommendation(
            type="duplicate_scene_files",
            target_type="scene",
            target_id="77",
            details={"scene_title": "Scene 77"},
            confidence=1.0,
        )
        keep_other = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="99:100",
            details={"scene_a_id": 99, "scene_b_id": 100},
            confidence=0.7,
        )

        rec_mod.stash_client.merge_scenes = AsyncMock(return_value={"id": "77"})

        resp = client.post(
            "/recommendations/actions/merge-scenes",
            json={"destination_id": "77", "source_ids": ["42"]},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert db.get_recommendation(dup_pair_a) is None
        assert db.get_recommendation(dup_pair_b) is None
        assert db.get_recommendation(dup_files_dest) is None
        assert db.get_recommendation(keep_other) is not None

    def test_merge_duplicate_scene_group_resolves_selected_and_blocks_unselected(self, client, db):
        rec_keep_77 = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77, "confidence": 90},
            confidence=0.9,
        )
        rec_keep_88 = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={"scene_a_id": 42, "scene_b_id": 88, "confidence": 80},
            confidence=0.8,
        )
        rec_skip_99 = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:99",
            details={"scene_a_id": 42, "scene_b_id": 99, "confidence": 70},
            confidence=0.7,
        )
        rec_mod.stash_client.merge_scenes = AsyncMock(return_value={"id": "42"})

        resp = client.post(
            "/recommendations/actions/merge-duplicate-scene-group",
            json={
                "source_scene_id": "42",
                "selected_match_scene_ids": ["77", "88"],
                "selected_recommendation_ids": [rec_keep_77, rec_keep_88],
                "unselected_recommendation_ids": [rec_skip_99],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        rec_mod.stash_client.merge_scenes.assert_awaited_once_with(["77", "88"], "42")

        assert db.get_recommendation(rec_keep_77).status == "resolved"
        assert db.get_recommendation(rec_keep_88).status == "resolved"
        assert db.get_recommendation(rec_skip_99).status == "resolved"
        assert db.is_dismissed("duplicate_scenes", "scene", "42:99") is True


class TestDuplicateSceneGroupActions:
    def test_delete_duplicate_scene_match_resolves_after_successful_delete(self, client, db):
        rec_id = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77},
            confidence=0.9,
        )
        dup_files_match = db.create_recommendation(
            type="duplicate_scene_files",
            target_type="scene",
            target_id="77",
            details={"scene_title": "Scene 77"},
            confidence=1.0,
        )
        rec_mod.stash_client.destroy_scene = AsyncMock(return_value=True)

        resp = client.post(
            "/recommendations/actions/delete-duplicate-scene-match",
            json={
                "source_scene_id": "42",
                "match_scene_id": "77",
                "recommendation_id": rec_id,
                "delete_file": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        rec = db.get_recommendation(rec_id)
        assert rec is not None
        assert rec.status == "resolved"
        assert rec.resolution_action == "deleted_match"
        assert db.get_recommendation(dup_files_match) is None

    def test_merge_source_into_duplicate_scene_match_resolves_keeper_and_siblings(self, client, db):
        keeper_rec = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77},
            confidence=0.9,
        )
        sibling_rec = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={"scene_a_id": 42, "scene_b_id": 88},
            confidence=0.8,
        )
        rec_mod.stash_client.merge_scenes = AsyncMock(return_value={"id": "77"})
        rec_mod.stash_client.destroy_scene = AsyncMock(return_value=True)

        resp = client.post(
            "/recommendations/actions/merge-source-into-duplicate-scene-match",
            json={
                "source_scene_id": "42",
                "keeper_match_scene_id": "77",
                "keeper_recommendation_id": keeper_rec,
                "other_matches": [
                    {"recommendation_id": sibling_rec, "scene_id": "88"},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        rec_mod.stash_client.merge_scenes.assert_awaited_once_with(["42"], "77")
        rec_mod.stash_client.destroy_scene.assert_awaited_once_with("88", delete_file=False)
        assert db.get_recommendation(keeper_rec).status == "resolved"
        assert db.get_recommendation(keeper_rec).resolution_action == "merged_source_into_match"
        assert db.get_recommendation(sibling_rec).status == "resolved"

    def test_delete_duplicate_scene_group_resolves_all_matches(self, client, db):
        rec_a = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77},
            confidence=0.9,
        )
        rec_b = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={"scene_a_id": 42, "scene_b_id": 88},
            confidence=0.8,
        )
        rec_mod.stash_client.destroy_scene = AsyncMock(return_value=True)

        resp = client.post(
            "/recommendations/actions/delete-duplicate-scene-group",
            json={"source_scene_id": "42", "recommendation_ids": [rec_a, rec_b], "delete_file": False},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert db.get_recommendation(rec_a).status == "resolved"
        assert db.get_recommendation(rec_b).status == "resolved"

    def test_dismiss_duplicate_scene_group_dismisses_all(self, client, db):
        rec_a = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:77",
            details={"scene_a_id": 42, "scene_b_id": 77},
            confidence=0.9,
        )
        rec_b = db.create_recommendation(
            type="duplicate_scenes",
            target_type="scene",
            target_id="42:88",
            details={"scene_a_id": 42, "scene_b_id": 88},
            confidence=0.8,
        )

        resp = client.post(
            "/recommendations/actions/dismiss-duplicate-scene-group",
            json={"recommendation_ids": [rec_a, rec_b], "reason": "Not duplicates"},
        )
        assert resp.status_code == 200
        assert resp.json()["dismissed_count"] == 2
        assert db.get_recommendation(rec_a).status == "dismissed"
        assert db.get_recommendation(rec_b).status == "dismissed"


class TestUpdateSceneAction:
    """Test POST /recommendations/actions/update-scene."""

    def test_update_scene_missing_scene_removes_stale_upstream_rec(self, client, db):
        rec_id = db.create_recommendation(
            type="upstream_scene_changes",
            target_type="scene",
            target_id="26240",
            details={"scene_id": "26240"},
            confidence=1.0,
        )
        rec_mod.stash_client.update_scene = AsyncMock(
            side_effect=RuntimeError(
                "GraphQL error: [{'message': 'scene with id 26240 not found', 'path': ['sceneUpdate']}]"
            )
        )

        resp = client.post(
            "/recommendations/actions/update-scene",
            json={"scene_id": "26240", "fields": {}},
        )
        assert resp.status_code == 404
        assert "removed stale upstream scene recommendation" in str(resp.json().get("detail", "")).lower()
        assert db.get_recommendation(rec_id) is None


class TestSearchEntitiesAction:
    """Test POST /recommendations/actions/search-entities."""

    def test_search_performer_returns_aliases_and_normalized_link_state(self, client):
        rec_mod.stash_client.search_performers = AsyncMock(return_value=[
            {
                "id": "11",
                "name": "Jane Doe",
                "disambiguation": "Performer",
                "alias_list": ["JD", "Jane D"],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-1"}
                ],
            }
        ])

        resp = client.post(
            "/recommendations/actions/search-entities",
            json={
                "entity_type": "performer",
                "query": "JD",
                "endpoint": "https://stashdb.org/graphql/",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["aliases"] == ["JD", "Jane D"]
        assert data["results"][0]["linked"] is True


class TestFindLinkedEntityAction:
    """Test POST /recommendations/actions/find-linked-entity."""

    def test_find_linked_performer_by_stash_id(self, client):
        rec_mod.stash_client._execute = AsyncMock(return_value={
            "findPerformers": {
                "performers": [
                    {
                        "id": "11",
                        "name": "Jane Doe",
                        "disambiguation": "Performer",
                        "alias_list": ["JD"],
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-1"}],
                    }
                ]
            }
        })

        resp = client.post(
            "/recommendations/actions/find-linked-entity",
            json={
                "entity_type": "performer",
                "endpoint": "https://stashdb.org/graphql",
                "stashbox_id": "perf-1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["id"] == "11"
        assert data["result"]["name"] == "Jane Doe"
        assert data["result"]["aliases"] == ["JD"]


class TestFingerprintReset:
    """POST /fingerprints/reset -- backs up + marks all scene fingerprints
    for refresh. Used by the Settings UI's Detection Resolution change
    modal (see stash-sense-settings.js's showDetectionSizeChangeModal)."""

    def test_backs_up_and_marks_all_for_refresh(self, client, db):
        db.create_scene_fingerprint(stash_scene_id=1, total_faces=2, frames_analyzed=60, db_version="2026.01.01")
        db.create_scene_fingerprint(stash_scene_id=2, total_faces=1, frames_analyzed=60, db_version="2026.01.01")

        resp = client.post("/recommendations/fingerprints/reset")

        assert resp.status_code == 200
        data = resp.json()
        assert data["fingerprints_backed_up"] == 2
        assert data["marked_for_refresh"] == 2
        assert db.get_scene_fingerprint(stash_scene_id=1)["db_version"] is None

    def test_empty_database_ok(self, client, db):
        resp = client.post("/recommendations/fingerprints/reset")

        assert resp.status_code == 200
        assert resp.json()["fingerprints_backed_up"] == 0

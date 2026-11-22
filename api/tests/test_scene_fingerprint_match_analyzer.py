"""Tests for SceneFingerprintMatchAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from analyzers.scene_fingerprint_match import SceneFingerprintMatchAnalyzer


def make_mock_db():
    """Create a mock RecommendationsDB."""
    db = MagicMock()
    db.is_dismissed.return_value = False
    db.create_recommendation.return_value = 1
    db.get_user_setting.return_value = None
    db.get_watermark.return_value = None
    db.set_watermark.return_value = None
    # Matches RecommendationsDB.get_endpoint_priorities()'s own real default
    # (empty list when unset) -- run() treats a MagicMock as truthy and
    # tries to enumerate() it, which isn't iterable by default.
    db.get_endpoint_priorities.return_value = []
    return db


def make_mock_stash():
    """Create a mock StashClientUnified."""
    stash = AsyncMock()
    stash.get_stashbox_connections.return_value = [
        {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"}
    ]
    return stash


def make_scene(scene_id, title, fingerprints, stash_ids=None, duration=1800.0, updated_at="2026-01-15T00:00:00Z"):
    """Helper to build a scene dict."""
    return {
        "id": scene_id,
        "title": title,
        "updated_at": updated_at,
        "files": [
            {
                "id": f"file-{scene_id}",
                "duration": duration,
                "fingerprints": fingerprints,
            }
        ],
        "stash_ids": stash_ids or [],
    }


def make_stashbox_match(scene_id, title, fingerprints, studio=None, performers=None, date=None, duration=None):
    """Helper to build a stash-box match result."""
    return {
        "id": scene_id,
        "title": title,
        "date": date,
        "duration": duration,
        "studio": studio,
        "performers": performers or [],
        "urls": [],
        "images": [],
        "fingerprints": fingerprints,
    }


class TestAnalyzerRun:
    """Test the full analyzer run pipeline."""

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_match(self):
        """A scene with matching fingerprints should produce a recommendation."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [
                {"type": "md5", "value": "abc123"},
                {"type": "oshash", "value": "def456"},
            ])],
            1,
        )

        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [
                {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"},
                {"hash": "def456", "algorithm": "OSHASH", "duration": 1800, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
            ],
            studio={"id": "s1", "name": "Studio A"},
            performers=[{"performer": {"id": "p1", "name": "Actor A"}, "as": None}],
            date="2024-01-15",
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 1
        db.create_recommendation.assert_called_once()
        call_kwargs = db.create_recommendation.call_args[1]
        assert call_kwargs["type"] == "scene_fingerprint_match"
        assert call_kwargs["target_type"] == "scene"
        assert "42|" in call_kwargs["target_id"]
        assert "sb-uuid-1" in call_kwargs["target_id"]

    @pytest.mark.asyncio
    async def test_skips_scene_already_linked_to_endpoint(self):
        """Scene with any existing stash_id should be skipped."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene",
                [{"type": "md5", "value": "abc123"}],
                stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "existing-uuid"}],
            )],
            1,
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0
        mock_sbc.find_scenes_by_fingerprints.assert_not_called()
        db.delete_pending_scene_fingerprint_for_scene.assert_called_once_with(scene_id="42")

    @pytest.mark.asyncio
    async def test_skips_scene_linked_to_different_endpoint(self):
        """Scene linked to another endpoint is still skipped (must be fully unlinked)."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene",
                [{"type": "md5", "value": "abc123"}],
                stash_ids=[{"endpoint": "https://fansdb.cc/graphql", "stash_id": "fansdb-uuid"}],
            )],
            1,
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0
        mock_sbc.find_scenes_by_fingerprints.assert_not_called()
        db.delete_pending_scene_fingerprint_for_scene.assert_called_once_with(scene_id="42")

    @pytest.mark.asyncio
    async def test_skips_dismissed_pair(self):
        """Dismissed (scene, stashbox_scene) pair should be skipped."""
        db = make_mock_db()
        db.is_dismissed.return_value = True
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])],
            1,
        )

        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [{"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"}],
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0
        db.create_recommendation.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_stashbox_connections_returns_zero(self):
        """If no stash-box endpoints configured, analyzer exits cleanly."""
        db = make_mock_db()
        stash = make_mock_stash()
        stash.get_stashbox_connections.return_value = []

        analyzer = SceneFingerprintMatchAnalyzer(stash, db)
        result = await analyzer.run()

        assert result.items_processed == 0
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_ambiguous_matches_not_high_confidence(self):
        """Multiple matches for same scene from same endpoint = not high confidence."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [
                {"type": "md5", "value": "abc123"},
                {"type": "oshash", "value": "def456"},
            ])],
            1,
        )

        match_a = make_stashbox_match("sb-1", "Match A", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
            {"hash": "def456", "algorithm": "OSHASH", "duration": 1800, "submissions": 2, "created": "2024-01-01", "updated": "2024-06-01"},
        ])
        match_b = make_stashbox_match("sb-2", "Match B", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 1, "created": "2024-01-01", "updated": "2024-06-01"},
        ])

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[match_a, match_b]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 2
        for call in db.create_recommendation.call_args_list:
            details = call[1]["details"]
            assert details["high_confidence"] is False


class TestEndpointPriority:
    """Confirmed live 2026-11-22: a tied 100%/high-confidence match on both
    StashDB and TPDB for the same local scene resolved to TPDB instead of
    the user's configured #1 priority (StashDB), because this analyzer
    never consulted get_endpoint_priorities() at all (every upstream_*
    analyzer already does, see base_upstream.py's run())."""

    @pytest.mark.asyncio
    async def test_endpoint_priority_recorded_in_details(self):
        """Each recommendation records its endpoint's priority rank so the
        recommendations_db.py sort can prefer it over raw confidence ties."""
        db = make_mock_db()
        db.get_endpoint_priorities.return_value = [
            "https://theporndb.net/graphql",
            "https://stashdb.org/graphql",
        ]
        stash = AsyncMock()
        stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "", "name": "StashDB"},
            {"endpoint": "https://theporndb.net/graphql", "api_key": "", "name": "TPDB"},
        ]
        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])], 1,
        )
        match = make_stashbox_match("sb-1", "Match", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 3,
             "created": "2024-01-01", "updated": "2024-06-01"},
        ])

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            await analyzer.run(incremental=False)

        # TPDB is rank 0 in get_endpoint_priorities, StashDB rank 1 -- despite
        # StashDB being first in get_stashbox_connections()'s own order.
        priorities_by_endpoint = {
            call[1]["details"]["endpoint"]: call[1]["details"]["endpoint_priority"]
            for call in db.create_recommendation.call_args_list
        }
        assert priorities_by_endpoint["https://theporndb.net/graphql"] == 0
        assert priorities_by_endpoint["https://stashdb.org/graphql"] == 1

    @pytest.mark.asyncio
    async def test_no_priorities_configured_falls_back_to_connection_order(self):
        """Confirmed live 2026-11-22: this system had never explicitly set
        stash-sense2's own endpoint_priorities setting -- get_endpoint_
        priorities() returns [] -- so the first fix attempt (defaulting
        every endpoint_priority to the same flat 999) left every endpoint
        tied and did nothing for exactly the system it was built for. With
        no explicit priority, get_stashbox_connections()'s own order (Stash's
        Settings > Stash-boxes list order -- what a user without the extra
        setting actually means by "my first configured stash-box") must be
        the implicit priority instead."""
        db = make_mock_db()
        stash = AsyncMock()
        stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "", "name": "StashDB"},
            {"endpoint": "https://theporndb.net/graphql", "api_key": "", "name": "TPDB"},
        ]
        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])], 1,
        )
        match = make_stashbox_match("sb-1", "Match", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 3,
             "created": "2024-01-01", "updated": "2024-06-01"},
        ])

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            await analyzer.run(incremental=False)

        priorities_by_endpoint = {
            call[1]["details"]["endpoint"]: call[1]["details"]["endpoint_priority"]
            for call in db.create_recommendation.call_args_list
        }
        assert priorities_by_endpoint["https://stashdb.org/graphql"] == 0
        assert priorities_by_endpoint["https://theporndb.net/graphql"] == 1


class TestAcceptAction:
    """Test the accept fingerprint match action logic."""

    @pytest.mark.asyncio
    async def test_accept_adds_stash_id_to_scene(self):
        """Accepting a match should add the stash_id to the local scene."""
        from recommendations_router import _accept_fingerprint_match
        from recommendations_db import Recommendation

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {
            "id": "42",
            "stash_ids": [
                {"endpoint": "https://other.org/graphql", "stash_id": "other-uuid"},
            ],
        }
        mock_stash.update_scene.return_value = {"id": "42"}

        mock_db = MagicMock()
        mock_db.get_recommendation.return_value = Recommendation(
            id=1, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="42|https://stashdb.org/graphql|sb-uuid-1",
            details={"local_scene_id": "42"},
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.67, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        mock_db.get_recommendations.return_value = []
        mock_db.resolve_recommendation.return_value = True

        await _accept_fingerprint_match(
            stash=mock_stash,
            db=mock_db,
            rec_id=1,
            scene_id="42",
            endpoint="https://stashdb.org/graphql",
            stash_id="sb-uuid-1",
        )

        # Verify stash_ids includes both old and new
        call_kwargs = mock_stash.update_scene.call_args[1]
        stash_ids = call_kwargs["stash_ids"]
        assert len(stash_ids) == 2
        assert {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-uuid-1"} in stash_ids

        # Verify recommendation was resolved
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")
        mock_db.dismiss_recommendation.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_sync_failure_never_blocks_the_link(self):
        """A metadata-sync failure must never undo or block the stash_id link
        that already succeeded -- bare, unconfigured mocks (get_stashbox_
        connections() etc. aren't wired up) make the sync step raise
        internally; the link + resolve must still go through, and the
        reported result is applied=False with a reason, not an exception."""
        from recommendations_router import _accept_fingerprint_match
        from recommendations_db import Recommendation

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {"id": "42", "stash_ids": []}
        mock_stash.update_scene.return_value = {"id": "42"}

        mock_db = MagicMock()
        mock_db.get_recommendation.return_value = Recommendation(
            id=1, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="42|https://stashdb.org/graphql|sb-uuid-1",
            details={"local_scene_id": "42"},
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=1.0, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        mock_db.resolve_recommendation.return_value = True
        mock_db.dismiss_pending_scene_fingerprint_for_scene.return_value = []

        result = await _accept_fingerprint_match(
            stash=mock_stash, db=mock_db, rec_id=1,
            scene_id="42", endpoint="https://stashdb.org/graphql", stash_id="sb-uuid-1",
        )

        mock_stash.update_scene.assert_called_once()
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")
        assert result["applied"] is False
        assert result["reason"] == "sync_failed"

    @pytest.mark.asyncio
    async def test_metadata_sync_applies_full_scene_recommendation(self):
        """On success, the metadata sync creates an upstream_scene_changes
        recommendation from the real diff/apply pipeline and applies it
        immediately -- confirming _accept_fingerprint_match actually wires
        UpstreamSceneAnalyzer + _apply_full_scene_recommendation together,
        not just that failures are swallowed safely."""
        from recommendations_router import _accept_fingerprint_match
        from recommendations_db import Recommendation

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {"id": "42", "stash_ids": []}
        mock_stash.update_scene.return_value = {"id": "42"}
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"}
        ]
        mock_stash.get_scene_for_upstream_sync.return_value = {
            "id": "42", "title": "Old Title", "performers": [], "tags": [], "studio": None,
        }

        mock_db = MagicMock()
        mock_db.get_recommendation.side_effect = [
            Recommendation(
                id=1, type="scene_fingerprint_match", status="pending",
                target_type="scene", target_id="42|https://stashdb.org/graphql|sb-uuid-1",
                details={"local_scene_id": "42"},
                resolution_action=None, resolution_details=None, resolved_at=None,
                confidence=1.0, source_analysis_id=None,
                created_at="2026-01-01", updated_at="2026-01-01",
            ),
            Recommendation(
                id=99, type="upstream_scene_changes", status="pending",
                target_type="scene", target_id="42",
                details={"scene_id": "42", "endpoint": "https://stashdb.org/graphql"},
                resolution_action=None, resolution_details=None, resolved_at=None,
                confidence=1.0, source_analysis_id=None,
                created_at="2026-01-01", updated_at="2026-01-01",
            ),
        ]
        mock_db.resolve_recommendation.return_value = True
        mock_db.dismiss_pending_scene_fingerprint_for_scene.return_value = []
        mock_db.get_enabled_fields.return_value = None
        mock_db.get_upstream_snapshot.return_value = None
        mock_db.is_dismissed.return_value = False
        mock_db.create_recommendation.return_value = 99

        with patch("recommendations_router.UpstreamSceneAnalyzer") as MockAnalyzer, \
                patch("recommendations_router._apply_full_scene_recommendation", new=AsyncMock(return_value={"success": True})) as mock_apply:
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer._build_name_lookups = AsyncMock()
            mock_analyzer._create_stashbox_client.return_value = AsyncMock()
            mock_analyzer._get_upstream_entity = AsyncMock(return_value={"id": "sb-uuid-1", "title": "New Title"})
            mock_analyzer._normalize_upstream.return_value = {"title": "New Title"}
            mock_analyzer._build_local_data.return_value = {"title": "Old Title"}
            mock_analyzer._get_default_fields.return_value = {"title"}
            mock_analyzer.entity_type = "scene"
            mock_analyzer._diff_fields.return_value = {"changes": [{"field": "title", "upstream_value": "New Title"}]}
            mock_analyzer._get_upstream_updated_at.return_value = "2026-01-01"
            mock_analyzer._build_recommendation_details.return_value = {"scene_id": "42", "endpoint": "https://stashdb.org/graphql"}
            mock_analyzer.create_recommendation.return_value = 99

            result = await _accept_fingerprint_match(
                stash=mock_stash, db=mock_db, rec_id=1,
                scene_id="42", endpoint="https://stashdb.org/graphql", stash_id="sb-uuid-1",
            )

        mock_apply.assert_called_once()
        assert result["applied"] is True

    @pytest.mark.asyncio
    async def test_metadata_sync_ambiguous_performer_reports_reason(self):
        """An ambiguous added performer must leave the link in place (already
        applied before this step runs) but report the reason so the frontend
        can point the user at the Upstream Scene Changes tab instead of
        silently discarding the metadata update."""
        from recommendations_router import _accept_fingerprint_match
        from recommendations_db import Recommendation

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {"id": "42", "stash_ids": []}
        mock_stash.update_scene.return_value = {"id": "42"}
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"}
        ]
        mock_stash.get_scene_for_upstream_sync.return_value = {"id": "42", "performers": [], "tags": [], "studio": None}

        mock_db = MagicMock()
        mock_db.get_recommendation.side_effect = [
            Recommendation(
                id=1, type="scene_fingerprint_match", status="pending",
                target_type="scene", target_id="42|https://stashdb.org/graphql|sb-uuid-1",
                details={"local_scene_id": "42"},
                resolution_action=None, resolution_details=None, resolved_at=None,
                confidence=1.0, source_analysis_id=None,
                created_at="2026-01-01", updated_at="2026-01-01",
            ),
            Recommendation(
                id=99, type="upstream_scene_changes", status="pending",
                target_type="scene", target_id="42", details={},
                resolution_action=None, resolution_details=None, resolved_at=None,
                confidence=1.0, source_analysis_id=None,
                created_at="2026-01-01", updated_at="2026-01-01",
            ),
        ]
        mock_db.resolve_recommendation.return_value = True
        mock_db.dismiss_pending_scene_fingerprint_for_scene.return_value = []
        mock_db.get_enabled_fields.return_value = None
        mock_db.get_upstream_snapshot.return_value = None
        mock_db.is_dismissed.return_value = False
        mock_db.create_recommendation.return_value = 99

        ambiguous_payload = {"ambiguous": [{"stashbox_id": "p1", "name": "Some Performer", "candidates": []}]}

        with patch("recommendations_router.UpstreamSceneAnalyzer") as MockAnalyzer, \
                patch("recommendations_router._apply_full_scene_recommendation", new=AsyncMock(return_value=ambiguous_payload)):
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer._build_name_lookups = AsyncMock()
            mock_analyzer._create_stashbox_client.return_value = AsyncMock()
            mock_analyzer._get_upstream_entity = AsyncMock(return_value={"id": "sb-uuid-1"})
            mock_analyzer._normalize_upstream.return_value = {}
            mock_analyzer._build_local_data.return_value = {}
            mock_analyzer._get_default_fields.return_value = set()
            mock_analyzer.entity_type = "scene"
            mock_analyzer._diff_fields.return_value = {"performer_changes": {"added": [{"id": "p1", "name": "Some Performer"}]}}
            mock_analyzer._get_upstream_updated_at.return_value = "2026-01-01"
            mock_analyzer._build_recommendation_details.return_value = {}
            mock_analyzer.create_recommendation.return_value = 99

            result = await _accept_fingerprint_match(
                stash=mock_stash, db=mock_db, rec_id=1,
                scene_id="42", endpoint="https://stashdb.org/graphql", stash_id="sb-uuid-1",
            )

        # The stash_id link + resolve already happened before this step --
        # an ambiguous performer only blocks the metadata sync, not the link.
        mock_stash.update_scene.assert_called_once()
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")
        assert result["applied"] is False
        assert result["reason"] == "ambiguous_performer"
        assert result["ambiguous"] == ambiguous_payload["ambiguous"]

    @pytest.mark.asyncio
    async def test_accept_dismisses_other_pending_matches_for_same_scene(self):
        """Accepting one match dismisses sibling pending matches for the same local scene.

        _accept_fingerprint_match() delegates this to a single bulk
        dismiss_pending_scene_fingerprint_for_scene() call (one SQL sweep
        keyed on the local scene id, excluding the just-accepted rec) --
        not per-sibling get_recommendations()/dismiss_recommendation()
        calls.
        """
        from recommendations_router import _accept_fingerprint_match
        from recommendations_db import Recommendation

        accepted_rec = Recommendation(
            id=1, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="42|https://stashdb.org/graphql|sb-1",
            details={"local_scene_id": "42"},
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.8, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {"id": "42", "stash_ids": []}
        mock_stash.update_scene.return_value = {"id": "42"}

        mock_db = MagicMock()
        mock_db.get_recommendation.return_value = accepted_rec
        mock_db.resolve_recommendation.return_value = True
        mock_db.dismiss_pending_scene_fingerprint_for_scene.return_value = [2]

        result = await _accept_fingerprint_match(
            stash=mock_stash,
            db=mock_db,
            rec_id=1,
            scene_id="42",
            endpoint="https://stashdb.org/graphql",
            stash_id="sb-1",
        )

        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")
        mock_db.dismiss_pending_scene_fingerprint_for_scene.assert_called_once()
        dismiss_call = mock_db.dismiss_pending_scene_fingerprint_for_scene.call_args
        assert dismiss_call[1]["scene_id"] == "42"
        assert dismiss_call[1]["exclude_rec_id"] == 1
        assert "Auto-dismissed after accepting scene fingerprint match" in dismiss_call[1]["reason"]

        # Callers (the API response, in turn the plugin's own list cache)
        # need the ids of any siblings that got silently dismissed as a
        # side effect, not just a count -- see accept_fingerprint_match's
        # own AcceptFingerprintMatchResponse.auto_dismissed_rec_ids.
        # _accept_fingerprint_match now returns a dict (not a bare list) so
        # the metadata-sync outcome can ride along too -- see
        # test_metadata_sync_failure_never_blocks_the_link below.
        assert result["auto_dismissed_rec_ids"] == [2]


class TestAcceptAllAction:
    """Test the accept-all fingerprint matches action."""

    @pytest.mark.asyncio
    async def test_accepts_only_high_confidence(self):
        """Only high_confidence recommendations should be accepted."""
        from recommendations_router import _accept_all_fingerprint_matches
        from recommendations_db import Recommendation

        high_conf_rec = Recommendation(
            id=1, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="42|https://stashdb.org/graphql|sb-1",
            details={
                "local_scene_id": "42",
                "endpoint": "https://stashdb.org/graphql",
                "stashbox_scene_id": "sb-1",
                "high_confidence": True,
            },
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.67, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        low_conf_rec = Recommendation(
            id=2, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="43|https://stashdb.org/graphql|sb-2",
            details={
                "local_scene_id": "43",
                "endpoint": "https://stashdb.org/graphql",
                "stashbox_scene_id": "sb-2",
                "high_confidence": False,
            },
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.33, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )

        mock_db = MagicMock()
        mock_db.get_recommendations.return_value = [high_conf_rec, low_conf_rec]
        mock_db.get_recommendation.return_value = high_conf_rec
        mock_db.resolve_recommendation.return_value = True
        mock_db.dismiss_recommendation.return_value = True

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {"id": "42", "stash_ids": []}
        mock_stash.update_scene.return_value = {"id": "42"}

        accepted = await _accept_all_fingerprint_matches(mock_stash, mock_db)

        assert accepted == 1
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")


class TestPreferWesternNamesStudio:
    """settings.py's prefer_western_names applied to the matched scene's
    own studio name -- stash-box's Studio type has no alias field, so the
    only source of a western name is a local studio already stash_id-
    linked to this same endpoint (see stash_client_unified.py's
    get_studios_for_endpoint, now also fetching aliases)."""

    @pytest.mark.asyncio
    async def test_swaps_to_local_studio_alias_when_setting_on(self):
        db = make_mock_db()
        stash = make_mock_stash()
        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])], 1,
        )
        stash.get_studios_for_endpoint.return_value = [
            {
                "id": "1462", "name": "カリビアンコム", "aliases": ["Caribbeancom"],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "s1"}],
            }
        ]
        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [{"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5,
              "created": "2024-01-01", "updated": "2024-06-01"}],
            studio={"id": "s1", "name": "カリビアンコム"},
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC, \
                patch("settings.get_setting", return_value=True):
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc
            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            await analyzer.run(incremental=False)

        details = db.create_recommendation.call_args[1]["details"]
        assert details["stashbox_studio"] == "Caribbeancom"
        assert details["stashbox_studio_original"] == "カリビアンコム"

    @pytest.mark.asyncio
    async def test_unchanged_when_setting_off(self):
        db = make_mock_db()
        stash = make_mock_stash()
        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])], 1,
        )
        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [{"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5,
              "created": "2024-01-01", "updated": "2024-06-01"}],
            studio={"id": "s1", "name": "カリビアンコム"},
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC, \
                patch("settings.get_setting", return_value=False):
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc
            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            await analyzer.run(incremental=False)

        details = db.create_recommendation.call_args[1]["details"]
        assert details["stashbox_studio"] == "カリビアンコム"
        assert details["stashbox_studio_original"] is None
        # get_studios_for_endpoint should not even be queried when the
        # setting is off -- no local-studio lookup needed at all.
        stash.get_studios_for_endpoint.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_linked_local_studio_leaves_name_unchanged(self):
        db = make_mock_db()
        stash = make_mock_stash()
        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])], 1,
        )
        stash.get_studios_for_endpoint.return_value = []  # nothing linked yet
        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [{"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5,
              "created": "2024-01-01", "updated": "2024-06-01"}],
            studio={"id": "s1", "name": "カリビアンコム"},
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC, \
                patch("settings.get_setting", return_value=True):
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc
            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            await analyzer.run(incremental=False)

        details = db.create_recommendation.call_args[1]["details"]
        assert details["stashbox_studio"] == "カリビアンコム"
        assert details["stashbox_studio_original"] is None

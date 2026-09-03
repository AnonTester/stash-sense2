"""Tests for the ambiguous-performer-identity handling added to the three
accept/create flows that resolve a StashBox performer to a local one:
accept_scene_face_matches (Face Recommendations), _apply_full_scene_recommendation
(Upstream Scene Changes), and create_performer_action (standalone/live
Identify "Add to Stash + Scene"). See recommendations_router.py's
PerformerIdentityAmbiguous/_check_performer_identity docstrings for the
underlying contract this all builds on -- confirmed live, matching by name/
alias alone silently linked an unrelated existing performer instead of
creating a new one, corrupting that performer's own stash_ids in the
process.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "test.db"))


@pytest.fixture(autouse=True)
def mock_connection_manager():
    """_resolve_scene_face_match_performer_id and _fetch_stashbox_performer_urls
    both call get_connection_manager() directly -- stub it so tests don't
    need a real StashBoxConnectionManager. get_client returns None so the
    URL hard-evidence check's on-demand stashbox fetch cleanly no-ops
    (falls through to "no URL evidence available", not an error)."""
    mgr = MagicMock()
    mgr.get_endpoint_url.return_value = "https://stashdb.org/graphql"
    mgr.get_client.return_value = None
    with patch("stashbox_connection_manager.get_connection_manager", return_value=mgr):
        yield mgr


def _mock_stash(**overrides):
    stash = MagicMock()
    stash.get_scene_by_id = AsyncMock(return_value={"id": "1", "performers": []})
    stash.update_scene_performers = AsyncMock(return_value=True)
    stash.search_performers = AsyncMock(return_value=[])
    stash.get_all_performers = AsyncMock(return_value=[])
    stash.create_performer = AsyncMock(return_value={"id": "999", "name": "Created"})
    stash.update_performer = AsyncMock(return_value={"id": "999"})
    stash.get_performer = AsyncMock(return_value={"id": "999", "name": "Created", "stash_ids": []})
    # _find_linked_entity_by_stash_id and _get_scene_tags_with_stash_ids
    # both go through _execute -- dispatch on the query text so one mock
    # can serve both.
    async def _execute(query, variables=None):
        if "findScene" in query:
            return {"findScene": {"tags": []}}
        return {"findPerformers": {"performers": []}}
    stash._execute = AsyncMock(side_effect=_execute)
    for k, v in overrides.items():
        setattr(stash, k, v)
    return stash


class TestAcceptSceneFaceMatchesAmbiguity:
    async def test_ambiguous_selection_blocks_the_whole_batch(self, db):
        """Confirmed decision: one ambiguous selection in a multi-select
        accept must block ALL of it, not just be skipped -- an accepted
        recommendation disappears from the pending queue, so a partial
        accept would hide the unresolved one from view."""
        from recommendations_router import accept_scene_face_matches, AcceptSceneFaceMatchesRequest, SceneFaceMatchSelection

        clean_id = db.create_recommendation(
            type="scene_face_match", target_type="scene", target_id="1|stashdb.org:clean-uuid",
            details={"scene_id": "1", "name": "Brand New Person", "stashdb_id": "clean-uuid", "endpoint": "stashdb.org"},
            confidence=0.9,
        )
        ambiguous_id = db.create_recommendation(
            type="scene_face_match", target_type="scene", target_id="1|stashdb.org:collide-uuid",
            details={"scene_id": "1", "name": "Jane Doe", "stashdb_id": "collide-uuid", "endpoint": "stashdb.org"},
            confidence=0.8,
        )

        stash = _mock_stash(search_performers=AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []},
        ]))

        import recommendations_router as rec_mod
        rec_mod.rec_db = db
        rec_mod.stash_client = stash

        result = await accept_scene_face_matches(AcceptSceneFaceMatchesRequest(
            scene_id="1",
            selections=[
                SceneFaceMatchSelection(recommendation_id=clean_id),
                SceneFaceMatchSelection(recommendation_id=ambiguous_id),
            ],
        ))

        assert result["success"] is False
        assert len(result["ambiguous"]) == 1
        assert result["ambiguous"][0]["recommendation_id"] == ambiguous_id
        assert result["ambiguous"][0]["candidates"][0]["id"] == "42"

        # Nothing applied -- not even the clean selection.
        stash.update_scene_performers.assert_not_called()
        assert db.get_recommendation(clean_id).status == "pending"
        assert db.get_recommendation(ambiguous_id).status == "pending"

    async def test_resubmitting_with_resolution_completes_the_accept(self, db):
        from recommendations_router import accept_scene_face_matches, AcceptSceneFaceMatchesRequest, SceneFaceMatchSelection

        rec_id = db.create_recommendation(
            type="scene_face_match", target_type="scene", target_id="1|stashdb.org:collide-uuid",
            details={"scene_id": "1", "name": "Jane Doe", "stashdb_id": "collide-uuid", "endpoint": "stashdb.org"},
            confidence=0.8,
        )

        stash = _mock_stash(search_performers=AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []},
        ]))
        stash.get_performer = AsyncMock(return_value={"id": "42", "name": "Main Name", "stash_ids": []})

        import recommendations_router as rec_mod
        rec_mod.rec_db = db
        rec_mod.stash_client = stash

        result = await accept_scene_face_matches(AcceptSceneFaceMatchesRequest(
            scene_id="1",
            selections=[
                SceneFaceMatchSelection(recommendation_id=rec_id, resolved_performer_id="42"),
            ],
        ))

        assert result["success"] is True
        stash.update_scene_performers.assert_called_once_with("1", ["42"])
        assert db.get_recommendation(rec_id).status == "resolved"


class TestAcceptSceneFaceMatchesStalePerformer:
    """A scene_face_match candidate whose local_performer_id was deleted or
    merged away in Stash directly (no hook/sync has caught up yet) used to
    crash this endpoint outright: Stash's own sceneUpdate raised a raw
    FOREIGN KEY error with no validation of its own, surfacing as an
    opaque 500 -- confirmed live (performer merged into a different id,
    "Internal server error" on accept). Regression coverage for the fix:
    validate before mutating, dismiss (not silently drop) the stale
    recommendation, and refresh the scene's own match data so a future
    scan can surface the correct candidate instead of repeating the dead
    one -- see accept_scene_face_matches's own comment."""

    async def test_stale_local_performer_id_is_dismissed_not_crashed(self, db):
        from recommendations_router import accept_scene_face_matches, AcceptSceneFaceMatchesRequest, SceneFaceMatchSelection

        rec_id = db.create_recommendation(
            type="scene_face_match", target_type="scene", target_id="1|local:2722",
            details={"scene_id": "1", "name": "Nattyprincessx", "local_performer_id": "2722"},
            confidence=0.9,
        )

        # get_performer(None) simulates Stash's findPerformer returning
        # null for an id that's been deleted/merged away.
        stash = _mock_stash(get_performer=AsyncMock(return_value=None))

        import recommendations_router as rec_mod
        rec_mod.rec_db = db
        rec_mod.stash_client = stash

        with patch("identification_router._identify_scene_impl", new=AsyncMock()) as mock_reidentify:
            result = await accept_scene_face_matches(AcceptSceneFaceMatchesRequest(
                scene_id="1",
                selections=[SceneFaceMatchSelection(recommendation_id=rec_id)],
            ))

        assert result["success"] is False
        assert result["stale_performers"] == ["2722"]

        # Never attempted the mutation that would have raised the FK error.
        stash.update_scene_performers.assert_not_called()

        # Dismissed (not left pending, not marked "accepted") -- it must
        # leave the pending queue since retrying the exact same selection
        # would just fail the exact same way again.
        rec = db.get_recommendation(rec_id)
        assert rec.status == "resolved"
        assert rec.resolution_action == "dismissed"

        # Scene's own match data was refreshed so a future scan can find
        # whoever this performer was actually merged into.
        mock_reidentify.assert_called_once()
        assert mock_reidentify.call_args[0][0].scene_id == "1"

    async def test_stale_performer_alongside_a_valid_one_still_accepts_the_valid_one(self, db):
        """Two candidates for the same scene, one stale and one still
        valid -- the valid one must not be collateral damage."""
        from recommendations_router import accept_scene_face_matches, AcceptSceneFaceMatchesRequest, SceneFaceMatchSelection

        stale_rec_id = db.create_recommendation(
            type="scene_face_match", target_type="scene", target_id="1|local:2722",
            details={"scene_id": "1", "name": "Stale Person", "local_performer_id": "2722"},
            confidence=0.9,
        )
        valid_rec_id = db.create_recommendation(
            type="scene_face_match", target_type="scene", target_id="1|local:42",
            details={"scene_id": "1", "name": "Valid Person", "local_performer_id": "42"},
            confidence=0.85,
        )

        async def _get_performer(performer_id):
            return None if performer_id == "2722" else {"id": "42", "name": "Valid Person", "stash_ids": []}

        stash = _mock_stash(get_performer=AsyncMock(side_effect=_get_performer))

        import recommendations_router as rec_mod
        rec_mod.rec_db = db
        rec_mod.stash_client = stash

        with patch("identification_router._identify_scene_impl", new=AsyncMock()):
            result = await accept_scene_face_matches(AcceptSceneFaceMatchesRequest(
                scene_id="1",
                selections=[
                    SceneFaceMatchSelection(recommendation_id=stale_rec_id),
                    SceneFaceMatchSelection(recommendation_id=valid_rec_id),
                ],
            ))

        assert result["success"] is True
        stash.update_scene_performers.assert_called_once_with("1", ["42"])
        assert db.get_recommendation(stale_rec_id).resolution_action == "dismissed"
        assert db.get_recommendation(valid_rec_id).resolution_action == "accepted"


class TestApplyFullSceneRecommendationAmbiguity:
    def _seed_rec(self, db, added_performers):
        return db.create_recommendation(
            type="upstream_scene_changes", target_type="scene", target_id="1",
            details={
                "scene_id": "1", "endpoint": "stashdb.org",
                "current_performer_ids": [],
                "performer_changes": {"added": added_performers, "removed": []},
            },
            confidence=0.9,
        )

    async def test_ambiguous_added_performer_blocks_everything_for_this_rec(self, db):
        """No field/tag/studio/scene mutation happens at all when an added
        performer is ambiguous -- performers are resolved before anything
        else in this recommendation is touched."""
        from recommendations_router import _apply_full_scene_recommendation

        rec_id = self._seed_rec(db, [{"id": "collide-uuid", "name": "Jane Doe", "aliases": []}])
        rec = db.get_recommendation(rec_id)

        stash = _mock_stash(search_performers=AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []},
        ]))

        result = await _apply_full_scene_recommendation(stash, db, rec)

        assert "ambiguous" in result
        assert result["ambiguous"][0]["stashbox_id"] == "collide-uuid"
        assert result["ambiguous"][0]["candidates"][0]["id"] == "42"
        # Never reached the tag-fetch step (a findScene query) -- only the
        # performer stash_id pre-checks (findPerformers) ran.
        queries = [c.args[0] for c in stash._execute.call_args_list]
        assert not any("findScene" in q for q in queries)
        assert db.get_recommendation(rec_id).status == "pending"

    async def test_resolutions_applies_the_users_explicit_choice(self, db):
        from recommendations_router import _apply_full_scene_recommendation

        rec_id = self._seed_rec(db, [{"id": "collide-uuid", "name": "Jane Doe", "aliases": []}])
        rec = db.get_recommendation(rec_id)

        stash = _mock_stash(search_performers=AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []},
        ]))
        stash.get_performer = AsyncMock(return_value={"id": "42", "name": "Main Name", "stash_ids": []})
        stash.update_scene = AsyncMock(return_value={"id": "1"})

        result = await _apply_full_scene_recommendation(
            stash, db, rec, resolutions={"collide-uuid": {"action": "link", "performer_id": "42"}},
        )

        assert "ambiguous" not in result
        assert result["action"] == "applied"
        stash.update_scene.assert_called_once()
        call_kwargs = stash.update_scene.call_args.kwargs
        assert call_kwargs["performer_ids"] == ["42"]
        assert db.get_recommendation(rec_id).status == "resolved"

    async def test_resolutions_create_action_forces_a_new_performer(self, db):
        """A resolved "create" action skips the identity check entirely --
        the user already looked at the candidates and said "none of these,
        make a new one"."""
        from recommendations_router import _apply_full_scene_recommendation

        rec_id = self._seed_rec(db, [{"id": "collide-uuid", "name": "Jane Doe", "aliases": []}])
        rec = db.get_recommendation(rec_id)

        stash = _mock_stash(search_performers=AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []},
        ]))
        stash.create_performer = AsyncMock(return_value={"id": "777", "name": "Jane Doe"})
        stash.update_scene = AsyncMock(return_value={"id": "1"})

        result = await _apply_full_scene_recommendation(
            stash, db, rec, resolutions={"collide-uuid": {"action": "create"}},
        )

        assert "ambiguous" not in result
        stash.create_performer.assert_called_once()
        call_kwargs = stash.update_scene.call_args.kwargs
        assert call_kwargs["performer_ids"] == ["777"]


class TestCreatePerformerActionAmbiguity:
    """The standalone /actions/create-performer endpoint -- used directly
    by the live Identify modal's "Add to Stash + Scene" button."""

    async def test_ambiguous_match_returns_candidates_instead_of_creating(self):
        from recommendations_router import create_performer_action, CreatePerformerRequest
        import recommendations_router as rec_mod

        stash = _mock_stash(search_performers=AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []},
        ]))
        rec_mod.stash_client = stash

        result = await create_performer_action(CreatePerformerRequest(
            stashbox_data={"name": "Jane Doe"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="collide-uuid",
        ))

        assert result["success"] is False
        assert result["ambiguous"] is True
        assert result["candidates"][0]["id"] == "42"
        stash.create_performer.assert_not_called()

    async def test_resolved_performer_id_links_without_rechecking(self):
        from recommendations_router import create_performer_action, CreatePerformerRequest
        import recommendations_router as rec_mod

        stash = _mock_stash()
        stash.get_performer = AsyncMock(return_value={"id": "42", "name": "Main Name", "stash_ids": []})
        rec_mod.stash_client = stash

        result = await create_performer_action(CreatePerformerRequest(
            stashbox_data={"name": "Jane Doe"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="collide-uuid",
            resolved_performer_id="42",
        ))

        assert result["success"] is True
        assert result["performer"]["id"] == "42"
        stash.search_performers.assert_not_called()
        stash.update_performer.assert_called_once_with(
            "42", stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "collide-uuid"}],
        )

    async def test_force_create_skips_the_ambiguity_check(self):
        from recommendations_router import create_performer_action, CreatePerformerRequest
        import recommendations_router as rec_mod

        stash = _mock_stash(search_performers=AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []},
        ]))
        stash.create_performer = AsyncMock(return_value={"id": "777", "name": "Jane Doe"})
        rec_mod.stash_client = stash

        result = await create_performer_action(CreatePerformerRequest(
            stashbox_data={"name": "Jane Doe"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="collide-uuid",
            force_create=True,
        ))

        assert result["success"] is True
        assert result["performer"]["id"] == "777"
        stash.search_performers.assert_not_called()

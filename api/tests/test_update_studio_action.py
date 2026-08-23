"""Tests for the update-studio action endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a test app with recommendations router."""
    from fastapi import FastAPI
    from recommendations_router import router, init_recommendations
    import tempfile
    import os

    app = FastAPI()
    app.include_router(router)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        init_recommendations(db_path, "http://localhost:9999", "test-key")
        yield app


class TestUpdateStudioEndpoint:
    def test_update_studio_name(self, app):
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "1"})
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "1",
                "fields": {"name": "New Name"},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            mock_stash.update_studio.assert_called_once()

    def test_update_studio_url(self, app):
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "1"})
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "1",
                "fields": {"url": "https://new-url.com"},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200

    def test_update_studio_merges_alias_add_meta_key(self, app):
        """Regression test: studios do have an aliases field (already
        queried elsewhere, e.g. _resolve_stashbox_studio_to_local), but this
        endpoint never translated the _alias_add meta-key into it like
        update_tag_fields/update_performer_fields do -- it reached
        stash.update_studio(**fields) unhandled, which isn't a real
        StudioUpdateInput field, failing the whole mutation for every
        studio name-normalization change sent through "Accept All"."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "1"})
            mock_stash._execute = AsyncMock(return_value={
                "findStudio": {"id": "1", "name": "Old Name", "aliases": ["Existing Alias"]}
            })
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "1",
                "fields": {"name": "New Name", "_alias_add": ["Old Name"]},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            call_kwargs = mock_stash.update_studio.call_args.kwargs
            assert "_alias_add" not in call_kwargs
            assert set(call_kwargs["aliases"]) == {"Existing Alias", "Old Name"}

    def test_update_studio_alias_add_dedupes_case_insensitively(self, app):
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "1"})
            mock_stash._execute = AsyncMock(return_value={
                "findStudio": {"id": "1", "name": "Studio", "aliases": ["Old Name"]}
            })
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "1",
                "fields": {"_alias_add": ["old name"]},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200
            call_kwargs = mock_stash.update_studio.call_args.kwargs
            assert call_kwargs["aliases"] == ["Old Name"]

    def test_update_studio_parent_found_locally(self, app):
        """When parent studio exists locally, resolve its ID."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "20"})
            mock_stash._execute = AsyncMock(return_value={
                "findStudios": {
                    "studios": [{"id": "10", "name": "Parent Studio"}]
                }
            })
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "20",
                "fields": {"parent_studio": "parent-stashbox-uuid"},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200


class TestResolveStashboxStudioToLocal:
    """Tests for _resolve_stashbox_studio_to_local parent studio resolution."""

    @pytest.fixture
    def app_and_db(self):
        from fastapi import FastAPI
        from recommendations_router import router, init_recommendations
        import tempfile, os
        app = FastAPI()
        app.include_router(router)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            init_recommendations(db_path, "http://localhost:9999", "test-key")
            yield app

    @pytest.mark.asyncio
    async def test_finds_studio_by_alias_match(self, app_and_db):
        """When upstream name matches a local studio alias, link instead of create.

        Reproduces: StashBox parent 'Jacquie et Michel' should match local
        'Jacquie & Michel TV' which has alias 'Jacquie et Michel'.
        """
        from recommendations_router import _resolve_stashbox_studio_to_local

        mock_stash = MagicMock()

        # Step 1: stash_box_id lookup returns no results
        # Step 2: search_studios returns the local studio that has the alias
        async def mock_execute(query, variables=None, **kwargs):
            if "stash_id_endpoint" in str(variables):
                return {"findStudios": {"studios": []}}
            raise AssertionError(f"Unexpected _execute call: {query}")
        mock_stash._execute = AsyncMock(side_effect=mock_execute)

        # search_studios returns a studio whose primary name differs but has matching alias
        mock_stash.search_studios = AsyncMock(return_value=[
            {
                "id": "42",
                "name": "Jacquie & Michel TV",
                "aliases": ["Jacquie et Michel"],
                "stash_ids": [],
            }
        ])
        mock_stash.update_studio = AsyncMock(return_value={"id": "42"})

        # Mock stashbox client to return upstream studio data
        mock_sbc = MagicMock()
        mock_sbc.get_studio = AsyncMock(return_value={
            "name": "Jacquie et Michel",
            "urls": [{"url": "https://jacquieetmichel.net"}],
        })

        with patch("stashbox_connection_manager.get_connection_manager") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = mock_sbc

            result = await _resolve_stashbox_studio_to_local(
                mock_stash, "parent-stashbox-uuid", "https://stashdb.org/graphql"
            )

        # Should find and link the existing studio, not try to create
        assert result == "42"
        mock_stash.update_studio.assert_called_once()
        # Should NOT have tried to create a new studio
        assert not hasattr(mock_stash, 'create_studio') or not mock_stash.create_studio.called

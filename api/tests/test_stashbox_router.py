"""Tests for the stashbox API router."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import stashbox_router as sb_mod
from stashbox_router import _map_stashbox_to_stash


# ==================== Pure function tests for _map_stashbox_to_stash ====================


class TestMapStashboxToStash:
    """Test _map_stashbox_to_stash() pure function."""

    def _base_performer(self, **overrides):
        """Create a minimal performer dict with optional overrides."""
        perf = {"name": "Test Performer"}
        perf.update(overrides)
        return perf

    def test_breast_type_natural(self):
        result = _map_stashbox_to_stash(
            self._base_performer(breast_type="NATURAL"),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.fake_tits == "Natural"

    def test_breast_type_fake(self):
        result = _map_stashbox_to_stash(
            self._base_performer(breast_type="FAKE"),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.fake_tits == "Augmented"

    def test_breast_type_none(self):
        result = _map_stashbox_to_stash(
            self._base_performer(),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.fake_tits is None

    def test_breast_type_case_insensitive(self):
        result = _map_stashbox_to_stash(
            self._base_performer(breast_type="natural"),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.fake_tits == "Natural"

    def test_measurements_full(self):
        result = _map_stashbox_to_stash(
            self._base_performer(cup_size="F", band_size=38, waist_size=24, hip_size=35),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.measurements == "38F-24-35"

    def test_measurements_partial_no_waist(self):
        result = _map_stashbox_to_stash(
            self._base_performer(cup_size="C", band_size=34, hip_size=36),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.measurements == "34C--36"

    def test_measurements_none_when_all_empty(self):
        result = _map_stashbox_to_stash(
            self._base_performer(),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.measurements is None

    def test_career_length_start_and_end(self):
        result = _map_stashbox_to_stash(
            self._base_performer(career_start_year=2015, career_end_year=2020),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.career_length == "2015-2020"

    def test_career_length_start_only(self):
        result = _map_stashbox_to_stash(
            self._base_performer(career_start_year=2015),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.career_length == "2015-"

    def test_career_length_none(self):
        result = _map_stashbox_to_stash(
            self._base_performer(),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.career_length is None

    def test_tattoos_formatting(self):
        result = _map_stashbox_to_stash(
            self._base_performer(tattoos=[
                {"location": "Left arm", "description": "Sleeve"},
                {"location": "Back", "description": "Dragon"},
            ]),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.tattoos == "Left arm: Sleeve; Back: Dragon"

    def test_piercings_formatting(self):
        result = _map_stashbox_to_stash(
            self._base_performer(piercings=[
                {"location": "Navel", "description": "Ring"},
            ]),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.piercings == "Navel: Ring"

    def test_tattoos_location_only(self):
        result = _map_stashbox_to_stash(
            self._base_performer(tattoos=[{"location": "Right ankle", "description": ""}]),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.tattoos == "Right ankle"

    def test_tattoos_description_only(self):
        result = _map_stashbox_to_stash(
            self._base_performer(tattoos=[{"location": "", "description": "Small heart"}]),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.tattoos == "Small heart"

    def test_tattoos_none(self):
        result = _map_stashbox_to_stash(
            self._base_performer(),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.tattoos is None

    def test_url_extraction(self):
        result = _map_stashbox_to_stash(
            self._base_performer(urls=[
                {"url": "https://twitter.com/test", "type": "TWITTER"},
                {"url": "https://instagram.com/test", "type": "INSTAGRAM"},
            ]),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.urls == ["https://twitter.com/test", "https://instagram.com/test"]

    def test_url_extraction_empty(self):
        result = _map_stashbox_to_stash(
            self._base_performer(),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.urls == []

    def test_image_selection_first(self):
        result = _map_stashbox_to_stash(
            self._base_performer(images=[
                {"url": "https://cdn.stashdb.org/img1.jpg"},
                {"url": "https://cdn.stashdb.org/img2.jpg"},
            ]),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.image == "https://cdn.stashdb.org/img1.jpg"

    def test_image_selection_no_images(self):
        result = _map_stashbox_to_stash(
            self._base_performer(),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.image is None

    def test_alias_mapping(self):
        result = _map_stashbox_to_stash(
            self._base_performer(aliases=["Alias One", "Alias Two"]),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.alias_list == ["Alias One", "Alias Two"]

    def test_alias_mapping_none(self):
        result = _map_stashbox_to_stash(
            self._base_performer(aliases=None),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.alias_list == []

    def test_stash_ids_construction(self):
        result = _map_stashbox_to_stash(
            self._base_performer(),
            "https://stashdb.org/graphql", "abc-uuid-123",
        )
        assert result.stash_ids == [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-uuid-123"}]

    def test_basic_field_mapping(self):
        result = _map_stashbox_to_stash(
            self._base_performer(
                name="Jane Doe",
                disambiguation="the actor",
                gender="FEMALE",
                birth_date="1990-01-15",
                death_date=None,
                ethnicity="Caucasian",
                country="US",
                eye_color="Blue",
                hair_color="Blonde",
                height=165,
            ),
            "https://stashdb.org/graphql", "abc123",
        )
        assert result.name == "Jane Doe"
        assert result.disambiguation == "the actor"
        assert result.gender == "FEMALE"
        assert result.birthdate == "1990-01-15"
        assert result.death_date is None
        assert result.ethnicity == "Caucasian"
        assert result.country == "US"
        assert result.eye_color == "Blue"
        assert result.hair_color == "Blonde"
        assert result.height_cm == 165


# ==================== Endpoint tests ====================


@pytest.fixture
def stashbox_client():
    """Create a test client for the stashbox router."""
    original_url = sb_mod._stash_url
    original_key = sb_mod._stash_api_key

    sb_mod._stash_url = ""
    sb_mod._stash_api_key = ""

    app = FastAPI()
    app.include_router(sb_mod.router)
    test_client = TestClient(app)

    yield test_client

    sb_mod._stash_url = original_url
    sb_mod._stash_api_key = original_key


class TestGetStashboxPerformer:
    """Test GET /stashbox/performer/{endpoint}/{id}."""

    def test_unknown_endpoint_returns_400(self, stashbox_client):
        with patch("stashbox_router._get_stashbox_client", return_value=None):
            resp = stashbox_client.get("/stashbox/performer/unknown_endpoint/abc123")
            assert resp.status_code == 400
            assert "Unknown or unconfigured endpoint" in resp.json()["detail"]

    def test_performer_not_found_returns_404(self, stashbox_client):
        mock_client = AsyncMock()
        mock_client.get_performer.return_value = None

        with patch("stashbox_router._get_stashbox_client", return_value=mock_client):
            resp = stashbox_client.get("/stashbox/performer/stashdb.org/abc123")
            assert resp.status_code == 404

    def test_success_returns_mapped_performer(self, stashbox_client):
        mock_client = AsyncMock()
        mock_client.get_performer.return_value = {
            "name": "Test Performer",
            "gender": "FEMALE",
            "breast_type": "NATURAL",
            "aliases": ["Alias1"],
        }

        with patch("stashbox_router._get_stashbox_client", return_value=mock_client), \
             patch("stashbox_router._get_endpoint_url", return_value="https://stashdb.org/graphql"):
            resp = stashbox_client.get("/stashbox/performer/stashdb.org/abc123")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Test Performer"
            assert data["fake_tits"] == "Natural"
            assert data["alias_list"] == ["Alias1"]


class TestSearchStashPerformers:
    """Test POST /stash/search-performers."""

    def test_missing_stash_url_returns_400(self, stashbox_client):
        resp = stashbox_client.post(
            "/stash/search-performers",
            json={"query": "test"},
        )
        assert resp.status_code == 400
        assert "STASH_URL not configured" in resp.json()["detail"]

    def test_search_with_configured_url(self, stashbox_client):
        sb_mod._stash_url = "http://localhost:9999"
        sb_mod._stash_api_key = "test-key"

        mock_stash = AsyncMock()
        mock_stash._execute.return_value = {
            "findPerformers": {
                "performers": [
                    {"id": "1", "name": "Test", "disambiguation": None, "alias_list": [], "image_path": None},
                ]
            }
        }

        # StashClientUnified is imported inside the endpoint function, so patch the source module
        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            resp = stashbox_client.post(
                "/stash/search-performers",
                json={"query": "Test"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["name"] == "Test"


class TestCreatePerformerNameTaken:
    """POST /stash/create-performer(-from-catalogue) when Stash already has a performer with that name:
    a 409 whose message names the blocking performer, not an unhandled 500 (the plugin showed a bare "Failed")."""

    TAKEN = RuntimeError("GraphQL error: [{'message': \"performer with name 'Jane Roe' already exists\", 'path': ['performerCreate']}]")

    def _mock_stash(self, blocking=True):
        mock_stash = AsyncMock()
        found = [{"id": "4242", "name": "Jane Roe", "disambiguation": "", "stash_ids": []}] if blocking else []

        async def execute(query, variables=None, **kwargs):
            if "performerCreate" in query:
                raise self.TAKEN
            return {"findPerformers": {"performers": found}}

        mock_stash._execute.side_effect = execute
        mock_stash.create_performer.side_effect = self.TAKEN
        return mock_stash

    def _post_stashbox(self, client, mock_stash):
        sb_mod._stash_url = "http://localhost:9999"
        sb_mod._stash_api_key = "test-key"
        stashbox = AsyncMock()
        stashbox.get_performer.return_value = {"name": "Jane Roe", "aliases": ["Jane Doe"]}
        with patch("stashbox_router._get_stashbox_client", return_value=stashbox), \
             patch("stashbox_router._get_endpoint_url", return_value="https://javstash.org/graphql"), \
             patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            return client.post(
                "/stash/create-performer",
                json={"endpoint": "javstash.org", "stashdb_id": "a2ae8711", "scene_id": "1"},
            )

    def test_stashbox_name_taken_returns_409_naming_the_blocker(self, stashbox_client):
        resp = self._post_stashbox(stashbox_client, self._mock_stash())
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "'Jane Roe' already exists" in detail
        assert "id 4242" in detail
        assert "Add as" in detail

    def test_stashbox_name_taken_without_lookup_result_still_409(self, stashbox_client):
        resp = self._post_stashbox(stashbox_client, self._mock_stash(blocking=False))
        assert resp.status_code == 409
        assert "'Jane Roe' already exists" in resp.json()["detail"]
        assert "id " not in resp.json()["detail"]

    def test_stashbox_other_create_errors_are_not_swallowed(self, stashbox_client):
        mock_stash = AsyncMock()
        mock_stash._execute.side_effect = RuntimeError("GraphQL error: boom")
        with pytest.raises(RuntimeError, match="boom"):
            self._post_stashbox(stashbox_client, mock_stash)

    def test_catalogue_name_taken_returns_409(self, stashbox_client):
        sb_mod._stash_url = "http://localhost:9999"
        sb_mod._stash_api_key = "test-key"
        with patch("stash_client_unified.StashClientUnified", return_value=self._mock_stash()):
            resp = stashbox_client.post(
                "/stash/create-performer-from-catalogue",
                json={"source": "iafd", "name": "Jane Roe", "scene_id": "1"},
            )
        assert resp.status_code == 409
        assert "id 4242" in resp.json()["detail"]

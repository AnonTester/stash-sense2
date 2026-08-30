"""Tests for upstream scene sync."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStashBoxSceneQuery:
    @pytest.mark.asyncio
    async def test_get_scene_returns_scene_data(self):
        """get_scene fetches a scene by ID with all required fields."""
        from stashbox_client import StashBoxClient

        mock_response = {
            "findScene": {
                "id": "scene-uuid-1",
                "title": "Test Scene",
                "details": "A test scene",
                "date": "2025-01-15",
                "urls": [{"url": "https://example.com/scene1", "site": {"name": "Example"}}],
                "studio": {"id": "studio-uuid-1", "name": "Test Studio"},
                "tags": [{"id": "tag-uuid-1", "name": "HD"}],
                "performers": [
                    {"performer": {"id": "perf-uuid-1", "name": "Jane Doe"}, "as": "Jane Smith"}
                ],
                "director": "John Director",
                "code": "TS-001",
                "deleted": False,
                "created": "2025-01-01T00:00:00Z",
                "updated": "2025-01-15T00:00:00Z",
            }
        }

        client = StashBoxClient("https://test.box/graphql", "key")
        client._execute = AsyncMock(return_value=mock_response)

        result = await client.get_scene("scene-uuid-1")
        assert result is not None
        assert result["title"] == "Test Scene"
        assert result["studio"]["id"] == "studio-uuid-1"
        assert len(result["performers"]) == 1
        assert result["performers"][0]["as"] == "Jane Smith"

    @pytest.mark.asyncio
    async def test_get_scene_returns_none_for_missing(self):
        """get_scene returns None when scene not found."""
        from stashbox_client import StashBoxClient

        client = StashBoxClient("https://test.box/graphql", "key")
        client._execute = AsyncMock(return_value={"findScene": None})

        result = await client.get_scene("nonexistent")
        assert result is None


class TestStashSceneQueries:
    @pytest.mark.asyncio
    async def test_get_scenes_for_endpoint(self):
        """get_scenes_for_endpoint returns scenes linked to a specific endpoint."""
        from stash_client_unified import StashClientUnified

        mock_response = {
            "findScenes": {
                "scenes": [
                    {
                        "id": "1",
                        "title": "Scene One",
                        "date": "2025-01-01",
                        "details": "Details here",
                        "director": "Director",
                        "code": "SC-001",
                        "urls": ["https://example.com/1"],
                        "studio": {"id": "10", "name": "Studio A", "stash_ids": []},
                        "performers": [{"id": "20", "name": "Perf A", "stash_ids": []}],
                        "tags": [{"id": "30", "name": "Tag A", "stash_ids": []}],
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                        ],
                    }
                ]
            }
        }

        client = StashClientUnified("http://localhost:9999", "key")
        client._execute = AsyncMock(return_value=mock_response)

        scenes = await client.get_scenes_for_endpoint("https://stashdb.org/graphql")
        assert len(scenes) == 1
        assert scenes[0]["title"] == "Scene One"
        assert scenes[0]["stash_ids"][0]["stash_id"] == "sb-scene-1"

    @pytest.mark.asyncio
    async def test_update_scene(self):
        """update_scene sends mutation with correct fields."""
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999", "key")
        client._execute = AsyncMock(return_value={"sceneUpdate": {"id": "1"}})

        result = await client.update_scene("1", title="New Title", date="2025-02-01")
        assert result["id"] == "1"
        # Verify the input dict was passed correctly
        call_args = client._execute.call_args
        # _execute is called as (query, {"input": input_dict}, priority=Priority.CRITICAL)
        # positional args: call_args[0][1] is the variables dict
        variables = call_args[0][1]
        input_dict = variables["input"]
        assert input_dict["id"] == "1"
        assert input_dict["title"] == "New Title"
        assert input_dict["date"] == "2025-02-01"


class TestSceneFieldMapper:
    def test_scene_field_config_registered(self):
        """Scene fields are registered in ENTITY_FIELD_CONFIGS."""
        from upstream_field_mapper import ENTITY_FIELD_CONFIGS
        assert "scene" in ENTITY_FIELD_CONFIGS
        cfg = ENTITY_FIELD_CONFIGS["scene"]
        assert "title" in cfg["default_fields"]
        assert "date" in cfg["default_fields"]
        assert "studio" in cfg["default_fields"]
        assert "performers" in cfg["default_fields"]
        assert "tags" in cfg["default_fields"]

    def test_normalize_upstream_scene_simple_fields(self):
        """normalize_upstream_scene extracts simple scalar fields."""
        from upstream_field_mapper import normalize_upstream_scene

        upstream = {
            "title": "Test Scene",
            "details": "Some details",
            "date": "2025-01-15",
            "director": "John",
            "code": "TS-001",
            "urls": [{"url": "https://example.com/1", "site": {"name": "Example"}}],
            "studio": {"id": "studio-1", "name": "Studio A"},
            "tags": [{"id": "tag-1", "name": "HD"}],
            "performers": [
                {"performer": {"id": "perf-1", "name": "Jane"}, "as": "Jane Smith"}
            ],
        }

        result = normalize_upstream_scene(upstream)
        assert result["title"] == "Test Scene"
        assert result["date"] == "2025-01-15"
        assert result["details"] == "Some details"
        assert result["director"] == "John"
        assert result["code"] == "TS-001"
        assert result["urls"] == ["https://example.com/1"]

    def test_normalize_upstream_scene_relational_fields(self):
        """normalize_upstream_scene extracts relational entity data."""
        from upstream_field_mapper import normalize_upstream_scene

        upstream = {
            "title": "Test",
            "details": None,
            "date": None,
            "director": None,
            "code": None,
            "urls": [],
            "studio": {"id": "studio-1", "name": "Studio A"},
            "tags": [
                {"id": "tag-1", "name": "HD"},
                {"id": "tag-2", "name": "POV"},
            ],
            "performers": [
                {"performer": {"id": "perf-1", "name": "Jane"}, "as": "Jane Smith"},
                {"performer": {"id": "perf-2", "name": "John"}, "as": None},
            ],
        }

        result = normalize_upstream_scene(upstream)
        assert result["studio"] == {"id": "studio-1", "name": "Studio A"}
        assert len(result["performers"]) == 2
        assert result["performers"][0] == {"id": "perf-1", "name": "Jane", "aliases": [], "gender": None, "as": "Jane Smith"}
        assert result["performers"][1] == {"id": "perf-2", "name": "John", "aliases": [], "gender": None, "as": None}
        assert len(result["tags"]) == 2
        assert result["tags"][0] == {"id": "tag-1", "name": "HD"}

    def test_diff_scene_simple_fields(self):
        """diff_scene_fields detects simple scalar changes."""
        from upstream_field_mapper import diff_scene_fields

        local = {"title": "Old Title", "date": "2025-01-01", "details": "", "director": "", "code": "", "urls": [],
                 "studio": None, "performers": [], "tags": []}
        upstream = {"title": "New Title", "date": "2025-01-01", "details": "", "director": "", "code": "", "urls": [],
                    "studio": None, "performers": [], "tags": []}

        result = diff_scene_fields(local, upstream, None, {"title", "date"})
        assert len(result["changes"]) == 1
        assert result["changes"][0]["field"] == "title"
        assert result["changes"][0]["upstream_value"] == "New Title"

    def test_diff_scene_relational_performers(self):
        """diff_scene_fields detects added/removed performers."""
        from upstream_field_mapper import diff_scene_fields

        local = {
            "title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
            "studio": None,
            "performers": [{"id": "perf-1", "name": "Jane", "as": None}],
            "tags": [],
        }
        upstream = {
            "title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
            "studio": None,
            "performers": [
                {"id": "perf-1", "name": "Jane", "as": None},
                {"id": "perf-2", "name": "John", "as": "Johnny"},
            ],
            "tags": [],
        }

        result = diff_scene_fields(local, upstream, None, {"performers"})
        assert len(result["performer_changes"]["added"]) == 1
        assert result["performer_changes"]["added"][0]["id"] == "perf-2"
        assert len(result["performer_changes"]["removed"]) == 0

    def test_diff_scene_no_changes(self):
        """diff_scene_fields returns empty results when nothing changed."""
        from upstream_field_mapper import diff_scene_fields

        data = {"title": "Same", "date": "2025-01-01", "details": "", "director": "", "code": "", "urls": [],
                "studio": {"id": "s1", "name": "S"}, "performers": [{"id": "p1", "name": "P", "as": None}],
                "tags": [{"id": "t1", "name": "T"}]}

        result = diff_scene_fields(data, data, None, {"title", "date", "studio", "performers", "tags"})
        assert result["changes"] == []
        assert result["studio_change"] is None
        assert result["performer_changes"]["added"] == []
        assert result["performer_changes"]["removed"] == []
        assert result["tag_changes"]["added"] == []
        assert result["tag_changes"]["removed"] == []

    def test_diff_scene_tags_order_and_case_do_not_trigger_changes(self):
        """Tag set comparison ignores ordering and stash-box ID case differences."""
        from upstream_field_mapper import diff_scene_fields

        local = {
            "title": "Same", "date": "", "details": "", "director": "", "code": "", "urls": [],
            "studio": None, "performers": [],
            "tags": [{"id": "tag-a"}, {"id": "TAG-B"}],
        }
        upstream = {
            "title": "Same", "date": "", "details": "", "director": "", "code": "", "urls": [],
            "studio": None, "performers": [],
            "tags": [{"id": "tag-b"}, {"id": "TAG-A"}],
        }

        result = diff_scene_fields(local, upstream, None, {"tags"})
        assert result["tag_changes"]["added"] == []
        assert result["tag_changes"]["removed"] == []

    def test_diff_scene_has_any_changes(self):
        """has_any_scene_changes returns True when there are changes."""
        from upstream_field_mapper import diff_scene_fields

        local = {"title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
                 "studio": None, "performers": [], "tags": []}
        upstream = {"title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
                    "studio": None, "performers": [], "tags": [{"id": "t1", "name": "New Tag"}]}

        result = diff_scene_fields(local, upstream, None, {"tags"})
        assert len(result["tag_changes"]["added"]) == 1


class TestUpstreamSceneAnalyzer:
    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                ],
            }
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.get_all_tags = AsyncMock(return_value=[])
        stash.get_all_studios = AsyncMock(return_value=[])
        return stash

    @pytest.mark.asyncio
    async def test_detects_title_change(self, mock_stash, rec_db):
        """Analyzer detects when upstream scene title differs from local."""
        upstream_data = {
            "title": "Upstream Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1
        assert recs[0].details["scene_name"] == "Local Title"
        changes = recs[0].details.get("changes", [])
        title_change = next((c for c in changes if c["field"] == "title"), None)
        assert title_change is not None
        assert title_change["upstream_value"] == "Upstream Title"

    @pytest.mark.asyncio
    async def test_no_changes_creates_no_recommendation(self, mock_stash, rec_db):
        """Analyzer creates no recommendation when upstream matches local."""
        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_detects_performer_addition(self, mock_stash, rec_db):
        """Analyzer detects when upstream has additional performers."""
        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                {"performer": {"id": "perf-1", "name": "Jane"}, "as": None}
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1
        assert len(recs[0].details["performer_changes"]["added"]) == 1

    @pytest.mark.asyncio
    async def test_keeps_added_performer_when_it_replaces_removed_stashbox_id(self, mock_stash, rec_db):
        """Do not prune added performer in merge/relink replacements (old ID removed, new ID added)."""
        mock_stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [
                    {
                        "id": "20",
                        "name": "Merged Performer",
                        "gender": "FEMALE",
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-old-id"}
                        ],
                    }
                ],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                ],
            }
        ])
        # Name lookup resolves this upstream name to the already-linked local performer.
        mock_stash.get_all_performers = AsyncMock(return_value=[
            {"id": "20", "name": "Merged Performer", "alias_list": []}
        ])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                {"performer": {"id": "perf-new-id", "name": "Merged Performer", "gender": "FEMALE"}, "as": None},
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1
        perf_changes = recs[0].details["performer_changes"]
        assert [p["id"] for p in perf_changes["removed"]] == ["perf-old-id"]
        assert [p["id"] for p in perf_changes["added"]] == ["perf-new-id"]

    @pytest.mark.asyncio
    async def test_filters_unselected_performer_genders(self, mock_stash, rec_db):
        """Performer add/remove changes are filtered by selected genders."""
        mock_stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [
                    {
                        "id": "20",
                        "name": "Male Existing",
                        "gender": "MALE",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-male-existing"}],
                    },
                    {
                        "id": "21",
                        "name": "Female Existing",
                        "gender": "FEMALE",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-female-existing"}],
                    },
                ],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                ],
            }
        ])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                # Male existing performer removed from upstream (should be filtered out)
                {"performer": {"id": "perf-female-existing", "name": "Female Existing", "gender": "FEMALE"}, "as": "F Alias"},
                # Male performer added upstream (should be filtered out)
                {"performer": {"id": "perf-male-new", "name": "Male New", "gender": "MALE"}, "as": None},
                # Female performer added upstream (should remain)
                {"performer": {"id": "perf-female-new", "name": "Female New", "gender": "FEMALE"}, "as": None},
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        def _gender_setting_value(key: str):
            if key == "upstream_scene_gender_male_enabled":
                return False
            return True

        with patch("stashbox_client.StashBoxClient") as MockSBC, \
                patch("analyzers.upstream_scene.get_setting", side_effect=_gender_setting_value):
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1
        perf_changes = recs[0].details["performer_changes"]

        assert [p["id"] for p in perf_changes["added"]] == ["perf-female-new"]
        assert perf_changes["removed"] == []
        assert perf_changes["alias_changed"] == []

    @pytest.mark.asyncio
    async def test_alias_only_scene_performer_change_is_ignored(self, mock_stash, rec_db):
        """Scene performer alias-only differences should not create recommendations."""
        mock_stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [
                    {
                        "id": "21",
                        "name": "Existing",
                        "gender": "FEMALE",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-existing"}],
                    },
                ],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                ],
            }
        ])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                {"performer": {"id": "perf-existing", "name": "Existing", "gender": "FEMALE"}, "as": "Alias Upstream"}
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_ignores_rec_when_only_unselected_gender_changes(self, mock_stash, rec_db):
        """No recommendation is created when all performer changes are filtered out."""
        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                {"performer": {"id": "perf-male-1", "name": "Male Added", "gender": "MALE"}, "as": None}
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        def _gender_setting_value(key: str):
            if key == "upstream_scene_gender_male_enabled":
                return False
            return True

        with patch("stashbox_client.StashBoxClient") as MockSBC, \
                patch("analyzers.upstream_scene.get_setting", side_effect=_gender_setting_value):
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_unknown_gender_respects_unknown_setting(self, mock_stash, rec_db):
        """Unknown/missing performer gender is controlled by the Unknown checkbox."""
        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                {"performer": {"id": "perf-unknown-1", "name": "Unknown Added"}, "as": None}
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        def _gender_setting_value(key: str):
            if key == "upstream_scene_gender_unknown_enabled":
                return False
            return True

        with patch("stashbox_client.StashBoxClient") as MockSBC, \
                patch("analyzers.upstream_scene.get_setting", side_effect=_gender_setting_value):
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_build_local_data_filters_stash_ids_by_endpoint(self, rec_db):
        """_build_local_data only matches stash_ids for the current endpoint."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://fansdb.cc/graphql", "api_key": "key"},
        ])
        # Performer has stash_ids for TWO endpoints — only fansdb should match
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Test",
                "date": "",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [
                    {
                        "id": "20",
                        "name": "Multi-Endpoint Performer",
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "stashdb-perf-1"},
                            {"endpoint": "https://fansdb.cc/graphql", "stash_id": "fansdb-perf-1"},
                        ],
                    }
                ],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://fansdb.cc/graphql", "stash_id": "fansdb-scene-1"}
                ],
            }
        ])

        # Upstream matches the fansdb performer ID — no changes expected
        upstream_data = {
            "title": "Test",
            "details": "",
            "date": "",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                {"performer": {"id": "fansdb-perf-1", "name": "Multi-Endpoint Performer"}, "as": None}
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(stash, rec_db)
            result = await analyzer.run()

        # No recommendation — fansdb IDs match correctly
        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_endpoint_normalization_avoids_false_tag_additions(self, rec_db):
        """Trailing slash endpoint variants should still match linked local tag stash_ids."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [],
                "tags": [
                    {
                        "id": "10",
                        "name": "Linked Tag",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql/", "stash_id": "tag-sb-1"}],
                    }
                ],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-1"}
                ],
            }
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.get_all_tags = AsyncMock(return_value=[])
        stash.get_all_studios = AsyncMock(return_value=[])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [{"id": "tag-sb-1", "name": "Linked Tag"}],
            "performers": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_ignores_tag_add_when_local_scene_already_has_matching_local_tag(self, rec_db):
        """Do not suggest scene-tag additions when matching local tag is already on scene."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [],
                "tags": [
                    {
                        "id": "10",
                        "name": "Femaleorgasm",
                        # No stash_ids link on tag itself, but tag is already assigned to scene.
                        "stash_ids": [],
                    }
                ],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-1"}
                ],
            }
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.get_all_tags = AsyncMock(return_value=[{"id": "10", "name": "Femaleorgasm"}])
        stash.get_all_studios = AsyncMock(return_value=[])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [{"id": "tag-sb-1", "name": "Femaleorgasm"}],
            "performers": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_ignores_tag_add_when_same_local_tag_name_has_different_lookup_id(self, rec_db):
        """Name-based suppression should prevent add rec even if global tag lookup points to another ID."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [],
                "tags": [
                    {"id": "10", "name": "Missionary", "stash_ids": []}
                ],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-1"}
                ],
            }
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        # Simulate duplicate-name situation where global lookup resolves to a different local ID.
        stash.get_all_tags_with_aliases = AsyncMock(return_value=[
            {"id": "99", "name": "Missionary", "aliases": []}
        ])
        stash.get_all_tags = AsyncMock(return_value=[])
        stash.get_all_studios = AsyncMock(return_value=[])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [{"id": "tag-sb-1", "name": "Missionary"}],
            "performers": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_ignores_studio_change_when_scene_already_has_same_local_studio_name(self, rec_db):
        """Do not suggest studio change when local scene already has the same studio assigned."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": {
                    "id": "10",
                    "name": "Black Is Better",
                    # Missing stashdb link on studio object triggers endpoint-ID mismatch path.
                    "stash_ids": [],
                },
                "performers": [],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-1"}
                ],
            }
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.get_all_tags = AsyncMock(return_value=[])
        stash.get_all_studios = AsyncMock(return_value=[
            {"id": "10", "name": "Black Is Better", "aliases": []}
        ])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": {"id": "studio-sb-1", "name": "Black Is Better"},
            "performers": [],
            "tags": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_studio_change_details_preserve_current_local_studio(self, rec_db):
        """Scene recommendations keep the assigned local studio for detail rendering."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": {
                    "id": "10",
                    "name": "Local Studio",
                    "stash_ids": [],
                },
                "performers": [],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-1"}
                ],
            }
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.get_all_tags = AsyncMock(return_value=[])
        stash.get_all_studios = AsyncMock(return_value=[])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": {"id": "studio-sb-1", "name": "Upstream Studio"},
            "performers": [],
            "tags": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(stash, rec_db)
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1
        details = recs[0].details
        assert details["current_studio"] == {"id": "10", "name": "Local Studio"}
        assert details["studio_change"]["local"] == {"id": "10", "name": "Local Studio"}
        assert details["studio_change"]["upstream"] == {"id": "studio-sb-1", "name": "Upstream Studio"}

    @pytest.mark.asyncio
    async def test_pending_scene_rec_rechecked_when_upstream_not_updated(self, rec_db):
        """Pending rec is re-compared in incremental mode and auto-resolved after local fix."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(side_effect=[
            # Run 1: local scene missing tag -> recommendation created
            [
                {
                    "id": "1",
                    "title": "Local Title",
                    "date": "2025-01-01",
                    "details": "",
                    "director": "",
                    "code": "",
                    "urls": [],
                    "studio": None,
                    "performers": [],
                    "tags": [],
                    "stash_ids": [
                        {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-1"}
                    ],
                }
            ],
            # Run 2: local scene updated to include same upstream tag
            [
                {
                    "id": "1",
                    "title": "Local Title",
                    "date": "2025-01-01",
                    "details": "",
                    "director": "",
                    "code": "",
                    "urls": [],
                    "studio": None,
                    "performers": [],
                    "tags": [
                        {
                            "id": "10",
                            "name": "Linked Tag",
                            "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-sb-1"}],
                        }
                    ],
                    "stash_ids": [
                        {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-1"}
                    ],
                }
            ],
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.get_all_tags = AsyncMock(return_value=[])
        stash.get_all_studios = AsyncMock(return_value=[])

        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [{"id": "tag-sb-1", "name": "Linked Tag"}],
            "performers": [],
            "deleted": False,
            # unchanged timestamp between runs -> would normally be skipped by watermark
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(stash, rec_db)
            await analyzer.run(incremental=True)
            recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
            assert len(recs) == 1

            await analyzer.run(incremental=True)

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0


class TestEntityCreation:
    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.create_studio = AsyncMock(return_value={"id": "100", "name": "New Studio"})
        stash.create_performer = AsyncMock(return_value={"id": "200", "name": "New Performer"})
        stash.create_tag = AsyncMock(return_value={"id": "300", "name": "New Tag"})
        stash.search_performers = AsyncMock(return_value=[])
        stash.update_performer = AsyncMock(return_value={"id": "200"})
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.search_tags = AsyncMock(return_value=[])
        stash.get_all_tags_with_aliases = AsyncMock(return_value=[])
        stash.update_tag = AsyncMock(return_value={"id": "300"})
        # _find_linked_entity_by_stash_id's raw GraphQL call -- empty by
        # default (no performer already linked to this stash_id).
        stash._execute = AsyncMock(return_value={"findPerformers": {"performers": []}})
        return stash

    @pytest.mark.asyncio
    async def test_create_performer_from_stashbox(self, mock_stash):
        """Creates a performer with stash_id link."""
        from recommendations_router import _create_performer_from_stashbox
        result = await _create_performer_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Jane Doe", "aliases": ["JD"], "gender": "FEMALE"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="perf-uuid-1",
        )
        assert result["id"] == "200"
        mock_stash.create_performer.assert_called_once()
        call_kwargs = mock_stash.create_performer.call_args[1]
        assert call_kwargs["name"] == "Jane Doe"
        assert call_kwargs["stash_ids"] == [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-uuid-1"}]

    @pytest.mark.asyncio
    async def test_create_performer_from_stashbox_alias_match_raises_ambiguous(self, mock_stash):
        """A name/alias match alone is never sufficient grounds to link or
        create -- confirmed live, this exact pattern silently linked a real
        StashBox performer's id onto an unrelated existing local performer
        whose alias list happened to contain the same name, corrupting that
        performer's own stash_ids in the process. Must raise instead of
        silently picking a side."""
        from recommendations_router import _create_performer_from_stashbox, PerformerIdentityAmbiguous
        mock_stash.search_performers = AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": []}
        ])

        with pytest.raises(PerformerIdentityAmbiguous) as exc_info:
            await _create_performer_from_stashbox(
                mock_stash,
                stashbox_data={"name": "Jane Doe"},
                endpoint="https://theporndb.net/graphql",
                stashbox_id="perf-uuid-2",
            )

        assert exc_info.value.candidates[0]["id"] == "42"
        mock_stash.create_performer.assert_not_called()
        mock_stash.update_performer.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_performer_from_stashbox_duplicate_error_raises_ambiguous(self, mock_stash):
        """If create fails with duplicate-name error, the now-visible
        same-named performer is still just a name match -- must raise
        ambiguous rather than assume it's the same performer and link it."""
        from recommendations_router import _create_performer_from_stashbox, PerformerIdentityAmbiguous
        mock_stash.search_performers = AsyncMock(side_effect=[
            [],
            [{"id": "42", "name": "Jane Doe", "alias_list": [], "stash_ids": []}],
        ])
        mock_stash.create_performer = AsyncMock(
            side_effect=RuntimeError(
                "GraphQL error: [{'message': \"performer with name 'Jane Doe' already exists\", 'path': ['performerCreate']}]"
            )
        )

        with pytest.raises(PerformerIdentityAmbiguous) as exc_info:
            await _create_performer_from_stashbox(
                mock_stash,
                stashbox_data={"name": "Jane Doe"},
                endpoint="https://stashdb.org/graphql",
                stashbox_id="perf-uuid-1",
            )

        assert exc_info.value.candidates[0]["id"] == "42"
        mock_stash.update_performer.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_performer_from_stashbox_hard_match_via_existing_stash_id_link(self, mock_stash):
        """A performer already linked to this exact (endpoint, stashbox_id)
        is hard evidence -- safe to resolve directly, no ambiguity, no
        name/alias matching involved at all."""
        from recommendations_router import _create_performer_from_stashbox
        mock_stash._execute = AsyncMock(return_value={
            "findPerformers": {"performers": [
                {"id": "42", "name": "Totally Different Name", "disambiguation": None,
                 "alias_list": [], "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-uuid-1"}]},
            ]}
        })

        result = await _create_performer_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Jane Doe"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="perf-uuid-1",
        )

        assert result["id"] == "42"
        mock_stash.create_performer.assert_not_called()
        mock_stash.search_performers.assert_not_called()
        mock_stash.update_performer.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_performer_from_stashbox_hard_match_via_matching_url(self, mock_stash):
        """A candidate's profile URL matching a local performer's own urls
        (www./scheme/trailing-slash tolerant) is hard evidence -- safe to
        link without asking, even though the name is only an alias match."""
        from recommendations_router import _create_performer_from_stashbox
        mock_stash.search_performers = AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": [],
             "urls": ["https://www.onlyfans.com/janedoe/"]},
        ])

        result = await _create_performer_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Jane Doe", "profile_url": "http://onlyfans.com/janedoe"},
            endpoint="https://theporndb.net/graphql",
            stashbox_id="perf-uuid-2",
        )

        assert result["id"] == "42"
        mock_stash.create_performer.assert_not_called()
        mock_stash.update_performer.assert_called_once_with(
            "42",
            stash_ids=[{"endpoint": "https://theporndb.net/graphql", "stash_id": "perf-uuid-2"}],
        )

    @pytest.mark.asyncio
    async def test_resolved_link_adds_a_new_profile_url_the_performer_did_not_have(self, mock_stash):
        """The explicit "link to this performer" resolution path (user
        picked a card after an ambiguous match) should also save the
        candidate's profile URL onto the performer if it's genuinely new --
        not just the stash_id."""
        from recommendations_router import CreatePerformerRequest, create_performer_action
        import recommendations_router as rec_mod

        mock_stash.get_performer = AsyncMock(return_value={
            "id": "42", "name": "Main Name", "stash_ids": [], "urls": ["https://twitter.com/janedoe"],
        })
        rec_mod.stash_client = mock_stash

        result = await create_performer_action(CreatePerformerRequest(
            stashbox_data={"name": "Jane Doe", "profile_url": "https://www.onlyfans.com/janedoe/"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="perf-uuid-1",
            resolved_performer_id="42",
        ))

        assert result["success"] is True
        assert mock_stash.update_performer.call_count == 2
        calls = {tuple(sorted(c.kwargs.keys())): c for c in mock_stash.update_performer.call_args_list}
        assert calls[("stash_ids",)].kwargs["stash_ids"] == [
            {"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-uuid-1"},
        ]
        assert calls[("urls",)].kwargs["urls"] == [
            "https://twitter.com/janedoe", "https://www.onlyfans.com/janedoe/",
        ]

    @pytest.mark.asyncio
    async def test_creating_a_new_performer_stores_stash_id_cover_image_and_urls(self, mock_stash):
        """Confirms the full create path already covers stash_id (at
        create time) plus cover image and urls (via the post-create
        enrichment fetch from the stashbox endpoint) -- nothing extra
        needed here, this is a coverage gap fix, not a behavior change."""
        from recommendations_router import _create_performer_from_stashbox

        mock_sbc = MagicMock()
        mock_sbc.get_performer = AsyncMock(return_value={
            "urls": [
                {"url": "https://onlyfans.com/janedoe", "type": "ONLYFANS"},
                {"url": "https://twitter.com/janedoe", "type": "TWITTER"},
            ],
            "images": [{"id": "img-1", "url": "https://stashdb.org/images/img-1.jpg"}],
        })
        mock_mgr = MagicMock()
        mock_mgr.get_client.return_value = mock_sbc

        with patch("stashbox_connection_manager.get_connection_manager", return_value=mock_mgr):
            result = await _create_performer_from_stashbox(
                mock_stash,
                stashbox_data={"name": "Jane Doe"},
                endpoint="https://stashdb.org/graphql",
                stashbox_id="perf-uuid-1",
            )

        assert result["id"] == "200"
        create_kwargs = mock_stash.create_performer.call_args.kwargs
        assert create_kwargs["stash_ids"] == [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-uuid-1"}]

        mock_stash.update_performer.assert_called_once()
        enrich_kwargs = mock_stash.update_performer.call_args.kwargs
        assert enrich_kwargs["urls"] == ["https://onlyfans.com/janedoe", "https://twitter.com/janedoe"]
        assert enrich_kwargs["image"] == "https://stashdb.org/images/img-1.jpg"


    @pytest.mark.asyncio
    async def test_create_tag_from_stashbox(self, mock_stash):
        """Creates a tag with stash_id link."""
        from recommendations_router import _create_tag_from_stashbox
        result = await _create_tag_from_stashbox(
            mock_stash,
            stashbox_data={"name": "HD", "description": "High definition"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="tag-uuid-1",
        )
        assert result["id"] == "300"
        mock_stash.create_tag.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tag_from_stashbox_links_existing_name_match(self, mock_stash):
        """Links stash_id to an existing local tag with same name instead of creating."""
        from recommendations_router import _create_tag_from_stashbox
        mock_stash.search_tags = AsyncMock(return_value=[
            {"id": "42", "name": "HD", "aliases": [], "stash_ids": []}
        ])

        result = await _create_tag_from_stashbox(
            mock_stash,
            stashbox_data={"name": "HD"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="tag-uuid-1",
        )

        assert result["id"] == "42"
        mock_stash.create_tag.assert_not_called()
        mock_stash.update_tag.assert_called_once_with(
            "42",
            stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
        )

    @pytest.mark.asyncio
    async def test_create_tag_from_stashbox_duplicate_error_falls_back_to_link(self, mock_stash):
        """If create fails with duplicate-name error, fallback links existing local tag."""
        from recommendations_router import _create_tag_from_stashbox
        mock_stash.search_tags = AsyncMock(side_effect=[
            [],
            [{"id": "42", "name": "Femaleorgasm", "aliases": [], "stash_ids": []}],
        ])
        mock_stash.create_tag = AsyncMock(
            side_effect=RuntimeError(
                "GraphQL error: [{'message': \"tag with name 'Femaleorgasm' already exists\", 'path': ['tagCreate']}]"
            )
        )

        result = await _create_tag_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Femaleorgasm"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="tag-uuid-1",
        )

        assert result["id"] == "42"
        mock_stash.update_tag.assert_called_once_with(
            "42",
            stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
        )

    @pytest.mark.asyncio
    async def test_create_tag_from_stashbox_alias_match_when_search_misses(self, mock_stash):
        """If search misses alias hits, fallback all-tags lookup still links existing tag."""
        from recommendations_router import _create_tag_from_stashbox
        mock_stash.search_tags = AsyncMock(return_value=[])
        mock_stash.get_all_tags_with_aliases = AsyncMock(return_value=[
            {
                "id": "88",
                "name": "Main Tag",
                "aliases": ["Ass Slapping"],
                "stash_ids": [],
            }
        ])

        result = await _create_tag_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Ass Slapping"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="tag-uuid-1",
        )

        assert result["id"] == "88"
        mock_stash.create_tag.assert_not_called()
        mock_stash.update_tag.assert_called_once_with(
            "88",
            stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
        )

    @pytest.mark.asyncio
    async def test_create_studio_from_stashbox_links_existing_name_match(self, mock_stash):
        """Links stash_id to existing local studio with same name instead of creating."""
        from recommendations_router import _create_studio_from_stashbox
        mock_stash.search_studios = AsyncMock(return_value=[
            {"id": "55", "name": "Manyvids: Dreaminskies", "aliases": [], "stash_ids": []}
        ])
        mock_stash.update_studio = AsyncMock(return_value={"id": "55"})

        result = await _create_studio_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Manyvids: Dreaminskies"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="studio-uuid-1",
        )

        assert result["id"] == "55"
        mock_stash.create_studio.assert_not_called()
        mock_stash.update_studio.assert_called_once_with(
            "55",
            stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}],
        )

    @pytest.mark.asyncio
    async def test_create_studio_from_stashbox_duplicate_error_falls_back_to_link(self, mock_stash):
        """If create fails with duplicate-name error, fallback links existing studio."""
        from recommendations_router import _create_studio_from_stashbox
        mock_stash.search_studios = AsyncMock(side_effect=[
            [],
            [{"id": "55", "name": "Manyvids: Dreaminskies", "aliases": [], "stash_ids": []}],
        ])
        mock_stash.update_studio = AsyncMock(return_value={"id": "55"})
        mock_stash.create_studio = AsyncMock(
            side_effect=RuntimeError(
                "GraphQL error: [{'message': \"studio with name 'Manyvids: Dreaminskies' already exists\", 'path': ['studioCreate']}]"
            )
        )

        result = await _create_studio_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Manyvids: Dreaminskies"},
            endpoint="https://stashdb.org/graphql",
            stashbox_id="studio-uuid-1",
        )

        assert result["id"] == "55"
        mock_stash.update_studio.assert_called_once_with(
            "55",
            stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}],
        )


class TestCatalogueSourcedPerformers:
    """Catalogue sources (seekfans, pornbox, etc -- see
    classify_universal_id's "catalogue" category) have no real stash-box
    connection: `endpoint` is just the source's own name, `stashbox_id` is
    that source's internal id, and there's no metadata API to fetch a
    cover/urls from after create -- see _create_performer_from_stashbox's
    own is_catalogue docstring. These mirror stashbox_router.py's existing
    create_performer_from_catalogue endpoint (used by the live Identify
    modal for the exact same case), just reached through the accept flows
    fixed for the alias-collision bug."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.create_performer = AsyncMock(return_value={"id": "200", "name": "New Performer"})
        stash.search_performers = AsyncMock(return_value=[])
        stash.update_performer = AsyncMock(return_value={"id": "200"})
        stash.get_all_performers = AsyncMock(return_value=[])
        stash._execute = AsyncMock(return_value={"findPerformers": {"performers": []}})
        return stash

    @pytest.mark.asyncio
    async def test_creates_with_image_and_urls_but_no_stash_ids(self, mock_stash):
        from recommendations_router import _create_performer_from_stashbox

        result = await _create_performer_from_stashbox(
            mock_stash,
            stashbox_data={
                "name": "Jane Doe", "source": "seekfans",
                "image_url": "https://seekfans.example/janedoe.jpg",
                "profile_url": "https://onlyfans.com/janedoe",
                "catalogue_url": "https://seekfans.example/model/janedoe",
            },
            endpoint="seekfans",
            stashbox_id="4821",
        )

        assert result["id"] == "200"
        # No stash_id hard-match pre-check for catalogue sources -- there's
        # no real linkage concept, so nothing to look up first.
        mock_stash._execute.assert_not_called()

        create_kwargs = mock_stash.create_performer.call_args.kwargs
        assert "stash_ids" not in create_kwargs
        assert create_kwargs["image"] == "https://seekfans.example/janedoe.jpg"
        # No surrounding brackets -- Stash's own UI already adds them for
        # display; a literal "(Seekfans)" here would render as "((Seekfans))".
        assert create_kwargs["disambiguation"] == "Seekfans"
        assert create_kwargs["alias_list"] == ["janedoe"]  # from the profile_url handle

        mock_stash.update_performer.assert_called_once_with(
            "200", urls=["https://onlyfans.com/janedoe", "https://seekfans.example/model/janedoe"],
        )

    @pytest.mark.asyncio
    async def test_alias_collision_still_raises_ambiguous(self, mock_stash):
        """The alias-collision protection applies identically regardless
        of source -- a catalogue candidate matching an unrelated local
        performer's alias must still be confirmed, not silently linked."""
        from recommendations_router import _create_performer_from_stashbox, PerformerIdentityAmbiguous
        mock_stash.search_performers = AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": [], "urls": []},
        ])

        with pytest.raises(PerformerIdentityAmbiguous) as exc_info:
            await _create_performer_from_stashbox(
                mock_stash,
                stashbox_data={"name": "Jane Doe", "source": "seekfans", "profile_url": "https://onlyfans.com/different-person"},
                endpoint="seekfans",
                stashbox_id="4821",
            )

        assert exc_info.value.candidates[0]["id"] == "42"
        mock_stash.create_performer.assert_not_called()

    @pytest.mark.asyncio
    async def test_hard_match_via_url_links_without_writing_stash_ids(self, mock_stash):
        """A local performer whose own urls already include this
        catalogue candidate's profile URL is hard evidence -- link, don't
        create a duplicate -- but never write a bogus stash_ids entry
        (there's no real "seekfans" stash-box connection to link to)."""
        from recommendations_router import _create_performer_from_stashbox
        mock_stash.search_performers = AsyncMock(return_value=[
            {"id": "42", "name": "Main Name", "alias_list": ["Jane Doe"], "stash_ids": [],
             "urls": ["https://onlyfans.com/janedoe"]},
        ])

        result = await _create_performer_from_stashbox(
            mock_stash,
            stashbox_data={"name": "Jane Doe", "source": "seekfans", "profile_url": "https://onlyfans.com/janedoe"},
            endpoint="seekfans",
            stashbox_id="4821",
        )

        assert result["id"] == "42"
        mock_stash.create_performer.assert_not_called()
        mock_stash.update_performer.assert_not_called()  # URL already present, nothing new to add

    @pytest.mark.asyncio
    async def test_scene_face_match_accept_creates_catalogue_performer_correctly(self, mock_stash):
        """End-to-end through the actual entry point that hits this: Face
        Recommendations accepting a seekfans-sourced candidate with no
        existing local match."""
        from recommendations_router import _resolve_scene_face_match_performer_id, SceneFaceMatchSelection
        from recommendations_db import Recommendation

        rec = Recommendation(
            id=1, type="scene_face_match", status="pending", target_type="scene", target_id="1|seekfans:4821",
            details={
                "scene_id": "1", "name": "Jane Doe", "stashdb_id": "4821", "endpoint": "seekfans",
                "source": "seekfans", "image_url": "https://seekfans.example/janedoe.jpg",
                "profile_url": "https://onlyfans.com/janedoe", "catalogue_url": None,
            },
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.8, source_analysis_id=None, created_at="", updated_at="",
        )
        selection = SceneFaceMatchSelection(recommendation_id=1)

        with patch("stashbox_connection_manager.get_connection_manager", return_value=MagicMock(
            get_endpoint_url=MagicMock(return_value=None),
        )):
            performer_id = await _resolve_scene_face_match_performer_id(mock_stash, rec, selection)

        assert performer_id == "200"
        create_kwargs = mock_stash.create_performer.call_args.kwargs
        assert "stash_ids" not in create_kwargs
        assert create_kwargs["image"] == "https://seekfans.example/janedoe.jpg"


class TestUpdateSceneAction:
    @pytest.mark.asyncio
    async def test_apply_scene_update_simple_fields(self):
        """_apply_scene_update passes simple fields to update_scene."""
        from recommendations_router import _apply_scene_update

        mock_stash = MagicMock()
        mock_stash.update_scene = AsyncMock(return_value={"id": "1"})

        await _apply_scene_update(mock_stash, scene_id="1", fields={
            "title": "New Title",
            "date": "2025-02-01",
        })

        mock_stash.update_scene.assert_called_once()
        call_args = mock_stash.update_scene.call_args
        assert call_args[0][0] == "1"  # scene_id
        assert call_args[1]["title"] == "New Title"

    @pytest.mark.asyncio
    async def test_apply_scene_update_with_relational_ids(self):
        """_apply_scene_update includes performer_ids, tag_ids, studio_id."""
        from recommendations_router import _apply_scene_update

        mock_stash = MagicMock()
        mock_stash.update_scene = AsyncMock(return_value={"id": "1"})

        await _apply_scene_update(
            mock_stash, scene_id="1", fields={"title": "Scene"},
            performer_ids=["10", "20"], tag_ids=["30"], studio_id="5"
        )

        call_args = mock_stash.update_scene.call_args
        assert call_args[1]["performer_ids"] == ["10", "20"]
        assert call_args[1]["tag_ids"] == ["30"]
        assert call_args[1]["studio_id"] == "5"


class TestSceneSyncIntegration:
    """End-to-end tests: analyzer + recommendation + action resolution."""

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42",
                "title": "Original Scene",
                "date": "2025-01-01",
                "details": "Original details",
                "director": "",
                "code": "",
                "urls": [],
                "studio": {
                    "id": "10",
                    "name": "Local Studio",
                    "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-sb-1"}],
                },
                "performers": [
                    {
                        "id": "20",
                        "name": "Existing Performer",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "perf-sb-1"}],
                    }
                ],
                "tags": [
                    {
                        "id": "30",
                        "name": "HD",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-sb-1"}],
                    }
                ],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "scene-sb-42"}
                ],
            }
        ])
        stash.get_all_performers = AsyncMock(return_value=[])
        stash.get_all_tags = AsyncMock(return_value=[])
        stash.get_all_studios = AsyncMock(return_value=[])
        return stash

    @pytest.mark.asyncio
    async def test_full_flow_with_simple_and_relational_changes(self, mock_stash, rec_db):
        """Complete flow: title change + new performer + new tag -> recommendation."""
        upstream_data = {
            "title": "Updated Scene Title",
            "details": "Original details",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": {"id": "studio-sb-1", "name": "Local Studio"},
            "performers": [
                {"performer": {"id": "perf-sb-1", "name": "Existing Performer"}, "as": None},
                {"performer": {"id": "perf-sb-2", "name": "New Performer"}, "as": "Stage Name"},
            ],
            "tags": [
                {"id": "tag-sb-1", "name": "HD"},
                {"id": "tag-sb-2", "name": "4K"},
            ],
            "deleted": False,
            "updated": "2025-02-01T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        assert result.recommendations_created == 1
        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1

        details = recs[0].details
        # Simple field change
        assert any(c["field"] == "title" and c["upstream_value"] == "Updated Scene Title" for c in details["changes"])
        # Performer addition
        assert len(details["performer_changes"]["added"]) == 1
        assert details["performer_changes"]["added"][0]["id"] == "perf-sb-2"
        assert details["performer_changes"]["added"][0]["as"] == "Stage Name"
        # Tag addition
        assert len(details["tag_changes"]["added"]) == 1
        assert details["tag_changes"]["added"][0]["id"] == "tag-sb-2"
        # No studio change (same studio)
        assert details["studio_change"] is None
        # Current entity IDs for merge-on-apply
        assert details["current_performer_ids"] == ["20"]
        assert details["current_tag_ids"] == ["30"]
        assert details["current_studio_id"] == "10"

    @pytest.mark.asyncio
    async def test_no_changes_auto_resolves_stale_rec(self, mock_stash, rec_db):
        """If upstream matches local, any stale pending rec is auto-resolved."""
        # First run: create a stale recommendation manually
        rec_db.create_recommendation(
            type="upstream_scene_changes",
            target_type="scene",
            target_id="42",
            details={"changes": [{"field": "title"}], "performer_changes": {"added": [], "removed": [], "alias_changed": []}, "tag_changes": {"added": [], "removed": []}},
            confidence=1.0,
        )
        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1

        # Upstream now matches local — no changes
        upstream_data = {
            "title": "Original Scene",
            "details": "Original details",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": {"id": "studio-sb-1", "name": "Local Studio"},
            "performers": [
                {"performer": {"id": "perf-sb-1", "name": "Existing Performer"}, "as": None},
            ],
            "tags": [
                {"id": "tag-sb-1", "name": "HD"},
            ],
            "deleted": False,
            "updated": "2025-02-01T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        # Stale rec should be auto-resolved
        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_incremental_uses_watermark(self, mock_stash, rec_db):
        """Incremental run skips scenes not updated since watermark."""
        # Set watermark
        rec_db.set_watermark(
            "upstream_scene_changes:https://stashdb.org/graphql",
            last_cursor="2025-03-01T00:00:00Z",
        )

        # Upstream scene updated BEFORE watermark
        upstream_data = {
            "title": "Updated Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "performers": [],
            "tags": [],
            "deleted": False,
            "updated": "2025-02-01T00:00:00Z",  # Before watermark
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run(incremental=True)

        # Should skip (updated before watermark)
        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

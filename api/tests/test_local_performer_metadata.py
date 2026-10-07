"""The local performer index keeps EVERY stash-box id and the performer's current details, and refreshes them when a
performer changes in Stash even though its cover image did not."""
import json
from types import SimpleNamespace

import numpy as np

from jobs.local_performer_sync_job import LocalPerformerSyncJob  # noqa: F401  (import check: shares the helpers)
from local_performer_index import (
    LocalPerformerIndex,
    endpoint_domain,
    extract_stash_ids,
    linked_stash_uids,
    performer_metadata,
    stash_image_fetch_url,
    sync_one_performer,
)

JAVSTASH = "https://javstash.org/graphql"
STASHDB = "https://stashdb.org/graphql"


def _performer(**over):
    base = {
        "id": "3073", "name": "Aimi Tomozaki", "alias_list": ["武藤つぐみ"], "disambiguation": "", "gender": "FEMALE",
        "country": "JP", "birthdate": None, "favorite": False, "urls": ["https://example.com/a"],
        "image_path": "http://stash/performer/3073/image?t=1",
        "stash_ids": [{"endpoint": JAVSTASH, "stash_id": "43456abb"}],
    }
    base.update(over)
    return base


class TestStashIds:
    def test_endpoint_domain(self):
        assert endpoint_domain(STASHDB) == "stashdb.org"
        assert endpoint_domain("https://www.fansdb.cc/graphql") == "fansdb.cc"
        assert endpoint_domain(None) is None

    def test_extract_keeps_every_endpoint(self):
        ids = extract_stash_ids(_performer(stash_ids=[
            {"endpoint": JAVSTASH, "stash_id": "j1"}, {"endpoint": STASHDB, "stash_id": "s1"},
            {"endpoint": STASHDB, "stash_id": "s2"},
        ]))
        assert ids == {"javstash.org": "j1", "stashdb.org": "s1"}

    def test_metadata_has_stashdb_id_and_details(self):
        meta = performer_metadata(_performer(stash_ids=[{"endpoint": STASHDB, "stash_id": "s1"}]))
        assert meta["stashdb_id"] == "s1"
        assert meta["aliases"] == ["武藤つぐみ"]
        assert meta["country"] == "JP" and meta["gender"] == "FEMALE"

    def test_javstash_only_performer_has_no_stashdb_id_but_a_link(self):
        meta = performer_metadata(_performer())
        assert meta["stashdb_id"] is None
        assert linked_stash_uids(meta) == ["javstash.org:43456abb"]

    def test_linked_uids_put_stashdb_first_and_read_old_entries(self):
        assert linked_stash_uids({"stash_ids": {"javstash.org": "j", "stashdb.org": "s"}}) == [
            "stashdb.org:s", "javstash.org:j"]
        assert linked_stash_uids({"stashdb_id": "s"}) == ["stashdb.org:s"]   # entry from before all ids were stored
        assert linked_stash_uids({"name": "x"}) == []


class TestUpdateMetadata:
    def _index(self, tmp_path):
        index = LocalPerformerIndex(tmp_path / "i.usearch", tmp_path / "m.json")
        meta = performer_metadata(_performer(name="武藤つぐみ"))
        index.upsert(3073, meta["name"], meta["stashdb_id"], "hash", "/img", np.ones(512, dtype=np.float32),
                     urls=meta["urls"], bbox={"x": 1}, details=meta)
        return index

    def test_upsert_stores_details(self, tmp_path):
        entry = self._index(tmp_path).mapping["3073"]
        assert entry["stash_ids"] == {"javstash.org": "43456abb"}
        assert entry["aliases"] == ["武藤つぐみ"] and entry["country"] == "JP"
        assert entry["image_hash"] == "hash" and entry["bbox"] == {"x": 1}

    def test_rename_and_new_link_reach_an_unchanged_cover(self, tmp_path):
        index = self._index(tmp_path)
        new = performer_metadata(_performer(stash_ids=[
            {"endpoint": JAVSTASH, "stash_id": "43456abb"}, {"endpoint": STASHDB, "stash_id": "12eff07a"}]))
        assert index.update_metadata(3073, new) is True
        entry = index.mapping["3073"]
        assert entry["name"] == "Aimi Tomozaki"
        assert entry["stashdb_id"] == "12eff07a"
        assert entry["stash_ids"] == {"javstash.org": "43456abb", "stashdb.org": "12eff07a"}
        # the embedding side is untouched
        assert entry["image_hash"] == "hash" and entry["bbox"] == {"x": 1} and entry["image_url"] == "/img"

    def test_no_change_reports_false(self, tmp_path):
        index = self._index(tmp_path)
        assert index.update_metadata(3073, performer_metadata(_performer(name="武藤つぐみ"))) is False

    def test_never_overwrites_cover_fields_and_unknown_id_is_ignored(self, tmp_path):
        index = self._index(tmp_path)
        index.update_metadata(3073, {"image_hash": "other", "bbox": None, "image_url": "/x", "_linked_uid": "z"})
        entry = index.mapping["3073"]
        assert (entry["image_hash"], entry["bbox"], entry["image_url"]) == ("hash", {"x": 1}, "/img")
        assert "_linked_uid" not in entry
        assert index.update_metadata(999, {"name": "n"}) is False

    def test_save_never_persists_runtime_annotations(self, tmp_path):
        index = self._index(tmp_path)
        index.mapping["3073"]["_linked_uid"] = "stashdb.org:x"
        index.save()
        assert "_linked_uid" not in json.loads((tmp_path / "m.json").read_text())["3073"]
        assert "_linked_uid" in index.mapping["3073"]       # still there in memory


class _Stash:
    api_key = "k"

    def __init__(self, performer):
        self.performer = performer

    async def get_performer(self, _id):
        return self.performer


class TestSyncOnePerformerUnchangedCover:
    async def test_a_rename_is_applied_without_re_embedding(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch
        index = LocalPerformerIndex(tmp_path / "i.usearch", tmp_path / "m.json")
        meta = performer_metadata(_performer(name="武藤つぐみ"))
        from local_performer_index import _image_fingerprint
        index.upsert(3073, meta["name"], None, _image_fingerprint(b"img"), "/img", np.ones(512, dtype=np.float32),
                     urls=meta["urls"], details=meta)

        resp = MagicMock(content=b"img", raise_for_status=MagicMock())
        client = AsyncMock()
        client.get = AsyncMock(return_value=resp)
        cls = MagicMock()
        cls.return_value.__aenter__ = AsyncMock(return_value=client)
        cls.return_value.__aexit__ = AsyncMock(return_value=False)
        generator = MagicMock()
        with patch("local_performer_index.httpx.AsyncClient", cls):
            status = await sync_one_performer(_Stash(_performer()), generator, index, 3073, "update")
            again = await sync_one_performer(_Stash(_performer()), generator, index, 3073, "update")

        assert status == "metadata_updated" and again == "unchanged"
        assert index.mapping["3073"]["name"] == "Aimi Tomozaki"
        generator.detect_faces.assert_not_called()


class TestStashImageFetchUrl:
    def test_rebased_onto_the_client_address_keeping_path_and_query(self):
        stash = SimpleNamespace(base_url="http://reachable:9999")
        assert stash_image_fetch_url("http://unreachable.internal/performer/5/image?t=1", stash) == \
            "http://reachable:9999/performer/5/image?t=1"

    def test_falls_back_to_stash_url_and_ignores_a_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("STASH_URL", "http://env-host:9999/")
        assert stash_image_fetch_url("http://x/performer/5/image", None) == "http://env-host:9999/performer/5/image"
        assert stash_image_fetch_url("http://x/performer/5/image", SimpleNamespace()) == \
            "http://env-host:9999/performer/5/image"

    def test_left_alone_without_an_address_or_a_path(self, monkeypatch):
        monkeypatch.delenv("STASH_URL", raising=False)
        assert stash_image_fetch_url("http://x/performer/5/image", None) == "http://x/performer/5/image"
        monkeypatch.setenv("STASH_URL", "http://env-host:9999")
        assert stash_image_fetch_url("", None) == ""


class TestHookPathUsesTheSameAddress:
    async def test_sync_one_performer_downloads_from_the_client_address(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch
        index = LocalPerformerIndex(tmp_path / "i.usearch", tmp_path / "m.json")
        stash = _Stash(_performer(image_path="http://unreachable.internal/performer/3073/image?t=1"))
        stash.base_url = "http://reachable:9999"
        client = AsyncMock()
        client.get = AsyncMock(return_value=MagicMock(content=b"img", raise_for_status=MagicMock()))
        cls = MagicMock()
        cls.return_value.__aenter__ = AsyncMock(return_value=client)
        cls.return_value.__aexit__ = AsyncMock(return_value=False)
        generator = MagicMock()
        generator.detect_faces.return_value = []        # no face: the test is about where the image came from
        with patch("local_performer_index.httpx.AsyncClient", cls), \
                patch("embeddings.load_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
            await sync_one_performer(stash, generator, index, 3073, "update")
        assert client.get.call_args[0][0] == "http://reachable:9999/performer/3073/image?t=1"

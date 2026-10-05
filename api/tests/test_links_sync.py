"""links_sync.py -- performer_links.json / aliases.json follow the release the local database is at.

These files were never delivered by any update path (not in the full-zip swap list, not carried by a delta), so an
install saw new link groups / aliases only if someone copied the files in by hand. Releases now carry them as a small
asset with their sha256 in the notes."""
import hashlib
import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import links_sync
from database_updater import DatabaseUpdater, RELEASE_FILES
from links_sync import find_links_asset, parse_link_hashes, stale_files, sync_link_files

pytestmark = pytest.mark.asyncio

NEW_LINKS = json.dumps([["stashdb.org:a", "adultfilmdatabase:2"]]).encode()
NEW_ALIASES = json.dumps({"stashdb.org:a": ["Nikolas Artem"]}).encode()
OLD_LINKS = json.dumps([["stashdb.org:a", "pornbox:1"]]).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _release(tag="2026.10.05", with_hashes=True, with_asset=True, links=NEW_LINKS, aliases=NEW_ALIASES):
    body = "Delta release.\n\nmin-sidecar-version: 0.33.0"
    if with_hashes:
        body += f"\nlinks-sha256: {_sha(links)}\naliases-sha256: {_sha(aliases)}"
    assets = [{"name": f"stash-sense2-links-{tag}.zip", "browser_download_url": "https://example/links.zip"}] if with_asset else []
    return {"tag_name": f"v{tag}", "body": body, "assets": assets}


def _zip_to(dest: Path, links=NEW_LINKS, aliases=NEW_ALIASES, drop=None):
    with zipfile.ZipFile(dest, "w") as zf:
        if drop != "performer_links.json":
            zf.writestr("performer_links.json", links)
        if drop != "aliases.json":
            zf.writestr("aliases.json", aliases)


def _fake_download(**zip_kwargs):
    async def _dl(url, dest):
        _zip_to(dest, **zip_kwargs)
    return _dl


def _data_dir(tmp_path, links=OLD_LINKS, aliases=b"{}"):
    d = tmp_path / "data"
    d.mkdir()
    if links is not None:
        (d / "performer_links.json").write_bytes(links)
    if aliases is not None:
        (d / "aliases.json").write_bytes(aliases)
    return d


class TestParsing:
    async def test_hashes_need_both_lines(self):
        assert parse_link_hashes(_release()["body"]) == {"performer_links.json": _sha(NEW_LINKS), "aliases.json": _sha(NEW_ALIASES)}
        assert parse_link_hashes("links-sha256: " + "a" * 64) is None
        assert parse_link_hashes(None) is None and parse_link_hashes("min-sidecar-version: 0.33.0") is None

    async def test_asset_lookup(self):
        assert find_links_asset(_release())["name"] == "stash-sense2-links-2026.10.05.zip"
        assert find_links_asset(_release(with_asset=False)) is None


class TestSync:
    async def test_stale_files_are_replaced_and_the_old_ones_kept_once(self, tmp_path):
        d = _data_dir(tmp_path)
        with patch("links_sync._download", _fake_download()):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["status"] == "updated" and sorted(r["changed"]) == ["aliases.json", "performer_links.json"]
        assert (d / "performer_links.json").read_bytes() == NEW_LINKS
        assert (d / "aliases.json").read_bytes() == NEW_ALIASES
        assert (d / "performer_links.json.prev").read_bytes() == OLD_LINKS
        assert not (d / ".links_sync_staging").exists() and not (d / ".links_sync_download.zip").exists()

    async def test_missing_files_are_installed_like_a_fresh_install_needs(self, tmp_path):
        d = _data_dir(tmp_path, links=None, aliases=None)
        with patch("links_sync._download", _fake_download()):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["status"] == "updated" and (d / "performer_links.json").exists() and (d / "aliases.json").exists()

    async def test_files_that_already_match_are_left_alone_without_a_download(self, tmp_path):
        d = _data_dir(tmp_path, links=NEW_LINKS, aliases=NEW_ALIASES)
        dl = AsyncMock()
        with patch("links_sync._download", dl):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["status"] == "in_sync" and dl.await_count == 0
        assert stale_files(d, parse_link_hashes(_release()["body"])) == []

    async def test_only_the_stale_file_is_replaced(self, tmp_path):
        d = _data_dir(tmp_path, links=NEW_LINKS, aliases=b"{}")
        with patch("links_sync._download", _fake_download()):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["changed"] == ["aliases.json"]
        assert not (d / "performer_links.json.prev").exists()

    async def test_a_checksum_mismatch_changes_nothing(self, tmp_path):
        d = _data_dir(tmp_path)
        with patch("links_sync._download", _fake_download(links=b'[["tampered"]]')):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["status"] == "failed" and "mismatch" in r["reason"]
        assert (d / "performer_links.json").read_bytes() == OLD_LINKS and (d / "aliases.json").read_bytes() == b"{}"

    async def test_an_incomplete_asset_changes_nothing(self, tmp_path):
        d = _data_dir(tmp_path)
        with patch("links_sync._download", _fake_download(drop="aliases.json")):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["status"] == "failed" and (d / "performer_links.json").read_bytes() == OLD_LINKS

    async def test_a_download_error_is_reported_not_raised(self, tmp_path):
        d = _data_dir(tmp_path)
        with patch("links_sync._download", AsyncMock(side_effect=RuntimeError("offline"))):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["status"] == "failed" and (d / "performer_links.json").read_bytes() == OLD_LINKS

    async def test_releases_without_the_asset_or_hashes_are_left_alone(self, tmp_path):
        d = _data_dir(tmp_path)
        for rel in (_release(with_asset=False), _release(with_hashes=False)):
            assert (await sync_link_files(d, rel, "2026.10.05"))["status"] == "skipped"
        assert (d / "performer_links.json").read_bytes() == OLD_LINKS

    async def test_a_release_the_local_database_is_not_at_is_skipped(self, tmp_path):
        d = _data_dir(tmp_path)
        assert (await sync_link_files(d, _release(tag="2026.10.05"), "2026.09.30"))["status"] == "skipped"
        assert (await sync_link_files(d, _release(tag="2026.10.05"), None))["status"] == "skipped"

    async def test_the_identities_of_changed_link_groups_are_reported(self, tmp_path):
        d = _data_dir(tmp_path)
        with patch("links_sync._download", _fake_download()):
            r = await sync_link_files(d, _release(), "2026.10.05")
        # old group {a, pornbox:1} was replaced by {a, adultfilmdatabase:2}: all three identities are affected
        assert r["link_uids"] == ["adultfilmdatabase:2", "pornbox:1", "stashdb.org:a"]

    async def test_a_fresh_install_reports_no_changed_identities(self, tmp_path):
        d = _data_dir(tmp_path, links=None, aliases=None)
        with patch("links_sync._download", _fake_download()):
            r = await sync_link_files(d, _release(), "2026.10.05")
        assert r["link_uids"] == []

    async def test_manifest_checksums_are_kept_true_when_present(self, tmp_path):
        d = _data_dir(tmp_path)
        (d / "manifest.json").write_text(json.dumps({"version": "2026.10.05", "checksums": {"performer_links.json": "sha256:old", "faces.json": "sha256:keep"}}))
        with patch("links_sync._download", _fake_download()):
            await sync_link_files(d, _release(), "2026.10.05")
        checks = json.loads((d / "manifest.json").read_text())["checksums"]
        assert checks["performer_links.json"] == f"sha256:{_sha(NEW_LINKS)}" and checks["faces.json"] == "sha256:keep"


class TestUpdaterIntegration:
    async def test_the_full_zip_swap_list_now_includes_both_files(self):
        assert {"performer_links.json", "aliases.json"} <= RELEASE_FILES

    def _updater(self, tmp_path, version="2026.10.05"):
        d = tmp_path / "data"
        d.mkdir(exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"version": version}))
        reload_fn = MagicMock(return_value=True)
        return DatabaseUpdater(data_dir=d, reload_fn=reload_fn), reload_fn

    async def test_startup_repair_syncs_and_reloads_only_when_something_changed(self, tmp_path):
        updater, reload_fn = self._updater(tmp_path)
        rel = _release()
        with patch("database_updater._fetch_releases", AsyncMock(return_value=[rel])), \
             patch("database_updater.sync_link_files", AsyncMock(return_value={"status": "updated", "reason": "", "changed": ["aliases.json"]})):
            r = await updater.sync_links_if_stale()
        assert r["status"] == "updated" and reload_fn.call_count == 1
        with patch("database_updater._fetch_releases", AsyncMock(return_value=[rel])), \
             patch("database_updater.sync_link_files", AsyncMock(return_value={"status": "in_sync", "reason": "", "changed": []})):
            await updater.sync_links_if_stale()
        assert reload_fn.call_count == 1

    async def test_scenes_matched_under_the_old_links_are_marked_for_rematching(self, tmp_path):
        updater, _ = self._updater(tmp_path)
        rec_db = MagicMock()
        rec_db.get_scene_ids_with_dirty_matches.return_value = {7, 9}
        rec_db.mark_fingerprints_for_refresh.return_value = 2
        done = {"status": "updated", "reason": "", "changed": ["performer_links.json"], "link_uids": ["stashdb.org:a"]}
        with patch("database_updater._fetch_releases", AsyncMock(return_value=[_release()])), \
             patch("database_updater.sync_link_files", AsyncMock(return_value=done)), \
             patch.object(DatabaseUpdater, "_get_rec_db_safe", return_value=rec_db):
            await updater.sync_links_if_stale()
        rec_db.get_scene_ids_with_dirty_matches.assert_called_once_with(["stashdb.org:a"])
        rec_db.mark_fingerprints_for_refresh.assert_called_once_with([7, 9])

    async def test_no_matching_release_or_a_listing_failure_is_harmless(self, tmp_path):
        updater, reload_fn = self._updater(tmp_path, version="2026.01.01")
        with patch("database_updater._fetch_releases", AsyncMock(return_value=[_release()])):
            assert (await updater.sync_links_if_stale())["status"] == "skipped"
        with patch("database_updater._fetch_releases", AsyncMock(side_effect=RuntimeError("rate limited"))):
            assert (await updater.sync_links_if_stale())["status"] == "failed"
        assert reload_fn.call_count == 0

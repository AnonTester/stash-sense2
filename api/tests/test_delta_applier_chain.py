"""Tests for delta_applier.py's chain-discovery logic (_walk_delta_chain,
find_delta_chain, find_full_bootstrap) -- kept separate from
test_delta_applier.py (scoped to apply_delta_db's own SQLite/usearch
apply logic). Mocks _fetch_releases directly (same style
test_database_updater.py already uses for find_delta_chain -- patching
the async function itself rather than mocking httpx, since these tests
care about the graph-walk, not the HTTP layer).

Added 2026-09-28 alongside find_full_bootstrap (the fresh-install/"force
full" backward-walk for a delta-only latest release, see stash-sense2-
data-gen's build/publish.py --delta-only) -- find_delta_chain itself had
no dedicated unit coverage before this either, only indirect exercise via
test_database_updater.py's own check_update() tests (which mock it out
entirely for anything not specifically testing it).
"""
from unittest.mock import AsyncMock, patch

import pytest

from delta_applier import find_delta_chain, find_full_bootstrap


def _release(tag, *, delta_from=None, full=False, min_sidecar=None):
    """One release() dict, GitHub API shape -- only the fields
    _fetch_releases' own callers actually read."""
    assets = []
    if delta_from is not None:
        assets.append({
            "name": f"stash-sense2-delta-{delta_from}-to-{tag}.zip",
            "browser_download_url": f"https://example/delta-{delta_from}-to-{tag}.zip",
            "size": 1_000_000,
        })
    if full:
        assets.append({
            "name": f"stash-sense2-data-{tag}.zip",
            "browser_download_url": f"https://example/full-{tag}.zip",
            "size": 900_000_000,
        })
    body = f"min-sidecar-version: {min_sidecar}" if min_sidecar else ""
    return {"tag_name": f"v{tag}", "assets": assets, "body": body}


pytestmark = pytest.mark.asyncio


class TestFindDeltaChain:
    async def test_none_current_version_is_fresh_install_no_chain(self):
        # find_full_bootstrap (not this function) handles a fresh install.
        assert await find_delta_chain("owner/repo", None) is None

    async def test_already_latest_returns_empty_chain(self):
        releases = [_release("2026.03.01", delta_from="2026.02.15", full=True)]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            chain = await find_delta_chain("owner/repo", "2026.03.01")
        assert chain == []

    async def test_single_hop_chain(self):
        releases = [_release("2026.03.01", delta_from="2026.02.15", full=True)]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            chain = await find_delta_chain("owner/repo", "2026.02.15")
        assert len(chain) == 1
        assert chain[0]["from_version"] == "2026.02.15"
        assert chain[0]["to_version"] == "2026.03.01"

    async def test_multi_hop_chain_is_oldest_to_newest(self):
        releases = [
            _release("2026.03.01", delta_from="2026.02.22"),
            _release("2026.02.22", delta_from="2026.02.15", full=True),
        ]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            chain = await find_delta_chain("owner/repo", "2026.02.15")
        assert [h["to_version"] for h in chain] == ["2026.02.22", "2026.03.01"]

    async def test_gap_in_chain_returns_none(self):
        # Latest's own delta hop points at a from_version with no release
        # of its own at all -- a genuine gap, not just "older than we asked".
        releases = [_release("2026.03.01", delta_from="2026.02.22", full=True)]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            chain = await find_delta_chain("owner/repo", "2026.01.01")
        assert chain is None

    async def test_delta_only_release_in_the_middle_of_the_chain_is_fine(self):
        # The whole point of --delta-only: a release with NO full zip must
        # still chain through cleanly for an already-installed sidecar.
        releases = [
            _release("2026.03.01", delta_from="2026.02.22"),  # delta-only
            _release("2026.02.22", delta_from="2026.02.15"),  # delta-only too
        ]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            chain = await find_delta_chain("owner/repo", "2026.02.15")
        assert len(chain) == 2

    async def test_min_sidecar_version_carried_per_hop(self):
        releases = [_release("2026.03.01", delta_from="2026.02.15", min_sidecar="0.36.0")]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            chain = await find_delta_chain("owner/repo", "2026.02.15")
        assert chain[0]["min_sidecar_version"] == "0.36.0"


class TestFindFullBootstrap:
    async def test_latest_release_already_has_a_full_zip(self):
        # The common/current case: no backward-walk needed at all.
        releases = [_release("2026.03.01", delta_from="2026.02.15", full=True)]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            result = await find_full_bootstrap("owner/repo")
        assert result["full_version"] == "2026.03.01"
        assert result["delta_chain"] == []

    async def test_walks_backward_past_delta_only_releases_to_the_full_zip(self):
        # The exact scenario --delta-only creates: latest (and the one
        # before it) are delta-only, but an older release still has a
        # full zip -- must find THAT one and chain the deltas forward.
        releases = [
            _release("2026.03.15", delta_from="2026.03.01"),  # delta-only (latest)
            _release("2026.03.01", delta_from="2026.02.15"),  # delta-only
            _release("2026.02.15", delta_from="2026.02.01", full=True),  # last full zip
        ]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            result = await find_full_bootstrap("owner/repo")
        assert result["full_version"] == "2026.02.15"
        assert result["full_download_url"] == "https://example/full-2026.02.15.zip"
        assert [h["to_version"] for h in result["delta_chain"]] == ["2026.03.01", "2026.03.15"]

    async def test_no_release_has_a_full_zip_returns_none(self):
        releases = [_release("2026.03.01", delta_from="2026.02.15")]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            result = await find_full_bootstrap("owner/repo")
        assert result is None

    async def test_no_releases_at_all_returns_none(self):
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=[])):
            result = await find_full_bootstrap("owner/repo")
        assert result is None

    async def test_gap_past_the_full_zip_degrades_to_full_zip_alone(self):
        # The full-zip release exists but the chain from there to latest
        # has a gap -- still usable (just install that full zip, no
        # trailing deltas), not a hard failure.
        releases = [
            _release("2026.03.01", delta_from="2026.02.22"),  # points at a version with no release
            _release("2026.02.15", full=True),  # a full zip, but not reachable via delta from latest
        ]
        with patch("delta_applier._fetch_releases", AsyncMock(return_value=releases)):
            result = await find_full_bootstrap("owner/repo")
        assert result["full_version"] == "2026.02.15"
        assert result["delta_chain"] is None

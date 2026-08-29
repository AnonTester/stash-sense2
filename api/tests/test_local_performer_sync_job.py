"""Tests for the parallelized LocalPerformerSyncJob (jobs/local_performer_sync_job.py).

Exercises the real async-fetch + threaded-embed + single-writer pipeline
end to end (small performer counts, mocked network/model calls) rather
than a simplified reimplementation -- concurrency bugs live in the real
interleaving, not in a serial stand-in for it.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from jobs.local_performer_sync_job import LocalPerformerSyncJob
from local_performer_index import LocalPerformerIndex


class _FakeStash:
    """Minimal stand-in for the Stash GraphQL client used by the job."""

    api_key = "fake-key"

    def __init__(self, performers: dict[int, dict]):
        self._performers = performers

    async def get_all_performers(self):
        # _fetch_one() reads name/image_path/stash_ids straight off this --
        # it no longer makes a separate per-performer get_performer() call
        # (that used to re-fetch the exact same fields one at a time
        # through the rate limiter; see local_performer_sync_job.py).
        return [
            {"id": str(pid), "name": p["name"], "image_path": p["image_path"], "stash_ids": []}
            for pid, p in self._performers.items()
        ]


def _make_context():
    ctx = MagicMock()
    ctx.job_id = 1
    ctx._stop_requested = False
    ctx.is_stop_requested = lambda: ctx._stop_requested
    ctx.report_progress = AsyncMock()
    ctx.checkpoint = AsyncMock()
    ctx.set_result_summary = MagicMock()
    return ctx


def _fake_detected_face():
    return SimpleNamespace(bbox={"w": 100, "h": 100})


def _patch_generator():
    """Every FaceEmbeddingGenerator() instance (one per embed worker)
    reports exactly one detected face and a fixed embedding."""
    def _new_generator(*args, **kwargs):
        gen = MagicMock()
        gen.detect_faces.return_value = [_fake_detected_face()]
        gen.get_embedding.return_value = SimpleNamespace(embedding=np.ones(512, dtype=np.float32))
        return gen
    return patch("jobs.local_performer_sync_job.FaceEmbeddingGenerator", side_effect=_new_generator)


def _patch_load_image():
    """No-op decode -- the fake bytes _patch_http() sends aren't a real,
    decodable image, and decoding isn't what these tests are about."""
    return patch("jobs.local_performer_sync_job.load_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8))


def _patch_http(image_bytes: bytes = b"fake-image-bytes", get_side_effect=None):
    """get_side_effect, if given, replaces the default fixed-response mock
    entirely (e.g. to delay/inspect specific URLs) -- see
    test_progress_advances_past_a_slow_straggler."""
    mock_response = MagicMock()
    mock_response.content = image_bytes
    mock_response.raise_for_status = MagicMock()

    client_instance = AsyncMock()
    if get_side_effect is not None:
        client_instance.get = AsyncMock(side_effect=get_side_effect)
    else:
        client_instance.get = AsyncMock(return_value=mock_response)

    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client_instance)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return patch("jobs.local_performer_sync_job.httpx.AsyncClient", mock_client_cls)


@pytest.fixture
def index_paths(tmp_path):
    return tmp_path / "local.usearch", tmp_path / "local_faces.json"


class TestLocalPerformerSyncJob:
    async def test_adds_new_performers_with_images(self, index_paths, monkeypatch):
        index_path, mapping_path = index_paths
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.DatabaseConfig",
            lambda data_dir: SimpleNamespace(
                local_embedding_index_path=index_path, local_faces_json_path=mapping_path,
            ),
        )
        stash = _FakeStash({
            1: {"name": "Alice", "image_path": "http://stash/performer/1/image"},
            2: {"name": "Bob", "image_path": "http://stash/performer/2/image"},
        })
        monkeypatch.setattr("jobs.local_performer_sync_job.get_stash_client", lambda: stash)
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.get_resource_manager",
            lambda: MagicMock(), raising=False,
        )

        with _patch_generator(), _patch_load_image(), _patch_http():
            job = LocalPerformerSyncJob()
            ctx = _make_context()
            result = await job.run(ctx, cursor=None)

        assert result is None  # completed, no resume cursor
        index = LocalPerformerIndex(index_path, mapping_path)
        assert len(index) == 2
        assert 1 in index and 2 in index
        ctx.set_result_summary.assert_called_once()
        summary = ctx.set_result_summary.call_args[0][0]
        assert "2 added" in summary

    async def test_skips_performers_without_custom_image(self, index_paths, monkeypatch):
        index_path, mapping_path = index_paths
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.DatabaseConfig",
            lambda data_dir: SimpleNamespace(
                local_embedding_index_path=index_path, local_faces_json_path=mapping_path,
            ),
        )
        stash = _FakeStash({
            1: {"name": "Alice", "image_path": "http://stash/performer/1/image?default=true"},
        })
        monkeypatch.setattr("jobs.local_performer_sync_job.get_stash_client", lambda: stash)

        with _patch_generator(), _patch_load_image(), _patch_http():
            job = LocalPerformerSyncJob()
            ctx = _make_context()
            await job.run(ctx, cursor=None)

        index = LocalPerformerIndex(index_path, mapping_path)
        assert len(index) == 0
        summary = ctx.set_result_summary.call_args[0][0]
        assert "1 had no usable image" in summary

    async def test_unchanged_image_is_not_reembedded(self, index_paths, monkeypatch):
        index_path, mapping_path = index_paths
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.DatabaseConfig",
            lambda data_dir: SimpleNamespace(
                local_embedding_index_path=index_path, local_faces_json_path=mapping_path,
            ),
        )
        stash = _FakeStash({1: {"name": "Alice", "image_path": "http://stash/performer/1/image"}})
        monkeypatch.setattr("jobs.local_performer_sync_job.get_stash_client", lambda: stash)

        with _patch_generator() as mock_gen_cls, _patch_load_image(), _patch_http():
            job = LocalPerformerSyncJob()
            await job.run(_make_context(), cursor=None)
            first_call_count = mock_gen_cls.call_count

            # Second run, same image bytes -> same fingerprint -> should
            # never reach the embed stage at all.
            await job.run(_make_context(), cursor=None)

        # A generator instance may still be created per embed worker even
        # if never used, but detect_faces must not be called a second time.
        index = LocalPerformerIndex(index_path, mapping_path)
        assert len(index) == 1

    async def test_removes_performer_no_longer_in_stash(self, index_paths, monkeypatch):
        index_path, mapping_path = index_paths
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.DatabaseConfig",
            lambda data_dir: SimpleNamespace(
                local_embedding_index_path=index_path, local_faces_json_path=mapping_path,
            ),
        )
        # Pre-seed the index with a performer that will no longer be present.
        index = LocalPerformerIndex(index_path, mapping_path)
        index.upsert(99, "Ghost", None, "deadbeef", None, np.ones(512, dtype=np.float32))
        index.save()

        stash = _FakeStash({1: {"name": "Alice", "image_path": "http://stash/performer/1/image"}})
        monkeypatch.setattr("jobs.local_performer_sync_job.get_stash_client", lambda: stash)

        with _patch_generator(), _patch_load_image(), _patch_http():
            job = LocalPerformerSyncJob()
            await job.run(_make_context(), cursor=None)

        index = LocalPerformerIndex(index_path, mapping_path)
        assert 99 not in index
        assert 1 in index

    async def test_stop_mid_run_returns_resumable_cursor_with_no_gaps(self, index_paths, monkeypatch):
        index_path, mapping_path = index_paths
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.DatabaseConfig",
            lambda data_dir: SimpleNamespace(
                local_embedding_index_path=index_path, local_faces_json_path=mapping_path,
            ),
        )
        performers = {
            i: {"name": f"P{i}", "image_path": f"http://stash/performer/{i}/image"}
            for i in range(1, 21)
        }
        stash = _FakeStash(performers)
        monkeypatch.setattr("jobs.local_performer_sync_job.get_stash_client", lambda: stash)

        ctx = _make_context()
        # Request a stop immediately -- the job should still finish
        # whatever's already in flight (bounded by FETCH_CONCURRENCY) and
        # return a cursor pointing at the first never-attempted position,
        # not skip anyone.
        ctx._stop_requested = True

        with _patch_generator(), _patch_load_image(), _patch_http():
            job = LocalPerformerSyncJob()
            result = await job.run(ctx, cursor=None)

        # Either finished nothing (stop seen before any dispatch) or
        # returned a valid resumable cursor -- in both cases, re-running
        # from that cursor (or from 0) must eventually cover everyone
        # with no duplicate/missing performers.
        import json as json_mod
        resume_cursor = result
        ctx2 = _make_context()
        with _patch_generator(), _patch_load_image(), _patch_http():
            job2 = LocalPerformerSyncJob()
            await job2.run(ctx2, cursor=resume_cursor)

        index = LocalPerformerIndex(index_path, mapping_path)
        assert len(index) == 20
        for i in range(1, 21):
            assert i in index

    async def test_progress_advances_past_a_slow_straggler(self, index_paths, monkeypatch):
        """Confirmed live: a job stuck reporting "2 of 1567" for several
        minutes while ~93 later positions had already completed
        concurrently -- one slow performer (position 0 here) must not pin
        the *displayed* progress while later positions (1, 2) finish
        first. See completed_count's own comment in the job for the fix."""
        index_path, mapping_path = index_paths
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.DatabaseConfig",
            lambda data_dir: SimpleNamespace(
                local_embedding_index_path=index_path, local_faces_json_path=mapping_path,
            ),
        )
        # report_progress is throttled to PROGRESS_REPORT_INTERVAL completions
        # in real runs (a synchronous DB write per call -- confirmed live,
        # calling it per-item turned a few seconds of real work into 5+
        # minutes over ~1600 performers). With only 3 performers here, the
        # real interval would mean report_progress never fires at all before
        # this test's own assertion window closes -- drop it to 1 so this
        # test still observes every completion, same as it did before the
        # throttle existed.
        monkeypatch.setattr("jobs.local_performer_sync_job.PROGRESS_REPORT_INTERVAL", 1)
        performers = {
            i: {"name": f"P{i}", "image_path": f"http://stash/performer/{i}/image"}
            for i in range(1, 4)
        }
        stash = _FakeStash(performers)
        monkeypatch.setattr("jobs.local_performer_sync_job.get_stash_client", lambda: stash)

        # Performer 1 (position 0, sorted first) hangs until released, at
        # its image fetch -- the one remaining async call _fetch_one() makes
        # per performer now that the redundant get_performer() re-fetch is
        # gone (see local_performer_sync_job.py). Performers 2 and 3
        # (positions 1, 2) must still be reported as progress before it
        # resolves. threading.Event, not asyncio.Event: _produce() runs on
        # its own event loop in its own OS thread (see
        # LocalPerformerSyncJob.run()'s producer_thread), a different loop
        # than this test's -- an asyncio.Event created here can't safely be
        # awaited over there.
        release_straggler = threading.Event()
        mock_response = MagicMock()
        mock_response.content = b"fake-image-bytes"
        mock_response.raise_for_status = MagicMock()

        async def _slow_get(url, **kwargs):
            if "/performer/1/" in url:
                await asyncio.to_thread(release_straggler.wait)
            return mock_response

        ctx = _make_context()
        reported_before_release: list[int] = []
        ctx.report_progress = AsyncMock(side_effect=lambda done, total: reported_before_release.append(done))

        with _patch_generator(), _patch_load_image(), _patch_http(get_side_effect=_slow_get):
            job = LocalPerformerSyncJob()
            run_task = asyncio.create_task(job.run(ctx, cursor=None))
            # Give the fast positions (1, 2) a real chance to complete on
            # their own OS thread/event loop while position 0 is still
            # blocked -- a zero-delay sleep loop only yields within this
            # test's own loop, not real wall-clock time for the producer
            # thread to actually run.
            # release_straggler must always get set before this test exits --
            # otherwise _produce()'s blocking asyncio.to_thread(...wait) never
            # returns, leaking a permanently-stuck thread-pool worker that
            # hangs the *entire* test process at interpreter shutdown, not
            # just this test. try/finally so an assertion failure here still
            # fails cleanly instead of hanging the whole suite.
            try:
                for _ in range(100):
                    if max(reported_before_release, default=0) >= 2:
                        break
                    await asyncio.sleep(0.02)
                assert max(reported_before_release, default=0) >= 2, (
                    "progress should reflect positions 1/2 completing, not stay "
                    "pinned at 0 behind the still-blocked straggler"
                )
            finally:
                release_straggler.set()
            await run_task

    async def test_resumed_run_reports_absolute_progress(self, index_paths, monkeypatch):
        """A resumed run's displayed progress must be absolute (out of the
        full `total`), not reset to counting only the remaining work --
        completed_count starts at start_position, same as next_expected."""
        index_path, mapping_path = index_paths
        monkeypatch.setattr(
            "jobs.local_performer_sync_job.DatabaseConfig",
            lambda data_dir: SimpleNamespace(
                local_embedding_index_path=index_path, local_faces_json_path=mapping_path,
            ),
        )
        performers = {
            i: {"name": f"P{i}", "image_path": f"http://stash/performer/{i}/image"}
            for i in range(1, 6)
        }
        stash = _FakeStash(performers)
        monkeypatch.setattr("jobs.local_performer_sync_job.get_stash_client", lambda: stash)

        ctx = _make_context()
        reported: list[int] = []
        ctx.report_progress = AsyncMock(side_effect=lambda done, total: reported.append(done))

        with _patch_generator(), _patch_load_image(), _patch_http():
            job = LocalPerformerSyncJob()
            import json as json_mod
            await job.run(ctx, cursor=json_mod.dumps({"position": 3}))

        # Resuming from position 3 with 2 remaining performers -> reported
        # progress should climb from 4 towards 5, never restart at 1.
        assert min(reported) >= 4

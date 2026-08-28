"""Tests for fingerprint generation job wrapper.

FingerprintGenerationJob now backs two distinct queue job types
(fingerprint_generation / fingerprint_refresh_outdated), with
refresh_outdated fixed at construction time by queue_manager.py's
_create_job_instance -- not hidden inside the resumption cursor anymore
(that made both Settings tab buttons/Operations tab react to "is *a*
fingerprint job running" instead of their own scope). See
jobs/fingerprint_job.py's docstring.
"""
import json

import pytest
from unittest.mock import MagicMock, patch
from base_job import JobContext


class TestFingerprintJob:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.update_job_progress = MagicMock()
        return db

    @pytest.fixture
    def ctx(self, mock_db):
        return JobContext(job_id=1, db=mock_db, queue_manager=None)

    @pytest.mark.asyncio
    async def test_runs_generator(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        mock_progress = MagicMock()
        mock_progress.processed_scenes = 5
        mock_progress.total_scenes = 10
        mock_progress.status.value = "completed"
        mock_progress.batch_completed = True
        mock_progress.current_offset = 5

        async def fake_generate(**kwargs):
            yield mock_progress

        mock_generator_cls = MagicMock()
        mock_generator_instance = MagicMock()
        mock_generator_instance.generate_all = fake_generate
        mock_generator_cls.return_value = mock_generator_instance

        with patch('jobs.fingerprint_job.SceneFingerprintGenerator', mock_generator_cls), \
             patch('jobs.fingerprint_job.get_stash_client', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_rec_db', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_db_version', return_value='2026.01.30'):
            job = FingerprintGenerationJob(refresh_outdated=False)
            await job.run(ctx)

        mock_generator_cls.assert_called_once()
        call_kwargs = mock_generator_cls.call_args[1]
        assert call_kwargs['db_version'] == '2026.01.30'
        ctx._db.update_job_progress.assert_called()

    @pytest.mark.asyncio
    async def test_stops_on_request(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        ctx.request_stop()
        with patch('jobs.fingerprint_job.SceneFingerprintGenerator') as mock_gen, \
             patch('jobs.fingerprint_job.get_stash_client', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_rec_db', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_db_version', return_value='2026.01.30'):
            job = FingerprintGenerationJob(refresh_outdated=False)
            result = await job.run(ctx)
        assert result is None
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_fails_without_db_version(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        with patch('jobs.fingerprint_job.get_db_version', return_value=None):
            job = FingerprintGenerationJob(refresh_outdated=False)
            with pytest.raises(RuntimeError, match="No face recognition database loaded"):
                await job.run(ctx)

    @staticmethod
    def _run_and_capture(ctx, job, cursor=None):
        captured = {}

        async def fake_generate(**kwargs):
            captured.update(kwargs)
            return
            yield  # pragma: no cover -- makes this an async generator

        mock_generator_instance = MagicMock()
        mock_generator_instance.generate_all = fake_generate

        async def _go():
            with patch('jobs.fingerprint_job.SceneFingerprintGenerator', return_value=mock_generator_instance), \
                 patch('jobs.fingerprint_job.get_stash_client', return_value=MagicMock()), \
                 patch('jobs.fingerprint_job.get_rec_db', return_value=MagicMock()), \
                 patch('jobs.fingerprint_job.get_db_version', return_value='2026.01.30'):
                await job.run(ctx, cursor=cursor)

        return _go(), captured

    @pytest.mark.asyncio
    async def test_missing_job_always_runs_refresh_outdated_false(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        job = FingerprintGenerationJob(refresh_outdated=False)
        coro, captured = self._run_and_capture(ctx, job)
        await coro

        assert captured['refresh_outdated'] is False
        assert captured['skip_errors'] is False

    @pytest.mark.asyncio
    async def test_refresh_job_always_runs_refresh_outdated_true(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        job = FingerprintGenerationJob(refresh_outdated=True)
        coro, captured = self._run_and_capture(ctx, job)
        await coro

        assert captured['refresh_outdated'] is True
        assert captured['skip_errors'] is False

    @pytest.mark.asyncio
    async def test_resuming_from_offset_cursor_sets_skip_errors_and_scope_unaffected(self, ctx):
        # Resuming a paused/crashed job (an offset/processed cursor) skips
        # previously-errored scenes -- but scope still comes purely from
        # which job type this is, never from the cursor.
        from jobs.fingerprint_job import FingerprintGenerationJob
        job = FingerprintGenerationJob(refresh_outdated=True)
        cursor = json.dumps({"offset": 2700, "processed": 2780})
        coro, captured = self._run_and_capture(ctx, job, cursor=cursor)
        await coro

        assert captured['refresh_outdated'] is True
        assert captured['start_offset'] == 2700
        assert captured['skip_errors'] is True

    @pytest.mark.asyncio
    async def test_checkpoint_cursor_has_no_scope_field(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob

        mock_progress = MagicMock()
        mock_progress.processed_scenes = 5
        mock_progress.total_scenes = 10
        mock_progress.status.value = "completed"
        mock_progress.batch_completed = True
        mock_progress.current_offset = 100

        async def fake_generate(**kwargs):
            yield mock_progress

        mock_generator_instance = MagicMock()
        mock_generator_instance.generate_all = fake_generate

        with patch('jobs.fingerprint_job.SceneFingerprintGenerator', return_value=mock_generator_instance), \
             patch('jobs.fingerprint_job.get_stash_client', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_rec_db', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_db_version', return_value='2026.01.30'):
            job = FingerprintGenerationJob(refresh_outdated=False)
            await job.run(ctx)

        checkpoint_cursor = json.loads(ctx._db.update_job_progress.call_args.kwargs['cursor'])
        assert checkpoint_cursor == {"offset": 100, "processed": 5}

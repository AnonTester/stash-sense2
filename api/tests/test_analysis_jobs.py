"""Tests for analysis job wrappers."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from base_job import JobContext
from jobs.analysis_jobs import AnalysisJob, FULL_RUN_CURSOR


class TestAnalysisJob:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.update_job_progress = MagicMock()
        return db

    @pytest.fixture
    def mock_stash(self):
        return MagicMock()

    @pytest.fixture
    def ctx(self, mock_db):
        return JobContext(job_id=1, db=mock_db, queue_manager=None)

    @pytest.mark.asyncio
    async def test_runs_analyzer(self, ctx, mock_stash, mock_db):
        mock_analyzer = MagicMock()
        mock_analyzer._items_total = None  # base.py default: no known upfront total
        mock_result = MagicMock()
        mock_result.recommendations_created = 5
        mock_result.recommendations_updated = 0
        mock_result.items_processed = 10
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with patch('jobs.analysis_jobs.ANALYZERS') as mock_reg, \
             patch('jobs.analysis_jobs.get_rec_db', return_value=mock_db), \
             patch('jobs.analysis_jobs.get_stash_client', return_value=mock_stash):
            mock_cls = MagicMock(return_value=mock_analyzer)
            mock_reg.get.return_value = mock_cls

            job = AnalysisJob("duplicate_performer")
            result = await job.run(ctx)

        mock_analyzer.run.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_full_cursor_runs_non_incremental(self, ctx, mock_stash, mock_db):
        mock_analyzer = MagicMock()
        mock_analyzer._items_total = None  # base.py default: no known upfront total
        mock_result = MagicMock()
        mock_result.recommendations_created = 0
        mock_result.recommendations_updated = 0
        mock_result.items_processed = 0
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with patch('jobs.analysis_jobs.ANALYZERS') as mock_reg, \
             patch('jobs.analysis_jobs.get_rec_db', return_value=mock_db), \
             patch('jobs.analysis_jobs.get_stash_client', return_value=mock_stash):
            mock_cls = MagicMock(return_value=mock_analyzer)
            mock_reg.get.return_value = mock_cls

            job = AnalysisJob("scene_fingerprint_match")
            await job.run(ctx, cursor=FULL_RUN_CURSOR)

        mock_analyzer.run.assert_called_once_with(incremental=False)

    @pytest.mark.asyncio
    async def test_respects_stop_request(self, ctx, mock_stash, mock_db):
        ctx.request_stop()
        with patch('jobs.analysis_jobs.get_rec_db', return_value=mock_db), \
             patch('jobs.analysis_jobs.get_stash_client', return_value=mock_stash):
            job = AnalysisJob("duplicate_performer")
            result = await job.run(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_reports_analyzer_known_total_when_larger(self, ctx, mock_stash, mock_db):
        # analyzer.set_items_total() (base.py) lets an analyzer report an
        # upfront known total (e.g. total local entities) that can exceed
        # items_processed so far -- the job should report that total
        # rather than just what's been processed, so progress UI reflects
        # "N of M", not "N of N".
        mock_analyzer = MagicMock()
        mock_analyzer._items_total = 100
        mock_result = MagicMock()
        mock_result.recommendations_created = 1
        mock_result.recommendations_updated = 0
        mock_result.items_processed = 40
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with patch('jobs.analysis_jobs.ANALYZERS') as mock_reg, \
             patch('jobs.analysis_jobs.get_rec_db', return_value=mock_db), \
             patch('jobs.analysis_jobs.get_stash_client', return_value=mock_stash):
            mock_reg.get.return_value = MagicMock(return_value=mock_analyzer)

            job = AnalysisJob("duplicate_performer")
            with patch.object(ctx, "report_progress", new=AsyncMock()) as mock_report:
                await job.run(ctx)

        mock_report.assert_called_once_with(40, 100)

    @pytest.mark.asyncio
    async def test_reports_processed_when_it_exceeds_known_total(self, ctx, mock_stash, mock_db):
        # A stale/undercounted _items_total shouldn't make the reported
        # total look smaller than what's already been processed.
        mock_analyzer = MagicMock()
        mock_analyzer._items_total = 10
        mock_result = MagicMock()
        mock_result.recommendations_created = 0
        mock_result.recommendations_updated = 0
        mock_result.items_processed = 25
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with patch('jobs.analysis_jobs.ANALYZERS') as mock_reg, \
             patch('jobs.analysis_jobs.get_rec_db', return_value=mock_db), \
             patch('jobs.analysis_jobs.get_stash_client', return_value=mock_stash):
            mock_reg.get.return_value = MagicMock(return_value=mock_analyzer)

            job = AnalysisJob("duplicate_performer")
            with patch.object(ctx, "report_progress", new=AsyncMock()) as mock_report:
                await job.run(ctx)

        mock_report.assert_called_once_with(25, 25)

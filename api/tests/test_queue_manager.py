"""Tests for QueueManager."""
import httpx
import pytest
from queue_manager import QueueManager, _friendly_error_message
from job_models import ResourceType, JobPriority


class TestQueueManager:
    @pytest.fixture
    def db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(str(tmp_path / "test.db"))

    @pytest.fixture
    def mgr(self, db):
        return QueueManager(db)

    def test_default_resource_slots(self, mgr):
        assert mgr.resource_slots[ResourceType.GPU] == 1
        assert mgr.resource_slots[ResourceType.NETWORK] == 1
        assert mgr.resource_slots[ResourceType.LIGHT] == 3

    def test_submit_job(self, mgr):
        job_id = mgr.submit("duplicate_performer", triggered_by="user")
        assert job_id is not None
        assert job_id > 0

    def test_submit_unknown_type_raises(self, mgr):
        with pytest.raises(ValueError, match="Unknown job type"):
            mgr.submit("nonexistent_type", triggered_by="user")

    def test_submit_duplicate_returns_none(self, mgr):
        mgr.submit("duplicate_performer", triggered_by="user")
        result = mgr.submit("duplicate_performer", triggered_by="user")
        assert result is None

    def test_submit_with_priority_override(self, mgr):
        job_id = mgr.submit("duplicate_performer", triggered_by="user",
                            priority=JobPriority.HIGH)
        job = mgr.get_job(job_id)
        assert job["priority"] == JobPriority.HIGH

    def test_get_queue_status(self, mgr):
        mgr.submit("duplicate_performer", triggered_by="user")
        status = mgr.get_status()
        assert status["queued"] == 1
        assert status["running"] == 0

    def test_cancel_queued_job(self, mgr):
        job_id = mgr.submit("duplicate_performer", triggered_by="user")
        mgr.cancel(job_id)
        job = mgr.get_job(job_id)
        assert job["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_should_job_yield_false_when_no_waiters(self, mgr, db):
        job_id = db.submit_job(type="duplicate_performer", priority=50, triggered_by="user")
        db.start_job(job_id)
        result = await mgr.should_job_yield(job_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_job_yield_true_when_higher_priority_waits(self, mgr, db):
        # Running job: low priority, LIGHT resource
        low_id = db.submit_job(type="duplicate_performer", priority=100, triggered_by="schedule")
        db.start_job(low_id)
        # Queued job: high priority, same resource
        db.submit_job(type="duplicate_scene_files", priority=10, triggered_by="user")
        result = await mgr.should_job_yield(low_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_job_yield_false_when_different_resource(self, mgr, db):
        # Running job: GPU resource
        gpu_id = db.submit_job(type="fingerprint_generation", priority=100, triggered_by="schedule")
        db.start_job(gpu_id)
        # Queued job: high priority, but LIGHT resource (different slot)
        db.submit_job(type="duplicate_performer", priority=10, triggered_by="user")
        result = await mgr.should_job_yield(gpu_id)
        assert result is False

    def test_startup_recovery(self, db):
        """Jobs left running after crash should be re-queued."""
        job_id = db.submit_job(type="test", priority=50, triggered_by="user")
        db.start_job(job_id)
        mgr = QueueManager(db)
        mgr.recover_on_startup()
        job = db.get_job(job_id)
        assert job["status"] == "queued"

    def test_cancel_stopping_job(self, mgr, db):
        """Stopping jobs should be force-cancelled."""
        job_id = mgr.submit("duplicate_performer", triggered_by="user")
        db.start_job(job_id)
        db.set_job_status(job_id, "stopping")
        mgr.cancel(job_id)
        job = mgr.get_job(job_id)
        assert job["status"] == "cancelled"

    def test_clear_history(self, mgr, db):
        job_id = mgr.submit("duplicate_performer", triggered_by="user")
        db.start_job(job_id)
        db.complete_job(job_id)
        count = mgr.clear_history()
        assert count == 1
        assert len(mgr.get_jobs()) == 0

    def test_clear_history_preserves_active(self, mgr, db):
        mgr.submit("duplicate_performer", triggered_by="user")
        count = mgr.clear_history()
        assert count == 0
        assert len(mgr.get_jobs()) == 1

    def test_request_shutdown(self, mgr):
        mgr.request_shutdown()
        assert mgr.is_shutting_down is True


from job_models import JOB_REGISTRY

class TestJobFactory:
    def test_create_analysis_job(self, tmp_path):
        from recommendations_db import RecommendationsDB
        db = RecommendationsDB(str(tmp_path / "test.db"))
        mgr = QueueManager(db)
        job = mgr._create_job_instance("duplicate_performer")
        from jobs.analysis_jobs import AnalysisJob
        assert isinstance(job, AnalysisJob)

    def test_create_fingerprint_job(self, tmp_path):
        from recommendations_db import RecommendationsDB
        db = RecommendationsDB(str(tmp_path / "test.db"))
        mgr = QueueManager(db)
        job = mgr._create_job_instance("fingerprint_generation")
        from jobs.fingerprint_job import FingerprintGenerationJob
        assert isinstance(job, FingerprintGenerationJob)

    def test_create_database_update_job(self, tmp_path):
        from recommendations_db import RecommendationsDB
        db = RecommendationsDB(str(tmp_path / "test.db"))
        mgr = QueueManager(db)
        job = mgr._create_job_instance("database_update")
        from jobs.database_update_job import DatabaseUpdateJob
        assert isinstance(job, DatabaseUpdateJob)

    def test_create_unknown_raises(self, tmp_path):
        from recommendations_db import RecommendationsDB
        db = RecommendationsDB(str(tmp_path / "test.db"))
        mgr = QueueManager(db)
        with pytest.raises(ValueError):
            mgr._create_job_instance("nonexistent")


class TestScheduleSeeding:
    def test_seeds_default_schedules(self, tmp_path):
        from recommendations_db import RecommendationsDB
        db = RecommendationsDB(str(tmp_path / "test.db"))
        mgr = QueueManager(db)
        mgr.seed_default_schedules()
        schedules = db.get_all_job_schedules()
        schedulable_types = [t for t, d in JOB_REGISTRY.items() if d.schedulable]
        assert len(schedules) == len(schedulable_types)

    def test_seeds_all_disabled_by_default(self, tmp_path):
        from recommendations_db import RecommendationsDB
        db = RecommendationsDB(str(tmp_path / "test.db"))
        mgr = QueueManager(db)
        mgr.seed_default_schedules()
        schedules = db.get_all_job_schedules()
        for schedule in schedules:
            assert schedule["enabled"] == 0, f"{schedule['type']} should be disabled by default"

    def test_seed_preserves_user_overrides(self, tmp_path):
        from recommendations_db import RecommendationsDB
        db = RecommendationsDB(str(tmp_path / "test.db"))
        db.upsert_job_schedule("duplicate_performer", True, 999.0, 50)
        mgr = QueueManager(db)
        mgr.seed_default_schedules()
        schedule = db.get_job_schedule("duplicate_performer")
        assert schedule["interval_hours"] == 999.0


class TestFriendlyErrorMessage:
    def test_stash_unavailable_names_stash_specifically(self):
        from fingerprint_generator import StashUnavailableError

        exc = StashUnavailableError(
            "Could not connect to the Stash instance at http://192.168.1.2:9997 "
            "after retrying for 5 minutes (12 attempts) -- last error: Connection refused"
        )

        message = _friendly_error_message(exc, job_row=None)

        assert "Stash instance" in message
        assert "192.168.1.2:9997" in message

    def test_stash_unavailable_includes_items_processed(self):
        from fingerprint_generator import StashUnavailableError

        exc = StashUnavailableError("Could not connect to the Stash instance at http://x")
        message = _friendly_error_message(exc, job_row={"items_processed": 42})

        assert "42 items processed" in message

    def test_connect_timeout_is_network_timeout(self):
        message = _friendly_error_message(httpx.ConnectTimeout("timed out"), job_row=None)
        assert message == "Network timeout after retries"

    def test_read_timeout_is_network_timeout(self):
        message = _friendly_error_message(httpx.ReadTimeout("timed out"), job_row=None)
        assert message == "Network timeout after retries"

    def test_connect_error_without_stash_context_is_generic(self):
        # A plain httpx.ConnectError not wrapped in StashUnavailableError
        # (e.g. from some other remote call) keeps the generic message --
        # only the fingerprint generator's Stash-specific retry path
        # raises StashUnavailableError.
        message = _friendly_error_message(httpx.ConnectError("refused"), job_row=None)
        assert message == "Network connection failed"

    def test_http_status_error_includes_status_code(self):
        request = httpx.Request("GET", "http://example.com")
        response = httpx.Response(503, request=request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)

        message = _friendly_error_message(exc, job_row=None)

        assert message == "HTTP 503 from remote"

    def test_generic_exception_falls_back_to_str(self):
        message = _friendly_error_message(ValueError("something broke"), job_row=None)
        assert message == "something broke"

    def test_generic_exception_truncated_at_200_chars(self):
        message = _friendly_error_message(ValueError("x" * 500), job_row=None)
        assert len(message) == 200

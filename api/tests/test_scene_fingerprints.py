"""Tests for scene fingerprint storage in recommendations DB."""



class TestSceneFingerprintSchema:
    """Tests for scene fingerprint table operations."""

    def test_create_fingerprint(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_scene_fingerprint(
            stash_scene_id=123,
            total_faces=5,
            frames_analyzed=40,
            fingerprint_status="complete",
        )

        assert fp_id is not None
        assert fp_id > 0

    def test_get_fingerprint_by_scene_id(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_scene_fingerprint(
            stash_scene_id=456,
            total_faces=3,
            frames_analyzed=40,
        )

        fp = db.get_scene_fingerprint(stash_scene_id=456)

        assert fp is not None
        assert fp["stash_scene_id"] == 456
        assert fp["total_faces"] == 3
        assert fp["frames_analyzed"] == 40

    def test_add_fingerprint_face(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_scene_fingerprint(stash_scene_id=789, total_faces=2, frames_analyzed=40)

        db.add_fingerprint_face(
            fingerprint_id=fp_id,
            performer_id="stashdb:abc-123",
            face_count=10,
            avg_confidence=0.85,
            proportion=0.5,
        )

        faces = db.get_fingerprint_faces(fp_id)

        assert len(faces) == 1
        assert faces[0]["performer_id"] == "stashdb:abc-123"
        assert faces[0]["face_count"] == 10
        assert faces[0]["proportion"] == 0.5

    def test_get_all_fingerprints(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_scene_fingerprint(stash_scene_id=1, total_faces=2, frames_analyzed=40)
        db.create_scene_fingerprint(stash_scene_id=2, total_faces=3, frames_analyzed=40)
        db.create_scene_fingerprint(stash_scene_id=3, total_faces=0, frames_analyzed=40)

        fps = db.get_all_scene_fingerprints()

        assert len(fps) == 3

    def test_fingerprint_upsert_updates_existing(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id1 = db.create_scene_fingerprint(stash_scene_id=100, total_faces=2, frames_analyzed=40)
        fp_id2 = db.create_scene_fingerprint(stash_scene_id=100, total_faces=5, frames_analyzed=40)

        # Should update, not create new
        assert fp_id1 == fp_id2

        fp = db.get_scene_fingerprint(stash_scene_id=100)
        assert fp["total_faces"] == 5


class TestMarkFingerprintsForRefresh:
    def test_marks_all_when_no_scene_ids(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")
        db.create_scene_fingerprint(stash_scene_id=1, total_faces=1, frames_analyzed=40, db_version="2026.01.01")
        db.create_scene_fingerprint(stash_scene_id=2, total_faces=1, frames_analyzed=40, db_version="2026.01.01")

        count = db.mark_fingerprints_for_refresh(None)

        assert count == 2
        assert db.get_scene_fingerprint(stash_scene_id=1)["db_version"] is None
        assert db.get_scene_fingerprint(stash_scene_id=2)["db_version"] is None

    def test_marks_only_given_scene_ids(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")
        db.create_scene_fingerprint(stash_scene_id=1, total_faces=1, frames_analyzed=40, db_version="2026.01.01")
        db.create_scene_fingerprint(stash_scene_id=2, total_faces=1, frames_analyzed=40, db_version="2026.01.01")

        count = db.mark_fingerprints_for_refresh([1])

        assert count == 1
        assert db.get_scene_fingerprint(stash_scene_id=1)["db_version"] is None
        assert db.get_scene_fingerprint(stash_scene_id=2)["db_version"] == "2026.01.01"


class TestResetSceneFingerprintsWithBackup:
    """reset_scene_fingerprints_with_backup() backs up scene_fingerprints +
    scene_fingerprint_faces to timestamped tables, then marks everything
    for refresh -- added for the Settings UI's Detection Resolution change
    modal, so existing fingerprints can be safely regenerated under a new
    detection_size instead of silently staying stale."""

    def test_backs_up_and_marks_all_for_refresh(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")
        fp1 = db.create_scene_fingerprint(stash_scene_id=1, total_faces=2, frames_analyzed=60, db_version="2026.01.01")
        fp2 = db.create_scene_fingerprint(stash_scene_id=2, total_faces=1, frames_analyzed=60, db_version="2026.01.01")
        db.add_fingerprint_face(fingerprint_id=fp1, performer_id="stashdb:abc", face_count=5)
        db.add_fingerprint_face(fingerprint_id=fp2, performer_id="stashdb:def", face_count=3)

        result = db.reset_scene_fingerprints_with_backup()

        assert result["fingerprints_backed_up"] == 2
        assert result["faces_backed_up"] == 2
        assert result["marked_for_refresh"] == 2
        assert db.get_scene_fingerprint(stash_scene_id=1)["db_version"] is None
        assert db.get_scene_fingerprint(stash_scene_id=2)["db_version"] is None

    def test_backup_tables_contain_pre_reset_data(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")
        db.create_scene_fingerprint(stash_scene_id=1, total_faces=4, frames_analyzed=60, db_version="2026.01.01")

        result = db.reset_scene_fingerprints_with_backup()

        with db._connection() as conn:
            row = conn.execute(
                f"SELECT stash_scene_id, total_faces, db_version FROM {result['backup_fingerprints_table']}"
            ).fetchone()
        assert row["stash_scene_id"] == 1
        assert row["total_faces"] == 4
        assert row["db_version"] == "2026.01.01"  # backup preserves the pre-reset value

    def test_empty_database_backs_up_zero(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        result = db.reset_scene_fingerprints_with_backup()

        assert result["fingerprints_backed_up"] == 0
        assert result["faces_backed_up"] == 0
        assert result["marked_for_refresh"] == 0

    def test_repeated_resets_use_distinct_backup_tables(self, tmp_path, monkeypatch):
        from recommendations_db import RecommendationsDB
        import recommendations_db as recommendations_db_module

        db = RecommendationsDB(tmp_path / "test.db")
        db.create_scene_fingerprint(stash_scene_id=1, total_faces=1, frames_analyzed=60)

        times = iter(["20260101_000000", "20260101_000001"])
        monkeypatch.setattr(recommendations_db_module.time, "strftime", lambda fmt: next(times))

        result1 = db.reset_scene_fingerprints_with_backup()
        result2 = db.reset_scene_fingerprints_with_backup()

        assert result1["backup_fingerprints_table"] != result2["backup_fingerprints_table"]

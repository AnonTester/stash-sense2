"""Tests for scene_face_embeddings caching (buffalo_l single-embedding
schema) and the v13->v14 migration that replaces the old legacy dual
facenet_vector/arcface_vector table.

See identification_router.py's _identify_scene_from_cache / recommendations_db.py's
schema-v14 migration for the code under test here.
"""
import sqlite3

import pytest

from recommendations_db import RecommendationsDB, SCHEMA_VERSION


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "test.db"))


class TestFaceEmbeddingCacheRoundTrip:
    def test_replace_then_get_round_trips_embedding_bytes(self, db):
        blob = b"\x01\x02\x03\x04" * 128  # stand-in for a 512-dim float32 vector
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
             "confidence": 0.9, "yaw": 5.0, "embedding": blob},
        ])

        rows = db.get_face_embeddings(1)

        assert len(rows) == 1
        assert rows[0]["embedding"] == blob
        assert rows[0]["frame_index"] == 0
        assert rows[0]["yaw"] == 5.0

    def test_replace_with_empty_list_clears_existing_rows(self, db):
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"x" * 8},
        ])
        db.replace_face_embeddings(1, [])

        assert db.get_face_embeddings(1) == []

    def test_get_face_embeddings_scoped_to_scene(self, db):
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"a" * 8},
        ])
        db.replace_face_embeddings(2, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.8, "yaw": None, "embedding": b"b" * 8},
        ])

        assert len(db.get_face_embeddings(1)) == 1
        assert len(db.get_face_embeddings(2)) == 1
        assert db.get_face_embeddings(1)[0]["embedding"] == b"a" * 8


class TestSchemaV14Migration:
    def _make_v13_db(self, path):
        """Build a schema_version=13 database with the legacy dual-vector
        scene_face_embeddings table and a populated scene_signal_cache row,
        simulating a real pre-fix install."""
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (13);

            CREATE TABLE scene_signal_cache (
                stash_scene_id INTEGER PRIMARY KEY,
                num_frames INTEGER NOT NULL,
                min_face_size INTEGER NOT NULL,
                min_face_confidence REAL NOT NULL,
                start_offset_pct REAL NOT NULL,
                end_offset_pct REAL NOT NULL,
                frames_analyzed INTEGER NOT NULL,
                body_shoulder_hip_ratio REAL,
                body_leg_torso_ratio REAL,
                body_arm_span_height_ratio REAL,
                body_confidence REAL,
                tattoos_detected INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO scene_signal_cache (stash_scene_id, num_frames, min_face_size,
                min_face_confidence, start_offset_pct, end_offset_pct, frames_analyzed)
            VALUES (1, 60, 60, 0.5, 0.02, 0.98, 60);

            CREATE TABLE scene_face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_scene_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                bbox_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                yaw REAL,
                facenet_vector BLOB NOT NULL,
                arcface_vector BLOB NOT NULL
            );

            -- Needed for the v15/v16 migrations further along this chain
            -- (scene_fingerprint_faces -> scene_fingerprint_matches, and
            -- an ALTER TABLE ADD COLUMN on scene_fingerprints itself) --
            -- everything else in recommendations_db.py's other tables
            -- still isn't needed for this test's own assertions.
            CREATE TABLE scene_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_scene_id INTEGER NOT NULL UNIQUE,
                total_faces INTEGER NOT NULL DEFAULT 0,
                frames_analyzed INTEGER NOT NULL DEFAULT 0,
                fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                db_version TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE scene_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL,
                proportion REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(fingerprint_id, performer_id)
            );

            -- recommendations_db.py's other tables aren't needed for this
            -- migration to run, since _migrate_schema applies steps
            -- sequentially by version regardless of unrelated tables.
        """)
        conn.commit()
        conn.close()

    def test_migration_drops_legacy_columns_and_stale_signal_cache(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        self._make_v13_db(db_path)

        db = RecommendationsDB(str(db_path))

        with sqlite3.connect(str(db_path)) as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == SCHEMA_VERSION

            cols = {row[1] for row in conn.execute("PRAGMA table_info(scene_face_embeddings)")}
            assert "embedding" in cols
            assert "facenet_vector" not in cols
            assert "arcface_vector" not in cols

            # The stale pre-fix scene_signal_cache row (written while face
            # caching was disabled) must not survive -- otherwise it would
            # look like a valid, compatible cache with zero cached faces,
            # which is indistinguishable from a genuine 0-face scene.
            remaining = conn.execute("SELECT COUNT(*) FROM scene_signal_cache").fetchone()[0]
            assert remaining == 0

        # And the new table is actually usable end-to-end post-migration.
        db.replace_face_embeddings(1, [
            {"frame_index": 0, "bbox": {}, "confidence": 0.9, "yaw": None, "embedding": b"z" * 8},
        ])
        assert db.get_face_embeddings(1)[0]["embedding"] == b"z" * 8

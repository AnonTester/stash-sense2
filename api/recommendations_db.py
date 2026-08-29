"""
SQLite Database Layer for Stash Sense Recommendations

Stores user-local recommendations, analysis state, and settings.
Separate from the distributed performers.db to allow independent updates.

See: docs/plans/2026-01-28-recommendations-engine-design.md
"""

import sqlite3
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Any


SCHEMA_VERSION = 17

# Caches the DB-independent, expensive-to-recompute part of scene
# fingerprinting (frame extraction + face detection+embedding)
# so a performer-database version bump only has to redo the cheap
# DB-dependent matching/re-ranking step, not the whole pipeline. Keyed by
# stash_scene_id directly (not fingerprint_id) so it survives
# scene_fingerprints rows being replaced on refresh. Shared between
# _create_schema (fresh installs) and _migrate_schema (existing installs)
# so the DDL only needs to be written once.
SCENE_SIGNAL_CACHE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS scene_signal_cache (
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

    CREATE TABLE IF NOT EXISTS scene_face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stash_scene_id INTEGER NOT NULL,
        frame_index INTEGER NOT NULL,
        bbox_json TEXT NOT NULL,
        confidence REAL NOT NULL,
        yaw REAL,
        embedding BLOB NOT NULL,
        is_sprite INTEGER NOT NULL DEFAULT 0,
        timestamp_sec REAL
    );
    CREATE INDEX IF NOT EXISTS idx_scene_face_emb_scene ON scene_face_embeddings(stash_scene_id);

    -- Presence of a row = this scene's sprite sheet has been successfully
    -- fetched and run through detection at least once (regardless of how
    -- many faces were found) -- the marker that lets sprite results be
    -- cached in scene_face_embeddings (is_sprite=1) the same way video-
    -- frame results already are. Separate from scene_signal_cache (which
    -- implies real ffmpeg extraction happened) because sprite processing
    -- can legitimately run before that (skip_frame_extraction's sprite-
    -- only identify) -- a scene can have this row without one there.
    CREATE TABLE IF NOT EXISTS scene_sprite_cache_status (
        stash_scene_id INTEGER PRIMARY KEY,
        checked_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS scene_tattoo_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stash_scene_id INTEGER NOT NULL,
        frame_index INTEGER NOT NULL,
        bbox_json TEXT NOT NULL,
        location_hint TEXT,
        confidence REAL NOT NULL,
        embedding BLOB NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_scene_tattoo_emb_scene ON scene_tattoo_embeddings(stash_scene_id);
"""


@dataclass
class Recommendation:
    """A recommendation for user action."""
    id: int
    type: str
    status: str  # 'pending', 'dismissed', 'resolved'
    target_type: str  # 'scene', 'performer', 'studio', 'file'
    target_id: str
    details: dict
    resolution_action: Optional[str]
    resolution_details: Optional[dict]
    resolved_at: Optional[str]
    confidence: Optional[float]
    source_analysis_id: Optional[int]
    created_at: str
    updated_at: str


@dataclass
class AnalysisRun:
    """Record of an analysis run."""
    id: int
    type: str
    status: str  # 'running', 'completed', 'failed'
    started_at: str
    completed_at: Optional[str]
    items_total: Optional[int]
    items_processed: Optional[int]
    recommendations_created: int
    cursor: Optional[str]
    error_message: Optional[str]


@dataclass
class RecommendationSettings:
    """Settings for a recommendation type."""
    type: str
    enabled: bool
    auto_dismiss_threshold: Optional[float]
    notify: bool
    interval_hours: Optional[int]
    last_run_at: Optional[str]
    next_run_at: Optional[str]
    config: Optional[dict]


class RecommendationsDB:
    """
    SQLite database for recommendations and analysis state.

    Usage:
        db = RecommendationsDB("stash_sense.db")

        # Create a recommendation
        rec_id = db.create_recommendation(
            type="duplicate_performer",
            target_type="performer",
            target_id="123",
            details={"duplicate_ids": ["123", "456"], "suggested_keeper": "123"}
        )

        # Get pending recommendations
        recs = db.get_recommendations(status="pending", type="duplicate_performer")

        # Resolve a recommendation
        db.resolve_recommendation(rec_id, action="merged", details={"kept_id": "123"})
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize database schema if needed."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                self._create_schema(conn)
            else:
                version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
                if version < SCHEMA_VERSION:
                    self._migrate_schema(conn, version)

    def _create_schema(self, conn: sqlite3.Connection):
        """Create the database schema."""
        conn.executescript(f"""
            -- Schema version tracking
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            );
            INSERT INTO schema_version (version) VALUES ({SCHEMA_VERSION});

            -- Core recommendations table
            CREATE TABLE recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details JSON NOT NULL,
                resolution_action TEXT,
                resolution_details JSON,
                resolved_at TEXT,
                confidence REAL,
                source_analysis_id INTEGER REFERENCES analysis_runs(id),
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(type, target_type, target_id)
            );
            CREATE INDEX idx_rec_status ON recommendations(status);
            CREATE INDEX idx_rec_type ON recommendations(type);
            CREATE INDEX idx_rec_target ON recommendations(target_type, target_id);

            -- Track analysis runs
            CREATE TABLE analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                items_total INTEGER,
                items_processed INTEGER,
                recommendations_created INTEGER DEFAULT 0,
                cursor TEXT,
                error_message TEXT
            );
            CREATE INDEX idx_analysis_type_status ON analysis_runs(type, status);

            -- User preferences per recommendation type
            CREATE TABLE recommendation_settings (
                type TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 1,
                auto_dismiss_threshold REAL,
                notify INTEGER DEFAULT 1,
                interval_hours INTEGER,
                last_run_at TEXT,
                next_run_at TEXT,
                config JSON
            );

            -- Dismissed targets (don't re-recommend)
            CREATE TABLE dismissed_targets (
                type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                dismissed_at TEXT DEFAULT (datetime('now')),
                reason TEXT,
                permanent INTEGER DEFAULT 0,
                PRIMARY KEY (type, target_type, target_id)
            );

            -- Track analysis watermarks for incremental runs
            CREATE TABLE analysis_watermarks (
                type TEXT PRIMARY KEY,
                last_completed_at TEXT,
                last_cursor TEXT,
                last_stash_updated_at TEXT,
                logic_version INTEGER DEFAULT 1
            );

            -- Scene fingerprints for duplicate detection
            CREATE TABLE scene_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_scene_id INTEGER NOT NULL UNIQUE,
                total_faces INTEGER NOT NULL DEFAULT 0,
                frames_analyzed INTEGER NOT NULL DEFAULT 0,
                fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                db_version TEXT,  -- Face recognition DB version used to generate this fingerprint
                used_sprite INTEGER NOT NULL DEFAULT 0,  -- whether the identify that produced the current data included sprite-tile detection
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_scene_fp_stash_id ON scene_fingerprints(stash_scene_id);
            CREATE INDEX idx_scene_fp_status ON scene_fingerprints(fingerprint_status);
            CREATE INDEX idx_scene_fp_db_version ON scene_fingerprints(db_version);

            -- Full per-person candidate match list within scene fingerprints
            -- (every entry of PersonResult.all_matches, ranked) -- lets
            -- recommendation generation and the live Identify button be
            -- served from stored data instead of re-running detect+embed+match.
            CREATE TABLE scene_fingerprint_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                person_id INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                match_rank INTEGER NOT NULL,
                is_best_match INTEGER NOT NULL DEFAULT 0,
                universal_id TEXT NOT NULL,
                stashdb_id TEXT,
                name TEXT,
                confidence REAL,
                distance REAL,
                country TEXT,
                image_url TEXT,
                endpoint TEXT,
                already_tagged INTEGER NOT NULL DEFAULT 0,
                local_performer_id TEXT,
                source TEXT,
                catalogue_url TEXT,
                profile_url TEXT,
                top_timestamps_sec TEXT,
                UNIQUE(fingerprint_id, person_id, match_rank)
            );
            CREATE INDEX idx_sfm_fingerprint ON scene_fingerprint_matches(fingerprint_id);
            CREATE INDEX idx_sfm_universal_id ON scene_fingerprint_matches(universal_id);

            -- Image fingerprints for gallery/image identification
            CREATE TABLE image_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_image_id TEXT NOT NULL UNIQUE,
                gallery_id TEXT,
                faces_detected INTEGER NOT NULL DEFAULT 0,
                db_version TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_image_fp_gallery ON image_fingerprints(gallery_id);

            -- Face entries within image fingerprints
            CREATE TABLE image_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_image_id TEXT NOT NULL REFERENCES image_fingerprints(stash_image_id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                confidence REAL,
                distance REAL,
                bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(stash_image_id, performer_id)
            );
            CREATE INDEX idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
            CREATE INDEX idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);

            -- Upstream sync snapshots
            CREATE TABLE upstream_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                local_entity_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                stash_box_id TEXT NOT NULL,
                upstream_data JSON NOT NULL,
                upstream_updated_at TEXT NOT NULL,
                fetched_at TEXT DEFAULT (datetime('now')),
                UNIQUE(entity_type, endpoint, stash_box_id)
            );
            CREATE INDEX idx_upstream_entity ON upstream_snapshots(entity_type, endpoint);
            CREATE INDEX idx_upstream_stash_box_id ON upstream_snapshots(stash_box_id);

            -- Per-field monitoring configuration
            CREATE TABLE upstream_field_config (
                endpoint TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                PRIMARY KEY (endpoint, entity_type, field_name)
            );

            -- User settings (key-value store)
            CREATE TABLE user_settings (
                key TEXT PRIMARY KEY,
                value JSON NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            -- Seed default settings
            INSERT INTO user_settings (key, value) VALUES ('normalize_enum_display', 'true');

            -- Duplicate scene candidates (work queue for scoring)
            CREATE TABLE duplicate_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_a_id INTEGER NOT NULL,
                scene_b_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                run_id INTEGER REFERENCES analysis_runs(id),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(scene_a_id, scene_b_id)
            );
            CREATE INDEX idx_dup_candidates_run ON duplicate_candidates(run_id);
            CREATE INDEX idx_dup_candidates_run_id ON duplicate_candidates(run_id, id);

            -- Job queue
            CREATE TABLE IF NOT EXISTS job_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                priority INTEGER NOT NULL,
                cursor TEXT,
                items_total INTEGER,
                items_processed INTEGER DEFAULT 0,
                error_message TEXT,
                result_summary TEXT,
                progress_label TEXT,
                resource_used TEXT,
                triggered_by TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                started_at TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
            CREATE INDEX IF NOT EXISTS idx_job_queue_type_status ON job_queue(type, status);

            -- Job schedules
            CREATE TABLE IF NOT EXISTS job_schedules (
                type TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                interval_hours REAL NOT NULL,
                priority INTEGER NOT NULL,
                last_run_at TEXT,
                next_run_at TEXT
            );
        """)
        conn.executescript(SCENE_SIGNAL_CACHE_SCHEMA)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Migrate schema from older version."""
        if from_version < 2:
            # Add scene fingerprint tables
            conn.executescript("""
                -- Scene fingerprints for duplicate detection
                CREATE TABLE IF NOT EXISTS scene_fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_scene_id INTEGER NOT NULL UNIQUE,
                    total_faces INTEGER NOT NULL DEFAULT 0,
                    frames_analyzed INTEGER NOT NULL DEFAULT 0,
                    fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                    db_version TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_scene_fp_stash_id ON scene_fingerprints(stash_scene_id);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_status ON scene_fingerprints(fingerprint_status);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_db_version ON scene_fingerprints(db_version);

                -- Face entries within scene fingerprints
                CREATE TABLE IF NOT EXISTS scene_fingerprint_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                    performer_id TEXT NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    avg_confidence REAL,
                    proportion REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(fingerprint_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_scene_fp_faces_fingerprint ON scene_fingerprint_faces(fingerprint_id);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_faces_performer ON scene_fingerprint_faces(performer_id);

                -- Update schema version
                UPDATE schema_version SET version = 3;
            """)

        if from_version == 2:
            # Add db_version column to scene_fingerprints
            conn.executescript("""
                ALTER TABLE scene_fingerprints ADD COLUMN db_version TEXT;
                CREATE INDEX IF NOT EXISTS idx_scene_fp_db_version ON scene_fingerprints(db_version);
                UPDATE schema_version SET version = 3;
            """)

        if from_version < 4:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS upstream_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    local_entity_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    stash_box_id TEXT NOT NULL,
                    upstream_data JSON NOT NULL,
                    upstream_updated_at TEXT NOT NULL,
                    fetched_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(entity_type, endpoint, stash_box_id)
                );
                CREATE INDEX IF NOT EXISTS idx_upstream_entity ON upstream_snapshots(entity_type, endpoint);
                CREATE INDEX IF NOT EXISTS idx_upstream_stash_box_id ON upstream_snapshots(stash_box_id);

                CREATE TABLE IF NOT EXISTS upstream_field_config (
                    endpoint TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    PRIMARY KEY (endpoint, entity_type, field_name)
                );

                ALTER TABLE dismissed_targets ADD COLUMN permanent INTEGER DEFAULT 0;

                UPDATE schema_version SET version = 4;
            """)

        if from_version < 5:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value JSON NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                INSERT OR IGNORE INTO user_settings (key, value) VALUES ('normalize_enum_display', 'true');

                UPDATE schema_version SET version = 5;
            """)

        if from_version < 6:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS image_fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_image_id TEXT NOT NULL UNIQUE,
                    gallery_id TEXT,
                    faces_detected INTEGER NOT NULL DEFAULT 0,
                    db_version TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_image_fp_gallery ON image_fingerprints(gallery_id);

                CREATE TABLE IF NOT EXISTS image_fingerprint_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_image_id TEXT NOT NULL REFERENCES image_fingerprints(stash_image_id) ON DELETE CASCADE,
                    performer_id TEXT NOT NULL,
                    confidence REAL,
                    distance REAL,
                    bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(stash_image_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);

                UPDATE schema_version SET version = 6;
            """)

        if from_version < 7:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS duplicate_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_a_id INTEGER NOT NULL,
                    scene_b_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    run_id INTEGER REFERENCES analysis_runs(id),
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(scene_a_id, scene_b_id)
                );
                CREATE INDEX IF NOT EXISTS idx_dup_candidates_run ON duplicate_candidates(run_id);
                CREATE INDEX IF NOT EXISTS idx_dup_candidates_run_id ON duplicate_candidates(run_id, id);

                UPDATE schema_version SET version = 7;
            """)

        if from_version < 8:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS job_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    priority INTEGER NOT NULL,
                    cursor TEXT,
                    items_total INTEGER,
                    items_processed INTEGER DEFAULT 0,
                    error_message TEXT,
                    triggered_by TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
                CREATE INDEX IF NOT EXISTS idx_job_queue_type_status ON job_queue(type, status);

                CREATE TABLE IF NOT EXISTS job_schedules (
                    type TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    interval_hours REAL NOT NULL,
                    priority INTEGER NOT NULL,
                    last_run_at TEXT,
                    next_run_at TEXT
                );

                UPDATE schema_version SET version = 8;
            """)

        if from_version < 9:
            conn.executescript("""
                ALTER TABLE analysis_watermarks ADD COLUMN logic_version INTEGER DEFAULT 1;

                UPDATE schema_version SET version = 9;
            """)

        if from_version < 10:
            conn.executescript("""
                ALTER TABLE job_queue ADD COLUMN result_summary TEXT;

                UPDATE schema_version SET version = 10;
            """)

        if from_version < 11:
            conn.executescript("""
                ALTER TABLE job_queue ADD COLUMN progress_label TEXT;

                UPDATE schema_version SET version = 11;
            """)

        if from_version < 12:
            conn.executescript(SCENE_SIGNAL_CACHE_SCHEMA + "UPDATE schema_version SET version = 12;")

        if from_version < 13:
            conn.executescript("""
                ALTER TABLE job_queue ADD COLUMN resource_used TEXT;

                UPDATE schema_version SET version = 13;
            """)

        if from_version < 14:
            # scene_face_embeddings was created (schema v12) with the old
            # dual facenet_vector/arcface_vector columns, sized for the
            # pre-buffalo_l dual-model pipeline, and was never actually
            # written to under v2 (the write path was disabled during the
            # buffalo_l migration -- see identification_router.py). Every
            # row in it predates buffalo_l and is in the wrong embedding
            # space regardless of db_version, so there's nothing worth
            # preserving -- drop and recreate with a single embedding
            # column instead of trying to convert existing rows.
            #
            # scene_signal_cache also gets wiped here even though its own
            # shape isn't changing: it's the "this scene was fully analyzed
            # and is safe to fast-path" signal consulted before ever
            # touching scene_face_embeddings (see is_scene_cache_compatible
            # callers), and every existing row was written while face-
            # embedding caching was disabled -- i.e. it would look
            # compatible while pointing at zero cached faces, which the
            # fast path can't distinguish from a scene that legitimately
            # has zero faces. Wiping both together forces exactly one real
            # pipeline run per scene going forward, after which both tables
            # populate together and stay consistent.
            conn.executescript("""
                DROP TABLE IF EXISTS scene_face_embeddings;
                DROP TABLE IF EXISTS scene_signal_cache;

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

                CREATE TABLE scene_face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_scene_id INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    bbox_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    yaw REAL,
                    embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scene_face_emb_scene ON scene_face_embeddings(stash_scene_id);

                UPDATE schema_version SET version = 14;
            """)

        if from_version < 15:
            # scene_fingerprint_faces stored one row per (fingerprint,
            # performer) -- only ever the single best match, no room for
            # alternate candidates. Replaced with scene_fingerprint_matches,
            # which stores every entry of a person's all_matches (ranked),
            # so downstream consumers (recommendation generation, the live
            # Identify button) can be served from this stored data instead
            # of re-running the whole detect+embed+match pipeline. See
            # identification_router.py's save_scene_fingerprint.
            #
            # Existing scene_fingerprint_faces rows are dropped outright
            # rather than migrated -- they only ever held a best-match
            # summary, which is exactly the subset save_scene_fingerprint
            # will repopulate the first time each scene is next identified.
            # duplicate_scenes.py's coverage (via get_fingerprints_with_faces/
            # generate_face_candidates, both rewritten to read the new table)
            # drops to 0 until then and rebuilds as scenes get re-identified
            # -- same one-time-cost shape as the v14 migration above.
            conn.executescript("""
                DROP TABLE IF EXISTS scene_fingerprint_faces;

                CREATE TABLE scene_fingerprint_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                    person_id INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    match_rank INTEGER NOT NULL,
                    is_best_match INTEGER NOT NULL DEFAULT 0,
                    universal_id TEXT NOT NULL,
                    stashdb_id TEXT,
                    name TEXT,
                    confidence REAL,
                    distance REAL,
                    country TEXT,
                    image_url TEXT,
                    endpoint TEXT,
                    already_tagged INTEGER NOT NULL DEFAULT 0,
                    local_performer_id TEXT,
                    source TEXT,
                    catalogue_url TEXT,
                    profile_url TEXT,
                    top_timestamps_sec TEXT,
                    UNIQUE(fingerprint_id, person_id, match_rank)
                );
                CREATE INDEX idx_sfm_fingerprint ON scene_fingerprint_matches(fingerprint_id);
                CREATE INDEX idx_sfm_universal_id ON scene_fingerprint_matches(universal_id);

                UPDATE schema_version SET version = 15;
            """)

        if from_version < 16:
            # Tracks whether the identify that produced a scene's *current*
            # stored data included sprite-tile detection. Needed so a bulk
            # Face Identification run (which defaults to sprites off, for
            # cost) never silently downgrades a scene Face Recommendations
            # already paid the sprite cost for -- see fingerprint_generator.py
            # (reads this to decide whether to preserve use_sprite=True on
            # a refresh) and scene_face_match.py (does an on-demand,
            # single-scene sprite top-up only for scenes still at 0).
            conn.executescript("""
                ALTER TABLE scene_fingerprints ADD COLUMN used_sprite INTEGER NOT NULL DEFAULT 0;

                UPDATE schema_version SET version = 16;
            """)

        if from_version < 17:
            # Sprite-tile embeddings used to be recomputed on every single
            # identify call (detect+embed is real, if smaller-than-video,
            # cost per call) and thrown away -- now cached the same way
            # video-frame embeddings already are, so that cost is paid once
            # per scene, ever. scene_sprite_cache_status distinguishes
            # "never checked yet" (retry live) from "checked, genuinely zero
            # sprite faces" (trust the cache) -- see its own comment above.
            conn.executescript("""
                ALTER TABLE scene_face_embeddings ADD COLUMN is_sprite INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE scene_face_embeddings ADD COLUMN timestamp_sec REAL;

                CREATE TABLE IF NOT EXISTS scene_sprite_cache_status (
                    stash_scene_id INTEGER PRIMARY KEY,
                    checked_at TEXT DEFAULT (datetime('now'))
                );

                UPDATE schema_version SET version = 17;
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ==================== Recommendations ====================

    def create_recommendation(
        self,
        type: str,
        target_type: str,
        target_id: str,
        details: dict,
        confidence: Optional[float] = None,
        source_analysis_id: Optional[int] = None,
    ) -> Optional[int]:
        """
        Create a recommendation. Returns ID if created, None if duplicate.
        """
        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO recommendations (
                        type, target_type, target_id, details, confidence, source_analysis_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (type, target_type, target_id, json.dumps(details), confidence, source_analysis_id)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Already exists
                return None

    def get_recommendation(self, rec_id: int) -> Optional[Recommendation]:
        """Get a recommendation by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM recommendations WHERE id = ?", (rec_id,)
            ).fetchone()
            if row:
                return self._row_to_recommendation(row)
        return None

    def delete_recommendation(self, rec_id: int) -> bool:
        """Delete a recommendation by ID. Returns True if a row was deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM recommendations WHERE id = ?",
                (rec_id,),
            )
            return bool(cursor.rowcount)

    def get_recommendations(
        self,
        status: Optional[str] = None,
        type: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Recommendation]:
        """Get recommendations with optional filtering."""
        query = "SELECT * FROM recommendations WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        if target_type:
            query += " AND target_type = ?"
            params.append(target_type)

        # Dismissed/resolved items are reviewed by recency, not confidence.
        # Pending items on confidence-ranked types are sorted by confidence.
        if status in ("dismissed", "resolved"):
            query += " ORDER BY COALESCE(resolved_at, updated_at) DESC, id DESC LIMIT ? OFFSET ?"
        elif type == "scene_fingerprint_match":
            query += (
                " ORDER BY COALESCE(json_extract(details, '$.high_confidence'), 0) DESC, "
                "COALESCE(confidence, 0) DESC, created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
        elif type == "duplicate_scenes":
            query += (
                " ORDER BY COALESCE("
                "confidence, "
                "CAST(json_extract(details, '$.confidence') AS REAL) / 100.0, "
                "0"
                ") DESC, created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
        else:
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_recommendation(row) for row in rows]

    def count_recommendations(self, status=None, type=None, target_type=None) -> int:
        """Count recommendations with optional filtering (for pagination totals)."""
        query = "SELECT COUNT(*) FROM recommendations WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        if target_type:
            query += " AND target_type = ?"
            params.append(target_type)
        with self._connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def get_recommendation_by_target(
        self,
        type: str,
        target_type: str,
        target_id: str,
        status: Optional[str] = None,
    ) -> Optional[Recommendation]:
        """Get a recommendation by target (uses idx_rec_target index). Returns first match or None."""
        query = "SELECT * FROM recommendations WHERE type = ? AND target_type = ? AND target_id = ?"
        params: list = [type, target_type, target_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " LIMIT 1"

        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return self._row_to_recommendation(row)
        return None

    def get_recommendation_counts(self) -> dict[str, dict[str, int]]:
        """Get counts by type and status."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT type, status, COUNT(*) as count FROM recommendations GROUP BY type, status"
            ).fetchall()

            counts = {}
            for row in rows:
                if row['type'] not in counts:
                    counts[row['type']] = {}
                counts[row['type']][row['status']] = row['count']
            return counts

    def resolve_recommendation(
        self,
        rec_id: int,
        action: str,
        details: Optional[dict] = None,
    ) -> bool:
        """Mark a recommendation as resolved. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET status = 'resolved',
                    resolution_action = ?,
                    resolution_details = ?,
                    resolved_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (action, json.dumps(details) if details else None, rec_id)
            )
            return cursor.rowcount > 0

    def dismiss_recommendation(self, rec_id: int, reason: Optional[str] = None, permanent: bool = False) -> bool:
        """Dismiss a recommendation and add to dismissed_targets."""
        with self._connection() as conn:
            # Get the recommendation first
            row = conn.execute(
                "SELECT type, target_type, target_id FROM recommendations WHERE id = ?",
                (rec_id,)
            ).fetchone()

            if not row:
                return False

            # Mark as dismissed
            conn.execute(
                """
                UPDATE recommendations
                SET status = 'dismissed', updated_at = datetime('now')
                WHERE id = ?
                """,
                (rec_id,)
            )

            # Add to dismissed_targets to prevent re-recommendation
            try:
                conn.execute(
                    """
                    INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row['type'], row['target_type'], row['target_id'], reason, int(permanent))
                )
            except sqlite3.IntegrityError:
                pass  # Already dismissed

            return True

    def add_recommendation_target_dismissal(
        self,
        rec_id: int,
        reason: Optional[str] = None,
        permanent: bool = False,
    ) -> bool:
        """Add a recommendation's target to dismissed_targets without changing status."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT type, target_type, target_id FROM recommendations WHERE id = ?",
                (rec_id,),
            ).fetchone()

            if not row:
                return False

            try:
                conn.execute(
                    """
                    INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row['type'], row['target_type'], row['target_id'], reason, int(permanent))
                )
            except sqlite3.IntegrityError:
                pass

            return True

    def batch_dismiss_by_type(self, rec_type: str, permanent: bool = False, reason: Optional[str] = None) -> int:
        """Dismiss all pending recommendations of a given type. Returns count dismissed."""
        with self._connection() as conn:
            # Get all pending recs of this type
            rows = conn.execute(
                "SELECT id, type, target_type, target_id FROM recommendations WHERE type = ? AND status = 'pending'",
                (rec_type,)
            ).fetchall()

            if not rows:
                return 0

            rec_ids = [row['id'] for row in rows]

            # Mark all as dismissed
            conn.execute(
                f"UPDATE recommendations SET status = 'dismissed', updated_at = datetime('now') WHERE id IN ({','.join('?' * len(rec_ids))})",
                rec_ids
            )

            # Add to dismissed_targets
            for row in rows:
                try:
                    conn.execute(
                        "INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent) VALUES (?, ?, ?, ?, ?)",
                        (row['type'], row['target_type'], row['target_id'], reason, int(permanent))
                    )
                except sqlite3.IntegrityError:
                    pass  # Already dismissed

            return len(rec_ids)

    def delete_pending_recommendations_by_type(self, rec_type: str) -> int:
        """Delete all pending recommendations for a type. Returns deleted row count."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM recommendations WHERE type = ? AND status = 'pending'",
                (rec_type,),
            )
            return cursor.rowcount or 0

    def dismiss_pending_scene_fingerprint_for_scene(
        self,
        scene_id: str,
        reason: Optional[str] = None,
        exclude_rec_id: Optional[int] = None,
    ) -> int:
        """
        Dismiss pending scene_fingerprint_match recommendations for one local scene.

        Matches by target_id prefix: "<scene_id>|...".
        Returns number of recommendations dismissed.
        """
        scene_prefix = f"{str(scene_id)}|%"

        with self._connection() as conn:
            query = (
                "SELECT id, type, target_type, target_id "
                "FROM recommendations "
                "WHERE type = 'scene_fingerprint_match' "
                "AND target_type = 'scene' "
                "AND status = 'pending' "
                "AND target_id LIKE ?"
            )
            params: list[Any] = [scene_prefix]
            if exclude_rec_id is not None:
                query += " AND id != ?"
                params.append(exclude_rec_id)

            rows = conn.execute(query, params).fetchall()
            if not rows:
                return 0

            rec_ids = [row["id"] for row in rows]
            conn.execute(
                f"UPDATE recommendations SET status = 'dismissed', updated_at = datetime('now') WHERE id IN ({','.join('?' * len(rec_ids))})",
                rec_ids,
            )

            # Keep dismissed_targets in sync with normal dismiss behavior.
            for row in rows:
                try:
                    conn.execute(
                        "INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent) VALUES (?, ?, ?, ?, ?)",
                        (row["type"], row["target_type"], row["target_id"], reason, 0),
                    )
                except sqlite3.IntegrityError:
                    pass

            return len(rec_ids)

    def dismiss_pending_scene_face_match_for_scene(
        self,
        scene_id: str,
        reason: Optional[str] = None,
        exclude_rec_ids: Optional[list[int]] = None,
    ) -> int:
        """
        Dismiss pending scene_face_match recommendations for one local scene.

        Matches by target_id prefix: "<scene_id>|...". Like
        dismiss_pending_scene_fingerprint_for_scene, but takes a list of ids
        to exclude (not just one) -- needed for "accept selected candidates,
        dismiss the rest of this scene's pending candidates" in one call.
        Returns number of recommendations dismissed.
        """
        scene_prefix = f"{str(scene_id)}|%"
        exclude_ids = [int(i) for i in (exclude_rec_ids or [])]

        with self._connection() as conn:
            query = (
                "SELECT id, type, target_type, target_id "
                "FROM recommendations "
                "WHERE type = 'scene_face_match' "
                "AND target_type = 'scene' "
                "AND status = 'pending' "
                "AND target_id LIKE ?"
            )
            params: list[Any] = [scene_prefix]
            if exclude_ids:
                query += f" AND id NOT IN ({','.join('?' * len(exclude_ids))})"
                params.extend(exclude_ids)

            rows = conn.execute(query, params).fetchall()
            if not rows:
                return 0

            rec_ids = [row["id"] for row in rows]
            conn.execute(
                f"UPDATE recommendations SET status = 'dismissed', updated_at = datetime('now') WHERE id IN ({','.join('?' * len(rec_ids))})",
                rec_ids,
            )

            for row in rows:
                try:
                    conn.execute(
                        "INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent) VALUES (?, ?, ?, ?, ?)",
                        (row["type"], row["target_type"], row["target_id"], reason, 0),
                    )
                except sqlite3.IntegrityError:
                    pass

            return len(rec_ids)

    def delete_pending_scene_fingerprint_for_scene(
        self,
        scene_id: str,
        exclude_rec_id: Optional[int] = None,
    ) -> int:
        """
        Delete pending scene_fingerprint_match recommendations for one local scene.

        Matches by target_id prefix: "<scene_id>|...".
        Returns number of deleted recommendations.
        """
        scene_prefix = f"{str(scene_id)}|%"

        with self._connection() as conn:
            query = (
                "DELETE FROM recommendations "
                "WHERE type = 'scene_fingerprint_match' "
                "AND target_type = 'scene' "
                "AND status = 'pending' "
                "AND target_id LIKE ?"
            )
            params: list[Any] = [scene_prefix]
            if exclude_rec_id is not None:
                query += " AND id != ?"
                params.append(exclude_rec_id)

            cursor = conn.execute(query, params)
            return cursor.rowcount or 0

    def delete_pending_duplicate_scene_recommendations_for_scene(
        self,
        scene_id: str,
    ) -> int:
        """
        Delete pending duplicate-scene recommendations involving a local scene.

        Covers:
        - duplicate_scene_files where target_id == scene_id
        - duplicate_scenes where target_id encodes "<scene_a_id>:<scene_b_id>"
          and legacy/details-based rows via details.scene_a_id/scene_b_id
        """
        scene_id_str = str(scene_id).strip()
        if not scene_id_str:
            return 0

        with self._connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM recommendations
                WHERE status = 'pending'
                  AND (
                    (
                      type = 'duplicate_scene_files'
                      AND target_type = 'scene'
                      AND target_id = ?
                    )
                    OR
                    (
                      type = 'duplicate_scenes'
                      AND target_type = 'scene'
                      AND (
                        target_id LIKE ?
                        OR target_id LIKE ?
                        OR CAST(json_extract(details, '$.scene_a_id') AS TEXT) = ?
                        OR CAST(json_extract(details, '$.scene_b_id') AS TEXT) = ?
                      )
                    )
                  )
                """,
                (
                    scene_id_str,
                    f"{scene_id_str}:%",
                    f"%:{scene_id_str}",
                    scene_id_str,
                    scene_id_str,
                ),
            )
            return cursor.rowcount or 0

    def is_dismissed(self, type: str, target_type: str, target_id: str) -> bool:
        """Check if a target has been dismissed."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ?
                """,
                (type, target_type, target_id)
            ).fetchone()
            return row is not None

    def is_permanently_dismissed(self, type: str, target_type: str, target_id: str) -> bool:
        """Check if a target has been permanently dismissed."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ? AND permanent = 1
                """,
                (type, target_type, target_id)
            ).fetchone()
            return row is not None

    def undismiss(self, type: str, target_type: str, target_id: str):
        """Remove soft dismissals for a target (does not remove permanent dismissals)."""
        with self._connection() as conn:
            conn.execute(
                """
                DELETE FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ? AND permanent = 0
                """,
                (type, target_type, target_id)
            )

    def update_recommendation_details(self, rec_id: int, details: dict) -> bool:
        """Update details on a pending recommendation. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET details = ?, updated_at = datetime('now')
                WHERE id = ? AND status = 'pending'
                """,
                (json.dumps(details), rec_id)
            )
            return cursor.rowcount > 0

    def reopen_recommendation(self, rec_id: int, details: dict) -> bool:
        """Reopen a dismissed recommendation with new details. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET status = 'pending', details = ?,
                    resolution_action = NULL, resolution_details = NULL,
                    resolved_at = NULL, updated_at = datetime('now')
                WHERE id = ? AND status = 'dismissed'
                """,
                (json.dumps(details), rec_id)
            )
            return cursor.rowcount > 0

    def _row_to_recommendation(self, row: sqlite3.Row) -> Recommendation:
        """Convert a database row to a Recommendation object."""
        return Recommendation(
            id=row['id'],
            type=row['type'],
            status=row['status'],
            target_type=row['target_type'],
            target_id=row['target_id'],
            details=json.loads(row['details']),
            resolution_action=row['resolution_action'],
            resolution_details=json.loads(row['resolution_details']) if row['resolution_details'] else None,
            resolved_at=row['resolved_at'],
            confidence=row['confidence'],
            source_analysis_id=row['source_analysis_id'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    # ==================== Analysis Runs ====================

    def start_analysis_run(self, type: str, items_total: Optional[int] = None) -> int:
        """Start a new analysis run. Returns run ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_runs (type, status, started_at, items_total)
                VALUES (?, 'running', datetime('now'), ?)
                """,
                (type, items_total)
            )
            return cursor.lastrowid

    def update_analysis_progress(
        self,
        run_id: int,
        items_processed: int,
        recommendations_created: int,
        cursor: Optional[str] = None,
    ):
        """Update analysis run progress."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET items_processed = ?, recommendations_created = ?, cursor = ?
                WHERE id = ?
                """,
                (items_processed, recommendations_created, cursor, run_id)
            )

    def fail_stale_analysis_runs(self) -> int:
        """Mark any 'running' analysis runs as failed (e.g. after sidecar restart). Returns count."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'failed', completed_at = datetime('now'),
                    error_message = 'Sidecar restarted while analysis was running'
                WHERE status = 'running'
                """
            )
            return cursor.rowcount

    def update_analysis_items_total(self, run_id: int, items_total: int):
        """Update the total items count for an analysis run."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE analysis_runs SET items_total = ? WHERE id = ?",
                (items_total, run_id)
            )

    def complete_analysis_run(self, run_id: int, recommendations_created: int):
        """Mark an analysis run as completed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'completed', completed_at = datetime('now'), recommendations_created = ?
                WHERE id = ?
                """,
                (recommendations_created, run_id)
            )

    def fail_analysis_run(self, run_id: int, error_message: str):
        """Mark an analysis run as failed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'failed', completed_at = datetime('now'), error_message = ?
                WHERE id = ?
                """,
                (error_message, run_id)
            )

    def get_analysis_run(self, run_id: int) -> Optional[AnalysisRun]:
        """Get an analysis run by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                return AnalysisRun(**dict(row))
        return None

    def get_recent_analysis_runs(self, type: Optional[str] = None, limit: int = 20) -> list[AnalysisRun]:
        """Get recent analysis runs."""
        query = "SELECT * FROM analysis_runs"
        params = []

        if type:
            query += " WHERE type = ?"
            params.append(type)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [AnalysisRun(**dict(row)) for row in rows]

    # ==================== Settings ====================

    def get_settings(self, type: str) -> Optional[RecommendationSettings]:
        """Get settings for a recommendation type."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM recommendation_settings WHERE type = ?", (type,)
            ).fetchone()
            if row:
                return RecommendationSettings(
                    type=row['type'],
                    enabled=bool(row['enabled']),
                    auto_dismiss_threshold=row['auto_dismiss_threshold'],
                    notify=bool(row['notify']),
                    interval_hours=row['interval_hours'],
                    last_run_at=row['last_run_at'],
                    next_run_at=row['next_run_at'],
                    config=json.loads(row['config']) if row['config'] else None,
                )
        return None

    def get_all_settings(self) -> list[RecommendationSettings]:
        """Get all recommendation settings."""
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM recommendation_settings").fetchall()
            return [
                RecommendationSettings(
                    type=row['type'],
                    enabled=bool(row['enabled']),
                    auto_dismiss_threshold=row['auto_dismiss_threshold'],
                    notify=bool(row['notify']),
                    interval_hours=row['interval_hours'],
                    last_run_at=row['last_run_at'],
                    next_run_at=row['next_run_at'],
                    config=json.loads(row['config']) if row['config'] else None,
                )
                for row in rows
            ]

    def upsert_settings(
        self,
        type: str,
        enabled: Optional[bool] = None,
        auto_dismiss_threshold: Optional[float] = None,
        notify: Optional[bool] = None,
        interval_hours: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        """Create or update settings for a recommendation type."""
        with self._connection() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT 1 FROM recommendation_settings WHERE type = ?", (type,)
            ).fetchone()

            if existing:
                updates = []
                params = []
                if enabled is not None:
                    updates.append("enabled = ?")
                    params.append(int(enabled))
                if auto_dismiss_threshold is not None:
                    updates.append("auto_dismiss_threshold = ?")
                    params.append(auto_dismiss_threshold)
                if notify is not None:
                    updates.append("notify = ?")
                    params.append(int(notify))
                if interval_hours is not None:
                    updates.append("interval_hours = ?")
                    params.append(interval_hours)
                if config is not None:
                    updates.append("config = ?")
                    params.append(json.dumps(config))

                if updates:
                    params.append(type)
                    conn.execute(
                        f"UPDATE recommendation_settings SET {', '.join(updates)} WHERE type = ?",
                        params
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO recommendation_settings (type, enabled, auto_dismiss_threshold, notify, interval_hours, config)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (type, int(enabled) if enabled is not None else 1,
                     auto_dismiss_threshold, int(notify) if notify is not None else 1,
                     interval_hours, json.dumps(config) if config else None)
                )

    # ==================== Watermarks ====================

    def get_watermark(self, type: str) -> Optional[dict]:
        """Get analysis watermark for incremental runs."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_watermarks WHERE type = ?", (type,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def set_watermark(
        self,
        type: str,
        last_cursor: Optional[str] = None,
        last_stash_updated_at: Optional[str] = None,
        logic_version: Optional[int] = None,
    ):
        """Update analysis watermark."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_watermarks (type, last_completed_at, last_cursor, last_stash_updated_at, logic_version)
                VALUES (?, datetime('now'), ?, ?, COALESCE(?, 1))
                ON CONFLICT(type) DO UPDATE SET
                    last_completed_at = datetime('now'),
                    last_cursor = COALESCE(?, last_cursor),
                    last_stash_updated_at = COALESCE(?, last_stash_updated_at),
                    logic_version = COALESCE(?, logic_version)
                """,
                (type, last_cursor, last_stash_updated_at, logic_version,
                 last_cursor, last_stash_updated_at, logic_version)
            )

    def delete_watermark(self, type: str):
        """Delete a watermark entry."""
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM analysis_watermarks WHERE type = ?", (type,)
            )

    # ==================== Upstream Snapshots ====================

    def upsert_upstream_snapshot(
        self,
        entity_type: str,
        local_entity_id: str,
        endpoint: str,
        stash_box_id: str,
        upstream_data: dict,
        upstream_updated_at: str,
    ) -> int:
        """
        Create or update an upstream snapshot. Returns the snapshot ID.
        Uses upsert on the unique constraint (entity_type, endpoint, stash_box_id).
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO upstream_snapshots (
                    entity_type, local_entity_id, endpoint, stash_box_id,
                    upstream_data, upstream_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, endpoint, stash_box_id) DO UPDATE SET
                    local_entity_id = excluded.local_entity_id,
                    upstream_data = excluded.upstream_data,
                    upstream_updated_at = excluded.upstream_updated_at,
                    fetched_at = datetime('now')
                RETURNING id
                """,
                (entity_type, local_entity_id, endpoint, stash_box_id,
                 json.dumps(upstream_data), upstream_updated_at)
            )
            return cursor.fetchone()[0]

    def get_upstream_snapshot(
        self,
        entity_type: str,
        endpoint: str,
        stash_box_id: str,
    ) -> Optional[dict]:
        """Get an upstream snapshot by its unique key. Returns dict with parsed upstream_data, or None."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM upstream_snapshots
                WHERE entity_type = ? AND endpoint = ? AND stash_box_id = ?
                """,
                (entity_type, endpoint, stash_box_id)
            ).fetchone()
            if row:
                result = dict(row)
                result["upstream_data"] = json.loads(result["upstream_data"])
                return result
        return None

    def delete_snapshots_for_endpoint(self, entity_type: str, endpoint: str) -> int:
        """Delete all upstream snapshots for an entity type + endpoint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM upstream_snapshots WHERE entity_type = ? AND endpoint = ?",
                (entity_type, endpoint)
            )
            return cursor.rowcount

    # ==================== Upstream Field Config ====================

    def get_enabled_fields(self, endpoint: str, entity_type: str) -> Optional[set[str]]:
        """
        Get the set of enabled field names for an endpoint/entity_type.
        Returns None if no config exists (caller should use defaults).
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT field_name, enabled FROM upstream_field_config
                WHERE endpoint = ? AND entity_type = ?
                """,
                (endpoint, entity_type)
            ).fetchall()
            if not rows:
                return None
            return {row["field_name"] for row in rows if row["enabled"]}

    def set_field_config(self, endpoint: str, entity_type: str, field_configs: dict[str, bool]):
        """
        Set field monitoring configuration for an endpoint/entity_type.
        Replaces all existing config for this endpoint/entity_type.
        field_configs maps field_name -> enabled bool.
        """
        with self._connection() as conn:
            # Delete existing config for this endpoint/entity_type
            conn.execute(
                """
                DELETE FROM upstream_field_config
                WHERE endpoint = ? AND entity_type = ?
                """,
                (endpoint, entity_type)
            )
            # Insert new config rows
            for field_name, enabled in field_configs.items():
                conn.execute(
                    """
                    INSERT INTO upstream_field_config (endpoint, entity_type, field_name, enabled)
                    VALUES (?, ?, ?, ?)
                    """,
                    (endpoint, entity_type, field_name, int(enabled))
                )

    # ==================== User Settings ====================

    def get_user_setting(self, key: str) -> Optional[Any]:
        """Get a user setting by key. Returns the parsed JSON value, or None if not found."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM user_settings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                val = row["value"]
                # SQLite NUMERIC affinity may auto-convert JSON numbers from
                # str to int/float. Return those directly; parse strings as JSON.
                if not isinstance(val, str):
                    return val
                return json.loads(val)
        return None

    def set_user_setting(self, key: str, value: Any):
        """Set a user setting. Creates or updates."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = datetime('now')
                """,
                (key, json.dumps(value))
            )

    def get_all_user_settings(self) -> dict[str, Any]:
        """Get all user settings as a dict."""
        with self._connection() as conn:
            rows = conn.execute("SELECT key, value FROM user_settings").fetchall()
            result = {}
            for row in rows:
                val = row["value"]
                result[row["key"]] = val if not isinstance(val, str) else json.loads(val)
            return result

    def delete_user_setting(self, key: str):
        """Delete a user setting by key."""
        with self._connection() as conn:
            conn.execute("DELETE FROM user_settings WHERE key = ?", (key,))

    # ==================== Endpoint Priorities ====================

    def get_endpoint_priorities(self) -> list[str]:
        """Get the ordered list of endpoint URLs, highest priority first.

        Returns empty list if no priorities are configured.
        """
        result = self.get_user_setting("endpoint_priorities")
        if isinstance(result, list):
            return result
        return []

    def set_endpoint_priorities(self, endpoints: list[str]):
        """Set the endpoint priority order. Index 0 = highest priority."""
        self.set_user_setting("endpoint_priorities", endpoints)

    # ==================== Disabled Endpoints ====================

    def get_disabled_endpoints(self) -> list[str]:
        """Get the list of disabled endpoint URLs."""
        result = self.get_user_setting("disabled_endpoints")
        if isinstance(result, list):
            return result
        return []

    def set_disabled_endpoints(self, endpoints: list[str]):
        """Set the list of disabled endpoint URLs."""
        self.set_user_setting("disabled_endpoints", endpoints)

    def is_endpoint_enabled(self, endpoint: str) -> bool:
        """Check if an endpoint is enabled (not in disabled list)."""
        return endpoint not in self.get_disabled_endpoints()

    # ==================== Scene Fingerprints ====================

    def create_scene_fingerprint(
        self,
        stash_scene_id: int,
        total_faces: int,
        frames_analyzed: int,
        fingerprint_status: str = "pending",
        db_version: Optional[str] = None,
        used_sprite: bool = False,
    ) -> int:
        """
        Create or update a scene fingerprint. Returns the fingerprint ID.
        Uses upsert - if fingerprint exists for scene, updates it.

        Args:
            stash_scene_id: The Stash scene ID
            total_faces: Total faces detected in the scene
            frames_analyzed: Number of frames analyzed
            fingerprint_status: Status ('pending', 'complete', 'error')
            db_version: Face recognition DB version used for this fingerprint
            used_sprite: Whether the identify producing this data included
                sprite-tile detection. Callers doing a pre-emptive "error"
                status write (see fingerprint_generator.py's _mark_scene_started)
                should leave this at the default -- it only matters once a
                fingerprint_status='complete' save reflects what actually ran.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scene_fingerprints (stash_scene_id, total_faces, frames_analyzed, fingerprint_status, db_version, used_sprite)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(stash_scene_id) DO UPDATE SET
                    total_faces = excluded.total_faces,
                    frames_analyzed = excluded.frames_analyzed,
                    fingerprint_status = excluded.fingerprint_status,
                    db_version = excluded.db_version,
                    used_sprite = excluded.used_sprite,
                    updated_at = datetime('now')
                RETURNING id
                """,
                (stash_scene_id, total_faces, frames_analyzed, fingerprint_status, db_version, int(used_sprite))
            )
            return cursor.fetchone()[0]

    def get_scene_fingerprint(self, stash_scene_id: int) -> Optional[dict]:
        """Get a scene fingerprint by stash scene ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scene_fingerprints WHERE stash_scene_id = ?",
                (stash_scene_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_all_scene_fingerprints(self, status: Optional[str] = None) -> list[dict]:
        """Get all scene fingerprints, optionally filtered by status."""
        with self._connection() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM scene_fingerprints WHERE fingerprint_status = ?",
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM scene_fingerprints").fetchall()
            return [dict(row) for row in rows]

    def replace_fingerprint_matches(self, fingerprint_id: int, matches: list[dict]) -> None:
        """Clear and bulk-insert the full candidate match list for a scene
        fingerprint -- every entry of every detected person's all_matches,
        not just the best one. Replaces the old add_fingerprint_face (which
        stored one best-match-only row per performer).

        Each dict in `matches`: person_id, frame_count, match_rank,
        is_best_match (bool), universal_id, stashdb_id, name, confidence,
        distance, country, image_url, endpoint, already_tagged (bool),
        local_performer_id, source, catalogue_url, profile_url,
        top_timestamps_sec (list[float], stored as JSON).
        """
        with self._connection() as conn:
            conn.execute("DELETE FROM scene_fingerprint_matches WHERE fingerprint_id = ?", (fingerprint_id,))
            conn.executemany(
                """
                INSERT INTO scene_fingerprint_matches (
                    fingerprint_id, person_id, frame_count, match_rank, is_best_match,
                    universal_id, stashdb_id, name, confidence, distance, country,
                    image_url, endpoint, already_tagged, local_performer_id,
                    source, catalogue_url, profile_url, top_timestamps_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        fingerprint_id, m["person_id"], m["frame_count"], m["match_rank"],
                        int(bool(m["is_best_match"])), m["universal_id"], m.get("stashdb_id"),
                        m.get("name"), m.get("confidence"), m.get("distance"), m.get("country"),
                        m.get("image_url"), m.get("endpoint"), int(bool(m.get("already_tagged"))),
                        m.get("local_performer_id"), m.get("source"), m.get("catalogue_url"),
                        m.get("profile_url"), json.dumps(m.get("top_timestamps_sec") or []),
                    )
                    for m in matches
                ],
            )

    def get_fingerprint_matches(self, fingerprint_id: int) -> list[dict]:
        """Get all stored candidate matches for a scene fingerprint, ordered
        by person then rank -- ready to be grouped back into per-person
        match lists by the caller."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scene_fingerprint_matches WHERE fingerprint_id = ? ORDER BY person_id, match_rank",
                (fingerprint_id,)
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["top_timestamps_sec"] = json.loads(d["top_timestamps_sec"]) if d.get("top_timestamps_sec") else []
                d["is_best_match"] = bool(d["is_best_match"])
                d["already_tagged"] = bool(d["already_tagged"])
                results.append(d)
            return results

    def get_fingerprinted_scene_ids(self) -> set[int]:
        """Scene IDs with a complete fingerprint.

        Used against Stash's own current scene ID list to find orphans --
        rows left behind for scenes since deleted from Stash -- which
        inflate this count if never cleaned up.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT stash_scene_id FROM scene_fingerprints WHERE fingerprint_status = 'complete'"
            ).fetchall()
            return {row[0] for row in rows}

    def get_all_fingerprint_scene_ids(self) -> set[int]:
        """Scene IDs with any scene_fingerprints row, regardless of status.

        Orphan detection needs this (not just the 'complete' subset above)
        -- a scene deleted from Stash mid-error-retry leaves an orphaned
        'error' row just as easily as a 'complete' one.
        """
        with self._connection() as conn:
            rows = conn.execute("SELECT stash_scene_id FROM scene_fingerprints").fetchall()
            return {row[0] for row in rows}

    def delete_fingerprints_for_scenes(self, scene_ids: list[int]) -> int:
        """Delete all fingerprint and signal-cache data for scenes that no
        longer exist in Stash. Returns the number of scene_fingerprints
        rows deleted.
        """
        if not scene_ids:
            return 0
        deleted = 0
        with self._connection() as conn:
            # Chunk to stay under SQLite's default ~999-variable-per-statement limit.
            for i in range(0, len(scene_ids), 500):
                chunk = scene_ids[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                fp_ids = [
                    row[0] for row in conn.execute(
                        f"SELECT id FROM scene_fingerprints WHERE stash_scene_id IN ({placeholders})", chunk
                    ).fetchall()
                ]
                if fp_ids:
                    fp_placeholders = ",".join("?" * len(fp_ids))
                    conn.execute(
                        f"DELETE FROM scene_fingerprint_matches WHERE fingerprint_id IN ({fp_placeholders})", fp_ids
                    )
                cursor = conn.execute(
                    f"DELETE FROM scene_fingerprints WHERE stash_scene_id IN ({placeholders})", chunk
                )
                deleted += cursor.rowcount
                conn.execute(f"DELETE FROM scene_signal_cache WHERE stash_scene_id IN ({placeholders})", chunk)
                conn.execute(f"DELETE FROM scene_face_embeddings WHERE stash_scene_id IN ({placeholders})", chunk)
                conn.execute(f"DELETE FROM scene_sprite_cache_status WHERE stash_scene_id IN ({placeholders})", chunk)
                conn.execute(f"DELETE FROM scene_tattoo_embeddings WHERE stash_scene_id IN ({placeholders})", chunk)
        return deleted

    # ==================== Scene signal cache (face) ====================
    # body_*/tattoos_detected columns and the scene_tattoo_embeddings table
    # (schema above) are no longer written -- body/tattoo identification
    # signals were removed. Left in the schema unused/NULL rather than
    # migrated away, since SQLite migrations on already-deployed databases
    # aren't worth the risk for dead columns.
    #
    # Caches the DB-independent, expensive part of scene analysis (frame
    # extraction + detection + embedding) separately from the match
    # results in scene_fingerprint_matches, so a performer-database version
    # bump can re-run just the cheap matching/re-ranking step instead of
    # the whole pipeline. See identification_router.py's /identify/scene.

    def get_scene_signal_cache(self, stash_scene_id: int) -> Optional[dict]:
        """Cache-meta row for a scene, or None if never cached."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scene_signal_cache WHERE stash_scene_id = ?",
                (stash_scene_id,),
            ).fetchone()
            return dict(row) if row else None

    def save_scene_signal_cache(
        self, stash_scene_id: int, *, num_frames: int, min_face_size: int,
        min_face_confidence: float, start_offset_pct: float, end_offset_pct: float,
        frames_analyzed: int,
    ) -> None:
        """Upsert the cache-meta row (detection params). body_*/tattoos_detected
        columns are no longer written (body/tattoo signals were removed) but
        stay in the schema as unused/NULL-or-default for existing rows."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO scene_signal_cache (
                    stash_scene_id, num_frames, min_face_size, min_face_confidence,
                    start_offset_pct, end_offset_pct, frames_analyzed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stash_scene_id) DO UPDATE SET
                    num_frames=excluded.num_frames, min_face_size=excluded.min_face_size,
                    min_face_confidence=excluded.min_face_confidence,
                    start_offset_pct=excluded.start_offset_pct, end_offset_pct=excluded.end_offset_pct,
                    frames_analyzed=excluded.frames_analyzed,
                    created_at=datetime('now')
                """,
                (
                    stash_scene_id, num_frames, min_face_size, min_face_confidence,
                    start_offset_pct, end_offset_pct, frames_analyzed,
                ),
            )

    def get_face_embeddings(self, stash_scene_id: int, is_sprite: Optional[bool] = None) -> list[dict]:
        """is_sprite=None returns both video-frame and sprite-tile rows;
        True/False filters to just one source."""
        query = "SELECT * FROM scene_face_embeddings WHERE stash_scene_id = ?"
        params: list = [stash_scene_id]
        if is_sprite is not None:
            query += " AND is_sprite = ?"
            params.append(1 if is_sprite else 0)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def replace_face_embeddings(self, stash_scene_id: int, faces: list[dict], is_sprite: bool = False) -> None:
        """Clear and bulk-insert cached face detections for a scene, scoped
        to one source at a time -- video-frame and sprite-tile results are
        computed/cached independently (see identification_router.py), so a
        write for one source must never delete the other's rows.

        Each dict: frame_index, bbox (dict), confidence, yaw, embedding
        (bytes, buffalo_l's single embedding vector), timestamp_sec
        (sprite tiles only -- the same "which frame did this come from"
        role frame_index plays for video, since sprite tiles don't have a
        meaningful ordinal index of their own).
        """
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM scene_face_embeddings WHERE stash_scene_id = ? AND is_sprite = ?",
                (stash_scene_id, 1 if is_sprite else 0),
            )
            conn.executemany(
                """
                INSERT INTO scene_face_embeddings (
                    stash_scene_id, frame_index, bbox_json, confidence, yaw, embedding, is_sprite, timestamp_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (stash_scene_id, f["frame_index"], json.dumps(f["bbox"]), f["confidence"],
                     f.get("yaw"), f["embedding"], 1 if is_sprite else 0, f.get("timestamp_sec"))
                    for f in faces
                ],
            )

    def is_sprite_cache_checked(self, stash_scene_id: int) -> bool:
        """True once this scene's sprite sheet has been successfully
        fetched and run through detection at least once (even if zero
        faces were found) -- see scene_sprite_cache_status's own comment."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM scene_sprite_cache_status WHERE stash_scene_id = ?",
                (stash_scene_id,),
            ).fetchone()
            return row is not None

    def mark_sprite_cache_checked(self, stash_scene_id: int) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO scene_sprite_cache_status (stash_scene_id, checked_at)
                VALUES (?, datetime('now'))
                ON CONFLICT(stash_scene_id) DO UPDATE SET checked_at = excluded.checked_at
                """,
                (stash_scene_id,),
            )

    def get_fingerprints_needing_refresh(self, current_db_version: str) -> list[dict]:
        """Get fingerprints that were generated with an older DB version."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scene_fingerprints
                WHERE db_version IS NULL OR db_version != ?
                ORDER BY stash_scene_id
                """,
                (current_db_version,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_scene_ids_without_fingerprints(self, scene_ids: list[int]) -> list[int]:
        """Given a list of scene IDs, return those without fingerprints."""
        if not scene_ids:
            return []
        with self._connection() as conn:
            placeholders = ",".join("?" * len(scene_ids))
            rows = conn.execute(
                f"""
                SELECT stash_scene_id FROM scene_fingerprints
                WHERE stash_scene_id IN ({placeholders})
                """,
                scene_ids
            ).fetchall()
            existing = {row[0] for row in rows}
            return [sid for sid in scene_ids if sid not in existing]

    def get_fingerprint_stats(self, current_db_version: Optional[str] = None) -> dict:
        """Get fingerprint coverage statistics."""
        with self._connection() as conn:
            stats = {}
            stats['total_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints"
            ).fetchone()[0]
            stats['complete_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'complete'"
            ).fetchone()[0]
            stats['pending_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'pending'"
            ).fetchone()[0]
            stats['error_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'error'"
            ).fetchone()[0]

            if current_db_version:
                stats['current_version_count'] = conn.execute(
                    "SELECT COUNT(*) FROM scene_fingerprints WHERE db_version = ?",
                    (current_db_version,)
                ).fetchone()[0]
                stats['needs_refresh_count'] = conn.execute(
                    "SELECT COUNT(*) FROM scene_fingerprints WHERE db_version IS NULL OR db_version != ?",
                    (current_db_version,)
                ).fetchone()[0]
                # Distinct from needs_refresh_count, which also counts rows
                # that were never completed at all (status='error' with a
                # stale/NULL db_version) -- this is specifically "already-
                # identified scenes whose db_version is stale," the set a
                # "refresh outdated" action should touch, as opposed to
                # "missing" (total_scenes - complete_fingerprints, computed
                # by the caller) which a "fingerprint missing" action should
                # touch. See recommendations_router.py's fingerprint status/
                # generate endpoints.
                stats['outdated_count'] = conn.execute(
                    "SELECT COUNT(*) FROM scene_fingerprints "
                    "WHERE fingerprint_status = 'complete' AND (db_version IS NULL OR db_version != ?)",
                    (current_db_version,)
                ).fetchone()[0]

            return stats

    def mark_fingerprints_for_refresh(self, scene_ids: Optional[list[int]] = None) -> int:
        """
        Mark fingerprints for refresh by clearing their db_version.
        If scene_ids is None, marks all fingerprints.
        Returns count of fingerprints marked.
        """
        with self._connection() as conn:
            if scene_ids is None:
                cursor = conn.execute(
                    "UPDATE scene_fingerprints SET db_version = NULL, updated_at = datetime('now')"
                )
            else:
                placeholders = ",".join("?" * len(scene_ids))
                cursor = conn.execute(
                    f"""
                    UPDATE scene_fingerprints
                    SET db_version = NULL, updated_at = datetime('now')
                    WHERE stash_scene_id IN ({placeholders})
                    """,
                    scene_ids
                )
            return cursor.rowcount

    def reset_scene_fingerprints_with_backup(self) -> dict:
        """Back up scene_fingerprints + scene_fingerprint_matches to
        timestamped tables, then mark every fingerprint for refresh (same
        effect as mark_fingerprints_for_refresh(None), so the next
        fingerprint generation run fully reprocesses every scene).

        For a detection-affecting settings change (e.g. detection_size) --
        existing fingerprints were computed under the old setting and
        won't reflect the new one until reprocessed; this gives a safety
        copy of the pre-change data before kicking that off. The backup
        tables are plain SQLite tables left in the same database (not
        automatically pruned) -- old ones accumulate if this is used
        repeatedly and would need manual cleanup.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_fp_table = f"scene_fingerprints_backup_{timestamp}"
        backup_faces_table = f"scene_fingerprint_matches_backup_{timestamp}"
        with self._connection() as conn:
            conn.execute(f"CREATE TABLE {backup_fp_table} AS SELECT * FROM scene_fingerprints")
            conn.execute(f"CREATE TABLE {backup_faces_table} AS SELECT * FROM scene_fingerprint_matches")
            fingerprints_backed_up = conn.execute(f"SELECT COUNT(*) FROM {backup_fp_table}").fetchone()[0]
            faces_backed_up = conn.execute(f"SELECT COUNT(*) FROM {backup_faces_table}").fetchone()[0]
            cursor = conn.execute(
                "UPDATE scene_fingerprints SET db_version = NULL, updated_at = datetime('now')"
            )
            marked_for_refresh = cursor.rowcount

        return {
            "backup_fingerprints_table": backup_fp_table,
            "backup_faces_table": backup_faces_table,
            "fingerprints_backed_up": fingerprints_backed_up,
            "faces_backed_up": faces_backed_up,
            "marked_for_refresh": marked_for_refresh,
        }

    # ==================== Image Fingerprints ====================

    def create_image_fingerprint(
        self,
        stash_image_id: str,
        gallery_id: Optional[str] = None,
        faces_detected: int = 0,
        db_version: Optional[str] = None,
    ) -> int:
        """
        Create or update an image fingerprint. Returns the fingerprint ID.
        Uses upsert - if fingerprint exists for image, updates it.

        Args:
            stash_image_id: The Stash image ID
            gallery_id: The gallery this image belongs to (optional)
            faces_detected: Number of faces detected in the image
            db_version: Face recognition DB version used for this fingerprint
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_fingerprints (stash_image_id, gallery_id, faces_detected, db_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(stash_image_id) DO UPDATE SET
                    gallery_id = COALESCE(excluded.gallery_id, gallery_id),
                    faces_detected = excluded.faces_detected,
                    db_version = excluded.db_version,
                    updated_at = datetime('now')
                RETURNING id
                """,
                (stash_image_id, gallery_id, faces_detected, db_version)
            )
            return cursor.fetchone()[0]

    def get_image_fingerprint(self, stash_image_id: str) -> Optional[dict]:
        """Get an image fingerprint by stash image ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_fingerprints WHERE stash_image_id = ?",
                (stash_image_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_gallery_image_fingerprints(self, gallery_id: str) -> list[dict]:
        """Get all image fingerprints for a gallery."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM image_fingerprints WHERE gallery_id = ?",
                (gallery_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def add_image_fingerprint_face(
        self,
        stash_image_id: str,
        performer_id: str,
        confidence: Optional[float] = None,
        distance: Optional[float] = None,
        bbox_x: Optional[float] = None,
        bbox_y: Optional[float] = None,
        bbox_w: Optional[float] = None,
        bbox_h: Optional[float] = None,
    ) -> int:
        """Add or update a face entry in an image fingerprint. Returns the face entry ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_fingerprint_faces (
                    stash_image_id, performer_id, confidence, distance,
                    bbox_x, bbox_y, bbox_w, bbox_h
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stash_image_id, performer_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    distance = excluded.distance,
                    bbox_x = excluded.bbox_x,
                    bbox_y = excluded.bbox_y,
                    bbox_w = excluded.bbox_w,
                    bbox_h = excluded.bbox_h
                RETURNING id
                """,
                (stash_image_id, performer_id, confidence, distance,
                 bbox_x, bbox_y, bbox_w, bbox_h)
            )
            return cursor.fetchone()[0]

    def get_image_fingerprint_faces(self, stash_image_id: str) -> list[dict]:
        """Get all face entries for an image fingerprint."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM image_fingerprint_faces WHERE stash_image_id = ?",
                (stash_image_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_image_fingerprint_faces(self, stash_image_id: str) -> int:
        """Delete all face entries for an image fingerprint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM image_fingerprint_faces WHERE stash_image_id = ?",
                (stash_image_id,)
            )
            return cursor.rowcount

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}
            stats['total_recommendations'] = conn.execute(
                "SELECT COUNT(*) FROM recommendations"
            ).fetchone()[0]
            stats['pending_recommendations'] = conn.execute(
                "SELECT COUNT(*) FROM recommendations WHERE status = 'pending'"
            ).fetchone()[0]
            stats['dismissed_count'] = conn.execute(
                "SELECT COUNT(*) FROM dismissed_targets"
            ).fetchone()[0]
            stats['analysis_runs_today'] = conn.execute(
                "SELECT COUNT(*) FROM analysis_runs WHERE date(started_at) = date('now')"
            ).fetchone()[0]
            return stats

    # ==================== Duplicate Candidates ====================

    def insert_candidate(
        self,
        scene_a_id: int,
        scene_b_id: int,
        source: str,
        run_id: int,
    ) -> Optional[int]:
        """Insert a candidate pair. Enforces canonical order (a < b). Returns ID or None if duplicate."""
        a, b = (min(scene_a_id, scene_b_id), max(scene_a_id, scene_b_id))
        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO duplicate_candidates (scene_a_id, scene_b_id, source, run_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (a, b, source, run_id),
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def insert_candidates_batch(
        self,
        candidates: list[tuple[int, int, str]],
        run_id: int,
    ) -> int:
        """Batch insert candidate pairs. Each tuple is (scene_a_id, scene_b_id, source).
        Enforces canonical order. Returns count inserted."""
        if not candidates:
            return 0
        rows = [(min(a, b), max(a, b), source, run_id) for a, b, source in candidates]
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO duplicate_candidates (scene_a_id, scene_b_id, source, run_id)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            return conn.execute(
                "SELECT COUNT(*) FROM duplicate_candidates WHERE run_id = ?", (run_id,)
            ).fetchone()[0]

    def get_candidates_batch(
        self,
        run_id: int,
        after_id: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        """Get a batch of candidates using cursor-based pagination."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM duplicate_candidates
                WHERE run_id = ? AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (run_id, after_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def count_candidates(self, run_id: int) -> int:
        """Count candidates for a run."""
        with self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM duplicate_candidates WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

    def clear_candidates(self, run_id: int) -> int:
        """Delete all candidates for a run. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM duplicate_candidates WHERE run_id = ?",
                (run_id,),
            )
            return cursor.rowcount

    def clear_all_candidates(self) -> int:
        """Delete ALL candidates from all runs. The candidates table is an
        ephemeral work queue — old rows block new inserts due to the
        UNIQUE(scene_a_id, scene_b_id) constraint which doesn't include run_id."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM duplicate_candidates")
            return cursor.rowcount

    def clear_orphaned_candidates(self) -> int:
        """Delete candidates with NULL run_id (from broken runs that passed run_id=None).
        Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM duplicate_candidates WHERE run_id IS NULL"
            )
            return cursor.rowcount

    def get_candidate_scene_ids(self, run_id: int) -> set[int]:
        """Get all distinct scene IDs that appear in candidates for a run."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT scene_a_id AS sid FROM duplicate_candidates WHERE run_id = ?
                UNION
                SELECT scene_b_id AS sid FROM duplicate_candidates WHERE run_id = ?
                """,
                (run_id, run_id),
            ).fetchall()
            return {row[0] for row in rows}

    def get_fingerprints_with_faces(
        self,
        scene_ids: Optional[set[int]] = None,
    ) -> dict:
        """
        Load all complete fingerprints with their best-match performers in a
        single JOIN query, derived from scene_fingerprint_matches (which
        holds every candidate match, not just the best one -- filtered here
        to is_best_match=1 to reproduce the old scene_fingerprint_faces
        "one best match per performer" shape duplicate_scenes.py depends on).
        Returns dict keyed by str(stash_scene_id) with structure:
        {stash_scene_id, total_faces, frames_analyzed, faces: {performer_id -> {face_count, avg_confidence, proportion}}}
        Optionally filtered to specific scene IDs.

        A performer can legitimately be the best match for more than one
        detected person in the same scene (e.g. clustering imperfectly split
        one real person into two groups) -- grouped here by summing
        frame_count and taking a frame_count-weighted average confidence,
        rather than the old table's silent last-write-wins (it upserted one
        row per performer per save, so an earlier person's row was simply
        overwritten by a later one with the same best match).
        """
        query = """
            SELECT sf.stash_scene_id, sf.total_faces, sf.frames_analyzed,
                   sfm.universal_id AS performer_id, sfm.frame_count, sfm.confidence
            FROM scene_fingerprints sf
            LEFT JOIN scene_fingerprint_matches sfm
                ON sf.id = sfm.fingerprint_id AND sfm.is_best_match = 1
            WHERE sf.fingerprint_status = 'complete'
        """
        with self._connection() as conn:
            if scene_ids:
                placeholders = ",".join("?" for _ in scene_ids)
                rows = conn.execute(
                    f"{query} AND sf.stash_scene_id IN ({placeholders}) ORDER BY sf.stash_scene_id",
                    list(scene_ids),
                ).fetchall()
            else:
                rows = conn.execute(f"{query} ORDER BY sf.stash_scene_id").fetchall()

        # Group by scene, then by performer within scene (summing across
        # persons that best-matched the same performer).
        result = {}
        for row in rows:
            scene_id = str(row["stash_scene_id"])
            if scene_id not in result:
                result[scene_id] = {
                    "stash_scene_id": row["stash_scene_id"],
                    "total_faces": row["total_faces"],
                    "frames_analyzed": row["frames_analyzed"],
                    "faces": {},
                }
            if row["performer_id"] is not None:
                faces = result[scene_id]["faces"]
                pid = row["performer_id"]
                if pid not in faces:
                    faces[pid] = {"performer_id": pid, "face_count": 0, "_confidence_weighted_sum": 0.0}
                faces[pid]["face_count"] += row["frame_count"]
                if row["confidence"] is not None:
                    faces[pid]["_confidence_weighted_sum"] += row["confidence"] * row["frame_count"]

        for scene in result.values():
            total_faces = scene["total_faces"] or 0
            for face in scene["faces"].values():
                face_count = face.pop("face_count")
                weighted_sum = face.pop("_confidence_weighted_sum")
                face["face_count"] = face_count
                face["avg_confidence"] = (weighted_sum / face_count) if face_count else None
                face["proportion"] = (face_count / total_faces) if total_faces else 0

        return result

    def generate_face_candidates(self) -> list[tuple[int, int]]:
        """
        Find all scene pairs that share an identified performer via SQL self-join
        on scene_fingerprint_matches (best matches only). Returns list of
        (scene_a_id, scene_b_id) tuples in canonical order (a < b).
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT sfa.stash_scene_id AS scene_a_id,
                                sfb.stash_scene_id AS scene_b_id
                FROM scene_fingerprint_matches fa
                JOIN scene_fingerprint_matches fb ON fa.universal_id = fb.universal_id
                JOIN scene_fingerprints sfa ON fa.fingerprint_id = sfa.id
                JOIN scene_fingerprints sfb ON fb.fingerprint_id = sfb.id
                WHERE fa.is_best_match = 1 AND fb.is_best_match = 1
                  AND sfa.fingerprint_status = 'complete'
                  AND sfb.fingerprint_status = 'complete'
                  AND sfa.stash_scene_id < sfb.stash_scene_id
                  AND fa.universal_id != 'unknown'
                """
            ).fetchall()
            return [(row[0], row[1]) for row in rows]

    def store_scene_phashes(self, phashes: list[tuple[int, str]]) -> int:
        """
        Store scene phashes in memory for duplicate candidate generation.
        Input: list of (stash_scene_id, phash_hex) tuples.
        Stores parsed data in self._phash_data for generate_phash_candidates().
        Returns count stored.
        """
        self._phash_data: list[tuple[int, int]] = []
        for scene_id, phash_hex in phashes:
            try:
                self._phash_data.append((scene_id, int(phash_hex, 16)))
            except (ValueError, TypeError):
                continue
        return len(self._phash_data)

    def generate_phash_candidates(self, max_distance: int = 10) -> list[tuple[int, int, int]]:
        """
        Find all scene pairs with phash Hamming distance <= max_distance.
        Uses data stored by store_scene_phashes().
        Returns list of (scene_a_id, scene_b_id, hamming_distance) in canonical order.
        """
        data = getattr(self, "_phash_data", None)
        if not data:
            return []

        candidates = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                xor = data[i][1] ^ data[j][1]
                dist = bin(xor).count("1")
                if dist <= max_distance:
                    a, b = data[i][0], data[j][0]
                    if a > b:
                        a, b = b, a
                    candidates.append((a, b, dist))

        return candidates

    # ========================================================================
    # Job Queue CRUD
    # ========================================================================

    def submit_job(
        self, type: str, priority: int, triggered_by: str,
        cursor: Optional[str] = None, items_total: Optional[int] = None,
    ) -> Optional[int]:
        """Submit a job to the queue. Returns job ID, or None if duplicate queued."""
        with self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM job_queue WHERE type = ? AND status IN ('queued', 'running', 'stopping')",
                (type,)
            ).fetchone()
            if existing:
                return None
            cursor_obj = conn.execute(
                """
                INSERT INTO job_queue (type, status, priority, cursor, items_total, triggered_by)
                VALUES (?, 'queued', ?, ?, ?, ?)
                """,
                (type, priority, cursor, items_total, triggered_by)
            )
            return cursor_obj.lastrowid

    def get_job(self, job_id: int) -> Optional[dict]:
        """Get a single job by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_jobs(self, status: Optional[str] = None, type: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Get jobs with optional filters."""
        query = "SELECT * FROM job_queue WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def get_queued_jobs(self) -> list[dict]:
        """Get all queued jobs ordered by priority (lowest number = highest priority)."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM job_queue WHERE status = 'queued' ORDER BY priority ASC, created_at ASC"
            ).fetchall()]

    def start_job(self, job_id: int, resource_used: Optional[str] = None):
        """Mark a job as running. `resource_used` records which device
        ("gpu"/"cpu") this specific run actually used, for job types whose
        resource classification (GPU) doesn't pin down the real device --
        it depends on the gpu_enabled setting and actual GPU availability
        at the moment the job started. Frozen at start time so history
        reflects what a job actually ran with even if settings change
        later (see embeddings.effective_device)."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'running', started_at = datetime('now'), "
                "resource_used = COALESCE(?, resource_used) WHERE id = ?",
                (resource_used, job_id)
            )

    def complete_job(self, job_id: int, result_summary: Optional[str] = None):
        """Mark a job as completed, storing an optional human-readable result summary."""
        with self._connection() as conn:
            conn.execute(
                """UPDATE job_queue
                   SET status = 'completed', completed_at = datetime('now'),
                       result_summary = COALESCE(?, result_summary)
                   WHERE id = ?""",
                (result_summary, job_id)
            )

    def fail_job(self, job_id: int, error_message: str):
        """Mark a job as failed."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'failed', completed_at = datetime('now'), error_message = ? WHERE id = ?",
                (error_message, job_id)
            )

    def cancel_job(self, job_id: int):
        """Cancel a queued job."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'cancelled', completed_at = datetime('now') WHERE id = ?",
                (job_id,)
            )

    def set_job_status(self, job_id: int, status: str):
        """Set job status directly."""
        with self._connection() as conn:
            conn.execute("UPDATE job_queue SET status = ? WHERE id = ?", (status, job_id))

    def update_job_progress(self, job_id: int, items_processed: Optional[int] = None,
                            items_total: Optional[int] = None, cursor: Optional[str] = None,
                            label: Optional[str] = None):
        """Update job progress fields. Only updates non-None fields."""
        updates = []
        params = []
        if items_processed is not None:
            updates.append("items_processed = ?")
            params.append(items_processed)
        if items_total is not None:
            updates.append("items_total = ?")
            params.append(items_total)
        if cursor is not None:
            updates.append("cursor = ?")
            params.append(cursor)
        if label is not None:
            updates.append("progress_label = ?")
            params.append(label)
        if not updates:
            return
        params.append(job_id)
        with self._connection() as conn:
            conn.execute(f"UPDATE job_queue SET {', '.join(updates)} WHERE id = ?", params)

    def requeue_interrupted_jobs(self) -> int:
        """Re-queue jobs left as running/stopping after a crash. Returns count.

        Clears stale progress fields so re-queued jobs don't appear already-finished.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """UPDATE job_queue
                SET status = 'queued', started_at = NULL, completed_at = NULL,
                    items_processed = 0, items_total = NULL
                WHERE status IN ('running', 'stopping')"""
            )
            return cursor.rowcount

    def delete_terminal_jobs(self) -> int:
        """Delete all completed/failed/cancelled jobs. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM job_queue WHERE status IN ('completed', 'failed', 'cancelled')"
            )
            return cursor.rowcount

    # ========================================================================
    # Job Schedules CRUD
    # ========================================================================

    def upsert_job_schedule(self, type: str, enabled: bool, interval_hours: float, priority: int):
        """Insert or update a job schedule.

        When enabling, sets next_run_at = now + interval to prevent immediate fire.
        When disabling, clears next_run_at.
        """
        with self._connection() as conn:
            if enabled:
                next_run_expr = "datetime('now', '+' || CAST(? * 3600 AS INTEGER) || ' seconds')"
                conn.execute(
                    f"""
                    INSERT INTO job_schedules (type, enabled, interval_hours, priority, next_run_at)
                    VALUES (?, 1, ?, ?, {next_run_expr})
                    ON CONFLICT(type) DO UPDATE SET
                        enabled = 1,
                        interval_hours = excluded.interval_hours,
                        priority = excluded.priority,
                        next_run_at = CASE
                            WHEN job_schedules.enabled = 1 THEN job_schedules.next_run_at
                            ELSE {next_run_expr}
                        END
                    """,
                    (type, interval_hours, priority, interval_hours, interval_hours)
                )
            else:
                conn.execute(
                    """
                    INSERT INTO job_schedules (type, enabled, interval_hours, priority, next_run_at)
                    VALUES (?, 0, ?, ?, NULL)
                    ON CONFLICT(type) DO UPDATE SET
                        enabled = 0,
                        interval_hours = excluded.interval_hours,
                        priority = excluded.priority,
                        next_run_at = NULL
                    """,
                    (type, interval_hours, priority)
                )

    def get_job_schedule(self, type: str) -> Optional[dict]:
        """Get schedule for a job type."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM job_schedules WHERE type = ?", (type,)).fetchone()
            return dict(row) if row else None

    def get_all_job_schedules(self) -> list[dict]:
        """Get all job schedules."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM job_schedules ORDER BY type").fetchall()]

    def update_schedule_last_run(self, type: str):
        """Update last_run_at to now and calculate next_run_at from interval."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE job_schedules
                SET last_run_at = datetime('now'),
                    next_run_at = datetime('now', '+' || CAST(interval_hours * 3600 AS INTEGER) || ' seconds')
                WHERE type = ?
                """,
                (type,)
            )

    def get_due_schedules(self) -> list[dict]:
        """Get enabled schedules that are past their next_run_at."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(
                """
                SELECT * FROM job_schedules
                WHERE enabled = 1 AND (next_run_at IS NULL OR next_run_at <= datetime('now'))
                """
            ).fetchall()]


# Convenience function
def open_recommendations_db(path: str | Path) -> RecommendationsDB:
    """Open or create a recommendations database."""
    return RecommendationsDB(path)

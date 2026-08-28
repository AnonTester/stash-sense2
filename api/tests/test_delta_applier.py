"""Tests for delta_applier.py's apply_delta_db -- specifically the
face_field_updates path (gender/age/image_sha256 backfilled server-side
onto a face that already shipped in an earlier release; see
stash-sense2-data-gen's build/export_delta.py for the generator side).
"""
import sqlite3

from delta_applier import apply_delta_db


def _make_performers_db(path, faces=()):
    """Minimal performers.db -- just enough schema for apply_delta_db to
    run against. `faces` is a list of (id, performer_id, embedding_index,
    gender, gender_confidence, estimated_age, image_sha256) tuples to
    pre-seed."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE performers (
            id INTEGER PRIMARY KEY, canonical_name TEXT, disambiguation TEXT,
            gender TEXT, country TEXT, ethnicity TEXT, birth_date TEXT,
            death_date TEXT, height_cm INTEGER, eye_color TEXT, hair_color TEXT,
            career_start_year INTEGER, career_end_year INTEGER, image_url TEXT,
            face_count INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT,
            stashdb_updated_at TEXT, inferred_gender TEXT, inferred_gender_confidence REAL
        );
        CREATE TABLE stashbox_ids (
            performer_id INTEGER, endpoint TEXT, stashbox_performer_id TEXT,
            PRIMARY KEY (endpoint, stashbox_performer_id)
        );
        CREATE TABLE aliases (
            id INTEGER PRIMARY KEY, performer_id INTEGER, alias TEXT, source_endpoint TEXT
        );
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY, performer_id INTEGER, embedding_index INTEGER UNIQUE,
            image_url TEXT, source_endpoint TEXT, quality_score REAL, created_at TEXT,
            yaw REAL, gender TEXT, gender_confidence REAL, estimated_age INTEGER,
            image_sha256 TEXT
        );
    """)
    for row in faces:
        conn.execute(
            "INSERT INTO faces (id, performer_id, embedding_index, gender, "
            "gender_confidence, estimated_age, image_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    conn.commit()
    conn.close()


def _make_delta_db(path, face_field_updates=None):
    """Minimal delta.db -- empty performers/faces/removed_faces (nothing
    to upsert this test), optionally a face_field_updates table."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE performers (
            endpoint TEXT, stashbox_id TEXT, action TEXT, name TEXT, disambiguation TEXT,
            gender TEXT, birth_date TEXT, death_date TEXT, ethnicity TEXT, country TEXT,
            eye_color TEXT, hair_color TEXT, height INTEGER, career_start_year INTEGER,
            career_end_year INTEGER, aliases_json TEXT, images_json TEXT, updated TEXT,
            merged_into_id TEXT, inferred_gender TEXT, inferred_gender_confidence REAL
        );
        CREATE TABLE faces (
            embedding_index INTEGER PRIMARY KEY, endpoint TEXT, stashbox_id TEXT,
            image_url TEXT, quality_score REAL, yaw REAL, embedding BLOB
        );
        CREATE TABLE removed_faces (embedding_index INTEGER PRIMARY KEY, reason TEXT);
    """)
    if face_field_updates is not None:
        conn.execute("""
            CREATE TABLE face_field_updates (
                embedding_index INTEGER PRIMARY KEY, gender TEXT, gender_confidence REAL,
                estimated_age INTEGER, image_sha256 TEXT
            )
        """)
        for row in face_field_updates:
            conn.execute(
                "INSERT INTO face_field_updates (embedding_index, gender, gender_confidence, "
                "estimated_age, image_sha256) VALUES (?, ?, ?, ?, ?)",
                row,
            )
    conn.commit()
    conn.close()


def _make_delta_db_with_custom_field_updates(path, columns, rows):
    """Like _make_delta_db, but face_field_updates has an arbitrary
    column set -- for simulating version skew (a delta built with a
    field this client doesn't know about yet, or missing one it does)."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE performers (
            endpoint TEXT, stashbox_id TEXT, action TEXT, name TEXT, disambiguation TEXT,
            gender TEXT, birth_date TEXT, death_date TEXT, ethnicity TEXT, country TEXT,
            eye_color TEXT, hair_color TEXT, height INTEGER, career_start_year INTEGER,
            career_end_year INTEGER, aliases_json TEXT, images_json TEXT, updated TEXT,
            merged_into_id TEXT, inferred_gender TEXT, inferred_gender_confidence REAL
        );
        CREATE TABLE faces (
            embedding_index INTEGER PRIMARY KEY, endpoint TEXT, stashbox_id TEXT,
            image_url TEXT, quality_score REAL, yaw REAL, embedding BLOB
        );
        CREATE TABLE removed_faces (embedding_index INTEGER PRIMARY KEY, reason TEXT);
    """)
    col_defs = ", ".join(f"{c} TEXT" for c in columns if c != "embedding_index")
    conn.execute(f"CREATE TABLE face_field_updates (embedding_index INTEGER PRIMARY KEY, {col_defs})")
    for row in rows:
        placeholders = ", ".join("?" * len(row))
        conn.execute(f"INSERT INTO face_field_updates ({', '.join(columns)}) VALUES ({placeholders})", row)
    conn.commit()
    conn.close()


class TestFaceFieldUpdates:
    def test_applies_field_updates_to_existing_face(self, tmp_path):
        _make_performers_db(
            tmp_path / "performers.db",
            faces=[(1, 1, 100, None, None, None, None)],
        )
        _make_delta_db(
            tmp_path / "delta.db",
            face_field_updates=[(100, "female", 0.97, 27, "abc123sha")],
        )

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["face_field_updates_applied"] == 1
        conn = sqlite3.connect(tmp_path / "performers.db")
        row = conn.execute(
            "SELECT gender, gender_confidence, estimated_age, image_sha256 FROM faces WHERE embedding_index = 100"
        ).fetchone()
        conn.close()
        assert row == ("female", 0.97, 27, "abc123sha")

    def test_does_not_affect_unrelated_faces(self, tmp_path):
        _make_performers_db(
            tmp_path / "performers.db",
            faces=[
                (1, 1, 100, None, None, None, None),
                (2, 1, 101, "male", 0.8, 30, "existing_sha"),
            ],
        )
        _make_delta_db(
            tmp_path / "delta.db",
            face_field_updates=[(100, "female", 0.97, 27, "abc123sha")],
        )

        apply_delta_db(tmp_path / "delta.db", tmp_path)

        conn = sqlite3.connect(tmp_path / "performers.db")
        untouched = conn.execute(
            "SELECT gender, gender_confidence, estimated_age, image_sha256 FROM faces WHERE embedding_index = 101"
        ).fetchone()
        conn.close()
        assert untouched == ("male", 0.8, 30, "existing_sha")

    def test_older_delta_without_table_is_a_noop(self, tmp_path):
        _make_performers_db(
            tmp_path / "performers.db",
            faces=[(1, 1, 100, None, None, None, None)],
        )
        _make_delta_db(tmp_path / "delta.db", face_field_updates=None)

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["face_field_updates_applied"] == 0
        conn = sqlite3.connect(tmp_path / "performers.db")
        row = conn.execute("SELECT gender FROM faces WHERE embedding_index = 100").fetchone()
        conn.close()
        assert row == (None,)

    def test_ensure_columns_migrates_legacy_faces_table(self, tmp_path):
        """A faces table predating gender/age/sha256 support (no such
        columns at all) must be migrated in place before the UPDATE runs,
        not error out."""
        conn = sqlite3.connect(tmp_path / "performers.db")
        conn.executescript("""
            CREATE TABLE performers (id INTEGER PRIMARY KEY, canonical_name TEXT, updated_at TEXT);
            CREATE TABLE stashbox_ids (performer_id INTEGER, endpoint TEXT, stashbox_performer_id TEXT,
                PRIMARY KEY (endpoint, stashbox_performer_id));
            CREATE TABLE aliases (id INTEGER PRIMARY KEY, performer_id INTEGER, alias TEXT, source_endpoint TEXT);
            CREATE TABLE faces (id INTEGER PRIMARY KEY, performer_id INTEGER,
                embedding_index INTEGER UNIQUE, image_url TEXT, source_endpoint TEXT,
                quality_score REAL, created_at TEXT, yaw REAL);
        """)
        conn.execute("INSERT INTO faces (id, performer_id, embedding_index) VALUES (1, 1, 100)")
        conn.commit()
        conn.close()

        _make_delta_db(
            tmp_path / "delta.db",
            face_field_updates=[(100, "female", 0.97, 27, "abc123sha")],
        )

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["face_field_updates_applied"] == 1
        conn = sqlite3.connect(tmp_path / "performers.db")
        row = conn.execute(
            "SELECT gender, gender_confidence, estimated_age, image_sha256 FROM faces WHERE embedding_index = 100"
        ).fetchone()
        conn.close()
        assert row == ("female", 0.97, 27, "abc123sha")

    def test_tolerates_a_field_this_client_predates(self, tmp_path):
        """A delta built with a newer BACKFILLABLE_FACE_FIELDS entry this
        client's own copy of the constant doesn't have yet -- the unknown
        column is silently ignored, known ones still applied."""
        _make_performers_db(
            tmp_path / "performers.db",
            faces=[(1, 1, 100, None, None, None, None)],
        )
        _make_delta_db_with_custom_field_updates(
            tmp_path / "delta.db",
            columns=["embedding_index", "gender", "gender_confidence", "estimated_age",
                     "image_sha256", "some_future_field"],
            rows=[(100, "female", 0.97, 27, "abc123sha", "unknown-to-this-client")],
        )

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["face_field_updates_applied"] == 1
        conn = sqlite3.connect(tmp_path / "performers.db")
        row = conn.execute(
            "SELECT gender, gender_confidence, estimated_age, image_sha256 FROM faces WHERE embedding_index = 100"
        ).fetchone()
        conn.close()
        assert row == ("female", 0.97, 27, "abc123sha")

    def test_tolerates_an_older_delta_missing_a_known_field(self, tmp_path):
        """A delta built before some field this client knows about
        existed -- that field is simply never written, known ones still
        applied."""
        _make_performers_db(
            tmp_path / "performers.db",
            faces=[(1, 1, 100, "male", 0.5, 20, "old_sha")],
        )
        _make_delta_db_with_custom_field_updates(
            tmp_path / "delta.db",
            columns=["embedding_index", "gender", "gender_confidence"],
            rows=[(100, "female", 0.97)],
        )

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["face_field_updates_applied"] == 1
        conn = sqlite3.connect(tmp_path / "performers.db")
        row = conn.execute(
            "SELECT gender, gender_confidence, estimated_age, image_sha256 FROM faces WHERE embedding_index = 100"
        ).fetchone()
        conn.close()
        # gender/gender_confidence updated; estimated_age/image_sha256 untouched (not in this older delta)
        assert row == ("female", 0.97, 20, "old_sha")

"""Tests for delta_applier.py's apply_delta_db -- specifically
`_upsert_face`'s upsert-safety (an existing embedding_index's fields
and/or vector can be legitimately replaced in place, not just a brand-new
face inserted). See stash-sense2-data-gen's build/export_delta.py (the
generator side) for why: its 2026-09-09 rework computes the delta as a
real diff of baseline vs. output, so a `faces`/`catalogue_faces` row can
represent an existing, already-published face whose gender/age/vector
changed server-side -- something the previous blind insert-only apply
path would crash on.
"""
import sqlite3

import numpy as np
import pytest
from usearch.index import Index

from delta_applier import apply_delta_db

# Needs a real usearch install to be meaningful -- under CI's conftest.py
# mock (usearch isn't in requirements.ci.txt), Index() becomes a MagicMock
# whose `in`/`[]` operations vacuously pass without checking anything real.
# Matches this repo's own convention for other usearch/ML-touching test
# files (test_model_manager.py, test_model_router.py) -- run via
# `make test-heavy` or in an environment with the real dependency.
pytestmark = pytest.mark.heavy

DIM = 512


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_performers_db(path, faces=()):
    """Minimal performers.db -- just enough schema for apply_delta_db to
    run against. `faces` is a list of (id, performer_id, embedding_index,
    image_url, source_endpoint, quality_score, yaw, gender,
    gender_confidence, estimated_age, image_sha256) tuples to pre-seed."""
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
        CREATE TABLE performer_urls (
            id INTEGER PRIMARY KEY, performer_id INTEGER, url TEXT, url_type TEXT, source_endpoint TEXT
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
            "INSERT INTO faces (id, performer_id, embedding_index, image_url, source_endpoint, "
            "quality_score, yaw, gender, gender_confidence, estimated_age, image_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        # id also doubles as a fresh performer row for face_count sync,
        # not exercised by these tests but keeps foreign-key-shaped data honest.
        conn.execute(
            "INSERT OR IGNORE INTO performers (id, canonical_name, face_count) VALUES (?, ?, 0)",
            (row[1], f"performer-{row[1]}"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO stashbox_ids (performer_id, endpoint, stashbox_performer_id) VALUES (?, 'stashdb', ?)",
            (row[1], f"perf-{row[1]}"),
        )
    conn.commit()
    conn.close()


def _seed_usearch_index(path, vectors: dict):
    index = Index(ndim=DIM, metric="cos")
    for embedding_index, vector in vectors.items():
        index.add(embedding_index, vector)
    index.save(str(path))


def _make_delta_db(path, faces=(), catalogue_faces=()):
    """Minimal delta.db -- empty performers/removed_faces (nothing to
    upsert in these tests), `faces` and `catalogue_faces` as lists of
    dicts matching build/export_delta.py's post-2026-09-09 DELTA_SCHEMA
    shape for those two tables."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE performers (
            id INTEGER, endpoint TEXT, stashbox_id TEXT, action TEXT, name TEXT, disambiguation TEXT,
            gender TEXT, birth_date TEXT, death_date TEXT, ethnicity TEXT, country TEXT,
            eye_color TEXT, hair_color TEXT, height INTEGER, career_start_year INTEGER,
            career_end_year INTEGER, aliases_json TEXT, images_json TEXT, updated TEXT,
            merged_into_id TEXT, inferred_gender TEXT, inferred_gender_confidence REAL
        );
        CREATE TABLE faces (
            embedding_index INTEGER PRIMARY KEY, endpoint TEXT, stashbox_id TEXT,
            image_url TEXT, quality_score REAL, yaw REAL,
            gender TEXT, gender_confidence REAL, estimated_age INTEGER, image_sha256 TEXT,
            embedding BLOB
        );
        CREATE TABLE removed_faces (embedding_index INTEGER PRIMARY KEY, reason TEXT);
    """)
    for f in faces:
        conn.execute(
            "INSERT INTO faces (embedding_index, endpoint, stashbox_id, image_url, quality_score, yaw, "
            "gender, gender_confidence, estimated_age, image_sha256, embedding) "
            "VALUES (:embedding_index, :endpoint, :stashbox_id, :image_url, :quality_score, :yaw, "
            ":gender, :gender_confidence, :estimated_age, :image_sha256, :embedding)",
            f,
        )
    if catalogue_faces:
        # catalogue_tables_present (apply_delta_db) gates on catalogue_performers
        # existing, not catalogue_faces specifically -- a real delta.db
        # always has all DELTA_SCHEMA tables (one executescript call), so
        # this must too, even empty, for the catalogue_faces loop to run.
        conn.execute("""
            CREATE TABLE catalogue_performers (
                id INTEGER PRIMARY KEY, canonical_name TEXT, gender TEXT, country TEXT,
                image_url TEXT, inferred_gender TEXT, inferred_gender_confidence REAL
            )
        """)
        conn.execute("""
            CREATE TABLE catalogue_performer_urls (
                performer_id INTEGER, url TEXT, url_type TEXT, source_endpoint TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE catalogue_faces (
                embedding_index INTEGER PRIMARY KEY, performer_id INTEGER, image_url TEXT,
                source_endpoint TEXT, quality_score REAL, yaw REAL,
                gender TEXT, gender_confidence REAL, estimated_age INTEGER, image_sha256 TEXT,
                embedding BLOB
            )
        """)
        for f in catalogue_faces:
            conn.execute(
                "INSERT INTO catalogue_faces (embedding_index, performer_id, image_url, source_endpoint, "
                "quality_score, yaw, gender, gender_confidence, estimated_age, image_sha256, embedding) "
                "VALUES (:embedding_index, :performer_id, :image_url, :source_endpoint, :quality_score, "
                ":yaw, :gender, :gender_confidence, :estimated_age, :image_sha256, :embedding)",
                f,
            )
    conn.commit()
    conn.close()


class TestUpsertFaceNewFace:
    def test_inserts_a_brand_new_stashbox_face(self, tmp_path):
        _make_performers_db(tmp_path / "performers.db", faces=[])
        conn = sqlite3.connect(tmp_path / "performers.db")
        conn.execute("INSERT INTO performers (id, canonical_name) VALUES (1, 'p1')")
        conn.execute("INSERT INTO stashbox_ids (performer_id, endpoint, stashbox_performer_id) VALUES (1, 'stashdb', 'perf-1')")
        conn.commit()
        conn.close()

        vec = _vector(1)
        _make_delta_db(tmp_path / "delta.db", faces=[{
            "embedding_index": 100, "endpoint": "stashdb", "stashbox_id": "perf-1",
            "image_url": "http://x/1.jpg", "quality_score": 0.9, "yaw": 0.0,
            "gender": "FEMALE", "gender_confidence": 0.9, "estimated_age": 25, "image_sha256": "sha1",
            "embedding": vec.tobytes(),
        }])

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["faces_added"] == 1
        assert result["faces_updated"] == 0
        conn = sqlite3.connect(tmp_path / "performers.db")
        row = conn.execute(
            "SELECT gender, estimated_age, image_sha256 FROM faces WHERE embedding_index = 100"
        ).fetchone()
        conn.close()
        assert row == ("FEMALE", 25, "sha1")

        index = Index(ndim=DIM, metric="cos")
        index.load(str(tmp_path / "face_embeddings.usearch"))
        assert 100 in index
        assert np.allclose(np.array(index[100]), vec, atol=2e-2)


class TestUpsertFaceExisting:
    def test_updates_fields_of_an_existing_face_without_duplicating_it(self, tmp_path):
        """The core regression test for this rework: a delta row can
        legitimately re-send an EXISTING embedding_index (e.g. corrected
        gender/age server-side) -- must UPDATE in place, never raise a
        UNIQUE-constraint error or create a second row."""
        vec = _vector(2)
        _make_performers_db(tmp_path / "performers.db", faces=[
            (1, 1, 100, "http://x/1.jpg", "stashdb", 0.9, 0.0, None, None, None, None),
        ])
        _seed_usearch_index(tmp_path / "face_embeddings.usearch", {100: vec})
        _make_delta_db(tmp_path / "delta.db", faces=[{
            "embedding_index": 100, "endpoint": "stashdb", "stashbox_id": "perf-1",
            "image_url": "http://x/1.jpg", "quality_score": 0.9, "yaw": 0.0,
            "gender": "FEMALE", "gender_confidence": 0.97, "estimated_age": 27, "image_sha256": "sha_new",
            "embedding": vec.tobytes(),  # unchanged vector, changed fields only
        }])

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["faces_added"] == 0
        assert result["faces_updated"] == 1
        conn = sqlite3.connect(tmp_path / "performers.db")
        rows = conn.execute("SELECT gender, estimated_age, image_sha256 FROM faces WHERE embedding_index = 100").fetchall()
        count = conn.execute("SELECT COUNT(*) FROM faces WHERE embedding_index = 100").fetchone()[0]
        conn.close()
        assert count == 1, "must UPDATE in place, never insert a duplicate row for an existing embedding_index"
        assert rows[0] == ("FEMALE", 27, "sha_new")

    def test_replaces_an_existing_faces_vector_in_place(self, tmp_path):
        """The whole reason this fix exists: an already-published face's
        embedding vector silently corrected server-side (e.g. a rotation-
        correction re-embed) must actually reach the client's usearch
        index, not just its SQL row."""
        old_vec = _vector(3)
        new_vec = _vector(4)
        assert not np.allclose(old_vec, new_vec)

        _make_performers_db(tmp_path / "performers.db", faces=[
            (1, 1, 100, "http://x/1.jpg", "stashdb", 0.9, 0.0, "FEMALE", 0.9, 25, "sha1"),
        ])
        _seed_usearch_index(tmp_path / "face_embeddings.usearch", {100: old_vec})
        _make_delta_db(tmp_path / "delta.db", faces=[{
            "embedding_index": 100, "endpoint": "stashdb", "stashbox_id": "perf-1",
            "image_url": "http://x/1.jpg", "quality_score": 0.9, "yaw": 0.0,
            "gender": "FEMALE", "gender_confidence": 0.9, "estimated_age": 25, "image_sha256": "sha1",
            "embedding": new_vec.tobytes(),
        }])

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["faces_updated"] == 1
        index = Index(ndim=DIM, metric="cos")
        index.load(str(tmp_path / "face_embeddings.usearch"))
        assert 100 in index
        assert np.allclose(np.array(index[100]), new_vec, atol=2e-2)
        assert not np.allclose(np.array(index[100]), old_vec, atol=1e-2)

    def test_does_not_affect_unrelated_faces(self, tmp_path):
        vec1, vec2 = _vector(5), _vector(6)
        _make_performers_db(tmp_path / "performers.db", faces=[
            (1, 1, 100, "http://x/1.jpg", "stashdb", 0.9, 0.0, None, None, None, None),
            (2, 1, 101, "http://x/2.jpg", "stashdb", 0.8, 0.0, "MALE", 0.8, 30, "existing_sha"),
        ])
        _seed_usearch_index(tmp_path / "face_embeddings.usearch", {100: vec1, 101: vec2})
        _make_delta_db(tmp_path / "delta.db", faces=[{
            "embedding_index": 100, "endpoint": "stashdb", "stashbox_id": "perf-1",
            "image_url": "http://x/1.jpg", "quality_score": 0.9, "yaw": 0.0,
            "gender": "FEMALE", "gender_confidence": 0.97, "estimated_age": 27, "image_sha256": "sha_new",
            "embedding": vec1.tobytes(),
        }])

        apply_delta_db(tmp_path / "delta.db", tmp_path)

        conn = sqlite3.connect(tmp_path / "performers.db")
        untouched = conn.execute(
            "SELECT gender, gender_confidence, estimated_age, image_sha256 FROM faces WHERE embedding_index = 101"
        ).fetchone()
        conn.close()
        assert untouched == ("MALE", 0.8, 30, "existing_sha")
        index = Index(ndim=DIM, metric="cos")
        index.load(str(tmp_path / "face_embeddings.usearch"))
        assert np.allclose(np.array(index[101]), vec2, atol=2e-2)


class TestUpsertFaceVersionSkew:
    def test_tolerates_a_delta_missing_the_new_gender_age_columns(self, tmp_path):
        """An older delta.db (built before gender/gender_confidence/
        estimated_age/image_sha256 were added to the faces table) --
        _upsert_face must still insert/update the row, writing NULL for
        the columns it doesn't have, not KeyError."""
        vec = _vector(7)
        _make_performers_db(tmp_path / "performers.db", faces=[])
        conn = sqlite3.connect(tmp_path / "performers.db")
        conn.execute("INSERT INTO performers (id, canonical_name) VALUES (1, 'p1')")
        conn.execute("INSERT INTO stashbox_ids (performer_id, endpoint, stashbox_performer_id) VALUES (1, 'stashdb', 'perf-1')")
        conn.commit()
        conn.close()

        delta_path = tmp_path / "delta.db"
        conn = sqlite3.connect(delta_path)
        conn.executescript("""
            CREATE TABLE performers (
                id INTEGER, endpoint TEXT, stashbox_id TEXT, action TEXT, name TEXT, disambiguation TEXT,
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
        conn.execute(
            "INSERT INTO faces (embedding_index, endpoint, stashbox_id, image_url, quality_score, yaw, embedding) "
            "VALUES (100, 'stashdb', 'perf-1', 'http://x/1.jpg', 0.9, 0.0, ?)",
            (vec.tobytes(),),
        )
        conn.commit()
        conn.close()

        result = apply_delta_db(delta_path, tmp_path)

        assert result["faces_added"] == 1
        conn = sqlite3.connect(tmp_path / "performers.db")
        row = conn.execute("SELECT gender, estimated_age FROM faces WHERE embedding_index = 100").fetchone()
        conn.close()
        assert row == (None, None)


class TestUpsertCatalogueFace:
    def test_updates_an_existing_catalogue_faces_fields_and_vector(self, tmp_path):
        old_vec, new_vec = _vector(8), _vector(9)
        conn = sqlite3.connect(tmp_path / "performers.db")
        conn.executescript("""
            CREATE TABLE performers (
                id INTEGER PRIMARY KEY, canonical_name TEXT, disambiguation TEXT,
                gender TEXT, country TEXT, image_url TEXT, face_count INTEGER DEFAULT 0,
                created_at TEXT, updated_at TEXT, inferred_gender TEXT, inferred_gender_confidence REAL
            );
            CREATE TABLE stashbox_ids (performer_id INTEGER, endpoint TEXT, stashbox_performer_id TEXT,
                PRIMARY KEY (endpoint, stashbox_performer_id));
            CREATE TABLE performer_urls (id INTEGER PRIMARY KEY, performer_id INTEGER, url TEXT, url_type TEXT, source_endpoint TEXT);
            CREATE TABLE faces (id INTEGER PRIMARY KEY, performer_id INTEGER, embedding_index INTEGER UNIQUE,
                image_url TEXT, source_endpoint TEXT, quality_score REAL, created_at TEXT, yaw REAL,
                gender TEXT, gender_confidence REAL, estimated_age INTEGER, image_sha256 TEXT);
        """)
        conn.execute("INSERT INTO performers (id, canonical_name) VALUES (500, 'Catalogue Performer')")
        conn.execute(
            "INSERT INTO faces (id, performer_id, embedding_index, image_url, source_endpoint, gender) "
            "VALUES (1, 500, 200, 'http://c/1.jpg', 'pornbox', 'FEMALE')"
        )
        conn.commit()
        conn.close()
        _seed_usearch_index(tmp_path / "face_embeddings.usearch", {200: old_vec})

        _make_delta_db(tmp_path / "delta.db", catalogue_faces=[{
            "embedding_index": 200, "performer_id": 500, "image_url": "http://c/1.jpg",
            "source_endpoint": "pornbox", "quality_score": 0.85, "yaw": 0.0,
            "gender": "FEMALE", "gender_confidence": 0.95, "estimated_age": 29, "image_sha256": "csha",
            "embedding": new_vec.tobytes(),
        }])

        result = apply_delta_db(tmp_path / "delta.db", tmp_path)

        assert result["catalogue_faces_added"] == 0
        assert result["catalogue_faces_updated"] == 1
        conn = sqlite3.connect(tmp_path / "performers.db")
        count = conn.execute("SELECT COUNT(*) FROM faces WHERE embedding_index = 200").fetchone()[0]
        row = conn.execute("SELECT estimated_age, image_sha256 FROM faces WHERE embedding_index = 200").fetchone()
        conn.close()
        assert count == 1
        assert row == (29, "csha")
        index = Index(ndim=DIM, metric="cos")
        index.load(str(tmp_path / "face_embeddings.usearch"))
        assert np.allclose(np.array(index[200]), new_vec, atol=2e-2)

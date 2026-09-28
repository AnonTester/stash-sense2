"""Tests for delta_applier._record_dirty_state -- the universal_id
resolution step apply_delta_chain runs once a whole delta chain is
applied, feeding delta_scope.py's dirty tables (see DELTA_SCOPE_SCHEMA).

Tested directly against _record_dirty_state rather than through the full
apply_delta_chain (HTTP download + zip extraction + multi-hop orchestration
has no test coverage of its own anywhere in this codebase yet, and would
mostly be testing plumbing unrelated to this function's own real risk: the
endpoint-priority-aware universal_id resolution, which needs real
before/after performers.db state either way). Needs a real usearch
install only insofar as export_db_to_json/_performer_universal_ids does
not (pure SQL) -- but _make_performers_db's schema below is shared in
spirit with test_delta_applier.py's own, so this stays marked heavy for
consistency with that file rather than asserting it's actually safe to
run without the real dependency stack.
"""
import sqlite3

import numpy as np
import pytest

from delta_applier import _record_dirty_state
from recommendations_db import RecommendationsDB

pytestmark = pytest.mark.heavy

DIM = 512


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


_SCHEMA = """
    CREATE TABLE performers (id INTEGER PRIMARY KEY, canonical_name TEXT);
    CREATE TABLE stashbox_ids (
        performer_id INTEGER, endpoint TEXT, stashbox_performer_id TEXT,
        PRIMARY KEY (endpoint, stashbox_performer_id)
    );
    CREATE TABLE performer_urls (
        id INTEGER PRIMARY KEY, performer_id INTEGER, url TEXT, url_type TEXT, source_endpoint TEXT
    );
    CREATE TABLE faces (
        id INTEGER PRIMARY KEY, performer_id INTEGER, embedding_index INTEGER UNIQUE, source_endpoint TEXT
    );
"""


def _make_db(path, performers=(), stashbox_ids=(), performer_urls=(), faces=()):
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    for pid, name in performers:
        conn.execute("INSERT INTO performers (id, canonical_name) VALUES (?, ?)", (pid, name))
    for pid, endpoint, sbid in stashbox_ids:
        conn.execute(
            "INSERT INTO stashbox_ids (performer_id, endpoint, stashbox_performer_id) VALUES (?, ?, ?)",
            (pid, endpoint, sbid),
        )
    for pid, source_endpoint in performer_urls:
        conn.execute(
            "INSERT INTO performer_urls (performer_id, url, url_type, source_endpoint) VALUES (?, 'http://x', 'catalogue', ?)",
            (pid, source_endpoint),
        )
    for pid, embedding_index, source_endpoint in faces:
        conn.execute(
            "INSERT INTO faces (performer_id, embedding_index, source_endpoint) VALUES (?, ?, ?)",
            (pid, embedding_index, source_endpoint),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def rec_db(tmp_path) -> RecommendationsDB:
    return RecommendationsDB(tmp_path / "stash_sense.db")


class TestUniversalIdPriorityResolution:
    def test_prefers_stashdb_over_theporndb_for_a_dual_linked_performer(self, tmp_path, rec_db):
        """The core risk this function exists to avoid: building a
        universal_id straight from whichever endpoint a delta row happens
        to mention would get this WRONG for a performer linked to more
        than one stash-box endpoint -- only the SAME priority rule
        faces.json/performers.json themselves use (stashdb > theporndb >
        ...) produces the uid a scene's stored matches actually key on."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _make_db(backup_dir / "performers.db",
                  performers=[(1, "Performer One")],
                  stashbox_ids=[(1, "stashdb", "sb-uuid-1"), (1, "theporndb", "tpdb-uuid-1")])

        _make_db(tmp_path / "performers.db",
                  performers=[(1, "Performer One")],
                  stashbox_ids=[(1, "stashdb", "sb-uuid-1"), (1, "theporndb", "tpdb-uuid-1")])
        conn = sqlite3.connect(tmp_path / "performers.db")

        # A delta that only touched performer 1 via their LOWER-priority
        # theporndb face -- must still resolve to the stashdb.org uid.
        _record_dirty_state(
            rec_db, conn, backup_dir,
            face_level_touched_performer_ids={1}, removed_performer_ids=set(),
            metadata_only_performer_ids=set(),
            upserted_face_vectors={100: (1, _vector(1).tobytes())},
            from_version="2026.09.01", to_version="2026.09.15",
        )
        conn.close()

        dirty_uids = rec_db.get_dirty_universal_ids(reasons=["needs_rematch"])
        assert dirty_uids == ["stashdb.org:sb-uuid-1"]
        vectors = rec_db.get_dirty_face_vectors()
        assert len(vectors) == 1
        assert vectors[0]["universal_id"] == "stashdb.org:sb-uuid-1"

    def test_removed_performer_resolves_its_old_uid_since_it_has_no_new_one(self, tmp_path, rec_db):
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _make_db(backup_dir / "performers.db",
                  performers=[(2, "Gone Performer")],
                  stashbox_ids=[(2, "theporndb", "tpdb-uuid-2")])

        _make_db(tmp_path / "performers.db")  # performer 2 no longer exists
        conn = sqlite3.connect(tmp_path / "performers.db")

        _record_dirty_state(
            rec_db, conn, backup_dir,
            face_level_touched_performer_ids=set(), removed_performer_ids={2},
            metadata_only_performer_ids=set(), upserted_face_vectors={},
            from_version="2026.09.01", to_version="2026.09.15",
        )
        conn.close()

        assert rec_db.get_dirty_universal_ids(reasons=["needs_rematch"]) == ["theporndb.net:tpdb-uuid-2"]

    def test_metadata_only_reason_kept_separate_from_needs_rematch(self, tmp_path, rec_db):
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _make_db(backup_dir / "performers.db",
                  performers=[(4, "Old Name")],
                  stashbox_ids=[(4, "stashdb", "sb-uuid-4")])
        _make_db(tmp_path / "performers.db",
                  performers=[(4, "New Name")],
                  stashbox_ids=[(4, "stashdb", "sb-uuid-4")])
        conn = sqlite3.connect(tmp_path / "performers.db")

        _record_dirty_state(
            rec_db, conn, backup_dir,
            face_level_touched_performer_ids=set(), removed_performer_ids=set(),
            metadata_only_performer_ids={4}, upserted_face_vectors={},
            from_version="2026.09.01", to_version="2026.09.15",
        )
        conn.close()

        assert rec_db.get_dirty_universal_ids(reasons=["needs_rematch"]) == []
        assert rec_db.get_dirty_universal_ids(reasons=["metadata_only"]) == ["stashdb.org:sb-uuid-4"]

    def test_catalogue_performer_uses_source_endpoint_and_own_id(self, tmp_path, rec_db):
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _make_db(backup_dir / "performers.db")
        _make_db(tmp_path / "performers.db",
                  performers=[(500, "Catalogue Performer")],
                  faces=[(500, 200, "pornbox")])
        conn = sqlite3.connect(tmp_path / "performers.db")

        _record_dirty_state(
            rec_db, conn, backup_dir,
            face_level_touched_performer_ids={500}, removed_performer_ids=set(),
            metadata_only_performer_ids=set(),
            upserted_face_vectors={200: (500, _vector(2).tobytes())},
            from_version="2026.09.01", to_version="2026.09.15",
        )
        conn.close()

        assert rec_db.get_dirty_universal_ids(reasons=["needs_rematch"]) == ["pornbox:500"]

    def test_sets_the_tracking_marker_to_from_version(self, tmp_path, rec_db):
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _make_db(backup_dir / "performers.db")
        _make_db(tmp_path / "performers.db")
        conn = sqlite3.connect(tmp_path / "performers.db")

        _record_dirty_state(
            rec_db, conn, backup_dir,
            face_level_touched_performer_ids=set(), removed_performer_ids=set(),
            metadata_only_performer_ids=set(), upserted_face_vectors={},
            from_version="2026.09.01", to_version="2026.09.15",
        )
        conn.close()

        assert rec_db.get_dirty_tracking_marker() == "2026.09.01"

    def test_unchanged_identity_is_recorded_once_not_twice(self, tmp_path, rec_db):
        """old_uid == new_uid (nothing about the performer's OWN identity
        changed, just a face) -- must not double-record the same uid
        under the same reason."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _make_db(backup_dir / "performers.db",
                  performers=[(1, "Performer One")], stashbox_ids=[(1, "stashdb", "sb-uuid-1")])
        _make_db(tmp_path / "performers.db",
                  performers=[(1, "Performer One")], stashbox_ids=[(1, "stashdb", "sb-uuid-1")])
        conn = sqlite3.connect(tmp_path / "performers.db")

        _record_dirty_state(
            rec_db, conn, backup_dir,
            face_level_touched_performer_ids={1}, removed_performer_ids=set(),
            metadata_only_performer_ids=set(),
            upserted_face_vectors={100: (1, _vector(3).tobytes())},
            from_version="2026.09.01", to_version="2026.09.15",
        )
        conn.close()

        assert rec_db.get_dirty_universal_ids(reasons=["needs_rematch"]) == ["stashdb.org:sb-uuid-1"]

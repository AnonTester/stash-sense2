"""RecommendationsDB bulk dirty-state writers: same semantics as the per-row ones, but one transaction. A catalogue-heavy
delta records tens of thousands of rows; one connection + commit per row took ~8 minutes and blocked the server."""
import numpy as np

from recommendations_db import RecommendationsDB


def _db(tmp_path):
    return RecommendationsDB(tmp_path / "rec.db")


def _vec(seed):
    return np.random.default_rng(seed).standard_normal(512).astype(np.float32).tobytes()


def test_bulk_universal_ids_first_writer_wins(tmp_path):
    db = _db(tmp_path)
    db.record_dirty_universal_ids([("stashdb.org:a", "needs_rematch", "2026.10.05"),
                                   ("stashdb.org:b", "metadata_only", "2026.10.05")])
    db.record_dirty_universal_ids([("stashdb.org:a", "needs_rematch", "2099.01.01")])  # INSERT OR IGNORE
    with db._connection() as conn:
        rows = {r["universal_id"]: (r["reason"], r["since_version"]) for r in conn.execute("SELECT * FROM dirty_universal_ids")}
    assert rows == {"stashdb.org:a": ("needs_rematch", "2026.10.05"), "stashdb.org:b": ("metadata_only", "2026.10.05")}


def test_bulk_face_vectors_replace_on_conflict(tmp_path):
    db = _db(tmp_path)
    db.record_dirty_face_vectors([(1, "u1", _vec(1), "v1"), (2, "u2", _vec(2), "v1")])
    db.record_dirty_face_vectors([(2, "u2b", _vec(3), "v2")])
    with db._connection() as conn:
        rows = {r["embedding_index"]: (r["universal_id"], bytes(r["embedding"]), r["since_version"]) for r in conn.execute("SELECT * FROM dirty_face_vectors")}
    assert rows[1] == ("u1", _vec(1), "v1")
    assert rows[2] == ("u2b", _vec(3), "v2")


def test_empty_batches_are_a_no_op(tmp_path):
    db = _db(tmp_path)
    db.record_dirty_universal_ids([])
    db.record_dirty_face_vectors([])
    with db._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM dirty_universal_ids").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM dirty_face_vectors").fetchone()[0] == 0

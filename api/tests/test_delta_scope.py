"""Tests for delta_scope.py -- the Refresh Outdated / Face Recommendations
delta-scoping decision (scenes_needing_rematch) and its supporting
dirty-state CRUD on RecommendationsDB (DELTA_SCOPE_SCHEMA).

Uses a real RecommendationsDB (sqlite on tmp_path), not mocks -- the whole
point of this module is real SQL joins + real numpy distance math, and its
own cold-start fallback specifically depends on telling "a real marker
string" apart from "nothing set yet", which a mock would paper over. No
usearch/insightface/onnxruntime needed (conftest.py's _ML_MODULES mocks
already make embeddings.py/recognizer.py/scene_matcher.py importable, and
this module's own vector math is plain numpy against raw bytes, never a
real usearch Index) -- runs under plain `make test-ci`, not `-m heavy`.
"""
import json
import sys

import numpy as np
import pytest

import face_config
from delta_scope import finalize_dirty_state, scenes_needing_rematch
from recommendations_db import RecommendationsDB

DIM = 512


@pytest.fixture(autouse=True)
def real_recognizer_and_embeddings_modules(monkeypatch):
    """delta_scope._lightweight_results_from_cache constructs REAL
    recognizer.RecognitionResult/embeddings.DetectedFace/FaceEmbedding
    instances (needed for real numpy math in the Tier-3 tests below) via a
    lazy `from recognizer import ...` inside the function -- but
    test_scene_matcher_logic.py replaces `sys.modules['recognizer']` with
    a bare Mock() at ITS OWN import time, permanently, for the rest of the
    whole pytest session (no teardown; deliberate there -- see that
    file's own top-of-file comment, which documents this exact class of
    cross-test sys.modules leakage biting a test once already). Whichever
    test file collects first decides what a later LAZY `from recognizer
    import X` call resolves to, which is exactly the kind of collection-
    order-dependent flakiness that bit that file before. Deleting the
    (possibly-mocked) sys.modules entries here forces a fresh, real
    import the next time anything in this file's own tests actually
    triggers one, restored by monkeypatch after each test -- this file's
    own tests never rely on `recognizer`/`embeddings` being mocked, so
    there's no equivalent reason to protect the other direction."""
    monkeypatch.delitem(sys.modules, "recognizer", raising=False)
    monkeypatch.delitem(sys.modules, "embeddings", raising=False)


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _nudge(vec: np.ndarray, amount: float) -> np.ndarray:
    """A vector at roughly cosine distance `amount` from `vec` -- close
    enough for threshold tests without hand-deriving exact geometry."""
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(DIM).astype(np.float32)
    noise -= (noise @ vec) * vec  # orthogonal component only
    noise /= np.linalg.norm(noise)
    # cosine distance 1-cos(theta) ~= amount for small theta with this construction
    theta = np.arccos(1 - amount)
    out = np.cos(theta) * vec + np.sin(theta) * noise
    return (out / np.linalg.norm(out)).astype(np.float32)


@pytest.fixture
def rec_db(tmp_path) -> RecommendationsDB:
    return RecommendationsDB(tmp_path / "stash_sense.db")


def _make_fingerprint(rec_db, scene_id: int, db_version: str, total_faces: int = 1) -> dict:
    fid = rec_db.create_scene_fingerprint(
        stash_scene_id=scene_id, total_faces=total_faces, frames_analyzed=1,
        fingerprint_status="complete", db_version=db_version,
    )
    return {**rec_db.get_scene_fingerprint(scene_id), "id": fid}


def _seed_match(rec_db, fingerprint_id: int, universal_id: str, person_id: int = 1) -> None:
    rec_db.replace_fingerprint_matches(fingerprint_id, [{
        "person_id": person_id, "frame_count": 1, "match_rank": 1, "is_best_match": True,
        "universal_id": universal_id,
    }])


def _seed_scene_embedding(rec_db, scene_id: int, vec: np.ndarray, frame_index: int = 0) -> None:
    rec_db.replace_face_embeddings(scene_id, [{
        "frame_index": frame_index, "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
        "confidence": 0.9, "yaw": 0.0, "embedding": vec.tobytes(),
    }], is_sprite=False)


class TestColdStart:
    def test_no_marker_set_treats_everything_as_must_rematch(self, rec_db):
        fp = _make_fingerprint(rec_db, 1, "2026.09.01")
        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert must_rematch == {1}
        assert safe == set()

    def test_no_marker_set_ignores_dirty_state_that_happens_to_exist(self, rec_db):
        # Shouldn't normally happen (dirty rows only get written alongside
        # the marker in the same call -- see delta_applier._record_dirty_state),
        # but if it ever did, cold start must still win: nothing is trusted
        # until the marker itself is known.
        rec_db.record_dirty_universal_id("stashdb.org:abc", "needs_rematch", "2026.09.01")
        fp = _make_fingerprint(rec_db, 1, "2026.09.01")
        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert must_rematch == {1}


class TestMarkerEligibility:
    def test_scene_older_than_marker_falls_back_to_must_rematch(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.01")  # last matched before tracking began
        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert must_rematch == {1}
        assert safe == set()

    def test_scene_at_or_after_marker_with_nothing_dirty_is_safe_to_bump(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.10", total_faces=0)
        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert safe == {1}
        assert must_rematch == set()

    def test_mixed_eligibility_splits_correctly(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        old = _make_fingerprint(rec_db, 1, "2026.09.01", total_faces=0)
        new = _make_fingerprint(rec_db, 2, "2026.09.15", total_faces=0)
        must_rematch, safe = scenes_needing_rematch(rec_db, [old, new])
        assert must_rematch == {1}
        assert safe == {2}


class TestTier2Identity:
    def test_scene_referencing_a_needs_rematch_uid_is_forced_to_rematch(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.10", total_faces=1)
        _seed_match(rec_db, fp["id"], "stashdb.org:abc")
        rec_db.record_dirty_universal_id("stashdb.org:abc", "needs_rematch", "2026.09.15")

        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert must_rematch == {1}

    def test_scene_referencing_an_unrelated_uid_is_unaffected(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.10", total_faces=0)
        _seed_match(rec_db, fp["id"], "stashdb.org:unrelated")
        rec_db.record_dirty_universal_id("stashdb.org:abc", "needs_rematch", "2026.09.15")

        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert safe == {1}

    def test_metadata_only_reason_does_not_trigger_tier2(self, rec_db):
        """A pure metadata edit (rename, etc.) is patched separately (see
        delta_applier._record_dirty_state's own comment) -- it must never
        force a real rematch on its own."""
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.10", total_faces=0)
        _seed_match(rec_db, fp["id"], "stashdb.org:abc")
        rec_db.record_dirty_universal_id("stashdb.org:abc", "metadata_only", "2026.09.15")

        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert safe == {1}


class TestTier3Vector:
    def test_scene_with_a_face_close_to_a_dirty_vector_is_forced_to_rematch(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.10", total_faces=1)
        base = _vec(1)
        _seed_scene_embedding(rec_db, 1, base)
        # A new/changed delta vector landing well inside max_distance of
        # this scene's own cached face -- exactly the "a new performer
        # could now match this scene" case.
        close = _nudge(base, face_config.MAX_DISTANCE * 0.3)
        rec_db.record_dirty_face_vector(999, "stashdb.org:new", close.tobytes(), "2026.09.15")

        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert must_rematch == {1}

    def test_scene_with_no_face_anywhere_near_a_dirty_vector_is_safe_to_bump(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.10", total_faces=1)
        _seed_scene_embedding(rec_db, 1, _vec(1))
        # An unrelated vector, far past max_distance -- cannot plausibly
        # become a new candidate for this scene.
        rec_db.record_dirty_face_vector(999, "stashdb.org:new", _vec(2).tobytes(), "2026.09.15")

        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert safe == {1}

    def test_zero_total_faces_short_circuits_without_checking_embeddings(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        fp = _make_fingerprint(rec_db, 1, "2026.09.10", total_faces=0)
        # Deliberately no scene_face_embeddings row at all for scene 1 --
        # if the short-circuit didn't fire, _scene_person_centroids would
        # just return [] anyway (same safe outcome), so this also doubles
        # as "no embeddings" coverage either way.
        rec_db.record_dirty_face_vector(999, "stashdb.org:new", _vec(1).tobytes(), "2026.09.15")

        must_rematch, safe = scenes_needing_rematch(rec_db, [fp])
        assert safe == {1}


class TestFinalizeDirtyState:
    def test_drains_universal_ids_and_vectors_up_to_version(self, rec_db):
        rec_db.record_dirty_universal_id("stashdb.org:abc", "needs_rematch", "2026.09.15")
        rec_db.record_dirty_face_vector(1, "stashdb.org:abc", _vec(1).tobytes(), "2026.09.15")

        finalize_dirty_state(rec_db, "2026.09.15")

        assert rec_db.get_dirty_universal_ids() == []
        assert rec_db.get_dirty_face_vectors() == []

    def test_does_not_drain_rows_newer_than_the_given_version(self, rec_db):
        rec_db.record_dirty_universal_id("stashdb.org:abc", "needs_rematch", "2026.09.20")

        finalize_dirty_state(rec_db, "2026.09.15")

        assert rec_db.get_dirty_universal_ids() == ["stashdb.org:abc"]


class TestDirtyTrackingMarker:
    def test_set_if_unset_keeps_the_first_value(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.20")
        assert rec_db.get_dirty_tracking_marker() == "2026.09.10"

    def test_push_forward_moves_past_a_later_version(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.10")
        rec_db.push_dirty_tracking_marker_forward("2026.09.20")
        assert rec_db.get_dirty_tracking_marker() == "2026.09.20"

    def test_push_forward_never_moves_backward(self, rec_db):
        rec_db.set_dirty_tracking_marker_if_unset("2026.09.20")
        rec_db.push_dirty_tracking_marker_forward("2026.09.10")
        assert rec_db.get_dirty_tracking_marker() == "2026.09.20"

    def test_push_forward_sets_it_when_nothing_was_set_yet(self, rec_db):
        rec_db.push_dirty_tracking_marker_forward("2026.09.10")
        assert rec_db.get_dirty_tracking_marker() == "2026.09.10"


class TestBumpFingerprintDbVersion:
    def test_updates_db_version_without_touching_matches(self, rec_db):
        fp = _make_fingerprint(rec_db, 1, "2026.09.01")
        _seed_match(rec_db, fp["id"], "stashdb.org:abc")

        rec_db.bump_fingerprint_db_version(1, "2026.09.15")

        updated = rec_db.get_scene_fingerprint(1)
        assert updated["db_version"] == "2026.09.15"
        assert len(rec_db.get_fingerprint_matches(fp["id"])) == 1

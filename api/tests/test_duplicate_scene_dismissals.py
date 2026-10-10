"""A duplicate-scene dismissal is a manual verdict ("these scenes do not match") that must survive later
re-scoring, so dismissing scenes together also covers every pair among them that had no row yet."""

import pytest

from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "test.db")


def _pair(db, a, b, status="pending", rec_type="duplicate_scenes"):
    rec_id = db.create_recommendation(
        type=rec_type, target_type="scene", target_id=f"{a}:{b}", details={"scene_a_id": a, "scene_b_id": b},
    )
    if status != "pending":
        with db._connection() as conn:
            conn.execute("UPDATE recommendations SET status = ? WHERE id = ?", (status, rec_id))
    return rec_id


def _dismissed(db, key, rec_type="duplicate_scenes"):
    return db.is_dismissed(rec_type, "scene", key)


def _status(db, rec_id):
    return db.get_recommendation(rec_id).status


class TestGroupDismissal:
    def test_pair_between_two_matches_of_the_same_source_is_covered(self, db):
        """Source 1 with matches 2 and 3 was dismissed; the 2:3 pair had no row (it scored under the
        threshold) and must not surface when a later database update raises its score."""
        ab, ac = _pair(db, 1, 2), _pair(db, 1, 3)

        result = db.dismiss_duplicate_scene_group([ab, ac], reason="Marked not duplicates")

        assert result == {"dismissed_count": 2, "extra_pairs_recorded": 1}
        assert _status(db, ab) == "dismissed" and _status(db, ac) == "dismissed"
        assert _dismissed(db, "1:2") and _dismissed(db, "1:3")
        assert _dismissed(db, "2:3")

    def test_covered_pair_is_never_recreated_by_the_analyzer_check(self, db):
        """The analyzer skips a target for which is_dismissed() is true -- the exact guard that was missing."""
        db.dismiss_duplicate_scene_group([_pair(db, 1, 2), _pair(db, 1, 3)])
        assert db.is_dismissed("duplicate_scenes", "scene", "2:3") is True

    def test_dismissal_reason_and_permanence_are_carried_to_the_extra_pairs(self, db):
        db.dismiss_duplicate_scene_group([_pair(db, 1, 2), _pair(db, 1, 3)], reason="Marked not duplicates")
        with db._connection() as conn:
            row = conn.execute(
                "SELECT reason, permanent FROM dismissed_targets WHERE target_id = '2:3'"
            ).fetchone()
        assert (row["reason"], row["permanent"]) == ("Marked not duplicates", 0)

    def test_a_pending_pair_elsewhere_stays_visible(self, db):
        """2:3 already has a pending recommendation of its own: the user has not judged it, so it is left
        for them (and gets no silent dismissal that the next analysis would turn into a disappearance)."""
        ab, ac, bc = _pair(db, 1, 2), _pair(db, 1, 3), _pair(db, 2, 3)

        result = db.dismiss_duplicate_scene_group([ab, ac])

        assert result["extra_pairs_recorded"] == 0
        assert _status(db, bc) == "pending"
        assert not _dismissed(db, "2:3")

    def test_groups_dismissed_together_connect_into_one_set(self, db):
        """Several selected groups (1:2, 1:3 and 3:4) form one connected set {1,2,3,4}: the three pairs
        none of them listed are covered."""
        ids = [_pair(db, 1, 2), _pair(db, 1, 3), _pair(db, 3, 4)]

        result = db.dismiss_duplicate_scene_group(ids)

        assert result["extra_pairs_recorded"] == 3
        for key in ("2:3", "2:4", "1:4"):
            assert _dismissed(db, key), key

    def test_unrelated_groups_are_not_connected(self, db):
        ids = [_pair(db, 1, 2), _pair(db, 3, 4)]

        result = db.dismiss_duplicate_scene_group(ids)

        assert result["extra_pairs_recorded"] == 0
        for key in ("1:3", "1:4", "2:3", "2:4"):
            assert not _dismissed(db, key), key

    def test_a_resolved_pair_is_left_alone(self, db):
        resolved = _pair(db, 2, 3, status="resolved")
        db.dismiss_duplicate_scene_group([_pair(db, 1, 2), _pair(db, 1, 3)])
        assert _status(db, resolved) == "resolved"
        assert not _dismissed(db, "2:3")

    def test_unknown_ids_and_empty_input(self, db):
        assert db.dismiss_duplicate_scene_group([]) == {"dismissed_count": 0, "extra_pairs_recorded": 0}
        assert db.dismiss_duplicate_scene_group([99999]) == {"dismissed_count": 0, "extra_pairs_recorded": 0}

    def test_other_recommendation_types_get_no_pair_closure(self, db):
        """Face-match recommendations are meant to resurface when new faces are identified."""
        ids = [_pair(db, 1, 2, rec_type="scene_face_match"), _pair(db, 1, 3, rec_type="scene_face_match")]

        result = db.dismiss_duplicate_scene_group(ids)

        assert result["extra_pairs_recorded"] == 0
        assert not _dismissed(db, "2:3", rec_type="scene_face_match")
        assert not _dismissed(db, "2:3")


class TestDismissAll:
    def test_dismiss_all_covers_pairs_among_the_dismissed_scenes(self, db):
        _pair(db, 1, 2), _pair(db, 1, 3)

        assert db.batch_dismiss_by_type("duplicate_scenes", reason="Batch dismissed by user") == 2

        assert _dismissed(db, "2:3")

    def test_dismiss_all_of_another_type_is_unchanged(self, db):
        _pair(db, 1, 2, rec_type="scene_face_match"), _pair(db, 1, 3, rec_type="scene_face_match")

        assert db.batch_dismiss_by_type("scene_face_match") == 2

        with db._connection() as conn:
            extra = conn.execute("SELECT count(*) FROM dismissed_targets WHERE target_id = '2:3'").fetchone()[0]
        assert extra == 0


class TestBackfill:
    def _setup(self, db):
        """1 was dismissed against 2, 3 and 4 (one review group). 2:3 is pending, 2:4 has no row, 3:4 is
        resolved; 7:8 is an unrelated dismissed pair."""
        for a, b in ((1, 2), (1, 3), (1, 4), (7, 8)):
            db.dismiss_recommendation(_pair(db, a, b), reason="User dismissed")
        pending = _pair(db, 2, 3)
        resolved = _pair(db, 3, 4, status="resolved")
        return pending, resolved

    def test_dry_run_counts_and_writes_nothing(self, db):
        pending, _ = self._setup(db)

        counts = db.backfill_duplicate_scene_dismissals(dry_run=True)

        assert counts["pending_recommendations"] == 1
        assert counts["pairs_without_recommendation"] == 1
        assert counts["pairs_left_alone"] == 1
        assert counts["applied"] is False
        assert _status(db, pending) == "pending"
        assert not _dismissed(db, "2:4")

    def test_apply_dismisses_pending_and_records_missing_pairs(self, db):
        pending, resolved = self._setup(db)

        counts = db.backfill_duplicate_scene_dismissals(dry_run=False)

        assert counts["applied"] is True
        assert _status(db, pending) == "dismissed" and _dismissed(db, "2:3")
        assert _dismissed(db, "2:4")
        assert _status(db, resolved) == "resolved" and not _dismissed(db, "3:4")
        assert not _dismissed(db, "1:7") and not _dismissed(db, "2:7")

    def test_apply_twice_changes_nothing_the_second_time(self, db):
        self._setup(db)
        db.backfill_duplicate_scene_dismissals(dry_run=False)

        again = db.backfill_duplicate_scene_dismissals(dry_run=False)

        assert again["pending_recommendations"] == 0 and again["pairs_without_recommendation"] == 0

    def test_a_rerun_does_not_widen_the_set_through_its_own_rows(self, db):
        """Dismissed 1:2, 1:3 and 3:4 (two reviews). Pass one covers 2:3 and 1:4. If those rows counted as
        reviews, a second pass would also cover 2:4 -- reviews that never happened together."""
        for a, b in ((1, 2), (1, 3), (3, 4)):
            db.dismiss_recommendation(_pair(db, a, b), reason="User dismissed")

        first = db.backfill_duplicate_scene_dismissals(dry_run=False)
        again = db.backfill_duplicate_scene_dismissals(dry_run=False)

        assert first["pairs_to_dismiss"] == 2 and _dismissed(db, "2:3") and _dismissed(db, "1:4")
        assert again["pairs_to_dismiss"] == 0
        assert not _dismissed(db, "2:4")

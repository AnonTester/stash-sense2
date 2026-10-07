"""A local performer linked through ANY stash-box (javstash, fansdb, ...) is the same person as the dataset performer
that carries that id: merged within one face's candidates, grouped across frames, and shown with the western name."""
from types import SimpleNamespace

import numpy as np

from matching import (
    CandidateMatch,
    build_stashbox_sibling_maps,
    fuse_local_results,
    local_display_name,
    local_identity,
    merge_local_candidates,
    resolve_local_identity,
    _resolve_local_link_uid,
)
from recognizer import FaceRecognizer
from scene_matcher import _match_is_tagged, _resolve_local_link

# dataset performer 92936: a StashDB and a javstash record of one person; its key in performers.json is the StashDB one
PERFORMERS = {"stashdb.org:12eff": {"name": "武藤つぐみ"}, "stashdb.org:solo": {"name": "Solo"}}
ROWS = [(92936, "stashdb.org", "12eff"), (92936, "javstash.org", "43456")]


def _maps():
    return build_stashbox_sibling_maps(ROWS, PERFORMERS)


class TestSiblingMaps:
    def test_other_ids_resolve_to_the_dataset_key(self):
        siblings, alt = _maps()
        assert siblings == {"stashdb.org:12eff": ["javstash.org:43456"]}
        assert alt == {"javstash.org:43456": "stashdb.org:12eff"}

    def test_performer_unknown_to_the_dataset_is_skipped(self):
        assert build_stashbox_sibling_maps([(1, "a.org", "x"), (1, "b.org", "y")], PERFORMERS) == ({}, {})


class TestResolveLocalIdentity:
    def test_javstash_only_local_performer_resolves_to_the_stashdb_key(self):
        siblings, alt = _maps()
        entry = {"stash_ids": {"javstash.org": "43456"}}
        primary, uids = resolve_local_identity(entry, PERFORMERS, alt, siblings)
        assert primary == "stashdb.org:12eff"
        assert uids == ["stashdb.org:12eff", "javstash.org:43456"]

    def test_own_stashdb_id_known_to_the_dataset(self):
        siblings, alt = _maps()
        primary, uids = resolve_local_identity({"stashdb_id": "12eff"}, PERFORMERS, alt, siblings)
        assert primary == "stashdb.org:12eff" and "javstash.org:43456" in uids

    def test_id_the_dataset_does_not_know_keeps_the_local_performers_own(self):
        primary, uids = resolve_local_identity({"stash_ids": {"fansdb.cc": "f"}}, PERFORMERS, {}, {})
        assert (primary, uids) == ("fansdb.cc:f", ["fansdb.cc:f"])

    def test_unlinked(self):
        assert resolve_local_identity({"name": "x"}, PERFORMERS, {}, {}) == (None, [])

    def test_local_identity_prefers_the_annotation_and_falls_back(self):
        assert local_identity({"_linked_uid": "a:1", "_linked_uids": ["a:1", "b:2"]}) == ("a:1", ["a:1", "b:2"])
        assert local_identity({"stashdb_id": "s"}) == ("stashdb.org:s", ["stashdb.org:s"])
        assert local_identity(None) == (None, [])


def _cand(uid, dist, name="n"):
    c = CandidateMatch(face_index=1, universal_id=uid, name=name, distance=dist, rank=1)
    c.combined_distance = dist
    return c


class TestMergeAndGrouping:
    MAPPING = {"3073": {"_linked_uid": "stashdb.org:12eff", "_linked_uids": ["stashdb.org:12eff", "javstash.org:43456"]}}

    def test_local_candidate_replaces_the_better_scoring_duplicate_it_is_linked_to(self):
        merged = merge_local_candidates([_cand("stashdb.org:12eff", 0.5)], [_cand("local:3073", 0.4)], self.MAPPING)
        assert [c.universal_id for c in merged] == ["local:3073"]

    def test_main_candidate_kept_when_it_scores_better(self):
        merged = merge_local_candidates([_cand("stashdb.org:12eff", 0.3)], [_cand("local:3073", 0.4)], self.MAPPING)
        assert [c.universal_id for c in merged] == ["stashdb.org:12eff"]

    def test_unrelated_stays_separate(self):
        merged = merge_local_candidates([_cand("stashdb.org:solo", 0.3)], [_cand("local:3073", 0.4)], self.MAPPING)
        assert {c.universal_id for c in merged} == {"stashdb.org:solo", "local:3073"}

    def test_grouping_key_is_the_dataset_key(self):
        assert _resolve_local_link_uid("local:3073", self.MAPPING) == "stashdb.org:12eff"
        assert _resolve_local_link_uid("local:9", {"9": {"name": "x"}}) == "local:9"      # unlinked stays local

    def test_scene_matcher_resolves_through_linked_ids(self):
        m = SimpleNamespace(universal_id="local:3073", local_performer_id="3073", stashdb_id="3073",
                            linked_universal_ids=["stashdb.org:12eff", "javstash.org:43456"])
        assert _resolve_local_link(m) == "stashdb.org:12eff"

    def test_scene_matcher_old_behaviour_without_linked_ids(self):
        m = SimpleNamespace(universal_id="local:3073", local_performer_id="3073", stashdb_id="abc")
        assert _resolve_local_link(m) == "stashdb.org:abc"
        unlinked = SimpleNamespace(universal_id="local:3073", local_performer_id="3073", stashdb_id="3073")
        assert _resolve_local_link(unlinked) == "local:3073"


class TestTagged:
    def test_tagged_through_any_linked_stash_id(self):
        m = SimpleNamespace(stashdb_id="12eff", linked_universal_ids=["stashdb.org:12eff", "javstash.org:43456"])
        assert _match_is_tagged(m, {"43456"}) is True
        assert _match_is_tagged(m, {"12eff"}) is True
        assert _match_is_tagged(m, {"other"}) is False
        assert _match_is_tagged(m, set()) is False

    def test_match_without_linked_ids(self):
        assert _match_is_tagged(SimpleNamespace(stashdb_id="x"), {"x"}) is True


class TestWesternNames:
    def test_local_performer_with_a_japanese_name_uses_its_own_western_alias(self):
        info = {"name": "星空もあ", "aliases": ["Moa Hoshizora", "星野"]}
        assert local_display_name("3071", info, {}, True) == ("Moa Hoshizora", "星空もあ")

    def test_falls_back_to_the_linked_dataset_performers_aliases(self):
        info = {"name": "星空もあ", "_linked_uid": "stashdb.org:61bb", "_linked_uids": ["stashdb.org:61bb"]}
        aliases = {"stashdb.org:61bb": ["Hoshino Kana", "Kana Hoshino"]}
        assert local_display_name("3071", info, aliases, True) == ("Hoshino Kana", "星空もあ")

    def test_setting_off_or_already_western_changes_nothing(self):
        assert local_display_name("1", {"name": "星空もあ", "aliases": ["Moa"]}, {}, False) == ("星空もあ", None)
        assert local_display_name("1", {"name": "Moa Hoshizora", "aliases": ["Other"]}, {}, True) == ("Moa Hoshizora", None)

    def test_fuse_local_results_applies_it(self):
        class Q:
            neighbors = np.array([3071])
            distances = np.array([0.4])
        mapping = {"3071": {"name": "星空もあ", "aliases": ["Moa Hoshizora"]}}
        cand = fuse_local_results(Q(), mapping, aliases={}, prefer_western_names=True)[0]
        assert (cand.name, cand.original_name) == ("Moa Hoshizora", "星空もあ")
        plain = fuse_local_results(Q(), mapping)[0]
        assert (plain.name, plain.original_name) == ("星空もあ", None)


class TestRecognizerWiring:
    def _fake(self, mapping):
        siblings, alt = _maps()
        return SimpleNamespace(
            local_performer_index=SimpleNamespace(mapping=mapping), performers=PERFORMERS,
            stashbox_siblings=siblings, stashbox_alt_primary=alt,
        )

    def test_annotation_resolves_every_local_performer(self):
        fake = self._fake({"3073": {"stash_ids": {"javstash.org": "43456"}}, "9": {"name": "unlinked"}})
        FaceRecognizer._annotate_local_identities(fake)
        entry = fake.local_performer_index.mapping["3073"]
        assert entry["_linked_uid"] == "stashdb.org:12eff"
        assert entry["_linked_uids"] == ["stashdb.org:12eff", "javstash.org:43456"]
        assert "_linked_uid" not in fake.local_performer_index.mapping["9"] or not fake.local_performer_index.mapping["9"]["_linked_uid"]

    def test_local_match_carries_the_dataset_stashdb_id_and_every_linked_id(self):
        fake = self._fake({"3073": {"name": "Aimi", "stash_ids": {"javstash.org": "43456"}}})
        FaceRecognizer._annotate_local_identities(fake)
        fields = FaceRecognizer._resolve_match_fields(fake, "local:3073")
        assert fields["stashdb_id"] == "12eff"                       # a StashDB uuid, from the dataset counterpart
        assert fields["local_performer_id"] == "3073"
        assert fields["linked_universal_ids"] == ["stashdb.org:12eff", "javstash.org:43456"]

    def test_unlinked_local_match_keeps_its_own_id(self):
        fake = self._fake({"9": {"name": "x"}})
        FaceRecognizer._annotate_local_identities(fake)
        fields = FaceRecognizer._resolve_match_fields(fake, "local:9")
        assert fields["stashdb_id"] == "9" and fields["linked_universal_ids"] == []

    def test_dataset_match_lists_the_other_stash_box_ids_of_the_same_performer(self):
        fake = self._fake({})
        fields = FaceRecognizer._resolve_match_fields(fake, "stashdb.org:12eff")
        assert fields["linked_universal_ids"] == ["stashdb.org:12eff", "javstash.org:43456"]
        assert FaceRecognizer._resolve_match_fields(fake, "stashdb.org:solo")["linked_universal_ids"] == []

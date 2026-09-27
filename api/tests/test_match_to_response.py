"""_match_to_response is the single chokepoint every identify surface (full
scene, current frame, selection, image, gallery-via-scene_matcher) converts
a PerformerMatch through -- see identification_router.py's own docstring on
it and scene_matcher.py's aggregate_matches/frequency_based_matching/etc.,
which all call it too. Covers universal_id and matched_embedding_index
passthrough, exposed for third-party consumers that need a match's identity
without reconstructing it from endpoint+stashdb_id themselves.
"""
from identification_router import _match_to_response
from recognizer import PerformerMatch


def _match(**overrides) -> PerformerMatch:
    defaults = dict(
        universal_id="stashdb.org:50459d16-aaaa-bbbb-cccc-000000000000",
        stashdb_id="50459d16-aaaa-bbbb-cccc-000000000000",
        name="Jane Doe",
        country=None,
        image_url=None,
        distance=0.2,
        combined_score=0.2,
    )
    defaults.update(overrides)
    return PerformerMatch(**defaults)


class TestMatchToResponseIdentityPassthrough:
    def test_universal_id_survives_to_the_response(self):
        resp = _match_to_response(_match())
        assert resp.universal_id == "stashdb.org:50459d16-aaaa-bbbb-cccc-000000000000"

    def test_matched_embedding_index_survives_to_the_response(self):
        resp = _match_to_response(_match(matched_embedding_index=1234))
        assert resp.matched_embedding_index == 1234

    def test_matched_embedding_index_defaults_to_none(self):
        # A PerformerMatch built without it (e.g. an older/local-index path
        # that never set it) must not error -- _match_to_response reads it
        # via getattr with a None default.
        resp = _match_to_response(_match())
        assert resp.matched_embedding_index is None

    def test_local_only_match_still_gets_its_own_universal_id(self):
        # A local-index match still gets *a* universal_id ("local:<id>") --
        # this conversion step doesn't special-case it. This test only pins
        # down that whatever PerformerMatch carries survives unmodified.
        resp = _match_to_response(_match(universal_id="local:42", local_performer_id="42"))
        assert resp.universal_id == "local:42"

    def test_overrides_still_win_over_the_source_match(self):
        # aggregate_matches/frequency_based_matching etc. pass
        # confidence/distance/top_timestamps_sec as overrides after
        # recomputing them across a whole cluster -- must still take
        # precedence over the single-frame PerformerMatch's own values.
        resp = _match_to_response(_match(distance=0.9), distance=0.05, confidence=0.95)
        assert resp.distance == 0.05
        assert resp.confidence == 0.95

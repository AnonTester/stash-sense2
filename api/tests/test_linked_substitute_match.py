"""build_linked_substitute_match keeps the reference-face pointer of the match it was built from -- except for a local
substitute, whose own cover photo is the right image there (see the method's docstring)."""
from types import SimpleNamespace

from recognizer import FaceRecognizer


def _fake(fields):
    return SimpleNamespace(
        performers={"fansdb.cc:abc": {"name": "Alina Lando"}},
        aliases={},
        local_performer_index=None,
        _resolve_match_fields=lambda uid: fields,
    )


def _fields(local_performer_id=None):
    return {
        "stashdb_id": "abc", "country": None, "image_url": None, "local_performer_id": local_performer_id,
        "source": None, "catalogue_url": None, "profile_url": None, "linked_universal_ids": [],
    }


def _template(embedding_index):
    return SimpleNamespace(distance=0.4, combined_score=0.4, matched_embedding_index=embedding_index)


class TestSubstituteEmbeddingIndex:
    def test_non_local_substitute_keeps_the_templates_reference_face(self):
        sub = FaceRecognizer.build_linked_substitute_match(_fake(_fields()), "fansdb.cc:abc", _template(612173))
        assert sub is not None
        assert sub.universal_id == "fansdb.cc:abc"
        assert sub.matched_embedding_index == 612173
        assert sub.combined_score == 0.4

    def test_template_without_an_index_gives_none(self):
        sub = FaceRecognizer.build_linked_substitute_match(_fake(_fields()), "fansdb.cc:abc", _template(None))
        assert sub.matched_embedding_index is None

    def test_local_substitute_never_carries_a_main_index_face(self):
        fake = _fake(_fields(local_performer_id="720"))
        fake.local_performer_index = SimpleNamespace(mapping={"720": {"name": "Rushia Toudou"}})
        fake.performers = {}
        sub = FaceRecognizer.build_linked_substitute_match(fake, "local:720", _template(612173))
        assert sub is not None
        assert sub.matched_embedding_index is None

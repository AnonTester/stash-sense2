"""Tests for FaceRecognizer._load_local_performer_index()/
reload_local_performer_index() (recognizer.py) -- refreshes just the
local performer index in place on an already-loaded recognizer, used
after a local performer sync instead of unloading the entire
face_recognition resource group (which also holds the buffalo_l models
and the main DB index, neither of which a local-index-only change
touches -- see main.py's refresh_local_performer_index() docstring for
the full story).

Constructs a bare object with just the attribute (db_config) these
methods touch rather than a real FaceRecognizer, since a real one needs a
loaded main DB index/model bundle -- neither method has any other
dependency on FaceRecognizer state.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from recognizer import FaceRecognizer


def _fake_recognizer(local_faces_json_path, local_embedding_index_path, existing_index=None):
    return SimpleNamespace(
        db_config=SimpleNamespace(
            local_faces_json_path=local_faces_json_path,
            local_embedding_index_path=local_embedding_index_path,
        ),
        local_performer_index=existing_index,
    )


class TestLoadLocalPerformerIndex:
    """The actual load-from-disk logic, shared by __init__ and
    reload_local_performer_index()."""

    def test_reloads_from_disk_when_files_exist(self, tmp_path):
        faces_json = tmp_path / "local_faces.json"
        faces_json.write_text("{}")
        embed_path = tmp_path / "local.usearch"
        fake_self = _fake_recognizer(faces_json, embed_path)
        fake_index = MagicMock()
        fake_index.__len__ = MagicMock(return_value=3)

        with patch("local_performer_index.LocalPerformerIndex", return_value=fake_index) as mock_cls:
            FaceRecognizer._load_local_performer_index(fake_self)

        mock_cls.assert_called_once_with(embed_path, faces_json)
        assert fake_self.local_performer_index is fake_index

    def test_resets_to_none_when_files_missing(self, tmp_path):
        faces_json = tmp_path / "local_faces.json"  # never created
        fake_self = _fake_recognizer(
            faces_json, tmp_path / "local.usearch", existing_index=MagicMock(),
        )

        FaceRecognizer._load_local_performer_index(fake_self)

        assert fake_self.local_performer_index is None

    def test_picks_up_a_second_sync_replacing_the_first_index(self, tmp_path):
        """The whole point: calling this twice (e.g. two performer syncs in
        a row) must reflect whatever's on disk *now*, not cache the first
        load."""
        faces_json = tmp_path / "local_faces.json"
        faces_json.write_text("{}")
        embed_path = tmp_path / "local.usearch"
        fake_self = _fake_recognizer(faces_json, embed_path)
        first_index = MagicMock()
        second_index = MagicMock()

        with patch("local_performer_index.LocalPerformerIndex", side_effect=[first_index, second_index]):
            FaceRecognizer._load_local_performer_index(fake_self)
            assert fake_self.local_performer_index is first_index
            FaceRecognizer._load_local_performer_index(fake_self)
            assert fake_self.local_performer_index is second_index


class TestReloadLocalPerformerIndex:
    def test_delegates_to_load_local_performer_index(self):
        fake_self = SimpleNamespace(_load_local_performer_index=MagicMock())

        FaceRecognizer.reload_local_performer_index(fake_self)

        fake_self._load_local_performer_index.assert_called_once_with()

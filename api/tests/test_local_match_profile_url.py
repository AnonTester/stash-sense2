"""Tests for recognizer.py's _local_match_profile_url() -- surfaces a
local-index match's own catalogue profile URL (e.g. javdatabase.com) for
display, when that performer has no real StashDB link to show instead.
"""
from recognizer import _local_match_profile_url


class TestLocalMatchProfileUrl:
    def test_returns_first_non_stashdb_url(self):
        local_info = {"urls": ["https://www.javdatabase.com/idols/mika-kitagawa/"]}
        assert _local_match_profile_url(local_info) == (
            "https://www.javdatabase.com/idols/mika-kitagawa/"
        )

    def test_skips_stashdb_url(self):
        local_info = {"urls": ["https://stashdb.org/performers/abc-123"]}
        assert _local_match_profile_url(local_info) is None

    def test_skips_stashdb_url_case_insensitive_picks_next(self):
        local_info = {
            "urls": [
                "https://StashDB.org/performers/abc-123",
                "https://www.javdatabase.com/idols/mika-kitagawa/",
            ]
        }
        assert _local_match_profile_url(local_info) == (
            "https://www.javdatabase.com/idols/mika-kitagawa/"
        )

    def test_no_urls_returns_none(self):
        assert _local_match_profile_url({}) is None
        assert _local_match_profile_url({"urls": []}) is None
        assert _local_match_profile_url({"urls": None}) is None

"""Tests for name_script.py -- western-script detection and the
prefer_western_names display-name swap (see that module's own docstring)."""
import pytest

from name_script import is_western_script, resolve_display_name


class TestIsWesternScript:
    def test_plain_ascii_name_is_western(self):
        assert is_western_script("Sasha Grey") is True

    def test_accented_latin_name_is_western(self):
        assert is_western_script("Renée Beaulieu") is True

    def test_japanese_kanji_is_not_western(self):
        assert is_western_script("田村美羽") is False

    def test_japanese_hiragana_is_not_western(self):
        assert is_western_script("あいかわ優衣") is False

    def test_cyrillic_is_not_western(self):
        assert is_western_script("Александра") is False

    def test_thai_is_not_western(self):
        assert is_western_script("สุดา") is False

    def test_trailing_fullwidth_punctuation_does_not_flip_a_western_name(self):
        # "Seiko。" -- a real dataset example (full-width Japanese period
        # on an otherwise-western romanized name). Punctuation must be
        # ignored, not counted as non-western.
        assert is_western_script("Seiko。") is True

    def test_hyphenated_western_name_is_western(self):
        assert is_western_script("Marie-Claire") is True

    def test_empty_string_is_not_western(self):
        assert is_western_script("") is False

    def test_none_is_not_western(self):
        assert is_western_script(None) is False

    def test_digits_only_is_not_western(self):
        # No letters at all -- nothing here worth preferring.
        assert is_western_script("12345") is False

    def test_majority_non_western_with_one_latin_letter_is_not_western(self):
        assert is_western_script("田村Aさん") is False


class TestResolveDisplayName:
    def test_setting_off_never_swaps(self):
        name, original = resolve_display_name(
            "javstash.org:1", "田村美羽", {"javstash.org:1": ["Miu Tamura"]}, prefer_western=False,
        )
        assert (name, original) == ("田村美羽", None)

    def test_already_western_name_is_unchanged(self):
        name, original = resolve_display_name(
            "javstash.org:1", "Sasha Grey", {"javstash.org:1": ["Some Alias"]}, prefer_western=True,
        )
        assert (name, original) == ("Sasha Grey", None)

    def test_non_western_name_with_western_alias_swaps(self):
        name, original = resolve_display_name(
            "javstash.org:1", "田村美羽",
            {"javstash.org:1": ["Akae Omiya", "Erika", "Mika Sakai"]},
            prefer_western=True,
        )
        assert (name, original) == ("Akae Omiya", "田村美羽")

    def test_first_western_alias_wins_not_a_later_one(self):
        name, original = resolve_display_name(
            "javstash.org:1", "田村美羽",
            {"javstash.org:1": ["浅野美希", "Akae Omiya", "Erika"]},
            prefer_western=True,
        )
        assert name == "Akae Omiya"

    def test_no_western_alias_available_leaves_name_unchanged(self):
        name, original = resolve_display_name(
            "javstash.org:1", "田村美羽",
            {"javstash.org:1": ["浅野美希", "中原りさこ"]},
            prefer_western=True,
        )
        assert (name, original) == ("田村美羽", None)

    def test_no_aliases_entry_at_all_leaves_name_unchanged(self):
        name, original = resolve_display_name(
            "javstash.org:1", "田村美羽", {}, prefer_western=True,
        )
        assert (name, original) == ("田村美羽", None)

    def test_none_aliases_dict_leaves_name_unchanged(self):
        name, original = resolve_display_name(
            "javstash.org:1", "田村美羽", None, prefer_western=True,
        )
        assert (name, original) == ("田村美羽", None)

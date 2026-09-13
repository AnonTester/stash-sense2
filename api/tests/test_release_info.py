"""Tests for release_info.py's changelog.txt parser.

No test existed for this before 2026-09-13, when a real five-month-old
parsing gap was found the hard way: changelog.txt's own convention
dropped the word "only" from version headers starting 2026-09-09
(0.33.0 onward), but _VERSION_HEADER_RE was never updated to match --
every entry since then was silently invisible to changelog_since()/
full_changelog(), with no error, no warning, just missing data. These
tests exist so a future format drift fails loudly instead of the same way.
"""
import pytest

import release_info


@pytest.fixture
def changelog(tmp_path, monkeypatch):
    """Point release_info at a throwaway changelog.txt for this test."""
    path = tmp_path / "changelog.txt"
    monkeypatch.setattr(release_info, "_CHANGELOG_PATH", path)

    def _write(text: str) -> None:
        path.write_text(text)

    return _write


class TestChangelogSince:
    def test_current_convention_no_only(self, changelog):
        """The active convention as of 2026-09-13: "(sidecar)"/"(plugin)",
        no "only" -- must parse, not just the legacy "only" form."""
        changelog(
            "## 2026-09-13\n"
            "\n"
            "### 0.34.0 (sidecar)\n"
            "- Feature: something new.\n"
        )
        entries = release_info.changelog_since("sidecar", "0.33.0")
        assert entries == [{"version": "0.34.0", "date": "2026-09-13", "bullets": ["Feature: something new."]}]

    def test_legacy_convention_with_only(self, changelog):
        """The older convention ("(sidecar only)"/"(plugin only)") must
        still parse -- most of the file's real history uses this."""
        changelog(
            "## 2026-09-05\n"
            "\n"
            "### 0.30.0 (sidecar only)\n"
            "- Fix: an old bug.\n"
        )
        entries = release_info.changelog_since("sidecar", "0.29.0")
        assert entries == [{"version": "0.30.0", "date": "2026-09-05", "bullets": ["Fix: an old bug."]}]

    def test_wrong_component_excluded(self, changelog):
        changelog(
            "## 2026-09-13\n"
            "\n"
            "### 0.34.0 (sidecar)\n"
            "- Feature: sidecar thing.\n"
            "\n"
            "### 0.25.0 (plugin)\n"
            "- Feature: plugin thing.\n"
        )
        sidecar_entries = release_info.changelog_since("sidecar", "0")
        plugin_entries = release_info.changelog_since("plugin", "0")
        assert [e["version"] for e in sidecar_entries] == ["0.34.0"]
        assert [e["version"] for e in plugin_entries] == ["0.25.0"]

    def test_version_at_or_below_since_excluded(self, changelog):
        changelog(
            "## 2026-09-13\n"
            "\n"
            "### 0.34.0 (sidecar)\n"
            "- Feature: newer.\n"
            "\n"
            "## 2026-09-10\n"
            "\n"
            "### 0.33.1 (sidecar)\n"
            "- Fix: older.\n"
        )
        entries = release_info.changelog_since("sidecar", "0.33.1")
        assert [e["version"] for e in entries] == ["0.34.0"]

    def test_bare_header_from_the_pre_split_era_applies_to_both_tracks(self, changelog):
        """A version header with no "(component)" tag at all is the
        project's oldest era, before sidecar and plugin were versioned
        separately -- one version number for the whole, undifferentiated
        project. It must show up under BOTH tracks with that same
        version, not be dropped and not corrupt a neighboring entry."""
        changelog(
            "## 2026-01-01\n"
            "\n"
            "### 0.14.6\n"
            "- Fix: something from before the split.\n"
            "\n"
            "### 0.14.5\n"
            "- Feature: even older, still unsplit.\n"
        )
        sidecar_entries = release_info.changelog_since("sidecar", "0")
        plugin_entries = release_info.changelog_since("plugin", "0")
        assert [e["version"] for e in sidecar_entries] == ["0.14.6", "0.14.5"]
        assert [e["version"] for e in plugin_entries] == ["0.14.6", "0.14.5"]
        assert sidecar_entries[0]["bullets"] == ["Fix: something from before the split."]
        assert plugin_entries[0]["bullets"] == ["Fix: something from before the split."]

    def test_bare_header_does_not_corrupt_a_neighboring_tagged_entry(self, changelog):
        changelog(
            "## 2026-01-02\n"
            "\n"
            "### 0.15.0 (sidecar only)\n"
            "- Fix: the real, correctly-tagged sidecar fix.\n"
            "\n"
            "### 0.14.6\n"
            "- Fix: an unrelated pre-split entry.\n"
        )
        entries = release_info.changelog_since("sidecar", "0")
        assert [e["version"] for e in entries] == ["0.15.0", "0.14.6"]
        assert entries[0]["bullets"] == ["Fix: the real, correctly-tagged sidecar fix."]

    def test_combined_header_splits_into_two_entries_with_shared_bullets(self, changelog):
        """"### X (sidecar) / Y (plugin)" -- a change landing in both
        components the same day, with two different version numbers,
        sharing one bullet list."""
        changelog(
            "## 2026-08-20\n"
            "\n"
            "### 0.32.1 (sidecar) / 0.24.1 (plugin)\n"
            "- Feature: shipped in both at once.\n"
        )
        sidecar_entries = release_info.changelog_since("sidecar", "0")
        plugin_entries = release_info.changelog_since("plugin", "0")
        assert sidecar_entries == [
            {"version": "0.32.1", "date": "2026-08-20", "bullets": ["Feature: shipped in both at once."]}
        ]
        assert plugin_entries == [
            {"version": "0.24.1", "date": "2026-08-20", "bullets": ["Feature: shipped in both at once."]}
        ]

    def test_combined_header_versions_compared_independently(self, changelog):
        """Each side of a combined header is filtered against
        `since_version` on its own -- one side can be newer than the
        caller's version while the other isn't."""
        changelog(
            "## 2026-08-20\n"
            "\n"
            "### 0.32.1 (sidecar) / 0.24.1 (plugin)\n"
            "- Feature: x.\n"
        )
        assert len(release_info.changelog_since("sidecar", "0.32.0")) == 1
        assert len(release_info.changelog_since("sidecar", "0.32.1")) == 0
        assert len(release_info.changelog_since("plugin", "0.24.0")) == 1
        assert len(release_info.changelog_since("plugin", "0.24.1")) == 0

    def test_missing_file_returns_empty(self, changelog):
        entries = release_info.changelog_since("sidecar", "0.1.0")
        assert entries == []

    def test_none_since_version_returns_empty(self, changelog):
        changelog("## 2026-09-13\n\n### 0.34.0 (sidecar)\n- Feature: x.\n")
        assert release_info.changelog_since("sidecar", None) == []


class TestFullChangelog:
    def test_returns_everything_regardless_of_current_version(self, changelog):
        """Unlike changelog_since(), full_changelog() must return real
        content even for a component already on the latest version --
        it exists specifically to serve a user-triggered "show me the
        changelog" action with no "current version" to diff against."""
        changelog(
            "## 2026-09-13\n"
            "\n"
            "### 0.34.0 (sidecar)\n"
            "- Feature: newest.\n"
            "\n"
            "## 2026-09-05\n"
            "\n"
            "### 0.30.0 (sidecar only)\n"
            "- Fix: oldest.\n"
        )
        entries = release_info.full_changelog("sidecar")
        assert [e["version"] for e in entries] == ["0.34.0", "0.30.0"]

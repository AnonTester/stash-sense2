"""Tests for performer_disambiguation.py - response-time disambiguation lookup from performers.db."""

import sqlite3

import pytest

import performer_disambiguation as pd


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "performers.db")
    conn.executescript("""
        CREATE TABLE performers (id INTEGER PRIMARY KEY, canonical_name TEXT, disambiguation TEXT);
        CREATE TABLE stashbox_ids (
            performer_id INTEGER, endpoint TEXT, stashbox_performer_id TEXT,
            PRIMARY KEY (endpoint, stashbox_performer_id));
        INSERT INTO performers VALUES (1, 'Linda', 'GGG 2002'), (2, 'Linda', NULL), (3, 'Linda', '  ');
        INSERT INTO stashbox_ids VALUES (1, 'stashdb', 'uuid-1'), (2, 'stashdb', 'uuid-2'), (1, 'theporndb', 'tpdb-1');
    """)
    conn.commit()
    conn.close()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    pd._cache.clear()
    pd._cache_token = None
    return tmp_path


def test_stashbox_and_catalogue_ids_resolve(data_dir):
    result = pd.get_disambiguations(
        ["stashdb.org:uuid-1", "stashdb.org:uuid-2", "theporndb.net:tpdb-1", "iafd:1", "iafd:2", "iafd:3"])
    assert result == {"stashdb.org:uuid-1": "GGG 2002", "theporndb.net:tpdb-1": "GGG 2002", "iafd:1": "GGG 2002"}


def test_local_unknown_and_empty_ids_are_skipped(data_dir):
    assert pd.get_disambiguations(["local:7", "stashdb.org:nope", "theporndb.org:tpdb-1", "iafd:abc", "", None]) == {}
    assert pd.get_disambiguation(None) is None
    assert pd.get_disambiguation("stashdb.org:uuid-1") == "GGG 2002"


def test_missing_database_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nowhere"))
    pd._cache.clear()
    pd._cache_token = None
    assert pd.get_disambiguation("stashdb.org:uuid-1") is None


def test_cache_follows_database_changes(data_dir):
    assert pd.get_disambiguation("stashdb.org:uuid-2") is None
    conn = sqlite3.connect(data_dir / "performers.db")
    conn.execute("UPDATE performers SET disambiguation = 'new' WHERE id = 2")
    conn.commit()
    conn.close()
    assert pd.get_disambiguation("stashdb.org:uuid-2") == "new"

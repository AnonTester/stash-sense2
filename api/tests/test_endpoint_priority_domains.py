"""Tests for FaceRecognizer._endpoint_priority_domains() (recognizer.py).

Regression coverage: an ordinary deployment with real stash-box
connections configured but no explicit save via the separate Endpoint
Priority reordering UI (Settings > ... > Endpoint priority) used to get an
EMPTY priority list back -- silently disabling every priority-based
decision downstream (matching.py's collapse_linked_candidates and
scene_matcher.py's linked-group display pick both then fall back to raw
match score for every group, which is how a linked group's pornbox entry
outranked its stashdb.org entry in practice). Fixed to mirror
GET /settings/endpoint-priorities' own fallback: any configured,
non-disabled connection missing from the explicit priority list is still
appended, in the connection manager's own default order.

Calls the unbound method against a bare object (no generator/index/etc.
needed -- this method only touches get_connection_manager() and the
lazily-imported get_rec_db()), same pattern as
test_recognizer_detection_pool.py.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import recognizer
import recommendations_router
from recognizer import FaceRecognizer


def _connections(*domains_by_endpoint):
    return [
        {"endpoint": ep, "domain": domain, "name": domain}
        for ep, domain in domains_by_endpoint
    ]


class TestEndpointPriorityDomainsFallback:
    def test_no_explicit_priority_falls_back_to_default_connection_order(self):
        conns = _connections(
            ("https://stashdb.org/graphql", "stashdb.org"),
            ("https://theporndb.net/graphql", "theporndb.net"),
        )
        conn_mgr = MagicMock()
        conn_mgr.get_connections.return_value = conns
        rec_db = MagicMock()
        rec_db.get_endpoint_priorities.return_value = []  # never explicitly saved
        rec_db.get_disabled_endpoints.return_value = []

        with patch.object(recognizer, "get_connection_manager", return_value=conn_mgr), \
             patch.object(recommendations_router, "get_rec_db", return_value=rec_db):
            domains = FaceRecognizer._endpoint_priority_domains(SimpleNamespace())

        assert domains == ["stashdb.org", "theporndb.net"]

    def test_explicit_priority_order_is_respected(self):
        conns = _connections(
            ("https://stashdb.org/graphql", "stashdb.org"),
            ("https://theporndb.net/graphql", "theporndb.net"),
        )
        conn_mgr = MagicMock()
        conn_mgr.get_connections.return_value = conns
        rec_db = MagicMock()
        rec_db.get_endpoint_priorities.return_value = ["https://theporndb.net/graphql", "https://stashdb.org/graphql"]
        rec_db.get_disabled_endpoints.return_value = []

        with patch.object(recognizer, "get_connection_manager", return_value=conn_mgr), \
             patch.object(recommendations_router, "get_rec_db", return_value=rec_db):
            domains = FaceRecognizer._endpoint_priority_domains(SimpleNamespace())

        assert domains == ["theporndb.net", "stashdb.org"]

    def test_partial_explicit_priority_appends_the_rest_in_default_order(self):
        conns = _connections(
            ("https://stashdb.org/graphql", "stashdb.org"),
            ("https://theporndb.net/graphql", "theporndb.net"),
            ("https://javstash.org/graphql", "javstash.org"),
        )
        conn_mgr = MagicMock()
        conn_mgr.get_connections.return_value = conns
        rec_db = MagicMock()
        # Only javstash.org was ever explicitly moved -- the other two were
        # never touched via the reordering UI.
        rec_db.get_endpoint_priorities.return_value = ["https://javstash.org/graphql"]
        rec_db.get_disabled_endpoints.return_value = []

        with patch.object(recognizer, "get_connection_manager", return_value=conn_mgr), \
             patch.object(recommendations_router, "get_rec_db", return_value=rec_db):
            domains = FaceRecognizer._endpoint_priority_domains(SimpleNamespace())

        assert domains == ["javstash.org", "stashdb.org", "theporndb.net"]

    def test_disabled_endpoint_excluded(self):
        conns = _connections(
            ("https://stashdb.org/graphql", "stashdb.org"),
            ("https://theporndb.net/graphql", "theporndb.net"),
        )
        conn_mgr = MagicMock()
        conn_mgr.get_connections.return_value = conns
        rec_db = MagicMock()
        rec_db.get_endpoint_priorities.return_value = []
        rec_db.get_disabled_endpoints.return_value = ["https://theporndb.net/graphql"]

        with patch.object(recognizer, "get_connection_manager", return_value=conn_mgr), \
             patch.object(recommendations_router, "get_rec_db", return_value=rec_db):
            domains = FaceRecognizer._endpoint_priority_domains(SimpleNamespace())

        assert domains == ["stashdb.org"]

    def test_exception_falls_back_to_empty_list(self):
        conn_mgr = MagicMock()
        conn_mgr.get_connections.side_effect = RuntimeError("boom")

        with patch.object(recognizer, "get_connection_manager", return_value=conn_mgr):
            domains = FaceRecognizer._endpoint_priority_domains(SimpleNamespace())

        assert domains == []

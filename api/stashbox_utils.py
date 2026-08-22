"""Shared StashBox client utilities.

Provides client creation and endpoint extraction used by both
the identification and stashbox routers.

Clients are sourced from the StashBoxConnectionManager which reads
endpoint config from Stash's settings API (auto-discovery).
"""

from typing import Optional

from stashbox_client import StashBoxClient
from stashbox_connection_manager import get_connection_manager


def _get_stashbox_client(endpoint_domain: str) -> Optional[StashBoxClient]:
    """Get a StashBox client for the given endpoint domain or URL.

    Reads from the StashBoxConnectionManager (backed by Stash config).
    """
    mgr = get_connection_manager()
    return mgr.get_client(endpoint_domain)


def _get_endpoint_url(endpoint_domain: str) -> Optional[str]:
    """Get the full endpoint URL for a domain.

    Reads from the StashBoxConnectionManager (backed by Stash config).
    """
    mgr = get_connection_manager()
    return mgr.get_endpoint_url(endpoint_domain)


def _extract_endpoint(universal_id: str | None) -> str | None:
    """Extract endpoint domain from universal_id (e.g. 'stashdb.org:uuid' -> 'stashdb.org')."""
    if universal_id and ":" in universal_id:
        return universal_id.split(":")[0]
    return None


def classify_universal_id(universal_id: str | None) -> str:
    """Classifies a universal_id's prefix into one of three shapes (see
    stash-sense2-data-gen's build/export_json.py, which is what actually
    produces these):
      - "stashbox":   "<endpoint-domain>:<stashbox_id>", e.g. "stashdb.org:0195...".
        A real stash-box performer -- prefix always contains a dot (every
        configured endpoint is a real domain).
      - "local":      "local:<performer_id>". The sidecar's own separate
        per-deployment local-performer-index feature.
      - "catalogue":  "<source_endpoint>:<performer_id>", e.g. "seekfans:4821".
        A performer discovered via a non-stash-box catalogue site -- no
        stashbox linkage, so no stashbox cover/metadata API to fall back
        on. Anything that isn't "local" and whose prefix has no dot lands
        here; a future catalogue source needs no change here, just its own
        handling wherever "catalogue" is branched on.
      - "unknown":    malformed or empty input.
    """
    endpoint = _extract_endpoint(universal_id)
    if not endpoint:
        return "unknown"
    if endpoint == "local":
        return "local"
    if "." in endpoint:
        return "stashbox"
    return "catalogue"

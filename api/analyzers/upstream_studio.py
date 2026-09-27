"""
Upstream Studio Changes Analyzer

Detects changes in stash-box linked studios by comparing upstream
data against local Stash data using 3-way diffing with stored snapshots.
"""

import logging
from typing import Optional

from .base_upstream import BaseUpstreamAnalyzer
from name_script import resolve_display_name
from stashbox_client import StashBoxClient
from upstream_field_mapper import (
    normalize_upstream_studio,
    diff_studio_fields,
    DEFAULT_STUDIO_FIELDS,
    STUDIO_FIELD_LABELS,
)

logger = logging.getLogger(__name__)


def _build_local_studio_data(studio: dict, endpoint: str = "") -> dict:
    """Extract comparable field values from a local Stash studio.

    For parent_studio, resolves the parent's stashbox ID for the given endpoint
    so it can be compared against the upstream stashbox UUID. If the parent isn't
    linked to this endpoint, parent_studio stays None — a numeric local ID can't
    be meaningfully compared against a stash-box UUID.
    """
    parent = studio.get("parent_studio")
    parent_studio_val = None
    parent_studio_name = None
    if parent and parent.get("id"):
        parent_studio_name = parent.get("name")
        # Resolve parent's stashbox ID for this endpoint
        for sid in parent.get("stash_ids", []):
            if sid["endpoint"] == endpoint:
                parent_studio_val = sid["stash_id"]
                break

    urls = studio.get("urls") or []
    if isinstance(urls, str):
        urls = [urls] if urls else []

    return {
        "name": studio.get("name"),
        "urls": urls,
        "parent_studio": parent_studio_val,
        "_parent_studio_name": parent_studio_name,
        # Not diffed (stash-box's own Studio type has no alias field to
        # compare against) -- purely for settings.py's prefer_western_names
        # display swap. See UpstreamStudioAnalyzer._build_recommendation_details.
        "_aliases": studio.get("aliases") or [],
    }


class UpstreamStudioAnalyzer(BaseUpstreamAnalyzer):
    """
    Detects upstream changes in stash-box linked studios.

    Compares studio data from configured stash-box endpoints against
    local Stash data, using stored snapshots for 3-way diffing.
    """

    type = "upstream_studio_changes"
    logic_version = 5  # v5: url (singular) → urls (array) field

    @property
    def entity_type(self) -> str:
        return "studio"

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        self._current_endpoint = endpoint
        return await self.stash.get_studios_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client: StashBoxClient, stashbox_id: str) -> Optional[dict]:
        return await stashbox_client.get_studio(stashbox_id)

    def _build_local_data(self, entity: dict) -> dict:
        return _build_local_studio_data(entity, self._current_endpoint)

    def _normalize_upstream(self, raw_data: dict) -> dict:
        return normalize_upstream_studio(raw_data)

    def _get_default_fields(self) -> set[str]:
        return DEFAULT_STUDIO_FIELDS

    def _get_field_labels(self) -> dict[str, str]:
        return STUDIO_FIELD_LABELS

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        changes = diff_studio_fields(local_data, upstream_data, snapshot, enabled_fields)

        # Suppress parent_studio changes when the local parent exists with the
        # same name as upstream — the parent is correct, just not linked to this
        # endpoint's stash_id. The resolution layer handles linking when needed.
        filtered = []
        for change in changes:
            if change["field"] == "parent_studio":
                local_name = (local_data.get("_parent_studio_name") or "").strip().lower()
                upstream_name = (upstream_data.get("_parent_studio_name") or "").strip().lower()
                if local_name and upstream_name and local_name == upstream_name:
                    continue
            filtered.append(change)
        return filtered

    def _build_recommendation_details(
        self, endpoint, endpoint_name, stash_box_id, local_entity, updated_at, changes,
    ) -> dict:
        """Same as the base implementation, plus settings.py's
        prefer_western_names: swaps studio_name (the recommendation's own
        card title/detail header) for this studio's own western alias
        when its real name isn't in Latin script. Only this display field
        swaps -- `changes`' own "Name" field diff (if any) still compares
        the real raw local/upstream names, so "did this actually change"
        stays correct. stash-box's Studio type has no alias field, so the
        only source of a western name here is this already-linked local
        studio's own aliases (see _build_local_studio_data's "_aliases")."""
        details = super()._build_recommendation_details(
            endpoint, endpoint_name, stash_box_id, local_entity, updated_at, changes,
        )
        try:
            from settings import get_setting
            prefer_western = bool(get_setting("prefer_western_names"))
        except RuntimeError:
            prefer_western = False
        if prefer_western:
            local_aliases = local_entity.get("aliases") or []
            display_name, original_name = resolve_display_name(
                str(local_entity["id"]), details["studio_name"],
                {str(local_entity["id"]): local_aliases}, prefer_western,
            )
            details["studio_name"] = display_name
            details["studio_name_original"] = original_name
        return details

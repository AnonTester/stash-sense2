"""Tests for identification_router.py's _process_sprite_frames() -- each
detected sprite-tile face must get its own distinct frame_index so its real
tile timestamp survives into frame_timestamps and can resolve a jump-to-
frame button, instead of every sprite face on a scene collapsing onto one
shared frame_index=-2 sentinel (only the single last-written timestamp for
that key would ever have been usable).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import identification_router as ir
from sprite_parser import SpriteFrame


class _NullAsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _fake_httpx_client(paths: dict):
    client = MagicMock()
    client.post = AsyncMock(return_value=SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"data": {"findScene": {"paths": paths}}},
    ))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestProcessSpriteFramesUniqueTimestamps:
    async def test_each_sprite_face_gets_a_distinct_frame_index_and_real_timestamp(self):
        sprite_frames = [
            SpriteFrame(image=object(), timestamp=5.0, index=0),
            SpriteFrame(image=object(), timestamp=15.0, index=1),
            SpriteFrame(image=object(), timestamp=25.0, index=2),
        ]
        faces_per_tile = [
            [SimpleNamespace(bbox={"w": 100, "h": 100}, confidence=0.9, yaw=0.0)],
            [SimpleNamespace(bbox={"w": 100, "h": 100}, confidence=0.9, yaw=0.0)],
            [SimpleNamespace(bbox={"w": 100, "h": 100}, confidence=0.9, yaw=0.0)],
        ]

        recognizer = MagicMock()
        recognizer.detect_faces_parallel.return_value = faces_per_tile
        recognizer.generator.get_embeddings_batch.return_value = [
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
            SimpleNamespace(embedding=[0.5, 0.6]),
        ]
        recognizer.recognize_face_v2.return_value = ([], None, None)

        with patch.object(ir, "httpx") as mock_httpx, \
                patch.object(ir, "fetch_sprite_from_stash", AsyncMock(return_value=sprite_frames)), \
                patch.object(ir, "_gpu_compute_lock", _NullAsyncCM), \
                patch.object(ir, "_recognizer", recognizer):
            mock_httpx.AsyncClient.return_value = _fake_httpx_client(
                {"sprite": "http://stash/scene/abc_sprite.jpg", "vtt": "http://stash/scene/abc_vtt.vtt"}
            )

            result = await ir._process_sprite_frames(
                "http://stash", "42", "apikey",
                min_face_size=50, min_face_confidence=0.5,
                match_config=object(), t_start=0.0,
            )

        assert result is not None
        extra_results, cache_rows = result

        frame_indices = [fi for fi, _ in extra_results]
        assert len(frame_indices) == 3
        assert len(set(frame_indices)) == 3, "sprite faces must not share one frame_index"
        assert all(fi <= -2 for fi in frame_indices), "sprite indices stay in the negative range below the -1 screenshot sentinel"

        timestamps_by_index = {row["frame_index"]: row["timestamp_sec"] for row in cache_rows}
        assert set(timestamps_by_index.values()) == {5.0, 15.0, 25.0}
        # cache_rows' frame_index values must match extra_results' 1:1 --
        # this is what lets _identify_scene_from_cache's later
        # reconstruction resolve the same timestamps a fresh run would.
        assert set(timestamps_by_index.keys()) == set(frame_indices)

    async def test_single_sprite_face_still_gets_a_real_timestamp(self):
        sprite_frames = [SpriteFrame(image=object(), timestamp=42.0, index=0)]
        faces_per_tile = [[SimpleNamespace(bbox={"w": 100, "h": 100}, confidence=0.9, yaw=0.0)]]

        recognizer = MagicMock()
        recognizer.detect_faces_parallel.return_value = faces_per_tile
        recognizer.generator.get_embeddings_batch.return_value = [SimpleNamespace(embedding=[0.1, 0.2])]
        recognizer.recognize_face_v2.return_value = ([], None, None)

        with patch.object(ir, "httpx") as mock_httpx, \
                patch.object(ir, "fetch_sprite_from_stash", AsyncMock(return_value=sprite_frames)), \
                patch.object(ir, "_gpu_compute_lock", _NullAsyncCM), \
                patch.object(ir, "_recognizer", recognizer):
            mock_httpx.AsyncClient.return_value = _fake_httpx_client(
                {"sprite": "http://stash/scene/abc_sprite.jpg", "vtt": "http://stash/scene/abc_vtt.vtt"}
            )

            extra_results, cache_rows = await ir._process_sprite_frames(
                "http://stash", "42", "apikey",
                min_face_size=50, min_face_confidence=0.5,
                match_config=object(), t_start=0.0,
            )

        assert cache_rows[0]["timestamp_sec"] == 42.0
        assert extra_results[0][0] == cache_rows[0]["frame_index"]

"""Tests for identification_router.py's _process_sprite_frames() -- each
detected sprite-tile face must get its own distinct frame_index so its real
tile timestamp survives into frame_timestamps and can resolve a jump-to-
frame button, instead of every sprite face on a scene collapsing onto one
shared frame_index=-2 sentinel (only the single last-written timestamp for
that key would ever have been usable).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import identification_router as ir
from sprite_parser import SpriteFrame

# _process_sprite_frames reads .image.shape to size the detector per-scene
# (see set_det_size_for_dims) -- a real (if tiny) array stands in for a
# genuine sprite tile crop; these tests otherwise don't care about pixel
# content, only frame_index/timestamp behavior.
_FAKE_TILE = np.zeros((90, 160, 3), dtype=np.uint8)


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
            SpriteFrame(image=_FAKE_TILE, timestamp=5.0, index=0),
            SpriteFrame(image=_FAKE_TILE, timestamp=15.0, index=1),
            SpriteFrame(image=_FAKE_TILE, timestamp=25.0, index=2),
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
        sprite_frames = [SpriteFrame(image=_FAKE_TILE, timestamp=42.0, index=0)]
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


class TestProcessSpriteFramesDetSize:
    """_process_sprite_frames sizes the detector to this scene's actual
    sprite-tile dimensions (see recognizer.set_det_size_for_dims's own
    docstring for why -- ~3x faster and slightly more accurate than the
    fixed default in a 500-scene production benchmark) and must restore it
    afterward, since det_size is process-wide shared state that a real
    video-frame identify running next would otherwise silently inherit."""

    async def test_sizes_detector_to_max_tile_dims_and_restores_after(self):
        # A shorter trailing tile (duration doesn't divide evenly into the
        # sprite grid) must not shrink the canvas below what the other
        # tiles need -- set_det_size_for_dims should see the batch max.
        sprite_frames = [
            SpriteFrame(image=np.zeros((90, 160, 3), dtype=np.uint8), timestamp=5.0, index=0),
            SpriteFrame(image=np.zeros((60, 160, 3), dtype=np.uint8), timestamp=15.0, index=1),
        ]
        faces_per_tile = [[], []]

        recognizer = MagicMock()
        call_order = []
        recognizer.set_det_size_for_dims.side_effect = lambda w, h: call_order.append(("set", w, h))
        recognizer.reset_det_size.side_effect = lambda: call_order.append(("reset",))
        recognizer.detect_faces_parallel.side_effect = lambda *a, **k: (
            call_order.append(("detect",)) or faces_per_tile
        )

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

        assert result == ([], [])
        recognizer.set_det_size_for_dims.assert_called_once_with(160, 90)
        recognizer.reset_det_size.assert_called_once()
        assert call_order == [("set", 160, 90), ("detect",), ("reset",)]

    async def test_restores_det_size_even_when_detection_raises(self):
        sprite_frames = [SpriteFrame(image=np.zeros((90, 160, 3), dtype=np.uint8), timestamp=5.0, index=0)]

        recognizer = MagicMock()
        recognizer.detect_faces_parallel.side_effect = RuntimeError("MIOPEN failure")

        with patch.object(ir, "httpx") as mock_httpx, \
                patch.object(ir, "fetch_sprite_from_stash", AsyncMock(return_value=sprite_frames)), \
                patch.object(ir, "_gpu_compute_lock", _NullAsyncCM), \
                patch.object(ir, "_recognizer", recognizer):
            mock_httpx.AsyncClient.return_value = _fake_httpx_client(
                {"sprite": "http://stash/scene/abc_sprite.jpg", "vtt": "http://stash/scene/abc_vtt.vtt"}
            )

            try:
                await ir._process_sprite_frames(
                    "http://stash", "42", "apikey",
                    min_face_size=50, min_face_confidence=0.5,
                    match_config=object(), t_start=0.0,
                )
                assert False, "expected the RuntimeError to propagate"
            except RuntimeError:
                pass

        recognizer.reset_det_size.assert_called_once()

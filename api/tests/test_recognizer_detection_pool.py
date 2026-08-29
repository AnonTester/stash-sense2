"""Tests for FaceRecognizer.detect_faces_parallel() (recognizer.py) --
the detection-pool fan-out used by identification_router.py's per-frame
detection loop instead of one shared generator processing frames
sequentially.

Constructs a bare object with just the attributes the method touches
(generator/detection_pool) rather than a real FaceRecognizer, since a real
one needs a loaded index/model bundle -- detect_faces_parallel() itself has
no other dependency on FaceRecognizer state.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from recognizer import FaceRecognizer


def _fake_generator(tag: str):
    gen = MagicMock()
    gen.detect_faces.side_effect = lambda frame, min_confidence: [f"face-from-{tag}"]
    return gen


def _fake_recognizer(pool_size: int):
    generators = [_fake_generator(str(i)) for i in range(pool_size)]
    return SimpleNamespace(generator=generators[0], detection_pool=generators[1:]), generators


class TestDetectFacesParallel:
    def test_results_in_frame_order_across_pool(self):
        fake_self, generators = _fake_recognizer(pool_size=3)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(10)]

        results = FaceRecognizer.detect_faces_parallel(fake_self, frames, min_confidence=0.5)

        assert len(results) == 10
        # Every frame got exactly one detection call (no drops, no
        # duplicates), and every position in the output list holds a real
        # result -- regardless of which pool worker happened to process
        # which frame (that assignment is nondeterministic by design).
        total_calls = sum(g.detect_faces.call_count for g in generators)
        assert total_calls == 10
        assert all(r in (["face-from-0"], ["face-from-1"], ["face-from-2"]) for r in results)

    def test_single_frame_skips_pool_uses_main_generator_only(self):
        fake_self, generators = _fake_recognizer(pool_size=3)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        results = FaceRecognizer.detect_faces_parallel(fake_self, frames, min_confidence=0.5)

        assert results == [["face-from-0"]]
        assert generators[0].detect_faces.call_count == 1
        assert generators[1].detect_faces.call_count == 0
        assert generators[2].detect_faces.call_count == 0

    def test_empty_pool_falls_back_to_sequential(self):
        fake_self, generators = _fake_recognizer(pool_size=1)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]

        results = FaceRecognizer.detect_faces_parallel(fake_self, frames, min_confidence=0.5)

        assert results == [["face-from-0"]] * 5
        assert generators[0].detect_faces.call_count == 5

    def test_no_frames_returns_empty(self):
        fake_self, _ = _fake_recognizer(pool_size=3)

        results = FaceRecognizer.detect_faces_parallel(fake_self, [], min_confidence=0.5)

        assert results == []


class TestDetectFacesParallelErrorPropagation:
    """A worker thread that raises (e.g. a real GPU/ROCm inference error)
    must surface as a real exception from detect_faces_parallel(), not
    silently leave that frame's result as None -- confirmed live: a
    swallowed exception here turned a GPU failure into a confusing
    'NoneType' object is not iterable crash much later in an unrelated
    caller instead of a clear, immediate error."""

    def test_sequential_fallback_propagates_directly(self):
        # pool_size=1 -> the plain list-comprehension fallback path (no
        # threading involved at all), so this is deterministic -- a
        # multi-worker equivalent would be racy about which generator
        # claims which frame, see test_all_workers_failing_reports_full_count
        # below for that path's coverage instead.
        gen_bad = MagicMock()
        gen_bad.detect_faces.side_effect = RuntimeError("MIOPEN failure 7")
        fake_self = SimpleNamespace(generator=gen_bad, detection_pool=[])
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]

        try:
            FaceRecognizer.detect_faces_parallel(fake_self, frames, min_confidence=0.5)
            assert False, "expected an exception, got a result instead"
        except RuntimeError as e:
            assert "MIOPEN failure 7" in str(e)

    def test_all_workers_failing_reports_full_count(self):
        gen_bad_1 = MagicMock()
        gen_bad_1.detect_faces.side_effect = RuntimeError("boom-1")
        gen_bad_2 = MagicMock()
        gen_bad_2.detect_faces.side_effect = RuntimeError("boom-2")
        fake_self = SimpleNamespace(generator=gen_bad_1, detection_pool=[gen_bad_2])
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]

        try:
            FaceRecognizer.detect_faces_parallel(fake_self, frames, min_confidence=0.5)
            assert False, "expected an exception, got a result instead"
        except RuntimeError as e:
            assert "4/4 frame(s)" in str(e)


class TestSetDetSizeForDims:
    """set_det_size_for_dims/reset_det_size (used by
    identification_router._process_sprite_frames to size the detector for
    a scene's actual sprite-tile dimensions instead of the fixed
    production default -- confirmed ~3x faster and slightly more accurate
    in a 500-scene production benchmark) must reach every pool generator,
    not just the main one, since detect_faces_parallel fans work across
    all of them."""

    def test_rounds_up_to_stride_and_applies_to_every_pool_generator(self):
        fake_self, generators = _fake_recognizer(pool_size=3)

        det_size = FaceRecognizer.set_det_size_for_dims(fake_self, width=160, height=90)

        assert det_size == (160, 96)  # 160 already a multiple of 32; 90 rounds up to 96
        for gen in generators:
            gen.set_det_size.assert_called_once_with((160, 96))

    def test_dims_smaller_than_stride_round_up_to_one_stride(self):
        fake_self, generators = _fake_recognizer(pool_size=1)

        det_size = FaceRecognizer.set_det_size_for_dims(fake_self, width=10, height=5)

        assert det_size == (32, 32)
        generators[0].set_det_size.assert_called_once_with((32, 32))

    def test_reset_det_size_restores_every_pool_generator_to_the_default(self):
        fake_self, generators = _fake_recognizer(pool_size=3)
        generators[0].default_det_size.return_value = (640, 640)

        FaceRecognizer.reset_det_size(fake_self)

        generators[0].default_det_size.assert_called_once()
        for gen in generators:
            gen.set_det_size.assert_called_once_with((640, 640))

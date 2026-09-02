"""Tests for embeddings.py -- buffalo_l detection+embedding wrapper.

buffalo_l does its own internal face alignment (see embeddings.py's module
docstring for why that isn't reimplemented standalone here), so unlike the
legacy FaceNet512/ArcFace pipeline there's no separate alignment step to
unit-test. These tests instead cover what this module actually owns:
image loading, device/provider selection, and the detect_faces() filtering/
yaw-estimation logic around InsightFace's `FaceAnalysis.get()` call.
"""
import asyncio
import io
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from embeddings import (
    DetectedFace,
    FaceEmbedding,
    FaceEmbeddingGenerator,
    GPU_COMPUTE_LOCK,
    MAX_ROLL_CORRECTION_ATTEMPTS,
    gpu_compute_lock,
    load_image,
    load_image_from_path,
)


def _mock_face(det_score=0.9, bbox=(10, 10, 60, 60), kps=None, embedding_dim=512):
    return SimpleNamespace(
        det_score=det_score,
        bbox=np.array(bbox, dtype=np.float32),
        kps=kps,
        normed_embedding=np.ones(embedding_dim, dtype=np.float32),
    )


def _generator_with_mock_analyzer(get_return):
    """A FaceEmbeddingGenerator whose face_analyzer.get() is pre-mocked,
    bypassing the real lazy-load (_face_analyzer set directly)."""
    generator = FaceEmbeddingGenerator(device="cpu")
    generator._face_analyzer = Mock(get=Mock(return_value=get_return))
    return generator


class TestLoadImage:
    def _png_bytes(self, mode="RGB", size=(8, 6)):
        buf = io.BytesIO()
        Image.new(mode, size).save(buf, format="PNG")
        return buf.getvalue()

    def test_load_image_returns_rgb_array(self):
        arr = load_image(self._png_bytes(mode="RGB", size=(8, 6)))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (6, 8, 3)

    def test_load_image_converts_non_rgb_to_rgb(self):
        # "L" (grayscale) and "RGBA" both need conversion to RGB
        arr = load_image(self._png_bytes(mode="L", size=(4, 4)))
        assert arr.shape == (4, 4, 3)

        arr_rgba = load_image(self._png_bytes(mode="RGBA", size=(4, 4)))
        assert arr_rgba.shape == (4, 4, 3)

    def test_load_image_from_path(self, tmp_path):
        path = tmp_path / "test.png"
        Image.new("RGB", (5, 5)).save(path)
        arr = load_image_from_path(str(path))
        assert arr.shape == (5, 5, 3)


class TestDeviceSelection:
    def test_auto_selects_gpu_when_rocm_available(self, monkeypatch):
        monkeypatch.setattr(
            sys.modules["onnxruntime"], "get_available_providers",
            lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"],
        )
        generator = FaceEmbeddingGenerator()
        assert generator.device == "gpu"

    def test_auto_selects_gpu_when_cuda_available(self, monkeypatch):
        monkeypatch.setattr(
            sys.modules["onnxruntime"], "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        generator = FaceEmbeddingGenerator()
        assert generator.device == "gpu"

    def test_auto_selects_cpu_when_no_gpu_provider(self, monkeypatch):
        monkeypatch.setattr(
            sys.modules["onnxruntime"], "get_available_providers",
            lambda: ["CPUExecutionProvider"],
        )
        generator = FaceEmbeddingGenerator()
        assert generator.device == "cpu"

    def test_explicit_device_skips_autodetect(self):
        generator = FaceEmbeddingGenerator(device="cpu")
        assert generator.device == "cpu"

    def test_gpu_enabled_setting_false_forces_cpu_even_with_gpu_provider(self, monkeypatch):
        monkeypatch.setattr(
            sys.modules["onnxruntime"], "get_available_providers",
            lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"],
        )
        import settings
        monkeypatch.setattr(settings, "get_setting", lambda key: False)
        generator = FaceEmbeddingGenerator()
        assert generator.device == "cpu"

    def test_gpu_enabled_setting_true_allows_gpu_autodetect(self, monkeypatch):
        monkeypatch.setattr(
            sys.modules["onnxruntime"], "get_available_providers",
            lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"],
        )
        import settings
        monkeypatch.setattr(settings, "get_setting", lambda key: True)
        generator = FaceEmbeddingGenerator()
        assert generator.device == "gpu"

    def test_settings_not_initialized_falls_back_to_hardware_autodetect(self, monkeypatch):
        """Standalone scripts/tests don't call init_settings() -- effective_device()
        must not blow up, and should behave like gpu_enabled=True (today's
        pre-fix behavior) rather than silently forcing CPU."""
        monkeypatch.setattr(
            sys.modules["onnxruntime"], "get_available_providers",
            lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"],
        )
        import settings
        def _raise(key):
            raise RuntimeError("Settings not initialized. Call init_settings() during startup.")
        monkeypatch.setattr(settings, "get_setting", _raise)
        generator = FaceEmbeddingGenerator()
        assert generator.device == "gpu"


class TestOrtProviders:
    def test_gpu_device_lists_gpu_providers_before_cpu_fallback(self):
        generator = FaceEmbeddingGenerator(device="gpu")
        providers = generator._ort_providers()
        assert providers[-1] == "CPUExecutionProvider"
        assert "ROCMExecutionProvider" in providers
        assert "CUDAExecutionProvider" in providers

    def test_cpu_device_lists_only_cpu(self):
        generator = FaceEmbeddingGenerator(device="cpu")
        assert generator._ort_providers() == ["CPUExecutionProvider"]


class TestDetectFaces:
    def test_returns_detected_face_per_result(self):
        generator = _generator_with_mock_analyzer([_mock_face()])
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert len(faces) == 1
        face = faces[0]
        assert isinstance(face, DetectedFace)
        assert face.confidence == pytest.approx(0.9)
        # rotation_applied: 0.0 -- no roll correction fired (this mock's
        # kps=None gives _face_roll_degrees() nothing to measure), see
        # TestRollCorrection below for the correction path itself.
        assert face.bbox == {"x": 10, "y": 10, "w": 50, "h": 50, "rotation_applied": 0.0}
        assert face.embedding.shape == (512,)

    def test_filters_faces_below_min_confidence(self):
        generator = _generator_with_mock_analyzer([
            _mock_face(det_score=0.3),
            _mock_face(det_score=0.9),
        ])
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image, min_confidence=0.5)

        assert len(faces) == 1
        assert faces[0].confidence == pytest.approx(0.9)

    def test_skips_crops_too_small(self):
        # bbox (0,0,5,5) -> 5x5 crop, below the 10px minimum in both dims
        generator = _generator_with_mock_analyzer([_mock_face(bbox=(0, 0, 5, 5))])
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert faces == []

    def test_clamps_bbox_to_image_bounds(self):
        # bbox extends past the 100x100 image on both edges
        generator = _generator_with_mock_analyzer([_mock_face(bbox=(-20, -20, 150, 150))])
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert len(faces) == 1
        assert faces[0].image.shape[:2] == (100, 100)

    def test_estimates_yaw_from_landmarks(self):
        # Nose centered between the eyes -> ~0 degrees yaw
        kps = np.array([
            [40.0, 40.0],  # left eye
            [60.0, 40.0],  # right eye
            [50.0, 55.0],  # nose (centered)
            [42.0, 70.0],
            [58.0, 70.0],
        ], dtype=np.float32)
        generator = _generator_with_mock_analyzer([_mock_face(kps=kps)])
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert faces[0].yaw == pytest.approx(0.0, abs=1.0)
        assert faces[0].landmarks is kps

    def test_yaw_is_none_without_landmarks(self):
        generator = _generator_with_mock_analyzer([_mock_face(kps=None)])
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert faces[0].yaw is None


def _rolled_kps():
    """~90 degree roll: mouth corners to the SIDE of the eyes rather than
    below them (eye-center (50,50) -> mouth-center (80,50), a horizontal
    eye-to-mouth vector)."""
    return np.array([
        [50.0, 35.0],  # left eye
        [50.0, 65.0],  # right eye
        [65.0, 50.0],  # nose
        [80.0, 35.0],  # mouth left
        [80.0, 65.0],  # mouth right
    ], dtype=np.float32)


def _upright_kps():
    """~0 degree roll: mouth corners below the eyes (same shape
    test_estimates_yaw_from_landmarks above already uses)."""
    return np.array([
        [40.0, 40.0], [60.0, 40.0], [50.0, 55.0], [42.0, 70.0], [58.0, 70.0],
    ], dtype=np.float32)


class TestRollCorrection:
    """detect_faces()'s in-plane rotation correction -- see that method's
    own docstring. These use side_effect (not the module-level
    _generator_with_mock_analyzer helper, which returns a fixed value on
    every call) since the whole point under test is what happens on the
    SECOND face_analyzer.get() call, after a rotated re-detection."""

    def test_corrects_large_roll_and_replaces_results(self):
        generator = FaceEmbeddingGenerator(device="cpu")
        rolled = _mock_face(det_score=0.7, kps=_rolled_kps())
        corrected = _mock_face(det_score=0.9, kps=_upright_kps())
        mock_analyzer = Mock(get=Mock(side_effect=[[rolled], [corrected]]))
        generator._face_analyzer = mock_analyzer
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert mock_analyzer.get.call_count == 2
        assert len(faces) == 1
        # The corrected pass's face, not the original rolled one.
        assert faces[0].confidence == pytest.approx(0.9)
        assert faces[0].bbox["rotation_applied"] == pytest.approx(-90.0, abs=1.0)

    def test_small_roll_does_not_trigger_a_second_detection_pass(self):
        generator = FaceEmbeddingGenerator(device="cpu")
        upright = _mock_face(det_score=0.9, kps=_upright_kps())
        mock_analyzer = Mock(get=Mock(return_value=[upright]))
        generator._face_analyzer = mock_analyzer
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert mock_analyzer.get.call_count == 1
        assert faces[0].bbox["rotation_applied"] == 0.0

    def test_falls_back_to_original_when_correction_finds_no_face(self):
        generator = FaceEmbeddingGenerator(device="cpu")
        rolled = _mock_face(det_score=0.7, kps=_rolled_kps())
        mock_analyzer = Mock(get=Mock(side_effect=[[rolled], []]))
        generator._face_analyzer = mock_analyzer
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert len(faces) == 1
        assert faces[0].confidence == pytest.approx(0.7)
        assert faces[0].bbox["rotation_applied"] == 0.0

    def test_falls_back_to_original_after_exhausting_all_attempts(self):
        # A rotation-correction pass that NEVER converges, across every
        # attempt the budget allows -- 1 initial call + MAX_ROLL_
        # CORRECTION_ATTEMPTS retries, all still rolled.
        generator = FaceEmbeddingGenerator(device="cpu")
        rolled = _mock_face(det_score=0.7, kps=_rolled_kps())
        still_rolled = _mock_face(det_score=0.5, kps=_rolled_kps())
        mock_analyzer = Mock(get=Mock(
            side_effect=[[rolled]] + [[still_rolled]] * MAX_ROLL_CORRECTION_ATTEMPTS
        ))
        generator._face_analyzer = mock_analyzer
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert mock_analyzer.get.call_count == 1 + MAX_ROLL_CORRECTION_ATTEMPTS
        assert len(faces) == 1
        # The ORIGINAL rolled face (0.7), not any of the still-bad
        # "corrected" attempts (0.5) -- a correction that never actually
        # converges within budget is discarded entirely.
        assert faces[0].confidence == pytest.approx(0.7)
        assert faces[0].bbox["rotation_applied"] == 0.0

    def test_second_attempt_converges_after_first_does_not(self):
        # Regression test for the real bug this iteration was added for:
        # a tightly-cropped ("just the head," not the full frame) upside-
        # down photo whose roll estimate was noisy enough that ONE
        # correction pass wasn't enough (-142 degrees -> -49 residual,
        # still outside ROLL_RESIDUAL_OK_DEG) -- a single-pass version
        # fell back to the original, badly-rotated detection and matched
        # the wrong performer. A second pass (correcting that -49 degree
        # residual) converged. This confirms the loop actually retries
        # instead of giving up after the first attempt.
        generator = FaceEmbeddingGenerator(device="cpu")
        initial = _mock_face(det_score=0.6, kps=_rolled_kps())
        still_rolled = _mock_face(det_score=0.65, kps=_rolled_kps())
        converged = _mock_face(det_score=0.9, kps=_upright_kps())
        mock_analyzer = Mock(get=Mock(side_effect=[[initial], [still_rolled], [converged]]))
        generator._face_analyzer = mock_analyzer
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = generator.detect_faces(image)

        assert mock_analyzer.get.call_count == 3
        assert len(faces) == 1
        assert faces[0].confidence == pytest.approx(0.9)
        # Both corrections accumulated (90 + 90 = 180), not just the last one.
        assert faces[0].bbox["rotation_applied"] == pytest.approx(-180.0, abs=1.0)


class TestGetEmbeddings:
    def test_get_embedding_reads_back_precomputed_embedding(self):
        generator = FaceEmbeddingGenerator(device="cpu")
        vec = np.arange(512, dtype=np.float32)
        face = DetectedFace(image=np.zeros((10, 10, 3)), bbox={}, confidence=0.9, embedding=vec)

        result = generator.get_embedding(face)

        assert isinstance(result, FaceEmbedding)
        assert np.array_equal(result.embedding, vec)

    def test_get_embeddings_batch_preserves_order(self):
        generator = FaceEmbeddingGenerator(device="cpu")
        faces = [
            DetectedFace(image=np.zeros((10, 10, 3)), bbox={}, confidence=0.9, embedding=np.full(512, i, dtype=np.float32))
            for i in range(3)
        ]

        results = generator.get_embeddings_batch(faces)

        assert [r.embedding[0] for r in results] == [0.0, 1.0, 2.0]


class TestGpuComputeLock:
    """GPU_COMPUTE_LOCK/gpu_compute_lock() -- serializes actual GPU
    inference across every caller in the process (identification_router.py's
    live endpoints, local_performer_sync_job.py's raw-thread worker,
    local_performer_index.py's hook handler). See embeddings.py's own
    docstring for why this exists (confirmed live: concurrent inference
    sessions produced real ROCm/MIOpen failures on the reference GPU) and
    why it's a plain threading.Lock rather than asyncio.Lock (shared with a
    raw-OS-thread caller, which can't use an asyncio.Lock safely)."""

    def test_is_a_plain_threading_lock_not_asyncio_lock(self):
        import threading
        assert isinstance(GPU_COMPUTE_LOCK, type(threading.Lock()))

    def test_raw_thread_can_acquire_and_release_directly(self):
        # local_performer_sync_job.py's _embed_worker does exactly this --
        # no event loop involved.
        acquired = GPU_COMPUTE_LOCK.acquire(timeout=1)
        try:
            assert acquired is True
        finally:
            GPU_COMPUTE_LOCK.release()

    @pytest.mark.asyncio
    async def test_async_wrapper_serializes_two_concurrent_callers(self):
        order = []

        async def _hold(name, delay):
            async with gpu_compute_lock():
                order.append(f"{name}-start")
                await asyncio.sleep(delay)
                order.append(f"{name}-end")

        await asyncio.gather(_hold("a", 0.02), _hold("b", 0.0))

        # Whichever ran first, it must fully finish (both -start and -end)
        # before the other one's -start -- true mutual exclusion, not just
        # interleaved-but-eventually-both-run.
        assert order in (
            ["a-start", "a-end", "b-start", "b-end"],
            ["b-start", "b-end", "a-start", "a-end"],
        )

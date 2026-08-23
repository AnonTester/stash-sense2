"""Tests for embeddings.py -- buffalo_l detection+embedding wrapper.

buffalo_l does its own internal face alignment (see embeddings.py's module
docstring for why that isn't reimplemented standalone here), so unlike the
legacy FaceNet512/ArcFace pipeline there's no separate alignment step to
unit-test. These tests instead cover what this module actually owns:
image loading, device/provider selection, and the detect_faces() filtering/
yaw-estimation logic around InsightFace's `FaceAnalysis.get()` call.
"""
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
        assert face.bbox == {"x": 10, "y": 10, "w": 50, "h": 50}
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

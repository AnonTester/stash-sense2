"""Tests for capability detection."""

import pytest

from capabilities import detect_capabilities


# -- File lists mirroring capabilities.py constants --

FACE_DATA_FILES = [
    "face_embeddings.usearch",
    "faces.json",
    "performers.json",
]

FACE_MODEL_FILES = [
    "models/buffalo_l/det_10g.onnx",
    "models/buffalo_l/w600k_r50.onnx",
]


def _create_files(directory, filenames):
    """Create empty sentinel files in *directory*, creating parent dirs as needed."""
    for name in filenames:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


class TestFreshInstall:
    """Empty directories -- only lightweight capabilities available."""

    def test_upstream_sync_always_true(self, data_dir, models_dir):
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["upstream_sync"] is True

    def test_duplicate_detection_always_true(self, data_dir, models_dir):
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["duplicate_detection_basic"] is True

    def test_identification_disabled(self, data_dir, models_dir):
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False


class TestWithFaceDataAndModels:
    """Face data + face models present -- identification enabled."""

    def test_identification_enabled(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is True


class TestFaceModelsButNoData:
    """Models installed but no data files -- identification disabled."""

    def test_identification_disabled(self, data_dir, models_dir):
        _create_files(models_dir, FACE_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False


class TestFaceDataButNoModels:
    """Data files imported but models not downloaded -- identification disabled."""

    def test_identification_disabled(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False


class TestPartialFaceData:
    """Some face data files missing -- identification disabled."""

    def test_missing_one_data_file(self, data_dir, models_dir):
        # Create all but one face data file
        _create_files(data_dir, FACE_DATA_FILES[:-1])
        _create_files(models_dir, FACE_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False

    def test_missing_one_model_file(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES[:1])
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False

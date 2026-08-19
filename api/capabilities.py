"""Capability detection based on installed models and data files.

Checks which features are available by looking for the required files
on disk. This avoids importing heavy ML libraries just to answer
"what can this instance do?"
"""

from pathlib import Path


# Face data files (shipped via database import)
_FACE_DATA_FILES = [
    "face_embeddings.usearch",
    "faces.json",
    "performers.json",
]

# Face recognition ONNX models (buffalo_l, downloaded by insightface's own
# model zoo mechanism into a buffalo_l/ subfolder under models_dir -- not
# the flat single-file layout the old FaceNet512/ArcFace models used)
_FACE_MODEL_FILES = [
    "models/buffalo_l/det_10g.onnx",
    "models/buffalo_l/w600k_r50.onnx",
]

# Tattoo detection ONNX models
_TATTOO_MODEL_FILES = [
    "tattoo_yolov5s.onnx",
    "tattoo_efficientnet_b0.onnx",
]

# Tattoo embedding data
_TATTOO_DATA_FILES = [
    "tattoo_embeddings.voy",
]


def _all_exist(directory: Path, filenames: list[str]) -> bool:
    """Return True if every file in *filenames* exists under *directory*."""
    return all((directory / f).is_file() for f in filenames)


def detect_capabilities(data_dir: Path, models_dir: Path) -> dict[str, bool]:
    """Detect available capabilities based on files on disk.

    Args:
        data_dir: Root data directory containing face/tattoo data files.
        models_dir: Directory containing ONNX model files.

    Returns:
        Dict mapping capability name to whether it is available.
    """
    has_face_data = _all_exist(data_dir, _FACE_DATA_FILES)
    has_face_models = _all_exist(models_dir, _FACE_MODEL_FILES)
    has_identification = has_face_data and has_face_models

    has_tattoo_models = _all_exist(models_dir, _TATTOO_MODEL_FILES)
    has_tattoo_data = _all_exist(data_dir, _TATTOO_DATA_FILES)
    has_tattoo_signal = has_identification and has_tattoo_models and has_tattoo_data

    return {
        "upstream_sync": True,
        "duplicate_detection_basic": True,
        "identification": has_identification,
        "tattoo_signal": has_tattoo_signal,
    }

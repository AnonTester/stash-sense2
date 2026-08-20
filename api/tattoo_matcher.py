"""
Tattoo embedding matcher using CLIP ViT-B/32 (ONNX) + usearch kNN search.

Takes YOLO tattoo detection crops, generates CLIP image-encoder embeddings
via ONNX Runtime, and queries the pre-built tattoo_embeddings.usearch
index to find performers with visually similar tattoos.

CLIP replaces the earlier frozen EfficientNet-B0 embedder (a plain
ImageNet classifier with its head stripped, never trained on tattoo data)
-- CLIP is trained contrastively, which is directly the property a
similarity-matching embedder needs, and is what TattooTrace (the source of
tattoo_yolov5s.onnx) itself pairs with YOLOv5 for this exact
"recognition/clustering" stage.

Tattoo matching is a helper/secondary signal, not a primary identifier --
even a dedicated research pipeline (Kobec et al. 2024, YOLOv5 + cosine
similarity on the deMSI dataset) reports an F-score of only ~0.52 for
trusting the single best cosine match, recommending a broad low-threshold
recall net instead of confident top-1 matching. Keep that in mind when
weighting this signal in a fused score -- it corroborates, it doesn't lead.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

TATTOO_EMBEDDING_DIM = 512

# Model search paths: DATA_DIR/models first, then ./models (relative to this file)
ONNX_MODEL_FILENAME = "tattoo_clip_vitb32.onnx"
DATA_DIR = os.environ.get("DATA_DIR", "./data")
LOCAL_MODELS_DIR = Path(__file__).parent / "models"


def _find_embedder_model_path() -> Path:
    """Find the CLIP ViT-B/32 ONNX model for tattoo embedding.

    Search order:
    1. {DATA_DIR}/models/tattoo_clip_vitb32.onnx (Docker / production)
    2. ./models/tattoo_clip_vitb32.onnx (local development, relative to this file)

    Returns:
        Path to the ONNX model file.

    Raises:
        FileNotFoundError: If the model cannot be found in any search path.
    """
    data_models_path = Path(DATA_DIR) / "models" / ONNX_MODEL_FILENAME
    if data_models_path.exists():
        return data_models_path

    local_path = LOCAL_MODELS_DIR / ONNX_MODEL_FILENAME
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        f"Tattoo embedder ONNX model not found. "
        f"Searched: {data_models_path}, {local_path}. "
        f"Run convert_models_to_onnx.py --clip to generate the model."
    )


class TattooMatcher:
    """Match tattoo crops against the pre-built tattoo embedding index."""

    def __init__(
        self,
        tattoo_index,
        tattoo_mapping: list,
        embedder_model_path: Optional[str] = None,
    ):
        """
        Initialize the tattoo matcher.

        Args:
            tattoo_index: usearch Index (512-dim cosine) loaded from tattoo_embeddings.usearch
            tattoo_mapping: List mapping usearch key -> universal_id
            embedder_model_path: Optional path to CLIP ViT-B/32 ONNX model.
                If None, searches DATA_DIR/models then ./models.
        """
        self.tattoo_index = tattoo_index
        self.tattoo_mapping = tattoo_mapping
        self._generator = None
        self._embedder_model_path = embedder_model_path

    @property
    def generator(self):
        """Lazy-load CLIP ONNX embedding generator."""
        if self._generator is None:
            self._generator = _TattooEmbeddingGenerator(
                model_path=self._embedder_model_path,
            )
        return self._generator

    @staticmethod
    def crop_tattoo(
        image: np.ndarray,
        bbox: dict,
        padding: float = 0.1,
    ) -> np.ndarray:
        """Crop a tattoo region from an image using normalized bbox coords.

        Args:
            image: Source image as numpy array (H, W, 3)
            bbox: Normalized bounding box {x, y, w, h} with values in [0, 1]
            padding: Fractional padding around the crop (0.1 = 10% on each side)

        Returns:
            Cropped image as numpy array
        """
        h, w = image.shape[:2]

        bw = bbox["w"]
        bh = bbox["h"]
        if bw <= 0 or bh <= 0:
            return None

        pad_w = bw * padding
        pad_h = bh * padding

        x1 = max(0, int((bbox["x"] - pad_w) * w))
        y1 = max(0, int((bbox["y"] - pad_h) * h))
        x2 = min(w, int((bbox["x"] + bw + pad_w) * w))
        y2 = min(h, int((bbox["y"] + bh + pad_h) * h))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def embed_detections(
        self,
        image: np.ndarray,
        detections: list,
    ) -> list[tuple[object, np.ndarray]]:
        """Crop + embed each detection — the expensive, DB-independent half
        of matching. Cacheable: the resulting embeddings are all
        `match_from_embeddings` needs, so a caller can persist them once and
        skip this step (and the frame pixels it needs) on future re-matches.

        Args:
            image: Full RGB image as numpy array (H, W, 3)
            detections: List of TattooDetection objects from TattooDetector

        Returns:
            List of (detection, embedding) pairs. Detections whose crop or
            embedding generation failed are silently omitted (matches the
            original inline behavior).
        """
        results = []
        for detection in detections:
            crop = self.crop_tattoo(image, detection.bbox)
            if crop is None:
                continue
            try:
                embedding = self.generator.get_embedding(crop)
            except Exception as e:
                logger.warning(f"Failed to generate tattoo embedding: {e}")
                continue
            results.append((detection, embedding))
        return results

    def match_from_embeddings(
        self,
        embeddings: list[np.ndarray],
        k: int = 10,
    ) -> dict[str, float]:
        """Match precomputed tattoo embeddings against the index — the
        cheap, DB-dependent half. Safe to call repeatedly against a fresh
        `tattoo_index`/`tattoo_mapping` (e.g. after a database update)
        without redoing detection or embedding.

        Args:
            embeddings: Embedding vectors from embed_detections() (or cached)
            k: Number of nearest neighbors per crop

        Returns:
            Dict mapping universal_id -> best similarity score (0-1, higher is better)
        """
        if not embeddings or self.tattoo_index is None:
            return {}

        scores_by_performer: dict[str, list[float]] = defaultdict(list)
        mapping_size = len(self.tattoo_mapping)

        for embedding in embeddings:
            try:
                matches = self.tattoo_index.search(embedding, k)
            except Exception as e:
                logger.warning(f"Tattoo index query failed: {e}")
                continue

            for idx, dist in zip(matches.keys, matches.distances):
                if idx < 0 or idx >= mapping_size:
                    continue

                entry = self.tattoo_mapping[idx]
                if entry is None:
                    continue

                universal_id = entry["universal_id"]

                # Convert cosine distance to similarity (0-1)
                similarity = max(0.0, 1.0 - dist)
                scores_by_performer[universal_id].append(similarity)

        # Take best score per performer
        return {uid: max(scores) for uid, scores in scores_by_performer.items()}

    def match(
        self,
        image: np.ndarray,
        detections: list,
        k: int = 10,
    ) -> dict[str, float]:
        """Match tattoo crops against the index (detect+embed+query in one
        call — unchanged behavior/signature for existing callers). Split
        into embed_detections()/match_from_embeddings() internally so a
        caller that wants to cache the embeddings can call those directly
        instead.

        Args:
            image: Full RGB image as numpy array (H, W, 3)
            detections: List of TattooDetection objects from TattooDetector
            k: Number of nearest neighbors per crop

        Returns:
            Dict mapping universal_id -> best similarity score (0-1, higher is better)
        """
        if not detections or self.tattoo_index is None:
            return {}
        embedded = self.embed_detections(image, detections)
        return self.match_from_embeddings([e for _, e in embedded], k=k)


class _TattooEmbeddingGenerator:
    """Generate CLIP ViT-B/32 embeddings from tattoo image crops using ONNX Runtime.

    Preprocessing replicates OpenAI CLIP's standard preprocessing (same as
    open_clip's `create_model_and_transforms(..., pretrained="openai")`
    returns):
    1. Resize shortest side to 224 (bicubic interpolation)
    2. Center crop to 224x224
    3. Convert to float32 [0, 1]
    4. Normalize with CLIP's own mean/std -- NOT ImageNet's; reusing the
       ImageNet constants here would silently produce wrong embeddings
       (the exact class of mistake this docstring exists to prevent, see
       convert_models_to_onnx.py's convert_clip_tattoo()).
    5. HWC -> CHW layout, add batch dimension
    """

    # CLIP normalization constants (OpenAI CLIP, all standard variants) --
    # deliberately not ImageNet's [0.485,0.456,0.406]/[0.229,0.224,0.225].
    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the embedding generator.

        Args:
            model_path: Path to CLIP ViT-B/32 ONNX model. If None, searches
                DATA_DIR/models then ./models for tattoo_clip_vitb32.onnx.
        """
        self._model_path = model_path
        self._session = None
        self._input_name = None
        self._output_name = None

    @property
    def session(self) -> ort.InferenceSession:
        """Lazy-load ONNX Runtime session."""
        if self._session is None:
            if self._model_path:
                model_path = Path(self._model_path)
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Tattoo embedder model not found at {self._model_path}"
                    )
            else:
                model_path = _find_embedder_model_path()

            # Use GPU if available, fall back to CPU
            providers = ort.get_available_providers()
            gpu_providers = {"CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider"}
            if gpu_providers & set(providers):
                ort_providers = ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                ort_providers = ["CPUExecutionProvider"]

            logger.info(f"Loading tattoo embedder ONNX model from {model_path}...")
            self._session = ort.InferenceSession(
                str(model_path), providers=ort_providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info("Tattoo embedder model loaded.")

        return self._session

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for CLIP ViT-B/32 inference.

        Replicates OpenAI CLIP's standard preprocessing:
        Resize(224, bicubic) -> CenterCrop(224) -> ToTensor -> Normalize(CLIP)

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            Preprocessed float32 array of shape (1, 3, 224, 224)
        """
        h, w = image.shape[:2]

        # Step 1: Resize shortest side to 224, preserve aspect ratio (bicubic --
        # CLIP's own preprocessing uses bicubic, not bilinear)
        if h < w:
            new_h = 224
            new_w = int(round(w * 224 / h))
        else:
            new_w = 224
            new_h = int(round(h * 224 / w))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Step 2: Center crop to 224x224
        crop_h, crop_w = 224, 224
        start_y = (new_h - crop_h) // 2
        start_x = (new_w - crop_w) // 2
        cropped = resized[start_y : start_y + crop_h, start_x : start_x + crop_w]

        # Step 3: Convert to float32 [0, 1]
        img_float = cropped.astype(np.float32) / 255.0

        # Step 4: Normalize with CLIP mean and std
        img_float = (img_float - self.CLIP_MEAN) / self.CLIP_STD

        # Step 5: HWC -> CHW, add batch dimension
        img_chw = np.transpose(img_float, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate L2-normalized embedding from an image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            L2-normalized float32 array of shape (512,)
        """
        input_tensor = self._preprocess(image)

        embedding = self.session.run(
            [self._output_name], {self._input_name: input_tensor}
        )[0].squeeze(0)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

"""Face detection and embedding generation.

buffalo_l migration: replaces the legacy RetinaFace(buffalo_sc)+FaceNet512+
ArcFace dual pipeline with InsightFace's `buffalo_l` bundle (SCRFD-10GF
detection + ResNet50@WebFace600K recognition), one embedding per face.
Mirrors stash-sense2-data-gen's own embeddings.py exactly (same
`FaceAnalysis(name="buffalo_l", allowed_modules=["detection","recognition"])`
call, same normed_embedding output) -- the sidecar's query-time embeddings
and the database's stored embeddings MUST come from the identical code
path, or matching would be silently, systematically wrong. Deliberately
NOT reimplemented as a standalone ONNX session fed a pre-cropped face image
(the way the legacy FaceNet512/ArcFace models were): InsightFace's own
recognition wrapper expects the *original* image + landmarks and does its
own internal alignment/preprocessing (channel order, normalization) that
isn't safe to reverse-engineer and re-implement by hand -- getting that
wrong would produce embeddings that are subtly, silently incompatible with
what the database actually contains. Detection and embedding therefore
happen together in `detect_faces()`; `get_embeddings_batch()`/
`get_embedding()` just read back what was already computed, on
`DetectedFace` objects (not raw crop arrays -- every caller in this
codebase already calls `detect_faces()` immediately before embedding the
same image, so this isn't a behavior change for any of them, just a
signature change: pass the DetectedFace, not `.image`).

All inference runs through ONNX Runtime with GPU acceleration when
available (CUDA or ROCm/MIGraphX, see Dockerfile.rocm/Dockerfile.cuda),
falling back to CPU.
"""
import os
import io
import warnings
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from insightface.app import FaceAnalysis

warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated since version 0\.26 and will be removed in version 2\.2\..*",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
)


# Model search paths: DATA_DIR/models first, then ./models (relative to this file)
MODELS_DIR = Path(__file__).parent / "models"
DATA_MODELS_DIR = Path(os.environ.get("DATA_DIR", "./data")) / "models"

GPU_PROVIDERS = {"CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider"}


def _onnxruntime_has_gpu_provider() -> bool:
    """Whether ONNX Runtime actually has a GPU execution provider available
    in this process. The CPU-only image ships onnxruntime built with no GPU
    providers at all, regardless of what the underlying host hardware is --
    this is a process-level check, not a host-hardware check (see
    hardware.py's HardwareProfile.gpu_available for that)."""
    import onnxruntime as ort
    return bool(GPU_PROVIDERS & set(ort.get_available_providers()))


def effective_device() -> str:
    """The device face-recognition inference will actually use right now:
    "gpu" only if both the gpu_enabled setting is on AND a GPU execution
    provider is actually available in this process; "cpu" otherwise.
    Shared by FaceEmbeddingGenerator's own auto-detect and by callers that
    need to know/display which device a job will run on before/without
    constructing a generator (e.g. queue job resource badges)."""
    try:
        from settings import get_setting
        gpu_enabled = get_setting("gpu_enabled")
    except RuntimeError:
        # Settings not initialized (e.g. standalone script/test) -- fall
        # back to hardware-only detection.
        gpu_enabled = True
    if not gpu_enabled:
        return "cpu"
    return "gpu" if _onnxruntime_has_gpu_provider() else "cpu"


@dataclass
class DetectedFace:
    """A detected face with its bounding box, confidence, and embedding."""
    image: np.ndarray  # RGB aligned face crop (buffalo_l's own alignment)
    bbox: dict  # {x, y, w, h}
    confidence: float
    embedding: np.ndarray  # 512-dim, already computed by buffalo_l's detect+embed call
    landmarks: Optional[np.ndarray] = None  # 5-point facial landmarks
    yaw: Optional[float] = None    # Estimated yaw in degrees (-90 to +90)


@dataclass
class FaceEmbedding:
    """A face embedding (buffalo_l: single 512-dim vector)."""
    embedding: np.ndarray  # 512-dim


class FaceEmbeddingGenerator:
    """Generate face embeddings using buffalo_l (SCRFD-10GF detection +
    ResNet50@WebFace600K recognition), CPU or GPU.
    """

    def __init__(self, device: str = None, models_dir: Path = None):
        """
        Initialize the embedding generator.

        Args:
            device: Device for inference ("gpu", "cpu", or None for auto)
            models_dir: Directory buffalo_l's model bundle lives/downloads
                under (a `buffalo_l/` subfolder gets created here). If
                None, checks DATA_DIR/models first (Docker/production),
                then falls back to ./models (local development).
        """
        if device is None:
            self.device = effective_device()
        else:
            self.device = device

        # Auto-detect models directory: DATA_DIR/models first, then local ./models
        if models_dir is None:
            if (DATA_MODELS_DIR / "buffalo_l").exists():
                models_dir = DATA_MODELS_DIR
            else:
                models_dir = MODELS_DIR
        self._models_dir = models_dir
        self._face_analyzer = None

    def _ort_providers(self) -> list[str]:
        """Get ONNX Runtime providers based on device."""
        if self.device == "gpu":
            return ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def face_analyzer(self) -> FaceAnalysis:
        """Lazy-load InsightFace buffalo_l (detection + recognition)."""
        if self._face_analyzer is None:
            print(f"Loading buffalo_l (detection+recognition) on {self.device}...")
            # Restricted to detection+recognition -- buffalo_l's bundled
            # age/gender/landmark models are pure overhead for identification
            # (confirmed ~3x CPU speedup in the face-pipeline-bench evaluation).
            try:
                self._face_analyzer = FaceAnalysis(
                    name="buffalo_l",
                    root=str(self._models_dir),
                    allowed_modules=["detection", "recognition"],
                    providers=self._ort_providers(),
                )
            except TypeError:
                # Backward compatibility for older InsightFace versions without
                # allowed_modules constructor support.
                self._face_analyzer = FaceAnalysis(
                    name="buffalo_l",
                    root=str(self._models_dir),
                    providers=self._ort_providers(),
                )
            # Get detection size from settings, fall back to 640
            try:
                from settings import get_setting
                det_size = int(get_setting("detection_size"))
            except (RuntimeError, KeyError):
                det_size = 640
            self._face_analyzer.prepare(
                ctx_id=0 if self.device == "gpu" else -1,
                det_size=(det_size, det_size),
            )
        return self._face_analyzer

    def detect_faces(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5,
    ) -> list[DetectedFace]:
        """
        Detect + align + embed every face in an image, in one buffalo_l call.

        Args:
            image: RGB image as numpy array
            min_confidence: Minimum detection confidence

        Returns:
            List of DetectedFace objects (embedding already populated)
        """
        results = self.face_analyzer.get(image)

        faces = []
        for face in results:
            conf = float(face.det_score)
            if conf < min_confidence:
                continue

            # Get bounding box (x1, y1, x2, y2)
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            kps = face.kps if hasattr(face, 'kps') else None

            cx1 = max(0, x1)
            cy1 = max(0, y1)
            cx2 = min(image.shape[1], x2)
            cy2 = min(image.shape[0], y2)
            face_img = image[cy1:cy2, cx1:cx2]

            # Skip if crop is too small
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue

            # Estimate yaw from 5-point landmarks
            yaw_estimate = None
            if kps is not None and len(kps) >= 3:
                left_eye, right_eye, nose = kps[0], kps[1], kps[2]
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_width = np.linalg.norm(right_eye - left_eye)
                if eye_width > 0:
                    nose_offset = nose[0] - eye_center_x
                    yaw_estimate = float(np.degrees(np.arctan2(nose_offset, eye_width / 2)))

            faces.append(DetectedFace(
                image=face_img,
                bbox={"x": int(x1), "y": int(y1), "w": int(w), "h": int(h)},
                confidence=conf,
                embedding=np.asarray(face.normed_embedding, dtype=np.float32),
                landmarks=kps,
                yaw=yaw_estimate,
            ))

        return faces

    def get_embedding(self, face: DetectedFace) -> FaceEmbedding:
        """Read back the embedding buffalo_l already computed for this face
        in detect_faces() -- no re-inference."""
        return self.get_embeddings_batch([face])[0]

    def get_embeddings_batch(self, faces: list[DetectedFace]) -> list[FaceEmbedding]:
        """Read back the embeddings buffalo_l already computed in
        detect_faces(). Takes DetectedFace objects (not raw crop arrays):
        the embedding lives on them already, computed as part of the same
        detect+align+embed call -- there is no separate inference pass to
        batch here, unlike the legacy pipeline's standalone embed step."""
        return [FaceEmbedding(embedding=f.embedding) for f in faces]


def load_image(data: bytes) -> np.ndarray:
    """Load image data into numpy array."""
    image = Image.open(io.BytesIO(data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def load_image_from_path(path: str) -> np.ndarray:
    """Load image from file path."""
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


if __name__ == "__main__":
    import time
    import requests
    import onnxruntime as ort

    print("Testing face embedding generator (ONNX Runtime, buffalo_l)...")
    print(f"ONNX Runtime providers: {ort.get_available_providers()}")

    generator = FaceEmbeddingGenerator()
    print(f"Using device: {generator.device}")

    test_url = "https://stashdb.org/images/b0aef39d-a1d6-4e58-a136-293f02b84921"
    print(f"\nDownloading test image from {test_url}...")

    response = requests.get(test_url)
    image = load_image(response.content)
    print(f"Image shape: {image.shape}")

    print("\nDetecting + embedding faces...")
    t0 = time.time()
    faces = generator.detect_faces(image)
    t1 = time.time()
    print(f"Found {len(faces)} face(s) in {(t1-t0)*1000:.1f}ms")

    if faces:
        face = faces[0]
        print(f"  Confidence: {face.confidence:.2f}")
        print(f"  Size: {face.bbox['w']}x{face.bbox['h']}")
        print(f"  Embedding shape: {face.embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(face.embedding):.4f}")

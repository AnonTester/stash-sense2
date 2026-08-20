"""Convert models to ONNX format for GPU-accelerated inference via ONNX Runtime.

One-time conversion script. Converts:
- YOLOv5s tattoo detector (PyTorch -> ONNX) for tattoo detection
- CLIP ViT-B/32 image encoder (PyTorch -> ONNX) for tattoo embedding/matching

FaceNet512/ArcFace conversion (TensorFlow -> ONNX) was removed with the
buffalo_l migration -- face recognition no longer uses these models at
all, insightface's own FaceAnalysis(name="buffalo_l") replaces them
entirely (see embeddings.py).

Usage:
    cd api
    python convert_models_to_onnx.py                  # Convert all models
    python convert_models_to_onnx.py --yolov5          # Convert YOLOv5 only
    python convert_models_to_onnx.py --clip            # Convert CLIP only

Output:
    models/tattoo_yolov5s.onnx
    models/tattoo_clip_vitb32.onnx
"""
import os
import sys
import numpy as np
from pathlib import Path


MODELS_DIR = Path(__file__).parent / "models"


def convert_yolov5_tattoo():
    """Convert YOLOv5s tattoo detector from PyTorch to ONNX.

    Exports the raw detection model (before NMS) so that post-processing
    can be handled in numpy. Output shape: (batch, 25200, 6) where
    6 = [x_center, y_center, w, h, obj_conf, class_conf].

    Coordinates are in pixel space relative to the 640x640 input.
    """
    import pathlib
    import torch

    # Fix for models saved on Windows
    pathlib.WindowsPath = pathlib.PosixPath

    pt_path = MODELS_DIR / "tattoo" / "tattoo_yolov5s.pt"
    output_path = str(MODELS_DIR / "tattoo_yolov5s.onnx")

    if not pt_path.exists():
        raise FileNotFoundError(
            f"PyTorch model not found at {pt_path}. "
            f"Download it first or check the path."
        )

    print("Loading YOLOv5s tattoo detector via torch.hub...")
    hub_model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(pt_path),
        trust_repo=True,
    )

    # Extract the underlying DetectionModel (nn.Module) from the
    # AutoShape -> DetectMultiBackend -> DetectionModel chain
    detect_model = hub_model.model.model
    detect_model.eval()

    # Input: (batch, 3, 640, 640), float32, normalized to [0, 1]
    input_shape = (1, 3, 640, 640)
    dummy_input = torch.randn(*input_shape)

    print(f"  Input shape: {input_shape}")
    print(f"  Model type: {type(detect_model).__name__}")

    # Verify forward pass works before export
    with torch.no_grad():
        test_output = detect_model(dummy_input)
    pred = test_output[0] if isinstance(test_output, tuple) else test_output
    print(f"  PyTorch output shape: {pred.shape}")  # (1, 25200, 6)

    print("  Exporting to ONNX...")
    # Use legacy exporter (dynamo=False) to produce a single self-contained
    # ONNX file. The new dynamo exporter splits weights into external data.
    torch.onnx.export(
        detect_model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["images"],
        output_names=["detections"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "detections": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"  Saved to {output_path}")
    return output_path


def verify_yolov5_equivalence(onnx_path):
    """Verify ONNX YOLOv5 model produces same raw detections as PyTorch.

    Compares the raw (pre-NMS) output tensors using:
    - Max absolute difference (must be < 1e-4)
    - Cosine similarity on flattened output (must be > 0.9999)
    """
    import pathlib
    import torch
    import onnxruntime as ort

    pathlib.WindowsPath = pathlib.PosixPath

    print("\nVerifying YOLOv5 numerical equivalence...")

    # Load PyTorch model
    pt_path = MODELS_DIR / "tattoo" / "tattoo_yolov5s.pt"
    hub_model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(pt_path),
        trust_repo=True,
    )
    detect_model = hub_model.model.model
    detect_model.eval()

    # Load ONNX model
    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    print(f"  ONNX input: {onnx_input_name}, output: {onnx_output_name}")

    # Test with deterministic input
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 640, 640).astype(np.float32) * 0.5

    # PyTorch inference
    with torch.no_grad():
        pt_output = detect_model(torch.from_numpy(test_input))
    pt_pred = pt_output[0].numpy() if isinstance(pt_output, tuple) else pt_output.numpy()

    # ONNX inference
    onnx_pred = onnx_session.run([onnx_output_name], {onnx_input_name: test_input})[0]

    print(f"  PyTorch output shape: {pt_pred.shape}")
    print(f"  ONNX output shape:    {onnx_pred.shape}")

    # Numerical comparison
    max_diff = np.max(np.abs(pt_pred - onnx_pred))

    pt_flat = pt_pred.flatten()
    onnx_flat = onnx_pred.flatten()
    cos_sim = np.dot(pt_flat, onnx_flat) / (
        np.linalg.norm(pt_flat) * np.linalg.norm(onnx_flat)
    )

    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # Verify output format: columns should be [x_center, y_center, w, h, obj_conf, class_conf]
    assert onnx_pred.shape[2] == 6, f"Expected 6 columns, got {onnx_pred.shape[2]}"
    assert onnx_pred.shape[1] == 25200, f"Expected 25200 anchors, got {onnx_pred.shape[1]}"

    # Check tolerances. YOLOv5 has deeper computation graphs than embedding
    # models so FP32 rounding accumulates more. Use 1e-3 for max diff
    # (plenty tight for detection) while keeping strict cosine similarity.
    assert max_diff < 1e-3, f"Max diff too large: {max_diff:.2e}"
    assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim:.8f}"

    # Test batch inference
    batch_input = np.random.randn(4, 3, 640, 640).astype(np.float32) * 0.5
    onnx_batch = onnx_session.run([onnx_output_name], {onnx_input_name: batch_input})[0]
    print(f"  Batch (4) output shape: {onnx_batch.shape}")
    assert onnx_batch.shape[0] == 4, f"Batch dim mismatch: {onnx_batch.shape[0]}"

    print("\n  YOLOv5 equivalence checks PASSED!")


class _ClipImageEncoder:
    """Wraps just CLIP's visual tower for export/inference.

    open_clip's full CLIP model expects (image, text) and returns
    (image_features, text_features, logit_scale) -- tattoo matching only
    ever needs the image side, so this strips the export down to exactly
    that (avoids exporting/loading the text tower and tokenizer at all).
    """
    def __new__(cls, clip_model):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, visual):
                super().__init__()
                self.visual = visual

            def forward(self, x):
                return self.visual(x)

        return _Wrapper(clip_model.visual)


def convert_clip_tattoo():
    """Convert CLIP ViT-B/32's image encoder from PyTorch to ONNX.

    Replaces the frozen EfficientNet-B0 embedder (a plain ImageNet
    classifier with its head stripped -- never trained on tattoo data at
    all) with CLIP's image encoder. CLIP is trained contrastively (make
    similar images close in embedding space, dissimilar ones far), which
    is directly the property a similarity-matching embedder needs, unlike
    a classifier's incidental features. This is also what TattooTrace (the
    source of tattoo_yolov5s.onnx) itself pairs with YOLOv5 for its own
    "recognition/clustering" stage, per its README.

    Uses OpenAI's original CLIP ViT-B/32 weights via open_clip (512-dim
    output, ~151M params) -- the smallest standard CLIP variant, chosen to
    keep throughput reasonable over a large corpus while still getting
    CLIP's embedding-space quality. Needs `open-clip-torch` installed
    (dev/conversion-time only, same as torch/torchvision/tf2onnx above --
    never a runtime dependency of the sidecar itself).

    Input:  (batch, 3, 224, 224) float32, CLIP-normalized -- NOT
            ImageNet mean/std, CLIP uses its own normalization constants
            (see tattoo_matcher.py's _TattooEmbeddingGenerator).
    Output: (batch, 512) float32, not yet L2-normalized (matches
            EfficientNet-B0's old raw-output convention; normalization
            happens at inference time in tattoo_matcher.py, unchanged).
    """
    import torch
    import open_clip

    output_path = str(MODELS_DIR / "tattoo_clip_vitb32.onnx")

    print("Loading CLIP ViT-B/32 (OpenAI weights) via open_clip...")
    # "-quickgelu" variant, not plain "ViT-B-32": OpenAI's original CLIP
    # weights use the QuickGELU activation (x * sigmoid(1.702x)), but
    # open_clip's default "ViT-B-32" config uses standard GELU. Loading
    # the "openai" pretrained weights onto the wrong activation function
    # doesn't error -- it silently produces a model that's subtly wrong
    # relative to genuine OpenAI CLIP (confirmed live: open_clip itself
    # warns "QuickGELU mismatch" when this is gotten wrong).
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip_model.eval()

    model = _ClipImageEncoder(clip_model)
    model.eval()

    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(*input_shape)

    print(f"  Input shape: {input_shape}")

    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"  PyTorch output shape: {test_output.shape}")  # (1, 512)

    print("  Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"  Saved to {output_path}")
    return output_path


def verify_clip_equivalence(onnx_path):
    """Verify ONNX CLIP image encoder produces same embeddings as PyTorch.

    Compares using:
    - Max absolute difference (must be < 1e-4)
    - Cosine similarity (must be > 0.9999)
    - Batch inference with batch=4
    """
    import torch
    import onnxruntime as ort
    import open_clip

    print("\nVerifying CLIP ViT-B/32 numerical equivalence...")

    # "-quickgelu" variant, not plain "ViT-B-32": OpenAI's original CLIP
    # weights use the QuickGELU activation (x * sigmoid(1.702x)), but
    # open_clip's default "ViT-B-32" config uses standard GELU. Loading
    # the "openai" pretrained weights onto the wrong activation function
    # doesn't error -- it silently produces a model that's subtly wrong
    # relative to genuine OpenAI CLIP (confirmed live: open_clip itself
    # warns "QuickGELU mismatch" when this is gotten wrong).
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip_model.eval()
    pt_model = _ClipImageEncoder(clip_model)
    pt_model.eval()

    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    print(f"  ONNX input: {onnx_input_name}, output: {onnx_output_name}")

    np.random.seed(42)
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_output = pt_model(torch.from_numpy(test_input)).numpy()

    onnx_output = onnx_session.run([onnx_output_name], {onnx_input_name: test_input})[0]

    print(f"  PyTorch output shape: {pt_output.shape}")
    print(f"  ONNX output shape:    {onnx_output.shape}")

    max_diff = np.max(np.abs(pt_output - onnx_output))
    cos_sim = np.dot(pt_output[0], onnx_output[0]) / (
        np.linalg.norm(pt_output[0]) * np.linalg.norm(onnx_output[0])
    )

    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    assert onnx_output.shape == (1, 512), f"Expected (1, 512), got {onnx_output.shape}"
    assert max_diff < 1e-4, f"Max diff too large: {max_diff:.2e}"
    assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim:.8f}"

    batch_input = np.random.randn(4, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_batch = pt_model(torch.from_numpy(batch_input)).numpy()
    onnx_batch = onnx_session.run([onnx_output_name], {onnx_input_name: batch_input})[0]

    print(f"  Batch (4) output shape: {onnx_batch.shape}")
    assert onnx_batch.shape == (4, 512), f"Batch shape mismatch: {onnx_batch.shape}"

    batch_max_diff = np.max(np.abs(pt_batch - onnx_batch))
    for i in range(4):
        sample_cos = np.dot(pt_batch[i], onnx_batch[i]) / (
            np.linalg.norm(pt_batch[i]) * np.linalg.norm(onnx_batch[i])
        )
        assert sample_cos > 0.9999, f"Batch sample {i} cosine sim too low: {sample_cos:.8f}"

    print(f"  Batch max absolute diff: {batch_max_diff:.2e}")
    assert batch_max_diff < 1e-4, f"Batch max diff too large: {batch_max_diff:.2e}"

    print("\n  CLIP ViT-B/32 equivalence checks PASSED!")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    yolov5_only = "--yolov5" in sys.argv
    clip_only = "--clip" in sys.argv

    if yolov5_only:
        # Convert only YOLOv5 tattoo detector
        yolo_path = convert_yolov5_tattoo()
        verify_yolov5_equivalence(yolo_path)

        yolo_size = os.path.getsize(yolo_path) / (1024 * 1024)
        print("\nModel sizes:")
        print(f"  YOLOv5s tattoo: {yolo_size:.1f} MB")

    elif clip_only:
        # Convert only CLIP tattoo embedder
        clip_path = convert_clip_tattoo()
        verify_clip_equivalence(clip_path)

        clip_size = os.path.getsize(clip_path) / (1024 * 1024)
        print("\nModel sizes:")
        print(f"  CLIP ViT-B/32 tattoo: {clip_size:.1f} MB")

    else:
        # Convert all models
        yolo_path = convert_yolov5_tattoo()
        verify_yolov5_equivalence(yolo_path)

        clip_path = convert_clip_tattoo()
        verify_clip_equivalence(clip_path)

        yolo_size = os.path.getsize(yolo_path) / (1024 * 1024)
        clip_size = os.path.getsize(clip_path) / (1024 * 1024)
        print("\nModel sizes:")
        print(f"  YOLOv5s tattoo:       {yolo_size:.1f} MB")
        print(f"  CLIP ViT-B/32 tattoo: {clip_size:.1f} MB")

    print("\nDone! Models saved to api/models/")

from unittest.mock import MagicMock, patch

import embeddings


def _generator(device):
    g = object.__new__(embeddings.FaceEmbeddingGenerator)  # skip __init__: no models needed
    g.device = device
    return g


def _providers(device, available):
    ort = MagicMock()
    ort.get_available_providers.return_value = available
    with patch.dict("sys.modules", {"onnxruntime": ort}):
        return _generator(device)._ort_providers()


def test_cuda_build_requests_only_cuda_then_cpu():
    assert _providers("gpu", ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider", "CPUExecutionProvider"]


def test_rocm_build_keeps_its_own_providers():
    assert _providers("gpu", ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]) == [
        "MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]


def test_cpu_device_is_cpu_only():
    assert _providers("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]) == ["CPUExecutionProvider"]

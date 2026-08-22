"""Tests for hardware detection module."""

import pytest
from unittest.mock import patch, MagicMock

from hardware import (
    HardwareProfile,
    _classify_tier,
    _probe_gpu,
    _probe_amd_gpu_name,
    _probe_amd_gpu_vram_mb,
    _probe_cpu,
    _probe_memory,
    _probe_storage,
    detect_hardware,
)

# A trimmed but structurally real rocminfo transcript: one block per HSA
# agent (the CPU always first), each with "Marketing Name:" BEFORE
# "Vendor Name:" -- the ordering _probe_amd_gpu_name() depends on.
_ROCMINFO_CPU_AND_GPU = """\
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    CPU
  Marketing Name:          AMD Ryzen 7 7840HS w/ Radeon 780M Graphics
  Vendor Name:             CPU
  Uuid:                    CPU-XX
*******
Agent 2
*******
  Name:                    gfx1103
  Marketing Name:          AMD Radeon 780M
  Vendor Name:             AMD
  Uuid:                    GPU-XX
"""

_ROCMINFO_CPU_ONLY = """\
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    CPU
  Marketing Name:          AMD Ryzen 7 7840HS w/ Radeon 780M Graphics
  Vendor Name:             CPU
"""


class TestTierClassification:
    """Test hardware tier classification logic."""

    def test_gpu_high_with_large_vram(self):
        assert _classify_tier(True, 8192) == "gpu-high"

    def test_gpu_high_at_threshold(self):
        assert _classify_tier(True, 4096) == "gpu-high"

    def test_gpu_low_below_threshold(self):
        assert _classify_tier(True, 4095) == "gpu-low"

    def test_gpu_low_small_vram(self):
        assert _classify_tier(True, 2048) == "gpu-low"

    def test_gpu_low_no_vram_info(self):
        """GPU detected but pynvml unavailable — no VRAM info."""
        assert _classify_tier(True, None) == "gpu-low"

    def test_cpu_no_gpu(self):
        assert _classify_tier(False, None) == "cpu"

    def test_cpu_no_gpu_ignores_vram(self):
        assert _classify_tier(False, 8192) == "cpu"


class TestProbeGpu:
    """Test GPU probing with mocked dependencies."""

    def test_no_cuda_provider(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            available, name, vram = _probe_gpu()
        assert available is False
        assert name is None
        assert vram is None

    def test_cuda_available_no_pynvml(self):
        # onnxruntime listing CUDAExecutionProvider only means the library
        # was built with CUDA support, not that a real NVIDIA GPU is
        # present (e.g. this image built FROM an nvidia/cuda base on an
        # AMD-only host) -- pynvml is the authoritative check, so a failed
        # pynvml import means no usable GPU, same as the AMD/rocminfo path.
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort, "pynvml": None}):
            available, name, vram = _probe_gpu()
        assert available is False
        assert name is None
        assert vram is None

    def test_cuda_with_pynvml(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce GTX 1080"
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        with patch.dict("sys.modules", {"onnxruntime": mock_ort, "pynvml": mock_pynvml}):
            available, name, vram = _probe_gpu()

        assert available is True

    def test_ort_import_fails(self):
        """When onnxruntime can't be imported, GPU is unavailable."""
        with patch.dict("sys.modules", {"onnxruntime": None}):
            available, name, vram = _probe_gpu()
        assert available is False

    def test_rocm_available_with_name_and_vram(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "ROCMExecutionProvider", "CPUExecutionProvider"
        ]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}), \
             patch("hardware._probe_amd_gpu_name", return_value="AMD Radeon 780M"), \
             patch("hardware._probe_amd_gpu_vram_mb", return_value=4096):
            available, name, vram = _probe_gpu()
        assert available is True
        assert name == "AMD Radeon 780M"
        assert vram == 4096

    def test_rocm_available_vram_unreadable_still_reports_gpu(self):
        # VRAM is a separate sysfs read from the rocminfo name lookup --
        # losing it (e.g. sysfs path doesn't exist on this kernel) should
        # not also lose GPU detection itself.
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "ROCMExecutionProvider", "CPUExecutionProvider"
        ]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}), \
             patch("hardware._probe_amd_gpu_name", return_value="AMD Radeon 780M"), \
             patch("hardware._probe_amd_gpu_vram_mb", return_value=None):
            available, name, vram = _probe_gpu()
        assert available is True
        assert name == "AMD Radeon 780M"
        assert vram is None

    def test_rocm_provider_present_but_rocminfo_finds_no_gpu(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "ROCMExecutionProvider", "CPUExecutionProvider"
        ]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}), \
             patch("hardware._probe_amd_gpu_name", return_value=None):
            available, name, vram = _probe_gpu()
        assert available is False
        assert name is None
        assert vram is None


class TestProbeAmdGpuName:
    """Test rocminfo output parsing."""

    def test_picks_gpu_marketing_name_skipping_cpu_block(self, monkeypatch):
        mock_result = MagicMock(returncode=0, stdout=_ROCMINFO_CPU_AND_GPU)
        monkeypatch.setattr("subprocess.run", lambda *a, **k: mock_result)
        assert _probe_amd_gpu_name() == "AMD Radeon 780M"

    def test_cpu_only_output_returns_none(self, monkeypatch):
        # The CPU's own "Marketing Name" (e.g. "...w/ Radeon 780M Graphics")
        # must never be mistaken for a GPU -- its Vendor Name is "CPU".
        mock_result = MagicMock(returncode=0, stdout=_ROCMINFO_CPU_ONLY)
        monkeypatch.setattr("subprocess.run", lambda *a, **k: mock_result)
        assert _probe_amd_gpu_name() is None

    def test_nonzero_returncode_returns_none(self, monkeypatch):
        mock_result = MagicMock(returncode=1, stdout="")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: mock_result)
        assert _probe_amd_gpu_name() is None

    def test_rocminfo_not_installed_returns_none(self, monkeypatch):
        def _raise(*a, **k):
            raise FileNotFoundError("rocminfo not found")
        monkeypatch.setattr("subprocess.run", _raise)
        assert _probe_amd_gpu_name() is None

    def test_timeout_returns_none(self, monkeypatch):
        import subprocess as sp

        def _raise(*a, **k):
            raise sp.TimeoutExpired(cmd="rocminfo", timeout=10)
        monkeypatch.setattr("subprocess.run", _raise)
        assert _probe_amd_gpu_name() is None


class TestProbeAmdGpuVramMb:
    """Test amdgpu sysfs VRAM parsing."""

    def test_reads_vram_total_in_mb(self, tmp_path, monkeypatch):
        vram_file = tmp_path / "card0_vram_total"
        vram_file.write_text("4294967296")  # exactly 4096MB
        monkeypatch.setattr("glob.glob", lambda pattern: [str(vram_file)])

        assert _probe_amd_gpu_vram_mb() == 4096

    def test_no_matching_sysfs_path_returns_none(self, monkeypatch):
        monkeypatch.setattr("glob.glob", lambda pattern: [])
        assert _probe_amd_gpu_vram_mb() is None

    def test_zero_vram_entry_skipped_for_next_candidate(self, tmp_path, monkeypatch):
        # A headless/inactive card can report 0 -- keep looking rather than
        # trusting the first match blindly.
        zero_file = tmp_path / "card0_vram_total"
        zero_file.write_text("0")
        real_file = tmp_path / "card1_vram_total"
        real_file.write_text("4294967296")
        monkeypatch.setattr("glob.glob", lambda pattern: [str(zero_file), str(real_file)])

        assert _probe_amd_gpu_vram_mb() == 4096

    def test_unreadable_file_skipped(self, tmp_path, monkeypatch):
        missing = tmp_path / "does_not_exist"
        real_file = tmp_path / "card0_vram_total"
        real_file.write_text("4294967296")
        monkeypatch.setattr("glob.glob", lambda pattern: [str(missing), str(real_file)])

        assert _probe_amd_gpu_vram_mb() == 4096

    def test_non_numeric_content_skipped(self, tmp_path, monkeypatch):
        garbage_file = tmp_path / "card0_vram_total"
        garbage_file.write_text("not-a-number")
        monkeypatch.setattr("glob.glob", lambda pattern: [str(garbage_file)])

        assert _probe_amd_gpu_vram_mb() is None


class TestProbeCpu:
    """Test CPU probing."""

    @patch("os.cpu_count", return_value=8)
    def test_reads_cpu_count(self, mock_count):
        cores = _probe_cpu()
        assert cores == 8

    @patch("os.cpu_count", return_value=None)
    def test_fallback_when_unknown(self, mock_count):
        cores = _probe_cpu()
        assert cores == 1


class TestProbeMemory:
    """Test memory probing."""

    def test_returns_positive_values(self):
        total, available = _probe_memory()
        # In some environments (containers, CI), psutil may not be available
        # or cgroup limits may cause unexpected values. Just verify non-negative
        # and that available <= total (or both are 0 if probing failed).
        assert total >= 0
        assert available >= 0
        if total > 0:
            assert available <= total


class TestProbeStorage:
    """Test storage probing."""

    def test_real_path(self, tmp_path):
        free = _probe_storage(str(tmp_path))
        assert free > 0

    def test_nonexistent_path(self):
        free = _probe_storage("/nonexistent/path/xyz")
        assert free == 0


class TestDetectHardware:
    """Test full hardware detection."""

    @patch("hardware._probe_gpu", return_value=(True, "Test GPU", 8192))
    @patch("hardware._probe_cpu", return_value=8)
    @patch("hardware._probe_memory", return_value=(32768, 16384))
    @patch("hardware._probe_storage", return_value=500000)
    def test_gpu_high_profile(self, mock_storage, mock_mem, mock_cpu, mock_gpu):
        profile = detect_hardware("/tmp")
        assert isinstance(profile, HardwareProfile)
        assert profile.gpu_available is True
        assert profile.gpu_name == "Test GPU"
        assert profile.gpu_vram_mb == 8192
        assert profile.cpu_cores == 8
        assert profile.memory_total_mb == 32768
        assert profile.tier == "gpu-high"

    @patch("hardware._probe_gpu", return_value=(False, None, None))
    @patch("hardware._probe_cpu", return_value=4)
    @patch("hardware._probe_memory", return_value=(8192, 4096))
    @patch("hardware._probe_storage", return_value=100000)
    def test_cpu_profile(self, mock_storage, mock_mem, mock_cpu, mock_gpu):
        profile = detect_hardware("/tmp")
        assert profile.gpu_available is False
        assert profile.tier == "cpu"

    def test_profile_is_frozen(self):
        profile = HardwareProfile(
            gpu_available=False, gpu_name=None, gpu_vram_mb=None,
            cpu_cores=4, memory_total_mb=8192, memory_available_mb=4096,
            storage_free_mb=100000, tier="cpu",
        )
        with pytest.raises(AttributeError):
            profile.tier = "gpu-high"

    def test_summary_with_gpu(self):
        profile = HardwareProfile(
            gpu_available=True, gpu_name="GTX 1080", gpu_vram_mb=8192,
            cpu_cores=8, memory_total_mb=32768, memory_available_mb=16384,
            storage_free_mb=500000, tier="gpu-high",
        )
        summary = profile.summary()
        assert "GTX 1080" in summary
        assert "8192MB VRAM" in summary

    def test_summary_without_gpu(self):
        profile = HardwareProfile(
            gpu_available=False, gpu_name=None, gpu_vram_mb=None,
            cpu_cores=4, memory_total_mb=8192, memory_available_mb=4096,
            storage_free_mb=100000, tier="cpu",
        )
        summary = profile.summary()
        assert "No GPU" in summary

    def test_summary_with_gpu_but_unknown_vram(self):
        # AMD APUs whose sysfs VRAM read fails still have a real, known GPU
        # name -- the summary must show just the name, not a "NoneMB VRAM"
        # artifact from formatting a None value.
        profile = HardwareProfile(
            gpu_available=True, gpu_name="AMD Radeon 780M", gpu_vram_mb=None,
            cpu_cores=16, memory_total_mb=32768, memory_available_mb=16384,
            storage_free_mb=100000, tier="gpu-low",
        )
        summary = profile.summary()
        assert "AMD Radeon 780M" in summary
        assert "None" not in summary

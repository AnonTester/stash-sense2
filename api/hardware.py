"""Hardware detection and profiling.

Probes GPU, CPU, memory, and storage at startup to build an immutable
hardware profile. Used by the settings system to pick tier-appropriate
defaults (batch sizes, concurrency, frame counts).

Runs once at startup. All probes have graceful fallbacks.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable hardware profile built at startup."""
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_mb: Optional[int]
    cpu_cores: int
    memory_total_mb: int
    memory_available_mb: int
    storage_free_mb: int
    tier: str  # "gpu-high", "gpu-low", "cpu"

    def summary(self) -> str:
        """One-line hardware summary for startup log."""
        if self.gpu_available:
            gpu_info = f"{self.gpu_name} ({self.gpu_vram_mb}MB VRAM)"
        else:
            gpu_info = "No GPU"
        return (
            f"{gpu_info}, {self.memory_total_mb}MB RAM, "
            f"{self.cpu_cores} cores, {self.storage_free_mb}MB free disk"
        )


def _classify_tier(gpu_available: bool, gpu_vram_mb: Optional[int]) -> str:
    """Classify hardware into a tier based on GPU availability and VRAM."""
    if not gpu_available:
        return "cpu"
    if gpu_vram_mb is not None and gpu_vram_mb >= 4096:
        return "gpu-high"
    return "gpu-low"


def _probe_gpu() -> tuple[bool, Optional[str], Optional[int]]:
    """Probe GPU via ONNX Runtime providers -- NVIDIA via pynvml, AMD via rocminfo.

    onnxruntime listing "CUDAExecutionProvider"/"ROCMExecutionProvider" only
    means the library was *built* with that support -- it says nothing about
    whether a real GPU of that vendor is actually present (e.g. this Docker
    image is built FROM an nvidia/cuda base image, so the provider always
    shows up even on an AMD-only host). pynvml/rocminfo are the authoritative
    checks: they actually talk to the driver, so if neither can init or find
    a device, there is no usable GPU regardless of what onnxruntime claims
    to support.

    Returns:
        (gpu_available, gpu_name, gpu_vram_mb)
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception as e:
        logger.debug("ONNX Runtime provider probe failed: %s", e)
        return False, None, None

    if "CUDAExecutionProvider" in providers:
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_vram_mb = mem_info.total // (1024 * 1024)
                return True, gpu_name, gpu_vram_mb
            finally:
                pynvml.nvmlShutdown()
        except Exception as e:
            logger.debug(f"pynvml found no usable NVIDIA GPU: {e}")

    if "ROCMExecutionProvider" in providers:
        gpu_name = _probe_amd_gpu_name()
        if gpu_name is not None:
            # VRAM isn't probed for AMD: rocminfo's pool sizes are shared
            # system memory on the APUs this was tested against (Radeon
            # 780M), not a fixed dedicated-VRAM figure, so reporting them
            # as gpu_vram_mb would be misleading rather than just missing.
            # _classify_tier() treats vram=None as "gpu-low", a safe
            # default until this can distinguish APUs from discrete cards.
            return True, gpu_name, None
        logger.debug("ROCMExecutionProvider available but rocminfo found no usable AMD GPU")

    return False, None, None


def _probe_amd_gpu_name() -> Optional[str]:
    """Get the marketing name of the first non-CPU device from `rocminfo`.

    rocminfo lists one block per device (the CPU itself, plus each GPU),
    each with its own "Marketing Name" and "Vendor Name" lines -- in that
    order (Marketing Name comes first within a block), so the vendor
    needed to decide whether to keep a given Marketing Name isn't known
    until a couple of lines after it. Buffer the most recent Marketing
    Name seen and confirm it once its block's Vendor Name line arrives.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        pending_name = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Marketing Name:"):
                pending_name = line.split(":", 1)[1].strip() or None
            elif line.startswith("Vendor Name:"):
                vendor = line.split(":", 1)[1].strip()
                if vendor and vendor != "CPU" and pending_name:
                    return pending_name
                pending_name = None
        return None
    except Exception as e:
        logger.debug("rocminfo probe failed: %s", e)
        return None


def _probe_cpu() -> int:
    """Probe CPU core count, respecting cgroup limits."""
    # Check cgroup v2 limit first (Docker containers)
    try:
        cgroup_path = Path("/sys/fs/cgroup/cpu.max")
        if cgroup_path.exists():
            content = cgroup_path.read_text().strip()
            parts = content.split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, quota // period)
    except Exception as e:
        logger.debug("cgroup CPU probe failed: %s", e)

    return os.cpu_count() or 1


def _probe_memory() -> tuple[int, int]:
    """Probe total and available memory in MB, respecting cgroup limits.

    Returns:
        (total_mb, available_mb)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_mb = mem.total // (1024 * 1024)
        available_mb = mem.available // (1024 * 1024)
    except Exception as e:
        logger.debug("psutil memory probe failed: %s", e)
        total_mb = 0
        available_mb = 0

    # Check cgroup v2 memory limit (Docker containers)
    try:
        cgroup_path = Path("/sys/fs/cgroup/memory.max")
        if cgroup_path.exists():
            content = cgroup_path.read_text().strip()
            if content != "max":
                cgroup_limit_mb = int(content) // (1024 * 1024)
                total_mb = min(total_mb, cgroup_limit_mb) if total_mb else cgroup_limit_mb
    except Exception as e:
        logger.debug("cgroup memory probe failed: %s", e)

    return total_mb, available_mb


def _probe_storage(data_dir: str) -> int:
    """Probe free disk space at the data directory in MB."""
    try:
        usage = shutil.disk_usage(data_dir)
        return usage.free // (1024 * 1024)
    except Exception as e:
        logger.debug("Storage probe failed for %s: %s", data_dir, e)
        return 0


def detect_hardware(data_dir: str = "./data") -> HardwareProfile:
    """Run all hardware probes and build an immutable profile.

    Args:
        data_dir: Path to the data directory for storage probe.

    Returns:
        Frozen HardwareProfile dataclass.
    """
    gpu_available, gpu_name, gpu_vram_mb = _probe_gpu()
    cpu_cores = _probe_cpu()
    memory_total_mb, memory_available_mb = _probe_memory()
    storage_free_mb = _probe_storage(data_dir)
    tier = _classify_tier(gpu_available, gpu_vram_mb)

    profile = HardwareProfile(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        cpu_cores=cpu_cores,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        storage_free_mb=storage_free_mb,
        tier=tier,
    )

    logger.warning(f"Hardware: {profile.summary()}")
    logger.warning(f"Resource tier: {tier}")

    return profile


# Module-level singleton, set during startup
_hardware_profile: Optional[HardwareProfile] = None


def get_hardware_profile() -> HardwareProfile:
    """Get the cached hardware profile. Must be called after detect_hardware()."""
    if _hardware_profile is None:
        raise RuntimeError("Hardware not yet detected. Call detect_hardware() during startup.")
    return _hardware_profile


def init_hardware(data_dir: str = "./data") -> HardwareProfile:
    """Detect hardware and cache the profile. Called once at startup."""
    global _hardware_profile
    _hardware_profile = detect_hardware(data_dir)
    return _hardware_profile

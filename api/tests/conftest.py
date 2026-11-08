"""Pytest configuration for async tests."""

import sys
from unittest.mock import MagicMock

import pytest

# Provide dummy modules for ML/GPU dependencies that aren't available in CI.
# This allows pytest to collect (import) heavy-marked test files without error;
# the @pytest.mark.heavy marker then skips them at runtime.
# Packages with submodule imports (e.g., insightface.app) need each level mocked.
_ML_MODULES = [
    "cv2",
    "onnxruntime",
    "usearch",
    "usearch.index",
    "insightface",
    "insightface.app",
    "insightface.model_zoo",
    "insightface.model_zoo.attribute",
    "insightface.utils",
    "insightface.utils.face_align",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except (ImportError, ModuleNotFoundError):
            sys.modules[_mod] = MagicMock()

# onnxruntime's stub needs get_available_providers to be a REAL, already-set attribute, not one
# MagicMock would auto-vivify on first access. monkeypatch.setattr()'s teardown doesn't trust its own
# getattr() result -- to avoid freezing a class descriptor (pytest's own bug #10644), it re-derives the
# "original value to restore" straight from the target's __dict__, and an attribute that only exists
# through MagicMock's __getattr__ auto-creation was never actually written into __dict__. That makes
# teardown think the attribute never existed and call delattr() instead of restoring it -- and
# unittest.mock treats a deleted attribute as PERMANENTLY gone, never auto-recreating it. The very first
# test anywhere in the session that does monkeypatch.setattr(sys.modules["onnxruntime"],
# "get_available_providers", ...) then poisons this shared singleton stub for every test and every
# module that runs after it, with no way back -- confirmed live 2026-11-08 (15 failures across
# test_embeddings.py/test_queue_dispatch.py/test_queue_router.py, all downstream of this exact mechanism).
# Pre-setting it here, as a plain assignment that genuinely lands in __dict__, gives monkeypatch a real
# value to find and restore instead.
if isinstance(sys.modules.get("onnxruntime"), MagicMock):
    sys.modules["onnxruntime"].get_available_providers = MagicMock(return_value=["CPUExecutionProvider"])


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()

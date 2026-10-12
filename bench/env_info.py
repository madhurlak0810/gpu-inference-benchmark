"""Record the hardware and software a run used: `python -m bench.env_info`.

Uses NVML only and never initialises CUDA in this process, so it is safe to
call before vLLM starts its own workers.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

import pynvml

from .config import REPO_ROOT

PACKAGES = (
    "torch", "transformers", "tokenizers", "vllm", "numpy", "pandas",
    "matplotlib", "httpx", "nvidia-ml-py", "PyYAML",
)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _gpu(index: int) -> dict:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = {
            "index": index,
            "name": _text(pynvml.nvmlDeviceGetName(handle)),
            "memory_total_gb": round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30, 2),
            "driver": _text(pynvml.nvmlSystemGetDriverVersion()),
            "device_count": pynvml.nvmlDeviceGetCount(),
        }
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            info["compute_capability"] = f"{major}.{minor}"
        except (AttributeError, pynvml.NVMLError):
            info["compute_capability"] = None
        return info
    finally:
        pynvml.nvmlShutdown()


def _packages() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _cuda_runtime() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.version.cuda


def _git() -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def collect(gpu_index: int = 0) -> dict:
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": _gpu(gpu_index),
        "cuda_runtime": _cuda_runtime(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _packages(),
        "git": _git(),
    }


if __name__ == "__main__":
    json.dump(collect(int(sys.argv[1]) if len(sys.argv) > 1 else 0), sys.stdout, indent=2)
    print()

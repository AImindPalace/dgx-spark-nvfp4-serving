"""Environment capture utilities for benchmark runs.

Captures vLLM version, torch version, CUDA version, system memory, vLLM serve
flags, and git commit. All subprocess-based so it can be tested with mocks on
any platform (the benchmark itself only runs on Linux/Spark, but tests run on
Windows).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class MemorySnapshot:
    total_gb: float
    used_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ServeFlags:
    gpu_memory_utilization: float
    enforce_eager: bool
    max_model_len: int
    speculative_config: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Environment:
    vllm_version: str
    torch_version: str
    cuda_version: str
    memory_total_gb: float
    memory_before_gb: float
    git_commit: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def capture_memory() -> MemorySnapshot:
    """Run ``free -b`` and return a MemorySnapshot."""
    output = subprocess.check_output(["free", "-b"], text=True)
    return parse_free_output(output)


def parse_free_output(output: str) -> MemorySnapshot:
    """Parse the Mem: line from ``free -b`` output.

    Column order: total used free shared buff/cache available
    Converts bytes → GB (divide by 1024³), rounded to 1 decimal place.
    """
    for line in output.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            total_bytes = int(parts[1])
            used_bytes = int(parts[2])
            gib = 1024 ** 3
            return MemorySnapshot(
                total_gb=round(total_bytes / gib, 1),
                used_gb=round(used_bytes / gib, 1),
            )
    raise RuntimeError("Could not parse 'free -b' output: no Mem: line found")


# ---------------------------------------------------------------------------
# vLLM serve flags
# ---------------------------------------------------------------------------

def capture_serve_flags() -> ServeFlags:
    """Detect a running vLLM server process and return its serve flags."""
    pid_output = subprocess.check_output(
        ["pgrep", "-f", "vllm.entrypoints"], text=True
    ).strip()
    pid = int(pid_output.splitlines()[0])

    with open(f"/proc/{pid}/cmdline", "rb") as fh:
        raw = fh.read()

    # /proc/<pid>/cmdline is null-separated; trailing null produces empty string
    args = [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]
    return parse_serve_args(args)


def parse_serve_args(args: list[str]) -> ServeFlags:
    """Walk a command-line arg list and extract vLLM serve flags.

    Defaults:
        --gpu-memory-utilization  0.40
        --enforce-eager           False  (presence of flag sets True)
        --max-model-len           4096
        --speculative-config      None   (JSON string → dict when present)
    """
    gpu_memory_utilization: float = 0.40
    enforce_eager: bool = False
    max_model_len: int = 4096
    speculative_config: dict | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--gpu-memory-utilization":
            if i + 1 < len(args):
                gpu_memory_utilization = float(args[i + 1])
                i += 2
                continue
        elif arg == "--enforce-eager":
            enforce_eager = True
        elif arg == "--max-model-len":
            if i + 1 < len(args):
                max_model_len = int(args[i + 1])
                i += 2
                continue
        elif arg == "--speculative-config":
            if i + 1 < len(args):
                speculative_config = json.loads(args[i + 1])
                i += 2
                continue
        i += 1

    return ServeFlags(
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        speculative_config=speculative_config,
    )


# ---------------------------------------------------------------------------
# Version / environment helpers
# ---------------------------------------------------------------------------

def capture_package_version(package: str) -> str:
    """Return the ``__version__`` of *package*, or ``"unknown"`` on failure."""
    try:
        result = subprocess.check_output(
            ["python", "-c", f"import {package}; print({package}.__version__)"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return result.strip()
    except Exception:
        return "unknown"


def capture_cuda_version() -> str:
    """Return ``torch.version.cuda``, or ``"unknown"`` on failure."""
    try:
        result = subprocess.check_output(
            ["python", "-c", "import torch; print(torch.version.cuda or 'unknown')"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return result.strip()
    except Exception:
        return "unknown"


def capture_git_commit() -> str:
    """Return the first 7 characters of the current git commit hash."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return result.strip()[:7]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def capture_environment() -> Environment:
    """Compose all capture helpers into an Environment snapshot."""
    mem = capture_memory()
    return Environment(
        vllm_version=capture_package_version("vllm"),
        torch_version=capture_package_version("torch"),
        cuda_version=capture_cuda_version(),
        memory_total_gb=mem.total_gb,
        memory_before_gb=mem.used_gb,
        git_commit=capture_git_commit(),
    )

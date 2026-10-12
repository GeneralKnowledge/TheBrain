"""Detect the actual machine instead of assuming it.

Only the standard library is required. Optional probes (torch, llama.cpp)
are attempted and reported as unavailable when they fail.
"""

from __future__ import annotations

import os
import platform
import resource
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GPUInfo:
    name: str
    vram_total_mb: int | None
    vram_free_mb: int | None
    driver: str | None
    compute_capability: str | None = None


@dataclass
class HardwareReport:
    os: str
    kernel: str
    python: str
    cpu_model: str
    logical_cpus: int
    physical_cores: int | None
    cpu_flags: list[str]
    ram_total_mb: int
    ram_available_mb: int
    swap_total_mb: int
    disk_free_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)
    torch_version: str | None = None
    torch_cuda: bool = False
    torch_cuda_version: str | None = None
    llama_cpp_version: str | None = None
    llama_cpp_gpu_offload: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            out[k] = int(v.strip().split()[0]) // 1024
    except OSError:
        pass
    return out


def _cpuinfo() -> tuple[str, int | None, list[str]]:
    model, flags, cores = platform.processor() or "unknown", [], set()
    try:
        phys = None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and model in ("unknown", "x86_64", ""):
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                wanted = {"avx", "avx2", "avx512f", "fma", "f16c", "sse4_2"}
                flags = sorted(wanted & set(line.split(":", 1)[1].split()))
            elif line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                cores.add((phys, line.split(":", 1)[1].strip()))
    except OSError:
        pass
    return model, (len(cores) or None), flags


def _nvidia_smi() -> tuple[list[GPUInfo], list[str]]:
    notes: list[str] = []
    exe = shutil.which("nvidia-smi")
    if not exe:
        if Path("/proc/driver/nvidia/version").exists():
            notes.append("NVIDIA kernel driver present but nvidia-smi missing")
        else:
            notes.append("no NVIDIA driver detected (nvidia-smi absent)")
        return [], notes
    q = "name,memory.total,memory.free,driver_version,compute_cap"
    try:
        res = subprocess.run(
            [exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], [f"nvidia-smi failed: {exc}"]
    if res.returncode != 0:
        return [], [f"nvidia-smi error: {res.stderr.strip()[:200]}"]
    gpus = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue

        def num(s: str) -> int | None:
            try:
                return int(float(s))
            except ValueError:
                return None

        gpus.append(GPUInfo(parts[0], num(parts[1]), num(parts[2]), parts[3],
                            parts[4] if len(parts) > 4 else None))
    return gpus, notes


def _probe_torch(report: HardwareReport) -> None:
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        report.notes.append("torch not installed: activation research unavailable")
        return
    report.torch_version = torch.__version__
    report.torch_cuda_version = torch.version.cuda
    try:
        report.torch_cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"torch CUDA probe failed: {exc}")


def _probe_llama(report: HardwareReport) -> None:
    try:
        import llama_cpp  # type: ignore
    except Exception:  # noqa: BLE001
        report.notes.append("llama-cpp-python not installed: GGUF backend unavailable")
        return
    report.llama_cpp_version = getattr(llama_cpp, "__version__", "unknown")
    fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if fn is not None:
        try:
            report.llama_cpp_gpu_offload = bool(fn())
        except Exception:  # noqa: BLE001
            report.llama_cpp_gpu_offload = None


def detect(data_dir: str | Path = ".", probe_optional: bool = True) -> HardwareReport:
    mem = _meminfo()
    model, cores, flags = _cpuinfo()
    try:
        os_name = platform.freedesktop_os_release().get("PRETTY_NAME", platform.system())
    except OSError:
        os_name = platform.system()
    path = Path(data_dir)
    while not path.exists() and path != path.parent:
        path = path.parent
    gpus, notes = _nvidia_smi()
    report = HardwareReport(
        os=os_name,
        kernel=platform.release(),
        python=platform.python_version(),
        cpu_model=model,
        logical_cpus=os.cpu_count() or 1,
        physical_cores=cores,
        cpu_flags=flags,
        ram_total_mb=mem.get("MemTotal", 0),
        ram_available_mb=mem.get("MemAvailable", 0),
        swap_total_mb=mem.get("SwapTotal", 0),
        disk_free_mb=shutil.disk_usage(path).free // (1024 * 1024),
        gpus=gpus,
        notes=notes,
    )
    if probe_optional:
        _probe_torch(report)
        _probe_llama(report)
    return report


def recommend(report: HardwareReport) -> dict[str, Any]:
    """Suggest a starting configuration. Suggestions are not verified results."""
    rec: dict[str, Any] = {"backend": "llamacpp", "n_gpu_layers": 0, "context_length": 1024}
    reasons = []
    if report.ram_available_mb < 1500:
        rec.update(backend="mock")
        reasons.append("less than 1.5 GB RAM available: use the mock backend or a 0.6B Q4 model")
    elif report.ram_available_mb < 3000:
        rec["quantization"] = "Q4_K_M"
        rec["context_length"] = 512
        reasons.append("under 3 GB RAM available: Q4 quantisation, 512-token context")
    else:
        rec["quantization"] = "Q4_K_M"
    vram = max((g.vram_total_mb or 0) for g in report.gpus) if report.gpus else 0
    if vram and report.llama_cpp_gpu_offload:
        # Qwen3-1.7B Q4_K_M is ~1.1 GB on disk; keep >=600 MB VRAM headroom.
        rec["n_gpu_layers"] = -1 if vram >= 2500 else max(0, int(28 * (vram - 600) / 1100))
        reasons.append(f"{vram} MB VRAM with llama.cpp GPU offload: suggested n_gpu_layers")
    elif vram:
        reasons.append("GPU present but llama.cpp build has no GPU offload: CPU inference")
    else:
        reasons.append("no usable GPU: CPU inference")
    rec["reasons"] = reasons
    return rec


def process_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

"""System/hardware introspection — GPU detection for the Learn feature."""

from __future__ import annotations

import shutil
import subprocess

from platrix.logging_conf import get_logger

logger = get_logger(__name__)


def gpu_info() -> dict:
    """Detect an NVIDIA GPU and whether training can use it.

    Returns a dict describing the accelerator so the dashboard can show it and
    the trainer can pick the device. We *detect and use* a GPU when the drivers
    are present; we never auto-install drivers (that can break the host).
    """
    info: dict = {
        "has_gpu": False,
        "name": None,
        "driver": None,
        "cuda_available": False,
        "device": "cpu",
        "note": "",
    }

    # 1) Is there a GPU + driver at all? (nvidia-smi)
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=6,
            )
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                info["has_gpu"] = True
                info["name"] = parts[0] if parts else None
                info["driver"] = parts[1] if len(parts) > 1 else None
        except Exception:  # noqa: BLE001
            pass

    # 2) Can PyTorch actually use it? (CUDA build present)
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device"] = "cuda"
            if not info["name"]:
                info["name"] = torch.cuda.get_device_name(0)
                info["has_gpu"] = True
    except Exception:  # noqa: BLE001
        pass

    # 3) Guidance
    if info["has_gpu"] and not info["cuda_available"]:
        info["note"] = (
            "GPU detected but PyTorch has no CUDA support. Install a CUDA build: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    elif info["cuda_available"]:
        info["note"] = "Training will run on the GPU."
    else:
        info["note"] = "No GPU detected — training runs on the CPU."
    return info

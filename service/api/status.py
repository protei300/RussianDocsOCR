"""Hardware, recognition runtime and queue state.

Hardware probes are each wrapped in a bare ``try/except`` returning ``{}`` so a
machine without a GPU — or without permission to read NVML — degrades to a
smaller status page instead of a 500.
"""
from __future__ import annotations

import logging
import platform
import sys
import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from service.api.deps import require_session
from service.core.config import get_settings
from service.core.database import DbSession, get_db
from service.ml import runtime
from service.repositories import documents as repo

log = logging.getLogger(__name__)
router = APIRouter(tags=["status"])

_STARTED_AT = time.time()


def _detect_cpu_name() -> str:
    """Cross-platform CPU model. Read once — it cannot change at runtime."""
    try:
        if sys.platform == "win32":
            import winreg
            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


_CPU_NAME = _detect_cpu_name()


def _server_stats() -> dict[str, Any]:
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\" if sys.platform == "win32" else "/")
        return {
            "cpu_pct": psutil.cpu_percent(interval=0.15),
            "cpu_name": _CPU_NAME,
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_used_gb": round(memory.used / 1e9, 1),
            "ram_total_gb": round(memory.total / 1e9, 1),
            "disk_used_gb": round(disk.used / 1e9, 1),
            "disk_total_gb": round(disk.total / 1e9, 1),
        }
    except Exception:
        log.debug("[API] psutil unavailable", exc_info=True)
        return {}


def _gpu_stats() -> dict[str, Any] | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return {
            # Older nvidia-ml-py returns bytes here, newer returns str.
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "utilization_pct": int(util.gpu),
            "vram_used_gb": round(memory.used / 1e9, 1),
            "vram_total_gb": round(memory.total / 1e9, 1),
            "temperature_c": int(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU)),
        }
    except Exception:
        return None


@router.get("/status")
def get_status(request: Request, db: DbSession = Depends(get_db),
               _user=Depends(require_session)) -> dict:
    settings = get_settings()
    info = runtime.device_info()
    stats = repo.stats(db)

    return {
        "server": _server_stats(),
        "gpu": _gpu_stats(),
        # `device` vs `ocr_device` are reported separately on purpose: with GPU
        # detectors the OCR engines still run on CPU, and a page that just says
        # "GPU active" invites a bug report the first time someone watches
        # nvidia-smi during recognition.
        "compute": info.as_dict(),
        "service": {
            "uptime_sec": int(time.time() - _STARTED_AT),
            "version": settings.git_commit,
            "documents_queued": stats.get("queued", 0),
            "documents_processing": stats.get("processing", 0),
            "documents_done": stats.get("done", 0),
            "documents_failed": stats.get("failed", 0),
            "documents_total": stats.get("total", 0),
            "recognised": stats.get("recognised", 0),
            "avg_processing_ms": stats.get("avg_processing_ms"),
            "data_dir_mb": round(db.disk_usage_bytes() / 1e6, 1),
        },
        # Which backend is actually live, so an operator can tell at a glance
        # whether what they are looking at will survive a restart.
        "storage": getattr(request.app.state, "storage_mode", None).as_dict()
        if getattr(request.app.state, "storage_mode", None) else
        {"backend": db.backend, "ephemeral": db.is_ephemeral},
    }

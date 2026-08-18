import importlib.util
import os
import sys
from pathlib import Path


def _enable_cuda_dlls() -> None:
    """Expose the CUDA/cuDNN runtime DLLs bundled with torch to onnxruntime.

    onnxruntime's CUDAExecutionProvider depends on cublasLt64_12.dll /
    cudnn64_9.dll etc., which ship inside ``torch/lib`` but are not on the
    Windows DLL search path by default — so the provider silently falls back to
    CPU. Registering that directory before onnxruntime is imported fixes it.
    Windows-only, idempotent, no-op if torch isn't installed.
    """
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("torch")  # locate torch without importing it
    if not spec or not spec.submodule_search_locations:
        return
    torch_lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    if not torch_lib.is_dir():
        return
    try:
        os.add_dll_directory(str(torch_lib))
    except (OSError, AttributeError):
        pass
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


_enable_cuda_dlls()

__version__ = '4.3.0'

from .pipeline.pipeline import Pipeline

__all__ = 'Pipeline', __version__

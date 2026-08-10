"""Download the model weights - thin wrapper over document_processing/fetch_models.py.

The real logic lives inside the package so a pip-installed copy can fetch its
own weights (see the docstring there). This wrapper keeps the documented
``python scripts/fetch_models.py`` entry point working from a repo checkout,
including before any dependencies are installed (the Docker model stage and CI
run it in a bare interpreter): it loads the implementation by file path instead
of importing ``document_processing``, whose __init__ pulls the ML stack.

Standard library only, on purpose.
"""
import importlib.util
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent / 'document_processing' / 'fetch_models.py'

_spec = importlib.util.spec_from_file_location('rdocs_fetch_models', _IMPL)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

if __name__ == '__main__':
    sys.exit(_module.main())

import gc
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Ensure CWD is the tests/ directory so relative image paths resolve correctly."""
    os.chdir(Path(__file__).parent)


@pytest.fixture(autouse=True, scope='module')
def _release_models_after_each_module():
    """Free a module's pipelines before the next module builds its own.

    A Pipeline holds every model of the library (~1 GB with the document detector of
    models-v10) and its objects reference each other, so a module-scoped fixture that
    goes out of scope is not freed until the cycle collector happens to run. The whole
    suite peaked at 8.8 GB and the CI runner of the private repository (7 GB) killed it
    (exit 143, 2026-10-08).
    """
    yield
    gc.collect()


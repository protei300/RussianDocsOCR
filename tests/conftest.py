import os
from pathlib import Path


def pytest_configure(config):
    """Ensure CWD is the tests/ directory so relative image paths resolve correctly."""
    os.chdir(Path(__file__).parent)

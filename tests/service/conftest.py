"""Pytest wiring for the service test-suite."""
import importlib.util
from pathlib import Path

import pytest

#: The service suite needs requirements-service.txt. Using the library alone is a
#: supported setup, so `pytest tests/` there must not die collecting these — and a
#: marker cannot help: markers are applied AFTER the module is imported, and the
#: import is what fails. Skipping at collection is the only stage early enough.
#:
#: This is a real condition rather than a flag, so it behaves the same locally and
#: in CI. The service job runs `pytest tests/service`, which exits 5 ("no tests
#: ran") if the dependencies are missing there — so this cannot hide a broken
#: install in the job that is supposed to catch it.
_REQUIRED = ("sqlalchemy", "jose", "fastapi")
_MISSING = [name for name in _REQUIRED if importlib.util.find_spec(name) is None]
if _MISSING:
    collect_ignore_glob = ["*"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: loads the real 215 MB model set; run with --runslow",
    )
    config.addinivalue_line(
        "markers",
        "service: needs requirements-service.txt (fastapi, sqlalchemy, jose)",
    )


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests that load the real recognition models")


def pytest_collection_modifyitems(config, items):
    # Mark everything under tests/service so the two suites can be selected with
    # `-m service` / `-m "not service"`. Doing it here rather than with a path
    # filter on the command line is not fussiness: tests/conftest.py chdirs to
    # tests/ during configure, so a relative `--ignore=tests/service` silently
    # matches nothing and the exclusion quietly does not happen.
    here = Path(__file__).parent
    for item in items:
        try:
            in_service = here in Path(str(item.fspath)).parents
        except (OSError, ValueError):
            in_service = False
        if in_service:
            item.add_marker(pytest.mark.service)

    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow (loads the real models)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

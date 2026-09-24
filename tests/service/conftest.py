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


# --- shared by the auth suites -------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_auth_process_state():
    """Clear the two pieces of auth state that live for the whole process.

    The failed-login counters are module-level (a single-process service keeps
    them in a dict), and every TestClient request comes from the same address,
    "testclient". Without this, failures from one test count against the next —
    the address-wide lockout fires in a test that never failed a login at all, and
    the suite becomes order-dependent. The per-process JWT secret is reset for the
    same reason: a test that forces the ephemeral secret must not leak it.
    """
    try:
        from service.core import auth as auth_core
    except Exception:                                   # library-only environment
        yield
        return
    auth_core._attempts.clear()
    auth_core._process_secret = None
    yield
    auth_core._attempts.clear()
    auth_core._process_secret = None


@pytest.fixture
def app_factory(monkeypatch):
    """Build the app with a given AUTH_MODE and a throwaway data directory.

    Extra environment can be passed as keyword arguments; ``None`` unsets.
    """
    import importlib
    import tempfile

    def build(mode: str | None, **env: str | None):
        base = {"DATA_DIR": tempfile.mkdtemp(), "JWT_SECRET": "test-secret",
                "SEED_SAMPLES": "-1", "WARMUP_IMAGE": "", "COMPUTE_DEVICE": "cpu",
                "AUTH_MODE": mode}
        base.update(env)
        for name, value in base.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

        from service.core.config import get_settings
        get_settings.cache_clear()
        import service.main as main
        importlib.reload(main)
        return main.app

    return build

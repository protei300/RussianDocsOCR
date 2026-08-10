"""Filesystem layout of the harness. Pure data — imports nothing heavy.

Resolved from this file rather than from the working directory: the CLI and the
checker get invoked from several places (PowerShell, Git Bash, a Go test, CI), and
a cwd-relative path would silently pick up the wrong models or the wrong goldens.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Repository root. Override with RDOCS_REPO when the harness runs against a
#: checkout other than the one it lives in.
REPO = Path(os.environ.get("RDOCS_REPO", Path(__file__).resolve().parents[1]))

CONFORMANCE = REPO / "conformance"
SPEC = CONFORMANCE / "spec"
CASES = CONFORMANCE / "cases"
REPORT = CONFORMANCE / "report"
PORTS_JSON = CONFORMANCE / "ports.json"

SAMPLES = REPO / "samples"
MODELS = REPO / "document_processing" / "models"
SEED_MANIFEST = REPO / "service" / "seed_data" / "manifest.json"


def case_dir(slug: str) -> Path:
    return CASES / slug


def stage_dir(slug: str) -> Path:
    return case_dir(slug) / "stages"

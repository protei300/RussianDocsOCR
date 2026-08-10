"""Which weight set the goldens were recorded against.

A golden is not a property of the code alone: every box, canvas shape and decoded
string is produced by a specific set of weights. Nothing recorded that, and it
cost a red CI nobody could read — Borders v4 landed on 3 Aug, the goldens stayed
on the previous weights, and the harness reported a four-pixel box shift and a
247 px Hausdorff distance with no hint that the models had moved underneath it.

So the goldens now carry the name of the set they were taken against, and the
runner refuses to grade a mismatch. The failure becomes one line naming the
cause instead of forty-four numeric differences that have to be diagnosed.

Pure: json and pathlib only. The checker imports this.
"""
from __future__ import annotations

import json
from pathlib import Path

from conformance.paths import CASES, REPO

#: Written by `refcli regen`, read by the runner.
PIN_FILE = CASES / "models.json"

MANIFEST = REPO / "document_processing" / "models.lock.json"


def installed_models_version() -> str | None:
    """The weight set currently on disk, per the fetch manifest."""
    if not MANIFEST.is_file():
        return None
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8")).get("models_version")
    except (json.JSONDecodeError, OSError):
        return None


def goldens_models_version() -> str | None:
    """The weight set the goldens were recorded against, if recorded at all."""
    if not PIN_FILE.is_file():
        return None
    try:
        return json.loads(PIN_FILE.read_text(encoding="utf-8")).get("models_version")
    except (json.JSONDecodeError, OSError):
        return None


def write_pin(models_version: str | None) -> None:
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(json.dumps({
        "models_version": models_version,
        "note": "The weight set these goldens were recorded against. Regenerating "
                "the goldens and changing the models are the same act: if these "
                "disagree with document_processing/models.lock.json, the numbers "
                "below cannot be compared.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mismatch_message() -> str | None:
    """Human-readable reason to refuse grading, or None when it is safe.

    Silent when either side is unknown: an older checkout without the pin, or a
    working copy without the manifest, should degrade to the previous behaviour
    rather than refuse to run.
    """
    installed = installed_models_version()
    recorded = goldens_models_version()
    if not installed or not recorded or installed == recorded:
        return None
    return (
        f"goldens were recorded against models {recorded}, but models {installed} "
        f"is installed.\n"
        f"Every box and canvas shape below depends on the weights, so the numbers "
        f"are not comparable.\n"
        f"  - to grade against {recorded}: check out those weights "
        f"(models.lock.json) and run scripts/fetch_models.py\n"
        f"  - to adopt {installed}: python -m conformance.refcli regen "
        f"(deliberate, reviewable, its own commit)"
    )

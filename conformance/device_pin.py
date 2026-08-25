"""Which device the goldens were recorded on.

A golden is not a property of the code alone. Beyond the weight set (see
`models_pin.py`) it is also a function of the *device* that produced it: the CUDA
and CPU providers agree on every discrete outcome — document type, box counts,
class labels, every OCR string — and disagree on geometry by whole pixels.

Measured on `models-v5`, reference against its own goldens, 330 stages over 9 cases:

* `--device cpu`, `--profile cpu`  -> PASS, 0 differences;
* `--device gpu`, `--profile cpu`  -> FAIL, 24 stages in 6 cases;
* `--device gpu`, `--profile gpu`  -> FAIL, 9 stages in 2 cases.

The GPU failure is NOT a stale tolerance. Coordinates reach 3 px against the
profile's 1 px allowance, but the part that no tolerance can cover is the canvas:
`viewmodel.canvas.width/height` differ by 1-2 px (702 vs 701, 502 vs 500, 620 vs
622) and the `borders.canvas` / `deskew.canvas` arrays differ in SHAPE. Both are
compared exactly on every profile by design — a canvas of a different size is a
different result — so loosening them would delete the check instead of fixing it.

Hence this pin: the runner refuses to grade on a device other than the one the
goldens were taken on, and says so in one line instead of producing dozens of
true-but-useless differences. Widening the ruler to fit today's drift would be an
edit of the reference in favour of the thing being measured; the drift itself is a
library question, tracked separately.

Pure: json and pathlib only. The checker imports this.
"""
from __future__ import annotations

import json
from pathlib import Path

from conformance.paths import CASES

#: Used when no case records a device at all — the goldens predate this pin, and
#: every one of them was CPU-generated (see spec/tolerances.md).
ASSUMED = "cpu"


def _case_files() -> list[Path]:
    return sorted(CASES.glob("*/case.json"))


def goldens_device() -> str:
    """The device every golden was recorded on.

    Cases disagreeing with each other would make one answer impossible, so that is
    reported as its own device string rather than silently resolved: a mixed set of
    goldens is a defect, not a configuration.
    """
    devices = set()
    for path in _case_files():
        try:
            args = json.loads(path.read_text(encoding="utf-8")).get("args") or {}
        except (json.JSONDecodeError, OSError):
            continue
        device = args.get("device")
        if device:
            devices.add(device)
    if not devices:
        return ASSUMED
    if len(devices) > 1:
        return "mixed:" + ",".join(sorted(devices))
    return devices.pop()


def mismatch_message(requested: str) -> str | None:
    """Human-readable reason to refuse grading, or None when it is safe."""
    golden = goldens_device()
    if golden.startswith("mixed:"):
        return ("the goldens do not agree on a device: "
                f"{golden.split(':', 1)[1]}\n"
                "Regenerate them from one device: python -m conformance.refcli regen")
    if requested == golden:
        return None
    return (
        f"goldens were recorded on {golden!r}, this run asks for {requested!r}.\n"
        "They are not comparable: the providers agree on every label, count and OCR\n"
        "string, and disagree on geometry by whole pixels — including canvas size,\n"
        "which is compared exactly on every profile. A GPU run therefore reports\n"
        "differences that are real and mean nothing about the implementation.\n"
        f"Grade on {golden!r}, or regenerate the goldens on {requested!r}."
    )

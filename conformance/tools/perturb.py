"""Deliberately break one constant, to prove the checker can fail.

A harness that has never been seen to fail is worth nothing: it may be comparing
the wrong files, ignoring the wrong paths, or silently passing everything. So M0
requires two proofs — the reference must score exactly zero differences against its
own goldens, and a single perturbed constant must make the checker fail AND name
the correct stage.

Usage:

    python -m conformance.tools.perturb --set TextFields.IOU=0.45
    python -m conformance.tools.perturb --restore

`--restore` uses `git checkout --`, so the file returns to its committed state
byte-for-byte rather than to whatever this script guesses it was.

Why this is a Python script and not two lines of PowerShell: `Set-Content -Encoding
utf8` writes a BOM, and `json.loads` rejects a BOM ("Unexpected UTF-8 BOM"). The
first attempt at this proof broke the pipeline's model loading instead of its NMS
threshold, and the checker dutifully reported an ERROR rather than a stage
divergence — the right answer to the wrong question. Writing JSON here, with
`ensure_ascii=False` and no BOM, removes that whole class of accident.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from conformance.paths import MODELS, REPO


def model_config(model: str) -> Path:
    path = MODELS / model / "ONNX" / "model.json"
    if not path.is_file():
        raise SystemExit(f"no such model config: {path}")
    return path


def _set_output_key(path: Path, key: str, value: float) -> object:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    out = cfg["Outputs"][0]
    if key not in out:
        raise SystemExit(f"{path.name}: output 0 has no key {key!r}; has {sorted(out)}")
    previous = out[key]
    out[key] = value
    # No BOM, LF newlines, trailing newline: exactly how the shipped configs look.
    path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return previous


def cmd_set(spec: str) -> int:
    try:
        target, raw = spec.split("=", 1)
        model, key = target.split(".", 1)
    except ValueError:
        raise SystemExit("expected --set <Model>.<Key>=<value>, e.g. TextFields.IOU=0.45")

    path = model_config(model)
    previous = _set_output_key(path, key, float(raw))
    print(f"{path.relative_to(REPO)}: {key} {previous} -> {raw}")
    print("restore with: python -m conformance.tools.perturb --restore")
    return 0


def cmd_restore() -> int:
    proc = subprocess.run(["git", "checkout", "--", str(MODELS.relative_to(REPO))],
                          cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return 1
    check = subprocess.run(["git", "diff", "--quiet", "--", str(MODELS.relative_to(REPO))],
                           cwd=REPO)
    print("restored" if check.returncode == 0 else "STILL MODIFIED — inspect manually")
    return 0 if check.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m conformance.tools.perturb")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", dest="spec", help="<Model>.<Key>=<value>")
    g.add_argument("--restore", action="store_true")
    args = p.parse_args(argv)
    return cmd_restore() if args.restore else cmd_set(args.spec)


if __name__ == "__main__":
    sys.exit(main())

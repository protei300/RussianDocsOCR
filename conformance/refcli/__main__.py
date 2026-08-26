"""The Python reference implementation of the conformance CLI.

Contract: `conformance/spec/cli.md`. Four subcommands — `info`, `recognize`,
`probe`, `regen`.

**stdout carries only the payload.** This matters more here than in any port,
because `document_processing` prints to stdout by design: every module prints
`[*] Loading model X!` when constructed, `process_img` prints
`[!] The document on picture has unknown type`, and `Pipeline.warmup` swallows
exceptions into a bare `print`. All of that is redirected to stderr for the whole
run, so a caller can pipe stdout into a JSON parser. A port that forgets this
produces output that looks like a serialisation bug.

`regen` and the checker share one code path by construction: `regen` writes what
`probe` and `recognize` would emit, so a golden can never disagree with a live run
for a reason other than a real behaviour change. Regeneration is always a
deliberate, reviewable commit — never a side effect of running the tests.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from conformance import cases as cases_mod
from conformance.paths import CASES, REPO, case_dir, stage_dir

# Exit codes, from spec/cli.md. `NOT_IMPLEMENTED` is the important one: a port
# under construction must be able to say "not yet" without being scored broken.
EXIT_OK = 0
EXIT_CRASH = 1
EXIT_NOT_IMPLEMENTED = 2
EXIT_INPUT = 3

#: Stages this implementation can emit. The reference emits all of them; a port
#: reports its own subset and the checker skips the difference.
# Every stage the reference emits, and therefore every stage the checker will grade it
# on. Keep it complete: a stage MISSING here is silently skipped rather than compared,
# so the reference stops self-checking on it and the hole is invisible — which is exactly
# what happened to `borders.segments` between its introduction and this list catching up.
# The `<Field>` entries are patterns, expanded by runner._claims_pattern.
STAGES_IMPLEMENTED = [
    "prepare", "doctype.label", "rotate", "quality",
    "borders.segments", "borders.canvas", "deskew.canvas", "fields.bbox",
    "address.lines", "words.<Field>.bbox", "ocr.<Field>.words", "join", "viewmodel",
]


def _quiet():
    """Redirect the library's stdout chatter to stderr for the duration."""
    return contextlib.redirect_stdout(sys.stderr)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _emit_json(payload: Any) -> None:
    """The payload, and nothing else, on stdout.

    ``ensure_ascii=False`` because the values are Cyrillic and escaping them would
    make every golden unreadable; ``sort_keys`` for a stable diff. Floats are
    already rounded by the producer (see spec/viewmodel.md), so no formatting is
    applied here.
    """
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #

def cmd_info(args: argparse.Namespace) -> int:
    with _quiet():
        import numpy as np
        import onnxruntime as ort

        import document_processing
        try:
            import cv2
            opencv = cv2.__version__
        except Exception:
            opencv = None

    _emit_json({
        "port": "python",
        "language": f"CPython {platform.python_version()}",
        "versions": {
            "runtime": platform.python_version(),
            "library": getattr(document_processing, "__version__", None),
            "onnxruntime": ort.__version__,
            "opencv": opencv,
            "numpy": np.__version__,
        },
        "device": args.device,
        "ocr_device": None,   # only known once a Pipeline exists; see recognize
        "providers": ort.get_available_providers(),
        "model_format": args.model_format,
        "ocr_mode": args.ocr,
        "stages_implemented": STAGES_IMPLEMENTED,
        "commit": _git_commit(),
        "platform": platform.platform(),
    })
    return EXIT_OK


# --------------------------------------------------------------------------- #
# shared pipeline plumbing
# --------------------------------------------------------------------------- #

def _build_pipeline(args: argparse.Namespace):
    with _quiet():
        from document_processing import Pipeline
        return Pipeline(model_format=args.model_format, device=args.device,
                        ocr=args.ocr, verbose=False)


def _process(pipeline, image: Path, args: argparse.Namespace):
    with _quiet():
        return pipeline.process_img(image, img_size=args.img_size, docconf=args.docconf)


def _viewmodel(pipeline, results, args: argparse.Namespace) -> dict[str, Any]:
    """Build the client-facing view model.

    Reuses ``service/ml/transform.py`` rather than reimplementing it: that module
    is the reference implementation of the shape defined in spec/viewmodel.md, and
    a second copy here would be free to drift from it.
    """
    with _quiet():
        from service.ml.transform import build_viewmodel
        return build_viewmodel(results, device=getattr(pipeline, "device", args.device),
                               include_debug=args.include_debug)


# --------------------------------------------------------------------------- #
# recognize
# --------------------------------------------------------------------------- #

def cmd_recognize(args: argparse.Namespace) -> int:
    image = Path(args.image)
    if not image.is_file():
        print(f"no such image: {image}", file=sys.stderr)
        return EXIT_INPUT

    pipeline = _build_pipeline(args)
    results = _process(pipeline, image, args)
    _emit_json(_viewmodel(pipeline, results, args))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #

def cmd_probe(args: argparse.Namespace) -> int:
    image = Path(args.image)
    if not image.is_file():
        print(f"no such image: {image}", file=sys.stderr)
        return EXIT_INPUT

    with _quiet():
        from document_processing.pipeline.probe import DirectoryStageSink

    dump = Path(args.dump_dir)
    pipeline = _build_pipeline(args)

    with DirectoryStageSink(dump, upto=args.upto) as sink:
        pipeline.probe = sink
        try:
            results = _process(pipeline, image, args)
        finally:
            # Always detach: a Pipeline is reused across calls, and a stale sink
            # would keep writing into a directory the next caller does not expect.
            pipeline.probe = None

        # `viewmodel` is a stage too, but the library cannot emit it -- the
        # transform lives outside document_processing. Emitted here so that
        # `probe` and `recognize` cannot disagree, and skipped when --upto stopped
        # earlier (asking for --upto rotate should not run the transform).
        if args.upto in (None, "viewmodel"):
            sink.emit("viewmodel", _viewmodel(pipeline, results, args))

    print(f"wrote {len(sink.index)} stage(s) to {dump}", file=sys.stderr)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# regen
# --------------------------------------------------------------------------- #

#: Stages whose payload is a full-resolution image. Their pixels are NOT committed
#: -- see `_digest_image_stages` for why.
IMAGE_STAGES = frozenset({"prepare", "rotate", "borders.canvas", "deskew.canvas"})


def _digest_image_stages(stages: Path, keep_pixels: bool) -> int:
    """Replace committed image pixels with a digest.

    Four image stages per case at roughly 2 MB each, across seven cases, is over
    50 MB of binary in git — for an open-source reference project that is a bad
    trade, and it grows every time a document type is added.

    A digest still delivers what the harness is for: the headline result is the
    FIRST DIVERGENT STAGE, and "prepare differs" is exactly that. What a digest
    cannot give is the *magnitude*, which is what relaxation R-02 needs — so when
    a digest mismatch appears, the developer regenerates pixels locally with
    `regen --with-pixels` (gitignored) and compares those. Localisation is
    committed; forensics are reproducible on demand.

    The digest is taken over the ARRAY BYTES, not the file: a .npy header contains
    padding to a 64-byte boundary, so two writers could produce byte-different
    files holding identical arrays.
    """
    import hashlib

    index_path = stages / "stages.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    converted = 0

    for entry in index["stages"]:
        if entry["kind"] != "npy" or entry["stage"] not in IMAGE_STAGES:
            continue
        npy_path = stages / entry["file"]
        if not npy_path.is_file():
            continue

        import numpy as np
        arr = np.load(npy_path, allow_pickle=False)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()

        digest_file = f"{entry['stage'].replace('/', '_')}.digest.json"
        (stages / digest_file).write_text(json.dumps({
            "stage": entry["stage"],
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256": digest,
            "note": "pixels are not committed; regenerate with "
                    "`python -m conformance.refcli regen --with-pixels` to compare magnitudes",
        }, indent=2), encoding="utf-8")

        if not keep_pixels:
            npy_path.unlink()
        entry["kind"] = "digest"
        entry["file"] = digest_file
        converted += 1

    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return converted


def cmd_regen(args: argparse.Namespace) -> int:
    selected = cases_mod.select(args.case, limit=args.limit)
    if not selected:
        print("no cases matched", file=sys.stderr)
        return EXIT_INPUT

    missing = [c.slug for c in selected if not c.exists()]
    if missing:
        print(f"sample images missing for: {', '.join(missing)}", file=sys.stderr)
        return EXIT_INPUT

    with _quiet():
        from document_processing.pipeline.probe import DirectoryStageSink

    # One Pipeline for every case: constructing it costs seconds, and reusing it
    # is also the realistic usage pattern (`process_img` rebinds `self.results`
    # and `self.ocr_options`, which is exactly why the service leases it).
    pipeline = _build_pipeline(args)
    CASES.mkdir(parents=True, exist_ok=True)

    written = []
    for case in selected:
        target = case_dir(case.slug)
        stages = stage_dir(case.slug)
        target.mkdir(parents=True, exist_ok=True)

        with DirectoryStageSink(stages) as sink:
            pipeline.probe = sink
            try:
                results = _process(pipeline, case.image, args)
            finally:
                pipeline.probe = None
            viewmodel = _viewmodel(pipeline, results, args)
            sink.emit("viewmodel", viewmodel)

        # Image pixels become digests unless explicitly kept (see the function's
        # docstring for the size argument).
        _digest_image_stages(stages, keep_pixels=args.with_pixels)

        (target / "viewmodel.json").write_text(
            json.dumps(viewmodel, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8")
        (target / "case.json").write_text(json.dumps({
            "slug": case.slug,
            "sample": case.sample,
            "doc_type": case.doc_type,
            "args": {"device": args.device, "ocr": args.ocr,
                     "img_size": args.img_size, "docconf": args.docconf,
                     "model_format": args.model_format},
            "expect": {"doc_type": viewmodel.get("doc_type")},
            "note": "Goldens are CPU-generated; see spec/tolerances.md for the "
                    "separate GPU numeric profile.",
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        written.append((case.slug, viewmodel.get("doc_type"), len(sink.index)))
        print(f"  {case.slug:48s} {viewmodel.get('doc_type'):22s} {len(sink.index)} stages",
              file=sys.stderr)

    # Record which weights produced these numbers. Regenerating the goldens and
    # changing the models are the same act; without this the runner cannot tell a
    # real regression from a model swap (see conformance/models_pin.py).
    from conformance import models_pin
    installed = models_pin.installed_models_version()
    models_pin.write_pin(installed)
    print(f"pinned to models {installed or 'unknown'}", file=sys.stderr)

    print(f"regenerated {len(written)} case(s) under {CASES}", file=sys.stderr)
    _emit_json([{"slug": s, "doc_type": d, "stages": n} for s, d, n in written])
    return EXIT_OK


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m conformance.refcli",
                                description="Python reference conformance CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="goldens are CPU-generated; see spec/tolerances.md")
        sp.add_argument("--ocr", default="accurate", choices=["accurate", "fast"])
        sp.add_argument("--img-size", type=int, default=1500, dest="img_size")
        sp.add_argument("--docconf", type=float, default=0.5)
        sp.add_argument("--model-format", default="ONNX", dest="model_format")
        sp.add_argument("--include-debug", action="store_true", dest="include_debug")

    sp = sub.add_parser("info", help="describe this implementation")
    common(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("recognize", help="emit the view model for one image")
    sp.add_argument("--image", required=True)
    common(sp)
    sp.set_defaults(func=cmd_recognize)

    sp = sub.add_parser("probe", help="dump per-stage intermediates for one image")
    sp.add_argument("--image", required=True)
    sp.add_argument("--dump-dir", required=True, dest="dump_dir")
    sp.add_argument("--upto", default=None,
                    help="stop AFTER this stage (inclusive); see spec/stages.md")
    common(sp)
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("regen", help="regenerate the golden files")
    sp.add_argument("--case", action="append", default=None,
                    help="slug substring; repeatable. Default: every case")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--with-pixels", action="store_true", dest="with_pixels",
                    help="also keep the full image arrays (large; gitignored) so "
                         "magnitudes and relaxation R-02 can be evaluated locally")
    common(sp)
    sp.set_defaults(func=cmd_regen)

    return p


def main(argv: list[str] | None = None) -> int:
    # stdout is a data channel, not a display: the contract says UTF-8 (spec/cli.md)
    # and the checker decodes it as UTF-8. Python otherwise writes it in the machine's
    # locale encoding, which on a Russian Windows is cp1251 — measured there, every
    # Cyrillic OCR string arrived at the checker as replacement characters and the
    # reference "failed" against its own goldens. Pinned here rather than only in the
    # checker so that the reference reads the same whoever launches it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):   # not a TextIOWrapper (redirected, tests)
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    except FileNotFoundError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except Exception:
        traceback.print_exc()
        return EXIT_CRASH


if __name__ == "__main__":
    sys.exit(main())

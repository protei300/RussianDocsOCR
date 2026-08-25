#!/usr/bin/env python
"""Pre-compute the seed documents so startup costs nothing.

Run once, by hand, when the samples or the recognition library change::

    python service/tools/build_seed_data.py

Recognises one sample per document type and writes the result plus the rendered
canvas into ``service/seed_data/``. The service then seeds those directly as
completed documents — the web UI shows real recognition results immediately
instead of spending a minute of GPU time re-deriving the same answers at every
start.

The output doubles as golden files: a port of this service to another language
must reproduce these payloads byte-for-byte from the same inputs.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from service.core.seed import _TYPE_ORDER  # noqa: E402
from service.ml import runtime  # noqa: E402
from service.ml.transform import build_search_text  # noqa: E402

SAMPLES = ROOT / "samples"
OUT = ROOT / "service" / "seed_data"


def _planned() -> list[dict]:
    """The cases to rebuild, taken from the existing manifest.

    The manifest is the single source of the case list - conformance derives its
    cases from it, so a second way of deciding what exists would be a second list
    by another name. Rediscovering material by scanning ``samples/`` was exactly
    that: it decided membership by what happened to sit in a folder, which is why
    material living anywhere else could not be used at all, and why an entry
    deliberately marked withdrawn would silently vanish on the next rebuild.

    Falls back to the old scan only when there is no manifest yet - bootstrapping
    a fresh checkout still has to work.
    """
    path = OUT / "manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    plan = []
    for type_name in _TYPE_ORDER:
        sample = next(iter(sorted((SAMPLES / type_name).glob("*.jpg"))), None)
        if sample is not None:
            rel = str(sample.relative_to(ROOT)).replace("\\", "/")
            plan.append({"slug": f"{type_name}-{sample.stem}", "sample": rel})
    return plan


def main() -> None:
    # Warm up on a real document, exactly as the service does at startup.
    # Without this the first document through the loop absorbs the whole
    # first-call cost — lazy CUDA kernel compilation, allocator growth, the OCR
    # graph's first dynamic width — and the timing baked into the fixture reads
    # as a multi-second document rather than the ~400 ms it actually takes.
    # Passing no image is not equivalent: `runtime` then feeds a synthetic grey
    # frame, which classifies as 'NONE' and short-circuits before OCR, so only
    # the head of the chain gets warm.
    warmup = next(iter(sorted((SAMPLES / _TYPE_ORDER[0]).glob("*.jpg"))), None)
    info = runtime.init_runtime(compute_device="auto", ocr_mode="accurate",
                                warmup_image=warmup)
    if info.state != "ready":
        raise SystemExit(f"recognition runtime failed to start: {info.error}")
    print(f"[BUILD] {info.device} ready, warmed up on {warmup.name if warmup else 'nothing'}")

    plan = _planned()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest = []
    for planned in plan:
        slug = planned["slug"]
        type_name = slug.split("-", 1)[0]

        # A withdrawn entry keeps its place in the list and loses only its data.
        # Dropping it instead would make the case disappear from conformance too,
        # and a case that vanishes is indistinguishable from one that never was.
        if planned.get("disabled"):
            manifest.append(planned)
            print(f"  {type_name:24} withdrawn, kept as a declared gap")
            continue

        sample = ROOT / planned["sample"]
        if not sample.is_file():
            print(f"  {type_name:24} sample {planned['sample']} is gone, skipped")
            continue

        payload, canvas = runtime.recognise(sample)
        entry_dir = OUT / slug
        entry_dir.mkdir()

        # The original is NOT copied — it already lives in samples/ and the
        # manifest records the path. Duplicating it would double the size of
        # this fixture set for no benefit.
        if canvas is not None:
            # RGB in, BGR out — see repositories/artifacts.save_canvas for why
            # skipping this swaps red and blue in every rendered document.
            bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            # JPEG, not PNG: this is a display image behind a box overlay, and
            # PNG made the committed fixtures roughly five times larger for a
            # difference nobody can see at these sizes.
            cv2.imwrite(str(entry_dir / "canvas.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            small_w = 96
            small_h = max(1, round(canvas.shape[0] * small_w / canvas.shape[1]))
            cv2.imwrite(str(entry_dir / "thumb.jpg"),
                        cv2.cvtColor(cv2.resize(canvas, (small_w, small_h),
                                                interpolation=cv2.INTER_AREA),
                                     cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])

        (entry_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

        filename = f"sample-{type_name}-{sample.name}"
        manifest.append({
            "slug": slug,
            "sample": str(sample.relative_to(ROOT)).replace("\\", "/"),
            "filename": filename,
            "original_ext": sample.suffix,
            "content_type": "image/jpeg",
            "size_bytes": sample.stat().st_size,
            "search_text": build_search_text(filename, payload),
        })
        size_kb = sum(f.stat().st_size for f in entry_dir.iterdir()) / 1024
        print(f"  {type_name:24} {payload['doc_type']:22} "
              f"{len(payload.get('fields') or [])} fields  {size_kb:.0f} KB")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    total_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\n{len(manifest)} document(s) -> {OUT}  ({total_mb:.1f} MB)")
    runtime.shutdown_runtime()


if __name__ == "__main__":
    main()

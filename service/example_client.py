#!/usr/bin/env python
"""Minimal API client — upload one image, wait for the result, print it.

Doubles as the worked example of calling this service from another program.
Standard library only, so it can be copied into any environment without
installing anything.

Usage::

    python service/example_client.py samples/INTPASSPORT_2011/12_CR_INTPASSPORT_2011.jpg
    python service/example_client.py D:/Grant/my_passp.jpg --key rdk_... --json
    python service/example_client.py photo.jpg --save-canvas out/

The API key defaults to the development one; override with ``--key`` or the
``RD_API_KEY`` environment variable.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8002"

#: No hardcoded fallback on purpose. When ``DEFAULT_API_KEY`` is not set the
#: service generates a random bootstrap key at every start, so any constant here
#: would be wrong most of the time and fail as a confusing 401. Empty means
#: "tell the user where to get it" instead.
DEFAULT_KEY = os.environ.get("RD_API_KEY", "")
KEY_HINT = ("No API key. Pass --key, or set RD_API_KEY.\n"
            "  Find the key in the service's startup log ([BOOT] ... API key),\n"
            "  or on the API keys page of the web UI.")


def _opener() -> urllib.request.OpenerDirector:
    """An opener that ignores system proxies.

    Necessary on machines behind a corporate proxy: with HTTP_PROXY set, even
    a request to 127.0.0.1 is routed through it and comes back as the proxy's
    HTML error page, which looks exactly like the service being broken.
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(url: str, key: str, *, method: str = "GET",
             body: bytes | None = None, content_type: str | None = None) -> bytes:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("X-API-Key", key)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with _opener().open(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:
            pass
        raise SystemExit(f"HTTP {exc.code} from {url}\n  {detail}") from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {url}: {exc.reason}\n"
                         f"  Is the service running?") from None


def upload(base: str, key: str, path: Path) -> dict:
    """POST the file as multipart/form-data, hand-built to avoid dependencies."""
    boundary = uuid.uuid4().hex
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode("utf-8"),
        f"Content-Type: {media}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    raw = _request(f"{base}/api/v1/documents", key, method="POST", body=body,
                   content_type=f"multipart/form-data; boundary={boundary}")
    return json.loads(raw)


def wait_for(base: str, key: str, doc_id: int, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last_step = ""
    while time.time() < deadline:
        doc = json.loads(_request(f"{base}/api/v1/documents/{doc_id}", key))
        if doc["status"] in ("done", "failed"):
            print()
            return doc
        progress = json.loads(_request(f"{base}/api/v1/documents/{doc_id}/progress", key) or b"null")
        step = (progress or {}).get("label", doc["status"])
        if step != last_step:
            print(f"\r  {step}…", end="", flush=True)
            last_step = step
        time.sleep(0.4)
    raise SystemExit(f"Timed out after {timeout}s waiting for document {doc_id}")


def print_report(doc: dict) -> None:
    print(f"  document type : {doc['doc_type'] or '—'}"
          f"{'' if doc['recognised'] else '  (not recognised)'}")
    print(f"  confidence    : {doc['doc_conf']}")
    print(f"  device / time : {doc['device']} / {doc['processing_ms']} ms")

    quality = {k: v for k, v in (doc.get("quality") or {}).items() if k != "DocConf"}
    if quality:
        print("  quality       : " + "  ".join(f"{k}={v}" for k, v in quality.items()))

    fields = doc.get("fields") or []
    if fields:
        print(f"\n  Fields ({len(fields)}):")
        width = max(len(f["display"]) for f in fields)
        for f in fields:
            conf = f"{f['conf']:.2f}" if f["conf"] is not None else "  — "
            boxes = ",".join(f["box_ids"]) or "—"
            print(f"    {f['display']:<{width}}  {conf}  {str(f['value'] or '—'):<34} [{boxes}]")

    address = doc.get("address")
    if address:
        print(f"\n  Address lines (geometry {'aligned' if address['aligned'] else 'UNALIGNED'}):")
        for line in address["lines"]:
            print(f"    [{line['kind']:>11}] {line['text'] or '(not recognised)'}")

    boxes = doc.get("boxes") or []
    if boxes:
        print(f"\n  Boxes: {len(boxes)} in {doc.get('coord_space')} space "
              f"({doc['canvas'].get('width')}x{doc['canvas'].get('height')} px)")
        ambiguous = [b["id"] for b in boxes if b.get("ambiguous")]
        if ambiguous:
            # Several boxes share one label: split fields, or the doubled
            # Licence_number on internal passports. The text is attached to the
            # highest-confidence one; the rest are flagged rather than repeated.
            print(f"    sharing a label with another box: {', '.join(ambiguous)}")

    timings = doc.get("timings") or {}
    if timings:
        slowest = sorted(((v, k) for k, v in timings.items() if k != "total"), reverse=True)[:4]
        print("\n  Slowest stages: "
              + ", ".join(f"{k} {v * 1000:.0f}ms" for v, k in slowest))

    if doc.get("error"):
        print(f"\n  ERROR [{doc.get('error_code')}]: {doc['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="image file to recognise")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"service base URL (default {DEFAULT_URL})")
    parser.add_argument("--key", default=DEFAULT_KEY, help="API key (or set RD_API_KEY)")
    parser.add_argument("--json", action="store_true", help="print the raw JSON instead of a report")
    parser.add_argument("--save-canvas", type=Path, metavar="DIR",
                        help="also download the corrected canvas image into DIR")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"No such file: {args.image}")
    if not args.key:
        raise SystemExit(KEY_HINT)

    base = args.url.rstrip("/")
    size_kb = args.image.stat().st_size / 1024
    print(f"→ uploading {args.image.name} ({size_kb:.0f} KB) to {base}")

    queued = upload(base, args.key, args.image)
    doc_id = queued["id"]
    position = queued.get("queue_position")
    print(f"  queued as #{doc_id}" + (f" (position {position})" if position else ""))

    started = time.perf_counter()
    doc = wait_for(base, args.key, doc_id)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print(f"✓ {doc['status']} in {elapsed:.1f}s wall clock\n")
        print_report(doc)

    if args.save_canvas:
        args.save_canvas.mkdir(parents=True, exist_ok=True)
        for kind in ("canvas", "original"):
            try:
                data = _request(f"{base}/api/v1/documents/{doc_id}/image/{kind}", args.key)
            except SystemExit:
                continue
            suffix = ".png" if kind == "canvas" else args.image.suffix
            out = args.save_canvas / f"{doc_id}_{kind}{suffix}"
            out.write_bytes(data)
            print(f"  saved {out} ({len(data) / 1024:.0f} KB)")

    print(f"\n  Web UI: {base}/documents/{doc_id}")
    sys.exit(0 if doc["status"] == "done" else 1)


if __name__ == "__main__":
    main()

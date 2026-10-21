"""Black-box check of in-memory recognition, against ANY of the four services.

    python -m conformance.memory_contract --port python
    python -m conformance.memory_contract --port go
    python -m conformance.memory_contract --port dotnet
    python -m conformance.memory_contract --port java

The contract is ports/MEMORY.md. Like auth_contract.py, this starts the service once
per scenario on a free port with a fresh DATA_DIR, drives it over HTTP only and stops
it — so the same checks grade the Python reference and every port.

Scenarios:

    memory     PROCESSING_MODE=memory — the document routes are gone, /recognize answers,
               and afterwards neither the data directory nor the log holds anything
               from the document
    stored     PROCESSING_MODE unset — /recognize works next to the document store and
               adds nothing to it
    mistyped   PROCESSING_MODE=memroy — must mean memory (fail toward privacy), and say why

What a black box cannot see — that buffers are zeroed, that the body is never spooled
to a temporary file outside DATA_DIR — each implementation proves in its own unit tests
(ports/MEMORY.md §8).
"""
from __future__ import annotations

import argparse
import base64
import os
import pathlib
import shlex
import sys
import time

from conformance.auth_contract import PRESETS, REPO, Report, Service, expect, expect_status

API_KEY = "rdk_memory_contract_key"
PIN = "1234"                    # the default AUTH_PIN
SAMPLE = REPO / "samples" / "DL_2011" / "1_CR_DL_2010.jpg"
#: Sent as the multipart filename; it must never appear in a log or on disk.
CANARY_NAME = "CANARY-FILENAME-7f3a.jpg"
JOURNAL_KEYS = {"id", "at", "outcome", "status_code", "processing_ms", "stored"}
OUTCOMES = {"recognised", "not_recognised", "rejected", "busy", "error"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_UPLOAD_MB = 2


class Svc(Service):
    _token: str | None = None

    def key(self) -> dict[str, str]:
        return {"X-API-Key": API_KEY}

    def session(self) -> dict[str, str]:
        """/status and /logs are for signed-in users, not API keys."""
        if self._token is None:
            resp = self.http.post("/api/v1/auth/pin-login", json={"pin": PIN})
            expect_status(resp, 200, "PIN sign-in")
            self._token = resp.json()["access_token"]
        return {"Authorization": f"Bearer {self._token}"}

    def recognize(self, body: bytes, content_type: str = "image/jpeg", images: bool = False,
                  auth: bool = True):
        headers = {"Content-Type": content_type, **(self.key() if auth else {})}
        return self.http.post("/api/v1/recognize", content=body, headers=headers,
                              params={"images": "1"} if images else None)

    def recognize_multipart(self, body: bytes, images: bool = False):
        return self.http.post("/api/v1/recognize", headers=self.key(),
                              files={"file": (CANARY_NAME, body, "image/jpeg")},
                              params={"images": "1"} if images else None)

    def wait_ready(self, seconds: float = 300) -> None:
        """Until the models are loaded /recognize answers 503 + Retry-After (busy) —
        correct, but not what the checks are about."""
        deadline = time.monotonic() + seconds
        image = SAMPLE.read_bytes()
        while time.monotonic() < deadline:
            resp = self.recognize(image)
            if resp.status_code != 503:
                return
            expect(resp.headers.get("Retry-After", "").isdigit(), "503 must carry Retry-After")
            time.sleep(2)
        raise RuntimeError("the recognition runtime never became ready")

    def journal(self) -> dict:
        resp = self.http.get("/api/v1/recognitions", headers=self.key())
        expect_status(resp, 200, "GET /recognitions")
        return resp.json()


def _secrets(answer: dict) -> list[str]:
    """Strings from a recognition that must never be written anywhere."""
    values = [f.get("value") for f in answer.get("fields", [])]
    return [v for v in values if isinstance(v, str) and len(v) >= 5]


# --------------------------------------------------------------------------- shared checks

def check_recognize(svc: Svc, r: Report, state: dict) -> None:
    image = SAMPLE.read_bytes()
    svc.wait_ready()

    def raw_body():
        resp = svc.recognize(image)
        expect_status(resp, 200, "POST /recognize with the image as the body")
        expect("no-store" in resp.headers.get("Cache-Control", ""),
               f"Cache-Control must be no-store, got {resp.headers.get('Cache-Control')!r}")
        body = resp.json()
        expect(body.get("stored") is False, f"stored must be false, got {body.get('stored')!r}")
        expect(body.get("recognised") is True and body.get("doc_type") == "DL_2011",
               f"expected DL_2011, got {body.get('doc_type')!r}")
        expect(isinstance(body.get("fields"), list) and len(body["fields"]) >= 8,
               f"expected the licence fields, got {len(body.get('fields') or [])}")
        expect("image_png_base64" not in (body.get("canvas") or {}),
               "the canvas image must be returned only when asked (?images=1)")
        for key in ("doc_type_base", "doc_type_era", "doc_conf", "boxes", "quality",
                    "timings", "processing_ms", "canvas"):
            expect(key in body, f"response lacks {key!r}")
        expect("DocConf" not in (body.get("quality") or {}), "DocConf belongs in doc_conf, not quality")
        state["answer"] = body
    r.check("raw image body is recognised and answered, stored=false, no-store", raw_body)

    def multipart_with_image():
        resp = svc.recognize_multipart(image, images=True)
        expect_status(resp, 200, "POST /recognize?images=1 as multipart")
        canvas = resp.json().get("canvas") or {}
        b64 = canvas.get("image_png_base64")
        expect(isinstance(b64, str) and b64, "?images=1 must return canvas.image_png_base64")
        png = base64.b64decode(b64)
        expect(png.startswith(PNG_MAGIC), "image_png_base64 is not a PNG")
        expect(canvas.get("width") and canvas.get("height"), "canvas width/height missing")
    r.check("multipart 'file' part is accepted; ?images=1 returns the canvas PNG", multipart_with_image)

    def refusals():
        expect_status(svc.recognize(image, auth=False), 401, "no credentials")
        expect_status(svc.recognize(b"%PDF-1.4\n%\xe2\xe3\n1 0 obj", "application/pdf"), 415, "a PDF")
        expect_status(svc.recognize(b"definitely not an image" * 10, "image/jpeg"), 415, "text sent as an image")
        expect_status(svc.recognize(b"", "image/jpeg"), 400, "an empty body")
        broken = image[:2000]                       # a JPEG header, then nothing
        resp = svc.recognize(broken)
        expect(resp.status_code in (422, 200), f"a truncated JPEG: expected 422 (or a decoder that "
               f"recovers: 200), got {resp.status_code}")
        big = b"\xff\xd8\xff\xe0" + b"\0" * (MAX_UPLOAD_MB * 1024 * 1024 + 200 * 1024)
        expect_status(svc.recognize(big), 413, f"a body over MAX_UPLOAD_MB={MAX_UPLOAD_MB}")
        resp = svc.http.post("/api/v1/recognize", headers=svc.key(),
                             files={"other": ("x.txt", b"hello", "text/plain")},
                             data={"note": "x"})
        expect(resp.status_code in (400, 415), f"multipart without an image part: {resp.status_code}")
    r.check("refusals: 401 / 415 PDF / 415 non-image / 400 empty / 413 too large", refusals)

    def journal():
        body = svc.journal()
        items, total, counts = body.get("items"), body.get("total"), body.get("counts")
        expect(isinstance(items, list) and items, "journal is empty after recognitions")
        expect(isinstance(total, int) and total >= len(items), f"total {total!r}")
        for row in items:
            expect(set(row) == JOURNAL_KEYS, f"journal row must have exactly {sorted(JOURNAL_KEYS)}, "
                   f"got {sorted(row)}")
            expect(row["outcome"] in OUTCOMES, f"outcome {row['outcome']!r}")
            expect(row["stored"] is False, "journal rows are stored=false")
        ids = [row["id"] for row in items]
        expect(ids == sorted(ids, reverse=True), f"newest first: {ids}")
        expect(items[0]["outcome"] == "rejected", f"latest (a refusal) outcome {items[0]['outcome']!r}")
        expect(any(row["outcome"] == "recognised" and row["status_code"] == 200 for row in items),
               "no recognised/200 row")
        expect(any(row["status_code"] == 413 for row in items), "a 413 refusal is journalled too")
        expect(all(row["status_code"] != 401 for row in items),
               "an unauthenticated request must not reach the journal")
        expect(isinstance(counts, dict) and counts.get("recognised", 0) >= 2,
               f"counts {counts!r}")
        expect(counts.get("total") == total, f"counts.total {counts.get('total')!r} vs total {total}")
    r.check("journal: outcome/status/time only, newest first, refusals included", journal)

    def nothing_logged():
        secrets = _secrets(state.get("answer") or {}) + [CANARY_NAME, "DL_2011"]
        logs = svc.log_text()
        resp = svc.http.get("/api/v1/logs", headers=svc.session(), params={"limit": 1000})
        expect_status(resp, 200, "GET /logs")
        logs += resp.text
        leaked = [s for s in secrets if s in logs]
        expect(not leaked, f"the log holds document data: {leaked}")
        lines = [ln for ln in svc.log_text().splitlines() if "[API] recognize:" in ln]
        expect(len(lines) >= 6, f"expected one '[API] recognize:' line per request, got {len(lines)}")
        for ln in lines:
            tail = ln.split("[API] recognize:", 1)[1]
            tail = tail.split('"', 1)[0].strip()
            expect(tail.startswith("outcome=") and " status=" in tail and " ms=" in tail
                   and len(tail.split()) == 3,
                   f"the recognize line must be exactly 'outcome=… status=… ms=…': {tail!r}")
    r.check("log: one line per request, outcome/status/ms only; no field value, filename or type",
            nothing_logged)

    def nothing_on_disk():
        secrets = [s.encode("utf-8") for s in _secrets(state.get("answer") or {})]
        secrets += [CANARY_NAME.encode(), image[200:264]]
        root = svc.data_dir / "data"
        found = []
        for path in (root.rglob("*") if root.exists() else []):
            if not path.is_file():
                continue
            blob = path.read_bytes()
            for s in secrets:
                if s in blob:
                    found.append(f"{path.relative_to(root)} holds {s[:24]!r}")
        expect(not found, "; ".join(found[:5]))
    r.check("DATA_DIR holds neither the image, the filename nor any recognised value",
            nothing_on_disk)


# --------------------------------------------------------------------------- scenarios

def scenario_memory(svc: Svc, r: Report) -> None:
    def config():
        body = svc.http.get("/api/v1/auth/config").json()
        expect(body.get("processing_mode") == "memory", f"auth/config processing_mode {body.get('processing_mode')!r}")
        status = svc.http.get("/api/v1/status", headers=svc.session())
        expect_status(status, 200, "GET /status")
        proc = status.json().get("processing") or {}
        expect(proc.get("mode") == "memory" and proc.get("reason") is None, f"status.processing {proc!r}")
        expect(isinstance(proc.get("recognitions"), dict), "status.processing.recognitions")
    r.check("auth/config and status report memory mode", config)

    def documents_gone():
        for method, path in (("GET", "/api/v1/documents"), ("GET", "/api/v1/documents/1"),
                             ("GET", "/api/v1/documents/1/image/original"),
                             ("DELETE", "/api/v1/documents/1")):
            resp = svc.http.request(method, path, headers=svc.key())
            expect_status(resp, 404, f"{method} {path}")
        resp = svc.http.post("/api/v1/documents", headers=svc.key(),
                             files={"file": (CANARY_NAME, SAMPLE.read_bytes(), "image/jpeg")})
        expect_status(resp, 404, "POST /documents (upload)")
    r.check("every document route answers 404", documents_gone)

    state: dict = {}
    check_recognize(svc, r, state)


def scenario_stored(svc: Svc, r: Report) -> None:
    def config():
        body = svc.http.get("/api/v1/auth/config").json()
        expect(body.get("processing_mode") == "stored", f"processing_mode {body.get('processing_mode')!r}")
    r.check("auth/config reports stored mode by default", config)

    def recognise_adds_nothing():
        before = svc.http.get("/api/v1/documents", headers=svc.key())
        expect_status(before, 200, "GET /documents")
        svc.wait_ready()
        resp = svc.recognize(SAMPLE.read_bytes())
        expect_status(resp, 200, "POST /recognize in stored mode")
        after = svc.http.get("/api/v1/documents", headers=svc.key()).json()
        expect(after.get("total") == before.json().get("total"),
               f"/recognize added a document: {before.json().get('total')} -> {after.get('total')}")
        rows = svc.journal().get("items") or []
        expect(rows and rows[0]["outcome"] == "recognised", "the journal holds the recognition")
    r.check("/recognize works next to the store and adds no document", recognise_adds_nothing)


def scenario_mistyped(svc: Svc, r: Report) -> None:
    def fails_toward_privacy():
        body = svc.http.get("/api/v1/auth/config").json()
        expect(body.get("processing_mode") == "memory",
               f"a mistyped PROCESSING_MODE must mean memory, got {body.get('processing_mode')!r}")
        proc = svc.http.get("/api/v1/status", headers=svc.session()).json().get("processing") or {}
        expect(proc.get("mode") == "memory", f"status mode {proc.get('mode')!r}")
        expect(isinstance(proc.get("reason"), str) and "memroy" in proc["reason"],
               f"status.processing.reason must name the bad value, got {proc.get('reason')!r}")
        expect_status(svc.http.get("/api/v1/documents", headers=svc.key()), 404, "GET /documents")
    r.check("PROCESSING_MODE=memroy -> memory, with the reason on /status", fails_toward_privacy)


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", choices=sorted(PRESETS), help="an implementation preset")
    parser.add_argument("--cmd", help="a command line with {port}, instead of a preset")
    parser.add_argument("--only", nargs="*", help="run only these scenarios")
    args = parser.parse_args()
    if not args.port and not args.cmd:
        parser.error("give --port or --cmd")
    cmd = shlex.split(args.cmd, posix=os.name != "nt") if args.cmd else PRESETS[args.port]
    os.environ.pop("PROCESSING_MODE", None)        # only the scenario decides it

    base = {"DEFAULT_API_KEY": API_KEY, "MAX_UPLOAD_MB": str(MAX_UPLOAD_MB)}
    scenarios = [
        ("memory", {**base, "PROCESSING_MODE": "memory"}, scenario_memory),
        ("stored", base, scenario_stored),
        ("mistyped", {**base, "PROCESSING_MODE": "memroy"}, scenario_mistyped),
    ]
    report = Report()
    for name, env, run in scenarios:
        if args.only and name not in args.only:
            continue
        print(f"\n== {name}  {env}")
        try:
            svc = Svc(cmd, env, f"mem-{name}")
        except Exception as error:                              # noqa: BLE001
            report.failed.append((f"{name}: start", str(error)))
            print(f"    ERROR could not start: {error}")
            continue
        try:
            run(svc, report)
        finally:
            svc.stop()

    print(f"\n{len(report.passed)} passed, {len(report.failed)} failed")
    for name, why in report.failed:
        print(f"  - {name}: {why}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())

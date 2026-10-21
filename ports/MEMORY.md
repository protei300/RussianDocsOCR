# In-memory recognition — the contract every service implements

Normative for the Go, .NET and Kotlin services, like [`AUTH.md`](AUTH.md). The reference
is the Python service: `service/core/processing.py`, `service/api/recognize.py`,
`service/repositories/recognitions.py`, `service/ml/runtime.py` (rule 11,
`recognise_image`, `wipe_arrays`), the middleware `_no_document_routes_in_memory_mode` and
`_announce_processing_mode` in `service/main.py`, `worker.start_worker(drain=…)`. **Why**
the mode exists is in [`docs/in-memory.md`](../docs/in-memory.md) (Russian); this file is
**what** to build.

**The executable form of this file is [`conformance/memory_contract.py`](../conformance/memory_contract.py).**
It starts a service three times (memory, stored, a mistyped mode), drives it over HTTP only
and checks every rule below that is visible from outside — including that afterwards
neither `DATA_DIR` nor the log holds the filename, the document type or any recognised
value. The reference passes it; a port is done when it does too, **and** its own unit
tests prove §7 (what a black box cannot see):

```bash
python -m conformance.memory_contract --port python    # must be green first
python -m conformance.memory_contract --port go
python -m conformance.memory_contract --port dotnet
python -m conformance.memory_contract --port java
```

The user's requirement, verbatim in substance: a document arrives through the API, is
recognised in memory, answered, the memory is explicitly cleared; a short log line without
anything about the document; in the UI one sees *that* something was recognised but
cannot open it. No images, no personal data, no artifacts.

---

## 1. Mode, chosen by configuration only

`PROCESSING_MODE`, trimmed and lower-cased; one function returns `(mode, reason)` and
nothing else reads the raw setting. **Resolution never fails, and a wrong value fails
toward privacy** — the opposite of `AUTH_MODE`, deliberately: somebody who typed `memroy`
asked for documents not to be stored.

| raw value | effective | `reason` |
|---|---|---|
| unset / empty | `stored` | `null` |
| `stored` | `stored` | `null` |
| `memory` | `memory` | `null` |
| anything else | `memory` | a sentence that quotes the raw value (the contract checks it contains it) |

Never a runtime setting, never on the settings page: a switch on a web page would let an
admin turn storage back on for a service that promised not to store.

At start, log one line: in memory mode at WARNING,
`[BOOT] PROCESSING_MODE=memory — documents are NOT stored: …`; if `reason` is set, log it
at WARNING too.

## 2. What memory mode switches off

| | stored | memory |
|---|---|---|
| `/api/v1/documents` and **everything under it** | as before | **404** `{"detail":"Document storage is disabled (PROCESSING_MODE=memory): documents are not stored and cannot be listed or opened. Recognise with POST /api/v1/recognize."}` — answered **before the body is read and before auth** (a middleware/filter, not a check inside the handler: a handler-level check runs after the framework has already parsed — and possibly spooled to disk — the upload; that an anonymous caller learns storage is off is not a secret worth a file on disk) |
| background worker | runs | **not started** (no stale-job reset, no drain loop). The recognition runtime is still built and warmed up |
| seed documents | as configured | **never** seeded |
| `POST /recognize`, `GET /recognitions` | available | available |
| auth, users, API keys, settings, status, logs | as before | as before |

The database/store still exists in memory mode — users, API keys and settings live
there. It just never receives a document row or artifact.

## 3. `POST /api/v1/recognize`

Guard: **API key, or a signed-in user with role operator or above** (`require_api_or_operator`,
same as uploading a document). Add it to the route-guard table every port already keeps
(AUTH.md §9) together with `GET /recognitions` → `require_api_or_viewer`.

Query: `images` — boolean, default false (`1`/`true` accepted). When true the response
carries the rectified canvas as base64 PNG.

### 3.1 Reading the body — never through the framework's upload machinery

Read the raw request stream yourself into **one buffer you own**, with a hard cap. The
framework's file-upload helpers spool to a temporary file on disk past a threshold —
that is a copy of a passport on disk before your code runs:

| stack | spools to disk | so |
|---|---|---|
| Python / Starlette | `UploadFile` (SpooledTemporaryFile, 1 MB) | read `request.stream()` |
| Go / net/http | `r.ParseMultipartForm(n)` / `r.FormFile` (temp files past `n`) | `io.ReadAll(io.LimitReader(r.Body, cap+1))`, then `multipart.NewReader(bytes.NewReader(buf), boundary)` and read the part into memory — `Reader.NextPart` does not spool, `ReadForm` does |
| .NET / ASP.NET Core | `IFormFile`, `Request.Form`, `ReadFormAsync` (past 64 KB) | read `Request.Body` into a byte array (or `ArrayPool` rented with the size known); parse multipart with `MultipartReader` over a `MemoryStream` of that array |
| Kotlin / Spring on Tomcat | `MultipartFile`, `@RequestPart`, the multipart resolver (`file-size-threshold`) | take `HttpServletRequest.getInputStream()` in a plain handler, **and** make sure the multipart resolver is not invoked for this path (`spring.servlet.multipart.enabled=false` or a filter-level exclusion) |

Cap: `MAX_UPLOAD_MB` MiB + 64 KiB for multipart framing. If `Content-Length` already
exceeds it, refuse before reading; while reading, refuse as soon as it is exceeded (and
zero what was read).

Two body shapes, both required:

* **raw**: the body is the image, `Content-Type: image/*` (or anything that is not
  multipart). This is what the web UI sends.
* **multipart/form-data**: the first part named `file`, or the first part with a
  `filename`. Copy only that part into its own buffer, zero the body buffer. **The
  filename is never read into anything that outlives the parse, never logged.**

### 3.2 Validation order and refusals

| check, in order | status | `detail` | outcome |
|---|---|---|---|
| no / bad credentials | 401 (403 for a viewer) | the usual auth answers | **not journalled** (auth runs first) |
| body over the cap | 413 | `Image exceeds the upload limit` | `rejected` |
| multipart without a boundary | 400 | `Malformed multipart body` | `rejected` |
| multipart without an image part | 400 | `No image in the multipart body — send it as the 'file' part` | `rejected` |
| empty image | 400 | `Empty upload` | `rejected` |
| image over `MAX_UPLOAD_MB` | 413 | `File exceeds the N MB limit` | `rejected` |
| PDF magic | 415 | `PDF is not supported — send a JPEG, PNG, WEBP, BMP or TIFF image` | `rejected` |
| not a recognised image magic | 415 | `Unsupported file type — expected an image` | `rejected` |
| decoder returns nothing | 422 | `The image could not be decoded — it may be corrupt` | `rejected` |
| no free pipeline within `RECOGNIZE_WAIT_SEC` (default 30) | 503 + `Retry-After: 5` | `All recognition pipelines are busy — retry shortly` | `busy` |
| runtime not built yet | 503 + `Retry-After: 10` | `The recognition runtime is not ready yet` | `busy` |
| over `job_timeout_sec` + `RECOGNIZE_WAIT_SEC` | 504 | `Recognition took too long` | `error` |
| anything else | 500 | `Recognition failed` | `error` — log the **exception type name only**, never its message (messages carry paths and values) |

The magic-byte sniffing is the same function the document upload already uses.

### 3.3 Recognition

Use the existing pool lease (the same one the worker uses; it serialises access to a
pipeline). Run on an executor/thread separate from the background worker's, so a request
does not queue behind a stored document. Inside the lease: recognise, build the payload
with the **same transform** the stored path uses (so the JSON is identical to a stored
document's detail), and, if `images`, encode the canvas to PNG — then, **still inside the
lease**, zero and release the pipeline's per-document state (§5).

### 3.4 Response — 200

`Content-Type: application/json`, **`Cache-Control: no-store`**. Keys:

```
stored: false
doc_type, doc_type_base, doc_type_era, recognised, doc_conf, device
canvas: {width, height, is_fallback?, image_png_base64? (only with ?images=1)}
coord_space, coord_space_note, boxes, fields, ocr, address
quality            (without DocConf — it is doc_conf)
timings, processing_ms
```

No `id`, no URL, nothing that points at a stored thing — there is none. `outcome` is
`recognised` when `recognised` is true, else `not_recognised`.

## 4. The journal and the log line

Every request that passed auth leaves exactly one journal entry and exactly one log line.

**Journal**: in process memory only — a bounded ring of `RECOGNITION_JOURNAL_MAX` (default
5000) entries, newest first, lost on restart, never written to the database or a file.
An entry is **exactly**:

```json
{"id": 12, "at": "2026-10-19T05:02:53.239750Z", "outcome": "recognised",
 "status_code": 200, "processing_ms": 639, "stored": false}
```

`id` is a per-process counter. Nothing else — not the filename, size, type, device, the
caller's identity or address.

`GET /api/v1/recognitions?limit=100&offset=0` (`limit` 1..1000) →
`{"items": [...], "total": N, "counts": {"recognised": a, "not_recognised": b,
"rejected": c, "busy": d, "error": e, "total": N}}`.

**Log line**, INFO, exactly:

```
[API] recognize: outcome=<outcome> status=<code> ms=<ms>
```

The contract parses it: after `[API] recognize:` there are exactly three tokens.
Library/pipeline stdout during the recognition is discarded, not forwarded (the Python
library prints field values). The HTTP access log line (method, path, status) is fine —
the path has no document data.

## 5. Explicit wiping

"Cleared" means overwritten with zeros, then released — not merely dropped:

| what | where |
|---|---|
| request body buffer | handler, `finally` |
| image bytes (the part copy) | after decode, and again in `finally` |
| decoded pixels (BGR, RGB, any resized copy you own) | `finally` of the decode/recognise step |
| the pipeline's per-document state: result object, MRZ zone, canvas, crops — every array reachable from it | inside the lease, `finally`, then set to null |
| PNG bytes of the canvas | handler `finally`, after the response body was built |

What must **not** be wiped: model weights, templates, centroids loaded before the first
document — anything the next document needs. Python's `wipe_arrays` only touches
writable arrays reachable from the per-document attributes; a port wipes the equivalent.

**Strings cannot be wiped** in any of the four languages (immutable in Python, Go, .NET,
the JVM): the recognised text and the serialised JSON are released and their memory
reused, not overwritten. Say so in the port's README instead of promising more. Go: avoid
`string(buf)` copies of the image; .NET: `CryptographicOperations.ZeroMemory` /
`Array.Clear`, and an `ArrayPool` rent must be returned with `clearArray: true`;
Kotlin: `java.util.Arrays.fill(buf, 0)`, `Mat.setTo(Scalar.all(0.0))` before `release()`.

After the handler: a garbage-collection hint is acceptable (Python calls `gc.collect()`),
not required.

## 6. Reporting the mode

* `GET /api/v1/auth/config` gains `"processing_mode": "stored" | "memory"` (public, like the
  rest of that response). The shared web bundle reads it: in memory mode it shows the
  journal and a "Recognise document" dialog instead of the document list, and never calls
  `/documents`.
* `GET /api/v1/status` gains `"processing": {"mode", "reason", "recognitions": counts}`.

## 7. Unit tests each port must carry (the black box cannot see these)

1. **No spooling**: a multipart body larger than the framework's spool threshold (≥ 2 MB)
   goes through `/recognize` while the temp-file API is made to fail or the temp directory
   is watched — no file appears. (Python: patches `SpooledTemporaryFile`,
   `NamedTemporaryFile` and `TemporaryFile` to raise.)
2. **Buffers zeroed**: with the recogniser replaced by a fake that keeps references to the
   buffers it got, after the request every one of them is all zeros — image bytes,
   pixels, PNG.
3. **Pipeline state scrubbed**: a fake pipeline whose result holds arrays; after
   `recognise_image` (and after it throws) the arrays are zero and the attribute is null,
   and the "weights" array is untouched.
4. **Memory mode short-circuits**: an oversized `POST /documents` in memory mode is
   answered 404 without the body being consumed.
5. The route-guard table includes both new routes.
6. Mode resolution table of §1, including the mistyped value.

Each must fail when the corresponding line of production code is removed — check by
deleting it once (the Python suite was mutation-checked 8/8).

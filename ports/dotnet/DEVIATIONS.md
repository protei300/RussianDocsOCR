# ports/dotnet — deviations

Numbered, so a reviewer can ask "which deviation is this?" and get an answer. `D-01`..`D-13`
are defined in [`../go/DEVIATIONS.md`](../go/DEVIATIONS.md) and apply to every port; this file
records **how each lands in .NET** and adds the ones that are .NET-specific (`N-01`..).

A deviation is not a licence to improvise. Each one below was either forced by the platform or
is an improvement deliberate enough to be written down — and if a future port meets a decision
that is in none of these files, that is a gap in the *design*, not a problem with that language.

---

## Shared deviations, as they land here

| # | Shared rule | In .NET |
|---|---|---|
| **D-01** | `viewmodel` lives on the library side, not with the service | `RussianDocs.DocumentProcessing/ViewModel/`. The conformance CLI needs it and must not depend on an HTTP service. |
| **D-02** | Go returns `(T, error)`; .NET and Kotlin throw | Exceptions, with the **same seven-kind taxonomy** (`ErrorKind`). The invariant that survives: one failing call per statement, and the lines *without* error handling appear in the same relative order as in Go. A C# file is shorter by exactly the check blocks. |
| **D-03** | The library's own `warmup()` cannot report failure | `PipelineRuntime.Warm` calls the ordinary path and lets the exception surface, where the reference swallows it into a `print`. |
| **D-04** | Kotlin methods are camelCase | n/a. |
| **D-05** | `Math.exp` on the JVM promotes `Float` to `Double` | n/a — .NET has `MathF`, used throughout. The rule it encodes (never widen float32) still applies and is followed. |
| **D-06** | An unknown `model.json` tag is an error, not a fall-through | `Loader`'s three switches throw naming the tag. The reference falls into `None` and produces a null dereference three stages later. |
| **D-07** | OpenCV must be **headless** | **Lands differently, and this is the interesting one.** gocv links highgui unconditionally, so the Go port compiles OpenCV with `WITH_GTK=OFF`. OpenCvSharp's native package is *also* not headless — `ldd` wants GTK 3, Pango, Cairo, ATK, X11, FreeType — but rebuilding it would mean rebuilding OpenCvSharp's extern wrapper too, so this port **installs GTK instead**. Go pays in build time; .NET pays in image size (2.31 GB against 902 MB). See `build/Dockerfile`. |
| **D-08** | Build the rotation matrix by hand | `Imaging/Geometry`. gocv cannot express a fractional centre; OpenCvSharp *can* (`Point2f`), but the hand-built matrix is kept so all ports compute the identical value, and it is verified to 1.6e-14. |
| **D-09** | Ship four MinGW DLLs beside the Windows binary | n/a — NuGet places the native assets under `runtimes/`, and the loader finds them. There is no System32 shadowing problem. |
| **D-10** | `model.json` must have no BOM | Applies unchanged. `Set-Content -Encoding utf8` in PowerShell 5.1 adds one and breaks parsing. |
| **D-11** | `OCRCyrillic` and `OCRLatin` are one type with a `script` field | `Modules/OcrEngine`. They share no state and override nothing, so there is nothing to break, and two copies of the field lists in four languages is four places to drift. |
| **D-12** | The CUDA provider may overwrite signal handlers | **Not reproduced here.** That is a Go-runtime issue (`yalue/onnxruntime_go#140`); .NET's `IHostApplicationLifetime` handles SIGTERM through the runtime, and the GPU container shuts down cleanly. Worth re-checking if graceful shutdown ever misbehaves under `--gpus`. |
| **D-13** | Report the providers actually **obtained**, not the ones advertised | `PipelineRuntime.Init` adds `CUDAExecutionProvider` to the published list only after a session has really built on it. |

---

## N-01 — One NuGet package for both devices, one build for both images

`Microsoft.ML.OnnxRuntime.Gpu` contains the CPU execution provider as well, so a single published
output runs on either host. Referencing both packages would put two copies of the native runtime
in the output and let the loader pick.

This is where .NET differs *usefully* from Go: Go links one specific `libonnxruntime`, so the Go
port needs two builds. Here the two Docker targets differ **only in the base image**.

A host without CUDA is not an error: the attempt loop falls back and reports the provider it got.

## N-02 — A custom date format cannot express the record format

The shared record format is UTC with up to nine fractional digits and a trailing `Z`. .NET cannot
write that:

- a `DateTime` tick is 100 ns, so **seven** `F`s is the maximum and nine throws;
- `T` and `Z` must be **quoted**, or they are read as format specifiers and throw.

Both are `FormatException`, not a misformat. The symptom was every record failing to persist
while the in-memory index looked perfectly fine — a service that worked right up until it
restarted. Seven digits is a *subset* of the format, so what this port writes still parses on the
Python and Go sides. `NullableUtcConverter.Pattern` is the one place that spells a timestamp, and
`Format(DateTime?)` is public because a `[JsonConverter]` attribute does not reach a value inside
a `Dictionary<string, object>` — a projection that formats its own timestamps would silently use
a different spelling.

The JVM has the same class of trap (`SimpleDateFormat`/`DateTimeFormatter` letter collisions), so
this is carried into `ports/java/PLAN.md`.

## N-03 — Body reads are async and waited on

Kestrel sets `AllowSynchronousIO = false`, so `StreamReader.ReadToEnd` on a request body throws
`InvalidOperationException`. Caught by the surrounding handler, that became a 400, so every PIN
login answered "expected a pin" and the login page merely said the PIN was wrong.

The handlers stay **synchronous** to match the Go port's shape, and the three that read a body
call the async API and wait on the completed task. The bodies are a few dozen bytes; the
alternative is a second, async copy of the whole guard chain.

## N-04 — `CachePrivateFileResult` instead of setting a header at the call site

`Results.File` writes the response as it executes, so a `Cache-Control` assigned afterwards never
reaches the client. The symptom would be a stale canvas after a reprocess, because the image URL
does not change. A four-line wrapper sets the header first.

## N-05 — `Pipeline.Results` is aliased at every use site

`Microsoft.AspNetCore.Http.Results` is in scope throughout the service project, so an unqualified
`Results` silently means the wrong type. `using PipelineResults = …` at the top of
`Ml/PipelineRuntime.cs`. Trivial, and exactly the kind of thing that costs an hour once.

## N-06 — `InvariantGlobalization=true` is a correctness setting

`double.ToString()` is culture-sensitive: a machine whose default culture is ru-RU writes `0,904`
where the wire contract requires `0.904` — every float in the view model, silently, and only on
that machine. An English CI runner never reproduces it. This makes the whole process invariant so
the bug cannot exist. Formatting boundaries still pass `CultureInfo.InvariantCulture` explicitly,
because a reader should not have to know the property is set. It also drops libicu from the
image, which is a side effect and not the reason.

## N-07 — Central Package Management and a committed lock file, but never a NuGet.config

Versions live in `Directory.Packages.props`; `packages.lock.json` is committed. A lock file
records resolved **versions and content hashes**, not feed URLs, which is precisely the
distinction that went wrong with `web/package-lock.json` — that file named a private mirror 132
times. No `NuGet.config` is committed, so restore follows whatever source the machine has
configured and nothing internal is recorded.

Corollary found by the first Docker build: `obj/` **does** record the feed URL, and
`.dockerignore` patterns match the whole path, so a bare `obj/` matches only a top-level
directory. Both `ports/dotnet/**/obj/` and `**/project.assets.json` are excluded.

## N-08 — No self-contained publish, no AOT

Both would work; neither is worth it. The runtime images already carry the ASP.NET runtime, so
self-contained duplicates it — and AOT is incompatible with the reflection OpenCvSharp's
marshalling uses. Framework-dependent publish is also what lets both Docker targets share one
build (N-01).

## N-09 — The conformance CLI pins threads; the service does not

`rdocs-conform` sets `IntraOpNumThreads = 1`. ONNX Runtime's CPU reductions split across threads,
so a different thread count legitimately shifts a result by ~1e-6 — inside the float tolerance,
but enough to flip an argmax on near-equal values, which is an exact-match failure with no float
anywhere near it. The service passes 0 (leave it to ORT) because it has no goldens to match and
wants the throughput.

## N-10 — `HttpClient` must be told not to proxy loopback

Not a port decision but a deployment and testing fact worth recording where somebody will find
it: `HttpClient` honours the system proxy even for `127.0.0.1`. Behind a corporate proxy every
request in the contract test left the machine and came back as a 403 with an HTML body —
twenty-three failures whose only visible symptom was `'<' is an invalid start of a value`.
`UseProxy = false` is set explicitly rather than trusting `NO_PROXY` to be present in whatever
environment runs the tests. The service's own `--healthcheck` probe talks to loopback for the
same reason.

## N-11 — Authentication: what differs from the reference, and why

The contract is [`../AUTH.md`](../AUTH.md) and `conformance/auth_contract.py` passes in full.
The differences, all deliberate:

- **Argon2 cannot be missing.** The reference has a fourth `AUTH_MODE` row for `argon2-cffi` not
  being installed; here `Konscious.Security.Cryptography.Argon2` (1.3.1, MIT) is a compile-time
  dependency, so the row does not exist. Konscious computes digests but does not parse PHC
  strings; `Auth/Passwords.cs` does, and both argon2-cffi vectors verify in the unit tests.
  Only `v=19` is accepted — Konscious implements nothing else, and every hash any of the four
  services writes says `v=19`.
- **Password length counts code points explicitly.** The rule is served to the UI as `.{8,}`, but
  a .NET `Regex` counts UTF-16 units, so four emoji would pass here and fail in Python. The server
  checks the length rule with a code-point count (in runs between line feeds, which is what `.`
  means); the other three rules are plain BMP character classes and use the pattern itself.
- **A malformed request body is a 400 with a sentence, not pydantic's 422 list.** AUTH.md §7
  allows either. The length limits (pin 1–32, username 1–64, password 1–256, display name ≤ 128)
  are the reference's. A non-integer `{id}` on `/users/…` **is** a 422 in pydantic's shape, as in
  the reference, because FastAPI declares it `int`.
- **Two error kinds were added**, `Forbidden` (403) and `TooManyAttempts` (429), and
  `Unauthorized` now passes its message through instead of a fixed "Not authenticated": the
  guards' texts are part of the contract. `NotFound` is unchanged, so the users routes write their
  `No such user` / `User accounts are disabled (AUTH_MODE=pin)` 404s directly.
- **Account timestamps are truncated to microseconds.** A `DateTime` would write seven fractional
  digits (N-02); Python writes six, and `users.json` is read by the reference too.
- **The mode is resolved once at startup**, not per call — the environment tier is immutable here,
  so it is the same answer computed fewer times.

## N-12 — What of the reference after 4.6.1 is carried, and what is not (2026-10-07)

Brought up to the reference of 2026-10-07 (models-v10 goldens, commit `601bac1`) in one pass; what
the reference gained since 4.5.0, and where this port stands:

**Carried, and graded by conformance:**

- the **`documents` stage** (decision #142): `Modules/DocumentDetector` finds the documents in the
  frame resized to `img_size`, `Recognizer.FindDocuments` scales the boxes back to the input image,
  `Recognizer.DocumentCrop` cuts the largest one plus 3 % of its longer side **at full resolution**,
  and only then is it shrunk to `img_size` (`prepare`). No document found, or no `DocDetect` in the
  weight set (models-v8 or older) → the whole frame, with a warning on stderr, as the
  reference does. `new Recognizer(detectDocuments: false)` is the same as `detect_documents=False`.
- the **gap guard** on the word split (`WORDS_MAX_GAP`, `LINE_MIN_INK`) with its flags
  (`Results.WordsFallback`, `Results.WordsNoInk`);
- the **whole-line date re-read** (`_reread_dates_whole`, `Results.DatesReadWhole`);
- the **quote read as a letter** next to the day (`Dates.DropQuoteLetters`, issue #23) and
  `strip_day_quote`;
- the **top-to-bottom order of the kept detections** and the **`_ru`/`_en` duplicate of one line**
  (`_paired_duplicate_indices`) in `SplitWords`;
- the **MRZ length self-check** (`MrzZone`: the line re-read from a wider crop). The 4.5.0 commit says
  it landed in all three ports, but this port had none of it; the models-v10 detector cuts the MRZ
  boxes narrower, and both external-passport cases lost the first and last characters of both lines
  until it was added.

**Carried since (2026-10-08, decision #144 - ports and Python ship together):** the STS and the pieces
that came with it, see N-13; `process_frame` / `pair_sides` (`Recognizer.RunFrame`, `Recognizer.PairSides`);
the **borders-first fallback on `NONE`** and the **early return on `NONE`** (`Recognizer.ReadDocument`:
for an unrecognised type the border detector runs once, the cut-out is classified again, and a type that is
still `NONE` ends the run with the stages `documents`, `prepare`, `doctype.label`, `rotate` and the
view model, as the reference does - checked against the reference on three frames with no document).
The fallback branch that RECOVERS a type (and, for an internal passport, redoes the two-page stitch) is
ported from the code but was not seen working: none of the frames tried was recovered by the reference.

**Not carried, on purpose:**

- The address page (`INTPASSPORTADDR`), and with it the `address_lines` key of the stage `quads` and
  `Results.AddressLineQuads` (always empty).

**A trap found on the way (2026-10-07):** `expand_quad` runs on a float32 array in the reference
(`order_points` and the page registrar's `page_quads` both return float32), so NumPy rounds to float32
at every step — the centroid, `quad - centre`, the product with the margin scale, the sum. This port
did it in double (`Geometry.ExpandQuad`); the corners then land up to ~6e-5 px elsewhere, and
`warpPerspective`, which quantises its interpolation weights to 1/32 px, moved 393 pixels of the
`BIRTHCERT_1998` canvas (max 8 grey levels) — a digest mismatch on `borders.canvas` and, through the
field detector, a 0.005-0.007 step in two word-box confidences of `Issue_date`. It looked like
OpenCV-version noise (R-02) and was not: `Geometry.ExpandQuadF32` is now used in `FixPerspective` and in
both page-expansion branches of `RegisterPages`, and the case runs clean. (The Kotlin port found the
cause first.) The older `ExpandQuad` is kept for reference only.

## N-13 - The vehicle registration certificate (STS), 2026-10-08

Ported in one pass with the reference (commits `7e421957`, `a7b12e81`, `4d8c2535`, `a1153af1`,
`dd37ea31`, `91eb0a9`): the field lists and engine routing (`OcrOptions` for `sts`, plus the `dlback`
branch that was missing), the VIN fix (`OcrCorrections.CheckVin`), the engine that follows the form year
(`OcrOptions.EngineByYear`), the wider crop of the special marks (`Recognizer.ReadMarginRows`), the glue
of words torn by a line break and the leasing flag (`Pipeline/StsMarks.cs`, `Ocr.GlueTorn`,
`Results.Leasing`, the conformance stage `leasing`, emitted for an STS back only), the straightening of
the card by its printed blank (`Recognizer.RegisterCard`, `PageRegistrar.CardSkew` / `CardFill` /
`WarpMatrix` / `PageMatrix`), and `RunFrame` / `PairSides`.

What the port had to get exactly, found by measurement:

- **The float32 chain of the reference.** The page quads are float32 (`PageQuads`, `QuadInImage`), and
  `native_scale` runs on them in float32 (`NativeScale`): the scale is a float32 value, and it multiplies
  the page matrix. A double value is up to 6e-8 off, which over a 1000 px page is ~6e-5 px and moves
  hundreds of pixels of a `warpPerspective` (weights quantised to 1/32 px). NumPy 1.x promotes
  `float32 * python float` to float64 and NumPy 2 does not - the reference runs 2.x, so the chain is float32.
- **LSD is bound.** OpenCvSharp4 has `LineSegmentDetector.Create/Detect` (it is a class of its own, not a
  `Cv2` method - the first port of the page registration looked for it in the wrong place). `Lsd` now runs
  the reference's detector, with its defaults, for `line_refine` and `line_dewarp`; the Canny+HoughLinesP
  stand-in is gone.
- **`line_dewarp` measures its cells the reference's way** (`LineDewarp.CellTilts`: the whole region
  rotated once per angle, a shared cumulative sum), not by calling the band routine per cell. The number
  of cells with a peak decides whether the page is dewarped at all, and on `STSBACK_1996` the shortcut
  dewarped a page the reference leaves alone.
- **The half-scale resizes pass `fx=fy=0.5` with an empty size**, as `cv2.resize(..., None, fx=0.5, ...)`
  does; passing the rounded size makes OpenCV recompute the scale, which differs for an odd side.
- A canvas rebuilt from the printed blank is not deskewed (`_deskew` in the reference returns it as it
  is). This holds for the passport canvas too; the port used to run the deskew on it, which was a no-op
  below the 2 degree threshold.

**Conformance, 2026-10-08, cpu, models-v10 goldens, 14 cases:** ten clean (the eight earlier ones and
`STS_1996`, `STSBACK_1996` - every stage), the two internal passports differ by the declared D-03, and two
STS cases are not clean: `STSBACK_2019` differs in one box edge (`fields.bbox[9]` y2, 846 against 845; one
pixel of its canvas differs by one grey level) and `STS_2019` differs from `borders.canvas` on (the canvas
is cut from the template here and from the Borders quad in the golden). Measured cause of both:

- **The goldens were recorded on Windows; the registration of a card is not reproducible on Linux.**
  OpenCV's SIFT keeps at most `nfeatures` keypoints with `std::nth_element`, and the order that leaves
  the keypoints in depends on the standard library that was built into OpenCV. The template keypoints of
  `sts_2019_new.jpg` come out in a different ORDER (the same 4972 points) in the Windows wheel and on
  Linux. MAGSAC samples correspondences by index, so it ends in a different homography when the
  consensus is not overwhelming: 1670 against 1760 coarse inliers on `STS_2019`, which flips the card-skew
  decision (0.0076 against 0.0118, the threshold is 0.01) and the canvas is cut from the template
  instead of kept from the Borders quad. Checked by running the REFERENCE's own `PageRegistrar` on Linux
  (opencv-python-headless 4.12.0 in the grading container): it gives this port's numbers, to the last
  digit, on both 2019 cases - coarse 1760 inliers, `H` 1.0989005002674674 ..., and for `STSBACK_2019`
  `H[0][0]` 0.7994596886243793 against 0.7994596889952418 recorded on Windows (that 4e-10 is the one
  moved pixel of its canvas and the one box edge). So the port agrees with the reference on the same
  platform; the golden of `STS_2019` cannot be reproduced on Linux by any port, Python included. It needs
  either a golden recorded where the grading runs or a declared deviation.

## N-14 - The way back from the canvas to the photo (PR #19), 2026-10-08

`document_processing/geometry.py` is ported as `Maps/Maps.cs` (`ScaleMap`, `OffsetMap`, `QuarterTurnsMap`,
`HomographyMap`, `BendMap` = `VerticalRemap`, `UnknownMap`, `ChainMap`, `PiecesMap`; the `...Map` suffix because
`Geometry` and `Homography` are taken in this library) and `Maps/Stitched.cs` (`stitched_geometry`, the
placements of `stitch_pages`). Every stage that changes the image records its map as the reference does:
the crop of the document and the resize (`ReadDocument`), the quarter turns (`QuarterTurnsMap`), the page warps
and the stitch (`Geometry.FixPerspective` -> `DocDetector.PredictTransform`), the page registration and the card
(`Recognizer.RegisterPages` / `RegisterCard`: the page's matrix, then the straightening homography and the
bend map, then the stitch), the deskew (`DocDeskewer.DeskewWithMap`), the field frame (`Field.Origin`, the size
as cut, which a series/number turn does not change) and the margin re-cut of the special marks. The public
result: `Results.Geometry`, `ToInput`, `FieldQuads`, `WordQuads`, `AddressLineQuads`; the stage `quads` is
emitted after `words.<Field>.bbox` and before the OCR, null for both keys when a stage left the way back
unknown (once per run, not per box).

Two traps the tests pin: the chain is read from the output to the input (a map built "forward" sends every box
off the photo), and the warps' half-pixel convention (`HomographyMap`). The fallback on `NONE` writes its maps
the way the reference does (the first border map and the second quarter turn, or the redone spread and the
turn of its canvas).

What differs from the reference, with the reason: the internal passport is fed to the field detector as the
stitched canvas, not page by page (see N-12 / D-03), so its quadrilaterals follow the field boxes of the same
two cases and differ with them by the same 1-3 px (the stage `quads` is in the scope of D-03, D-07 and D-08).
A spread that is not registered is deskewed as one canvas here and page by page in the reference, so its map
differs too; the registered one (the default) is not deskewed in either.

Verified three ways: the maps against what OpenCV does to a marked pixel (quarter turns, perspective, bend,
stitch), the quadrilaterals of a real photo against the canvas patch (a driving licence, an internal passport,
an STS front and a back: the residual shift of the best match stays under half a pixel, and the test goes red
with the chain reversed or the half pixel dropped), and the conformance stage `quads` with the strict 1e-3.

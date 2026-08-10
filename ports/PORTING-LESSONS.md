# Lessons from the Go port

For whoever writes `ports/dotnet`, `ports/java` or `ports/cpp` — including a future Claude with
no memory of how the Go port went.

**This file is not a design document.** The design is normative and lives elsewhere; read those
first, in this order:

| document | answers |
|---|---|
| [`conformance/spec/`](../conformance/spec/) | what the contract IS (stages, tolerances, view model) |
| [`CONVENTIONS.md`](CONVENTIONS.md) | how to write a port — the rules |
| [`MAPPING.md`](MAPPING.md) | which file becomes which, and switch-case order |
| [`ports/go/DEVIATIONS.md`](go/DEVIATIONS.md) | where divergence is licensed (D-01…D-13) |
| [`ports/go/ARCHITECTURE.md`](go/ARCHITECTURE.md) | how the finished Go implementation works |

What follows is what those four could not tell you, because it was only learned by finishing.

---

## 1. The defect that kept recurring: a check that could not fail

Five separate times, something reported success while verifying nothing. This was by far the most
expensive class of bug in the project — more than every numeric trap combined — because a green
check does not merely fail to find a defect, it actively argues that there is none.

| where | how it was vacuous | found by |
|---|---|---|
| `doctype.label` golden | emitted `meta_results['DocTypeAngles']`, a key that does not exist → golden was `null`, and `null == null` passed | Go produced a real value and there was nothing to compare it to |
| `STAGES_IMPLEMENTED` | omitted `borders.segments`, so the REFERENCE skipped a stage in its own self-check — 26 stages reported PASS where 44 were gradeable | adding a new stage and noticing the count |
| tolerance `abs <= 1e-3` | `0.904 - 0.903` is `1.0000000000000009e-3` in binary, so a bare `>` rejected exactly the difference the tolerance existed to admit | a real failure, not review |
| GPU coordinate allowance | positional rows (`fields.bbox[11][2]`) have no leaf name, so nothing matched `COORDINATE_LEAVES` and a pixel coordinate was graded at 1e-2 | the first GPU run, failing by exactly 1.0 px |
| models image `verify` stage | not on the default build path, so `docker build` skipped the assertion and reported success | running the stage by hand and seeing it had never run |

**The rule that follows.** Every check must be shown to fail. Not argued to be capable of
failing — *shown*, once, by breaking something on purpose and watching it go red. It costs two
minutes. `conformance/runner` already institutionalises part of this with `vacuous_reason()`, which
warns when a golden is `null` or empty; extend that habit to anything you add.

Concretely, for a new port:

- After the first stage passes, corrupt one constant in your port and confirm the runner names
  **that** stage.
- Any build-time assertion you write: break its threshold once, watch the build stop, restore.
- A stage whose golden is `null`, `[]` or `{}` is not evidence of agreement. Treat it as a bug in
  the instrumentation until proven otherwise.

## 2. Measure, don't estimate — every estimate I made was wrong

Not "imprecise". Wrong in a way that would have misled a decision:

| claim | reality | error |
|---|---|---|
| "Go starts 29× faster" | 1.5–1.9× | the 15.5 s baseline was a GPU build, dominated by CUDA session creation |
| "OCR on GPU may be a Go advantage" | **13.7× slower** | the spike measured a dynamic-width penalty RATIO over an amortising loop, not GPU vs CPU |
| "CPU image ≈ 390 MB" | 902 MB | the estimate omitted OpenCV (93 MB) and `samples/` (53 MB) |
| "GPU image ≈ 3.0 GB, Python 7.72 GB" | 6.73 vs 7.72 GB — 13 % | CUDA and cuDNN dominate both; the language does not reach them |
| "the redundant colour conversion explains the slow upload" | removing it changed nothing | plausible mechanism, wrong one |

Two of those were mine and stated confidently in documentation, where they would have outlived the
session. **Write the number you measured, name the tool you measured it with, and say when a
comparison is not apples-to-apples** — the GPU/CPU image comparison is the clearest case: 902 MB
against 7.72 GB is real but compares a CPU-only image with a GPU-capable one, and saying so is the
difference between an honest figure and a sales figure.

Related: **one measurement is not a measurement.** Per-document means for `python-cpu` differed by
17 % between two rounds while the medians differed by 2 ms. Report medians for latency, run at
least two rounds, and keep the raw numbers.

## 3. Where the residual risk actually lives (this reorders your milestones)

Measured in the spike and confirmed by the port: **the whole path from file bytes to model output
is bit-identical between languages, for free.** JPEG decode, resize, letterbox, tensor layout,
inference — 21 of 21 artifacts, ~14 million pixels, zero difference. The heavy lifting is the same
C++ (ONNX Runtime, OpenCV) in every language, and the host language does not reach it.

Therefore **100 % of the convergence risk sits in the code you write by hand**: NMS, CTC decoding,
contour selection, geometry, sorting, rounding, crop bounds.

Consequence for planning: the milestones that feel foundational (decode, resize, tensors) are the
safe ones. The dangerous ones are borders/deskew, text fields/word splitting, and OCR
post-processing. Do not spend your care budget on the early ones.

## 4. What the conformance harness cannot see

It is a strong instrument and it did localise real defects to a stage within minutes. But it runs
**one document per process** through a CLI, and that shape is blind to an entire class:

- **Resource leaks.** `runtime.Recognise` never closed its `Results`. Every intermediate stayed
  alive — 12.7 MB per document, unbounded, 663 MB → 6932 MB over 460 documents. The CLI defers
  `Close` and exits, so conformance passed throughout. **Only a soak with memory sampled BETWEEN
  rounds finds this**, and a leak is indistinguishable from an allocator plateau in a single
  measurement — they differ only in the SHAPE of the curve across rounds.
- **The service wire contract.** Seven mismatches with the shared SPA shipped while conformance was
  green, including a settings page that reported success and stored nothing.
- **Everything in the container.** Eight build/run defects, all invisible to the harness.
- **Anything on the upload/HTTP path.** The 3× upload cost is not a stage.

Plan a soak (`build.ps1 -Soak` equivalent) and a container run as first-class milestones, not as
afterthoughts.

## 5. The contract is owned by the shared frontend, not by your port

`web/` is reused **unchanged** by every port. That makes the SPA a normative consumer, and it reads
~60 named fields. Every mismatch found was a 200 OK that no page could use:

`items` vs `entries` · `limit` vs `n` · missing `note` · missing `warning` · missing `schema` ·
a flat settings body where the page sends `{values: …}` · `page_size` clamped where the reference
answers 422.

None were visible from the server side. What eventually worked: `contract_test.go`, asserting
response **keys** against lists transcribed from the named `.vue`/`.ts` sources. Write that test in
your port on day one of the service milestone, not after the user finds the third mismatch.

And when in doubt about an error body, **capture it from the running reference** rather than
writing what looks right. The 422 body for a bad query parameter is pydantic's list-of-objects
shape, not the hand-written `{"detail": "<string>"}` used everywhere else — an inconsistency in the
reference that a port must reproduce, and one nobody would guess.

## 6. Ordering that worked

`M7` — the first milestone where the port is comparable end-to-end — was seventh of eleven, on
purpose. Everything before it was verified against a partial golden via `--upto <stage>`. This was
the single best structural decision: no milestone ended in "probably fine".

Suggested order, unchanged for the next port:

```
M1 imaging + tensors + config     ← validates decode & resize, the two silent killers
M2 inference + loader + doctype   ← validates the ONNX binding and the whole loader design
M3 quality group + parallel prim  ← rehearse the concurrency shape on cheap models
M4 borders + deskew               ← tolerance policy gets its real test here
M5 text fields + word splitting   ← sorting and crop bounds bite here
M6 OCR                            ← per-word exact comparison
M7 view model + recognize         ← first end-to-end parity, CPU
M8 GPU + soak                     ← the leak hunt belongs here
M9 service (errors first!)        ← + the SPA contract test
M10 Docker (build it, don't write it)
M10.5 ARCHITECTURE.md
```

`api/errors` genuinely must come first inside M9: status codes, the error body shape and the
401-not-403 rule are inherited by every handler after it.

## 7. Effort, honestly

The Go port took roughly one working day per two milestones once the harness existed, and the
harness itself (M0) was a day. The parts that consumed time were not the parts that looked hard:

- Writing the pipeline: fast, because CONVENTIONS had already decided everything.
- Numeric traps: **cheap when pre-written, expensive when discovered.** The traps recorded in
  CONVENTIONS §6 before any Go code existed cost nothing; `FloorDiv`, `convexHull` orientation and
  banker's rounding were found by failures and cost hours each.
- Docker: eight fixes, most of a day, entirely in places the file itself had flagged as unverified.
- The leak: found in an afternoon only because a benchmark happened to sample memory.

**The next port should be materially faster** — the design is settled, the goldens exist, the
harness is proven, and the traps are written down. What will NOT be faster: the platform-specific
half (packaging, native libraries, container, threading defaults). Budget that separately.

---

## 8. .NET specifics — verified facts, not expectations

Checked against nuget.org while preparing the port:

- **`Microsoft.ML.OnnxRuntime` (CPU) exists at 1.21.0 only** in the 1.21 line — no 1.21.1. The GPU
  package has 1.21.0/1.21.1/1.21.2. This mirrors exactly what bit the Go Dockerfile, where the
  linux-x64 CPU tarball is absent from the 1.21.1 release. **Pin 1.21.0 for both.** The reference
  runs 1.21.1; the spike measured 1.21.1 against 1.28.0 on CPU and got bit-identical tensors, so a
  patch step inside 1.21 cannot move numbers that two minors did not.
- **`OpenCvSharp4` 4.13.0 exists**, the same OpenCV the Go port is verified against, with official
  native runtimes for `win-x64` and `linux-x64`. **This removes the from-source OpenCV build
  entirely** — the longest and most fragile stage of the Go image.
- **OpenCvSharp versions are `<opencv>.<date>`**, and there are 18 distinct `4.13.0.*` builds.
  Pinning "4.13.0" pins nothing; pin the full four-part version.
- Package sizes, for a size estimate that is not a guess: ORT CPU 120 MB, ORT GPU (linux) 87 MB,
  OpenCvSharp4 1.7 MB + native linux-x64 49 MB / win 40 MB.
- The binding is **first-party**, so there is no `ORT_API_VERSION` handshake to get wrong and no
  cgo. .NET is the lowest-risk of the three remaining ports for that reason alone.

Traps to expect, from the conventions rather than from experience:

- `Math.Round` defaults to **ToEven**, which is what NumPy does — so .NET is *right by default*
  where Go needed `math.RoundToEven`. Do not "fix" it.
- `List.Sort`/`Array.Sort` are **unstable**; LINQ `OrderBy` is stable. Every sort in the pipeline
  must be stable (CONVENTIONS §6.3). Prefer `OrderBy`.
- `Mat[Rect]` / `new Mat(mat, rect)` **throws** on out-of-range, where Python's slice clamps. The
  `ClampedCrop` equivalent is mandatory and must be the only crop path.
- CPython's float `//` is not `Math.Floor(a/b)` — port `tensor.FloorDiv` faithfully (CONVENTIONS
  §6.10b). This one is a one-pixel canvas difference that moves every downstream box.
- `Task`/`async` replaces goroutines, but the parallel group must keep its shape: one launch per
  member in source order, one `Task.WhenAll`, positional collection (CONVENTIONS §3).
- Errors: D-02 licenses exceptions instead of `(T, error)`. Keep the **line order** of the
  reference; the C# file is shorter by exactly the error checks.

Open questions for the .NET port are collected in [`dotnet/PLAN.md`](dotnet/PLAN.md).

### 8b. What the .NET port then actually hit (written after finishing it)

The list above was written before any .NET code existed. It aged well — `RoundToEven`, stable
sorts, `ClampedCrop` and `FloorDiv` all landed as predicted and cost nothing, which is the whole
argument for writing a trap list in advance. What it **missed** is more useful:

- **`OpenCvSharp4.official.runtime.linux-x64` is not headless.** The claim above that the NuGet
  package "removes the from-source OpenCV build entirely" is true for the *build* and false for
  the *image*: `ldd` on the shipped `libOpenCvSharpExtern.so` wants GTK 3, Pango, Cairo, ATK, X11
  and FreeType — sixteen sonames. So D-07 applies to every port; it only changes *currency*. Go
  pays in build time (compiles OpenCV with `WITH_GTK=OFF`), .NET pays in image size (2.31 GB
  against Go's 902 MB). The failure mode is worth memorising because it names nothing useful:
  `The type initializer for 'OpenCvSharp.Internal.NativeMethods' threw an exception`. **Put an
  `ldd`-resolves assertion in every runtime stage** — it converts that into a build failure
  listing exactly what is missing.
- **A custom date format cannot express the shared record format.** Two independent traps in one
  nine-character string, and both *throw* rather than misformat: `T` and `Z` must be quoted, and
  **seven** fractional digits is the maximum because a tick is 100 ns. The symptom was every
  record failing to persist while the in-memory index looked perfectly fine — a service that
  worked right up until it restarted. **The JVM shares this class of trap.**
- **The web framework may refuse synchronous IO.** Kestrel sets `AllowSynchronousIO = false`, so
  reading a request body with `ReadToEnd` throws — and a surrounding catch turned that into a 400,
  so every PIN login reported a wrong PIN. Any framework may do this; check before writing three
  handlers that read a body.
- **`.dockerignore` patterns match the whole path.** A bare `obj/` matches only a top-level
  directory, so the host's Windows `project.assets.json` landed on top of the container's and
  `dotnet publish` failed with "Package OpenCvSharp4 was not found" — a message about a missing
  package whose cause was packaging. Use `**/`-prefixed patterns. Every language has a build
  directory that must not travel, and in .NET's case it also records the package feed URL.
- **An HTTP client may proxy loopback.** `HttpClient` honours the system proxy for `127.0.0.1`, so
  behind a corporate proxy the contract test produced twenty-three failures whose only symptom was
  `'<' is an invalid start of a value`. Disable the proxy explicitly in tests rather than trusting
  `NO_PROXY` to be set.
- **A `[JsonConverter]` attribute does not reach a value inside a dictionary of `object`.** Any
  projection that builds a map by hand must format its own timestamps, or it silently uses a
  different spelling than the record format. The general form of this — *attribute-driven
  serialisation stops at the boundary of a dynamic container* — applies to Jackson too.
- **`Results.File`-style helpers write the response as they execute**, so a header assigned
  afterwards never ships. Symptom: a stale image after a reprocess, because the URL is unchanged.
- **`Microsoft.AspNetCore.Http.Results` collides with the library's `Pipeline.Results`.** Alias
  one. Trivial, and exactly the sort of thing that costs an hour once.

Effort, for calibration against §7: the whole .NET port — M1 through M10, library and service and
Docker — took appreciably less than the Go port, and almost all of the saving was in M1–M4
(no toolchain spike, no cgo, no OpenCV build) and M10. **M9, the service, took about the same**,
because re-typing 7 000 lines of wire contract is re-typing 7 000 lines of wire contract, and the
defects it surfaced were all platform-specific rather than logic-specific — which is the design
working as intended.

## 9. Kotlin/JVM specifics — flagged now, decided later

- Use the **official `org.opencv` Java bindings, not JavaCV.** They mirror the C++/Python API
  method-for-method, which is the whole point; JavaCV's abstractions would force per-call design
  decisions and break mechanical re-typing.
- **D-05 is a real hazard**: `Math.exp` promotes `Float` to `Double`. Every accumulation must stay
  in float32, elementwise: `exp(x.toDouble()).toFloat()`. It shows up first at
  `borders.protomask`.
- `sortedBy` is stable — good. `Math.rint` is round-half-to-even — also good.
- `submat` throws like .NET; same `ClampedCrop` requirement.
- The JVM has no equivalent of "ship the DLLs beside the binary" (D-09), but it has its own version
  of the problem: native library extraction from the JAR, and `-Djava.library.path`.
- Coroutines replace goroutines cleanly (`coroutineScope` + `awaitAll`), but the **per-session
  mutex is still mandatory** — it is a property of ONNX Runtime's CUDA provider, not of Go.
- **HTTP framework: Spring Boot** (decided — most popular on the JVM). It brings DI and annotations
  that Go's `net/http` and .NET's minimal API do not have, so the rule already applied to FastAPI
  applies here too: *framework magic must not leak into logic.* Keep validation, filter parsing and
  key checking as explicit functions and controllers as thin adapters, or that logic ends up living
  nowhere and the next port reinvents it.
- **Verified, and the opposite of what I expected:** the JVM ORT artifacts publish 1.21.1 for BOTH
  CPU and GPU, so Kotlin can pin the reference's exact version — the CPU/GPU gap that forced Go and
  .NET onto 1.21.0 does not exist here.
- **Also verified, and worse than expected:** there is no official `org.opencv` on Maven Central at
  all, and every third-party republisher stops at 4.9/4.10. 4.13.0 is mandatory (contour
  approximation changed in 4.8), so the JVM port must build OpenCV itself with `-DBUILD_JAVA=ON` —
  inheriting the long Docker stage that .NET escapes entirely.

---

## 10. Five things I would tell myself before starting

1. **Write the trap list before the code.** Everything in CONVENTIONS §6 that was written in
   advance cost nothing to honour; everything discovered later cost hours.
2. **Break every check once.** If you have not seen it red, you do not have a check.
3. **Soak before declaring victory.** Conformance passing and the service being correct are
   different claims, and the difference is measured in gigabytes.
4. **Do not estimate anything you can measure in ten minutes** — and if you must estimate, label it
   as an estimate in the same sentence.
5. **The container is part of the work.** "Written but not built" is worth roughly nothing; the
   first real build cost eight fixes, and every one was in a place already flagged as risky.

# The Kotlin/JVM port — plan

Status: **PREPARATION ONLY. No code written.** Planned deliberately *before* the .NET port starts,
so that [`../CONVENTIONS.md`](../CONVENTIONS.md) absorbs the constraints of both languages in
one pass instead of being amended twice. Implementation follows .NET.

The directory is `ports/java/` rather than `ports/kotlin/` because the artefact is a JVM build and
the layout is Gradle's; the language inside it is Kotlin.

Read first: [`../PORTING-LESSONS.md`](../PORTING-LESSONS.md) — §9 is specifically about this port —
then the four Go documents.

---

## 0. Decisions inherited, not re-opened

| | |
|---|---|
| Dependencies | **public only.** Nothing internal, and nothing committed may name an internal host |
| Naming | neutral — no company name in an open project |
| Tolerance | floats `abs <= 1e-3`, discrete outputs exact |
| Milestone order | identical to Go and .NET (M0…M10.5) |

Decisions taken by the user, 2026-08-05:

| | |
|---|---|
| HTTP framework | **Spring Boot.** The instruction was "take the most popular", and on the JVM that is Spring Boot by a wide margin — Ktor leads only within the Kotlin-first subset. For an open library, recognisability to contributors outweighs symmetry with the other ports. **Cost, accepted knowingly:** Spring's DI and annotation magic does not map onto Go's `net/http` or .NET's minimal API, so the existing rule for FastAPI applies unchanged — *framework magic must not leak into logic.* Validation, filter parsing and key checking stay explicit functions; controllers are thin adapters. Otherwise that logic ends up nowhere and the fourth port has to reinvent it |
| Build tool | **Gradle**, Kotlin DSL, version catalogue |
| JDK | **21 LTS** — verified present on this machine (Oracle 21.0.1) |
| OpenCV | **build it ourselves and document it** (see §1: there is no alternative) |

## 1. The one real decision this port still owns: the OpenCV binding

**Use the official `org.opencv` Java bindings. Not JavaCV, not Bytedeco's wrappers.**

The reason is the project's governing constraint rather than taste: the official bindings mirror the
C++/Python API method-for-method, so `Imgproc.warpPerspective(...)` is a re-typing of
`cv2.warpPerspective(...)`. JavaCV introduces its own abstractions, and each one forces a per-call
design decision — which is precisely what "mechanical re-typing with zero design decisions" forbids.

### Checked, not assumed: there is no usable published artifact

Searched Maven Central during preparation. **No official `org.opencv:opencv` exists at all.** All
24 `a:opencv` hits are third-party republishers, and the newest relevant ones stop well short of the
version this project needs:

| artifact | version | verdict |
|---|---|---|
| `org.openpnp:opencv` | 4.9.0-0 | the common republisher — **4.9, not 4.13** |
| `org.bytedeco:opencv` | 4.10.0 | JavaCV's; excluded by design above |
| `us.ihmc:opencv`, `io.github.kamilszewc:opencv`, … | 4.7–4.10 | third-party, older still |

4.13.0 is **not a preference**: contour approximation changed in 4.8, and the goldens encode 4.13
behaviour. A port on 4.9 would fail `borders.segments` for a reason nothing in the code explains.

**Therefore: build OpenCV with `-DBUILD_JAVA=ON` ourselves, and document it** (the user's decision,
and the only route left). Practical consequences, all inherited from the Go port which already does
this:

- The Docker image gets the long OpenCV stage back — the one thing .NET escapes. Budget it.
- Build it **headless** (D-07) and with **bundled codecs** (`BUILD_JPEG/PNG/TIFF/WEBP=ON`), for the
  same two reasons the Go image needs both: highgui links unconditionally, and Debian and Ubuntu
  disagree on codec SONAMEs.
- Local development needs the same JAR plus the native library. Document the exact cmake line and
  keep it identical to the Go Dockerfile's, so the two cannot drift apart in OpenCV configuration
  while both claim 4.13.0.
- Anyone compiling the port needs a C++ toolchain and cmake. State that in the README rather than
  letting them discover it.

Still to establish at M1: whether the Java binding exposes the ~20 functions the pipeline needs, in
particular `Imgproc.boxPoints`, `Imgproc.convexHull` **with its orientation argument**,
`Imgproc.findContours` and `Core.rotate`. The Go spike found `BoxPoints` present and
`RotatedRectangleIntersection` absent; the JVM inventory may differ.

## 2. ONNX Runtime — the low-risk half

`com.microsoft.onnxruntime:onnxruntime` is **first-party** and published to Maven Central, with a
separate `onnxruntime_gpu` artifact. So, as in .NET: no API-version handshake, no JNI to write.

**Checked: the asymmetry that bit Go and .NET does NOT apply here.** Both
`com.microsoft.onnxruntime:onnxruntime` and `:onnxruntime_gpu` publish **1.21.0 and 1.21.1**, so this
port can pin **1.21.1** — the exact version the reference runs, which neither of the other two ports
can do. Worth noting as the one place where the JVM has it easiest, and worth having checked rather
than assumed: the expectation going in was the opposite.

## 3. Layout

Two Gradle modules mirroring the Go module's two binaries and the .NET solution's two runnable
projects. The library/service split is **enforced**: `:service` depends on `:docproc`, never the
reverse (this is rule 10 of `service/ml/runtime.py`, and it is what keeps the service testable
without 215 MB of models).

```
ports/java/
  settings.gradle.kts
  build.gradle.kts                  ← versions in one place: the lockfile analogue
  gradle/libs.versions.toml         ← Gradle version catalogue
  docproc/src/main/kotlin/…/        ← config imaging tensors preprocess postprocess
                                      inference models modules pipeline viewmodel
  service/src/main/kotlin/…/        ← config model store repo auth logging
                                      settingsschema runtime api worker seed
  conform/                          ← CLI: info / recognize / probe
  servicehost/                      ← the web service
```

`viewmodel` sits in `docproc` — D-01, same as Go and .NET, because the conformance CLI needs it and
must not depend on the HTTP service.

## 4. Milestones

Identical order and exit criteria to .NET; only the per-language risks differ.

| # | milestone | JVM-specific first-time risks |
|---|---|---|
| **M0** | CLI skeleton, `ports.json` entry | a Gradle `application` start script is a shell wrapper — the `cmd` in `ports.json` must point at something directly executable on both platforms; a fat JAR plus `java -jar` may be simpler than the wrapper |
| **M1** | imaging, tensors, config | **settle the OpenCV binding and version** (§1); `submat` throws where Python clamps → `ClampedCrop`; port `FloorDiv` faithfully; native library loading from the JAR |
| **M2** | inference, loader, doctype | `<U64` label decode from `centers.npz`; `OnnxTensor` shape/type mapping; outputs matched by NAME |
| **M3** | quality group | `coroutineScope` + `awaitAll`, positional collection, launches in source order; the per-session mutex is **still mandatory** — it is a property of ORT's CUDA provider, not of Go |
| **M4** | borders, deskew | **D-05: `Math.exp` promotes `Float` to `Double`.** Every accumulation stays float32, elementwise: `exp(x.toDouble()).toFloat()`. It surfaces first at `borders.protomask`. `convexHull` orientation argument |
| **M5** | text fields, word splitting | `sortedBy` is stable — good; `Math.rint` is half-to-even — good. Crop bounds again |
| **M6** | OCR | Kotlin `String` is UTF-16: the alphabet must be indexed by **code point**, not by `Char`. `-inf` masking. `check_ddmmyyyy`'s third outcome |
| **M7** | view model, `recognize` | JSON: pick one serializer and write **every** name by hand (`@SerialName`). kotlinx.serialization omits nulls by default — must be configured `explicitNulls = true`, or the 14-key contract breaks silently |
| **M8** | GPU + **soak** | `Mat.release()` is manual and the JVM finalizer will not save you; a leak here looks like the Go one. Sample memory **between** rounds, and read RSS from the OS — the JVM heap will not show native `Mat` allocations at all, exactly as `runtime.MemStats` did not in Go |
| **M9** | service | Ktor or Spring Boot — see the open question. `errors` first, then the SPA contract test |
| **M10** | Docker, built and run | JRE base image; if OpenCV must be self-built, this stage inherits the Go image's longest step |
| **M10.5** | `ARCHITECTURE.md` | same structure, with coroutines and explicit `release()` in place of goroutines and `defer` |

## 5. Traps that are the JVM's own

Not in `CONVENTIONS.md` yet because Go has no equivalent. **These belong in it** — the whole reason
this plan is being written before the .NET port starts.

- **`Float`→`Double` promotion** (already D-05, but it deserves a §6 entry: it is a numeric
  determinism trap, not merely a deviation).
- **Locale.** `String.format` and `toString()` on floats are locale-sensitive on the JVM just as in
  .NET. One `ru-RU` default locale turns every decimal point into a comma throughout the JSON.
  Force `Locale.ROOT` at every formatting boundary. **Same trap in two of the three remaining
  languages, and absent from Go — which is exactly how it would have been missed.**
- **Native memory is invisible to the JVM.** Heap metrics, GC logs and profilers will all show a
  healthy process while `Mat` allocations grow without bound. Measure RSS from the OS.
- **`Mat` has no `IDisposable` equivalent.** Kotlin's `use` works only on `Closeable`, which
  `org.opencv.core.Mat` is not. Either wrap it in a `Closeable` type of our own — the analogue of Go's
  `imaging.Image` — or accept manual `release()` everywhere. **Wrap it.** The Go leak happened with a
  wrapper in place; without one it is close to certain.

### 5b. Added after finishing the .NET port

Each of these cost real time in .NET and has a JVM analogue. Written here so the JVM port pays for
them at design time instead.

- **Date formatting is a whole trap family, not one trap.** .NET lost a debugging session to a
  nine-character format string with **two** independent faults: unquoted `T`/`Z` were read as
  format specifiers, and nine fractional digits exceeded the platform's maximum of seven. Both
  *throw*, and the symptom was every record silently failing to persist while the in-memory index
  looked perfectly healthy — a service that works right up until it restarts. The JVM has the same
  class of collision (`DateTimeFormatter` letter patterns, and `SimpleDateFormat` is worse).
  **Write one formatting method, comment the reason, and assert a round-trip in a unit test on day
  one.** The record format is UTC, up to nine fractional digits, trailing `Z`, and fewer digits is
  a valid subset — so if the JVM can only manage some other precision, that is fine as long as it
  is *fewer*.
- **Assume the web framework refuses synchronous IO until proved otherwise.** Kestrel does. Reading
  a request body the obvious way threw, a surrounding catch turned it into a 400, and every PIN
  login reported an incorrect PIN. Spring's reactive stack has the same shape of restriction; check
  before writing the three handlers that read a body.
- **Put an `ldd`-resolves assertion in every Docker runtime stage.** D-07 is not "Go needs headless
  OpenCV"; it is "**every** port needs headless OpenCV, and only the currency differs". Since this
  port builds OpenCV itself (`-DBUILD_JAVA=ON`), pass `-DWITH_GTK=OFF -DWITH_QT=OFF` and get Go's
  bargain rather than .NET's. The failure without it is a native type-initialiser error that names
  neither OpenCV nor a library.
- **Docker-ignore patterns match the whole path.** `build/` and `.gradle/` need `**/` prefixes, or
  the host's build output lands on top of the container's — .NET's equivalent produced "Package X
  was not found", a message about a missing package whose cause was packaging.
- **Disable the proxy explicitly in tests that call loopback.** .NET's `HttpClient` proxies
  `127.0.0.1` by default; behind a corporate proxy that was twenty-three failures whose only
  symptom was `'<' is an invalid start of a value`. The JVM's `HttpClient` behaves the same way when
  `java.net.useSystemProxies` is set. Do not rely on `NO_PROXY` being present.
- **Attribute/annotation-driven serialisation stops at a dynamic container.** A `[JsonConverter]`
  on a property does not apply to the same value inside a `Map<String, Object>`, and Jackson's
  `@JsonFormat` behaves identically. Any hand-built projection must format its own timestamps or it
  silently ships a different spelling than the record format.
- **A response helper may commit the response as it executes**, so a header set afterwards never
  ships. Symptom: a stale image after a reprocess, because the URL is unchanged.
- **Write the SPA contract test with the very first endpoint.** .NET's 23 tests run in 200 ms
  because they never construct the recognition runtime — which also proves, for free, that the
  service is usable while models load. In Spring, resist the pull towards a full context: what is
  being tested is the wire shape, not the framework.

## 6. Resolved; and what remains genuinely unknown

All four questions are answered — see the decisions table at the top. What is still open is not a
decision but a measurement, and it belongs to M1:

- **Which of the ~20 OpenCV functions the Java binding actually exposes**, and whether any needs
  reimplementing the way `BoxPoints` might have. Establish by probe, as the Go spike did (T8), before
  writing pipeline code against them.
- **Gradle is not installed on this machine.** The wrapper (`gradlew`) downloads its own distribution,
  which means one more thing that must traverse the corporate proxy — the same class of friction as
  `GOPROXY` and npm. Expect to configure it; do not commit any internal host to do so.

## 7. Not in scope

Same exclusions as Go and .NET: `INTPASSPORTADDR` (no anonymised sample ⇒ no golden ⇒ ungradeable),
`ocrGpuBatch`, the OpenVINO and CoreML runtimes, the SQL store, and every item under
`MAPPING.md` §"Not ported".

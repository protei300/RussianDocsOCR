# ports/java — deviations

Numbered, so a reviewer can ask "which deviation is this?" and get an answer. `D-01`..`D-13` are
defined in [`../go/DEVIATIONS.md`](../go/DEVIATIONS.md) and apply to every port; this file records
**how each lands on the JVM** and adds the ones that are this platform's own (`J-01`..).

A deviation is not a licence to improvise. Each was either forced by the platform or is an improvement
deliberate enough to be written down — and if a future port meets a decision that is in none of these
files, that is a gap in the *design*, not a problem with that language.

---

## Shared deviations, as they land here

| # | Shared rule | On the JVM |
|---|---|---|
| **D-01** | `viewmodel` lives on the library side | `:docproc`, module `viewmodel`. The conformance CLI needs it and must not depend on an HTTP service |
| **D-02** | Go returns `(T, error)`; .NET and Kotlin throw | Exceptions, with the **same seven-kind taxonomy**. The invariant that survives: one failing call per statement, and the lines *without* error handling appear in the same relative order as in Go |
| **D-03** | The library's own `warmup()` cannot report failure | The runtime layer calls the ordinary path and lets the exception surface |
| **D-04** | Kotlin methods are camelCase | Applies here, mechanically |
| **D-05** | `Math.exp` promotes `Float` to `Double` | **Real and load-bearing.** Every accumulation stays float32, elementwise. Where OpenCV can do it, `Core.exp` on a `CV_32F` Mat keeps the depth — verified in the spike — which is how `borders.protomask` avoids the promotion entirely |
| **D-06** | An unknown `model.json` tag is an error, not a fall-through | The loader's three `when` blocks throw naming the tag. The reference falls into `None` and produces a null dereference three stages later |
| **D-07** | OpenCV must be **headless** | Applies, and this port gets the *cheap* version of it: since it builds OpenCV anyway, `-DWITH_GTK=OFF -DWITH_QT=OFF` costs nothing and the configure summary confirms `GUI: NONE`. Go pays the same way; .NET pays in image size instead, because its prebuilt native package is not headless |
| **D-08** | Build the rotation matrix by hand | **DOES NOT APPLY.** Verified in the spike: `Imgproc.getRotationMatrix2D` takes a double-valued `Point`, so a fractional centre is expressible and the real function can be called. Go needs the hand-built matrix only because gocv takes an integer `image.Point`; a naive integer version was measured to shift the deskew variance array by 3.8e-3 relative, above the 1e-3 policy. One fewer deviation, and one fewer place to get wrong |
| **D-09** | Ship four MinGW DLLs beside the Windows binary | **Applies in a worse form, and gets a better fix.** See `J-01` |
| **D-10** | `model.json` must have no BOM | Applies unchanged |
| **D-11** | `OCRCyrillic` and `OCRLatin` are one type with a `script` field | Applies. They share no state and override nothing |
| **D-12** | The CUDA provider may overwrite signal handlers | Not expected to apply — that is a Go-runtime issue, and the JVM installs its own handlers early. To be confirmed at M8 rather than assumed |
| **D-13** | Report the providers actually **obtained**, not the ones advertised | Applies. `BuildInfo.availableProviders` is explicitly documented as the advertised list; what bound is reported separately after a session builds |

---

## J-01 — `System.load` in dependency order, because "beside the binary" is not available

A MinGW-built `libopencv_java4130.dll` imports `libstdc++-6.dll` and `libgcc_s_seh-1.dll`. Windows
resolves those **by base name**: the executable's directory first, then `System32`, then `PATH`. On a
machine where `System32` carries its own copies — as the development machine does, for all four — the
foreign copies win and the load fails with `The specified procedure could not be found`, which names
nothing useful.

D-09's fix was to ship the DLLs beside the binary. Here the "binary" is `java.exe`, and copying files
into a JDK installation is not something a deployment can do.

**The fix is to load them explicitly, in dependency order, before the JNI library** — each
`System.load` registers the module under its base name, so the later import binds to the copy we chose.
Verified with the copies removed from the JDK again. `RDOCS_TOOLCHAIN_BIN` selects the directory;
ignored on non-Windows.

## J-02 — ONNX Runtime versus the JDK's own C runtime: diagnose, do not attempt to fix

The same shadowing rule, in the one place code cannot reach: `C:\Program Files\Java\jdk-21\bin` ships
`msvcp140.dll` at 14.31.31103.0 (VS 2022 17.1) while the system has 14.50.35719.0, and `jvm.dll` loads
the JDK's copy **before `main()` runs**. ONNX Runtime, built against a newer runtime, then fails
`DllMain` with Windows error 1114.

It survives every plausible remedy — the CPU artefact fails identically to the GPU one, loading from
disk rather than `%TEMP%` changes nothing, a scrubbed `PATH` changes nothing, every import is present
and current, and the very same file loads fine in a .NET process. Only replacing the JDK's three CRT
files makes it work.

So `NativeLibraries.loadOnnxRuntime` catches the `UnsatisfiedLinkError` and rethrows it with the
diagnosis and the fix. **A warning up front was tried first and removed**: it fired whenever the JDK
bundled a CRT at all, which is nearly every JDK including the working ones, so it cried wolf on a
healthy setup. A check that fires when it is wrong trains the reader to ignore the time it is right.

Recommendation: use a Temurin, Zulu or Corretto build of 21. Cannot occur on Linux or in Docker.

## J-03 — A fat jar, not Gradle's start scripts

`conformance/ports.json` names ONE executable that the checker runs with `exec`. Gradle's `application`
plugin produces a shell wrapper plus a `.bat`, which needs a shell and differs per platform — awkward
to name in a single `cmd` entry, and a source of "works on my machine". `java -jar` on a self-contained
jar is the same invocation everywhere.

Cost, accepted: the fat jar is 534 MB, because the GPU ONNX Runtime artefact is 468 MB of it. It is
build output and is not committed.

## J-04 — kotlinx.serialization with `explicitNulls`, and every wire name written by hand

The view model's contract is fourteen keys with nulls **present**, and kotlinx.serialization omits
nulls by default — so the default configuration silently drops keys the SPA reads. `explicitNulls` must
be on wherever the view model is emitted.

Every `@SerialName` is written by hand, for the reason every wire name in every port is: four languages
have four default naming policies, and `stages_implemented` arriving as `stagesImplemented` is a key the
checker simply does not find.

## J-05 — The GPU artefact only, so one build serves both devices

`onnxruntime_gpu` contains the CPU provider as well, so `:conform` and `:service` depend on it alone.
Depending on both would put two copies of the native runtime on the classpath and let the loader pick.

Unlike .NET — where one NuGet package carries both — the JVM publishes them separately, so this is a
choice rather than a default. A host without CUDA is not an error: the attempt loop falls back and
reports the provider it actually got.

## J-06 — `dnn` is not in the OpenCV module list

Neither the Go nor the .NET port calls OpenCV's `NMSBoxes`: both implement NMS by hand, because the
reference's suppression has a specific stable-argsort order and a specific tie-break (on equal
confidence, keep the LARGEST original index) that OpenCV reproduces in neither. `dnn` is one of the
largest modules, and dropping it shrinks every image.

Checked rather than inherited — `PORTING-LESSONS.md` claimed "gocv has NMSBoxes", which was true and
irrelevant.

## J-07 — `WITH_LAPACK=OFF` when building OpenCV

Not an optimisation. On a machine with MSYS2's OpenBLAS installed, OpenCV's LAPACK detection finds it
and then demands a Fortran compiler: `No CMAKE_Fortran_COMPILER could be found`, and the configure
fails for a reason that has nothing to do with anything this port uses. Nothing in the module set needs
LAPACK.

## J-08 — `OrtEnvironment.getVersion()` is an instance method

Trivial, and recorded because it is exactly the asymmetry that gets "fixed" by hardcoding a literal:
Python and .NET expose the ORT version statically, the JVM binding does not. Likewise
`OrtProvider.getName()` — not the enum's Kotlin `name` — yields the canonical
`CPUExecutionProvider`/`CUDAExecutionProvider` spellings the other three ports and
`conformance/spec/cli.md` use. The enum constants are `CPU` and `CUDA`, and reporting those would have
this port describe the same machine in a different vocabulary than its siblings.

## J-09 — Kotlin NESTS block comments, so a path with a `/` before a `*` breaks the file

`/**` inside a KDoc block opens a NESTED comment in Kotlin, and so does the two-character sequence in
`service/repositories/*`. The file then fails to parse at its LAST line with `Syntax error: Unclosed
comment` and no hint about where the real problem is. C#, Go and C++ do not nest, so this trap belongs
to this port alone — and it is easy to hit, because a doc comment naming a source path with a wildcard
is exactly what the other three ports' comments contain.

Paths in KDoc are therefore written without a trailing wildcard: "the reference's
`service/repositories` package".

## J-10 — `System.out` is not UTF-8 on Windows

`System.out` encodes with the console codepage, so every Cyrillic character in a log line or a
`recognize` payload became `?`. Both the conformance CLI and `logging/LogRing.kt` construct their own
`PrintStream(FileOutputStream(FileDescriptor.out), true, "UTF-8")` instead.

This is not the same as `file.encoding`, which has defaulted to UTF-8 since JDK 18: the console stream
is separate and still follows the codepage. The .NET port needed no equivalent, and Go writes bytes.

## J-11 — Logging is hand-written, not SLF4J/Logback

The ring buffer behind `GET /logs` is the deliverable, an appender wrapping it would be more
configuration than code, and the two-sink rule — stdout at the configured level, the ring at EVERY
level — stays visible in one file instead of spread across a `logback.xml`. Same reasoning as the
hand-rolled JWT, and the same choice the Go and .NET ports made.

Spring's own startup lines still go through Logback and look different, which is why
`application.properties` turns `logging.level.root` down to WARN: two log formats interleaved in one
terminal is worse than one quiet framework.

## J-12 — Spring MVC carries requests and nothing else

No Spring Security, no `@ConfigurationProperties`, no Jackson for the wire format, no
`@RequestParam(defaultValue=…)` for validated parameters, no `@ResponseStatus` on exceptions. Each is
load-bearing rather than taste:

- `@RequestParam Int` answers a Spring-shaped 400 where the contract requires a pydantic-shaped 422 with
  `detail` as a LIST (`api/ApiErrors.kt`), so handlers read the raw string through `QueryParams`.
- Jackson's naming policy differs from the ~60 wire names the shared SPA reads; kotlinx-serialization
  with hand-written `@SerialName` is the same discipline as the other three ports. Jackson stays on the
  classpath only for Spring's own plumbing.
- A filter chain would move the 401-versus-403 decision away from the single file that documents it.
- `@ConfigurationProperties` would hide each default away from the field it belongs to, and the Go port
  reads `Settings.load` line for line.

Handlers are therefore `(HttpServletRequest) -> ResponseEntity<*>`, and `ApiRoutes.kt` is a table of
mappings with one `guard` call each.

## J-13 — Extension functions instead of C#'s `partial class`

Kotlin has no partial classes, and the API surface is split across three files in the Go and .NET ports
(`router` / `documents` / `misc`) so the three can be read side by side. `ApiDocuments.kt` and
`ApiMisc.kt` are therefore extension functions on `ApiServer`, whose members they reach are `internal`.

## J-14 — A Spring Boot fat jar's code source is a NESTED url

`javaClass.protectionDomain.codeSource.location` inside a `bootJar` is
`jar:nested:/…/rdocs-service.jar/!BOOT-INF/classes/`, and `File(uri)` answers
`IllegalArgumentException: URI is not hierarchical`. `ModelPaths.root()` evaluated that candidate
eagerly in a list literal, so the exception escaped before the working-directory candidate was tried and
the service reported **"URI is not hierarchical" as its reason for having no recognition at all**.

The conformance CLI never saw it: its fat jar is flat, so the location is a plain file. The probe now
requires the `file` protocol and is wrapped, which is the general rule — *a location probe must not be
able to throw*.

## J-15 — `SpringApplication.run` RETURNS; `app.Run()` blocks

The .NET entry point ends with `app.Run(); return 0;` because ASP.NET Core blocks there. Copied
literally, the Kotlin `main` called `exitProcess` on a value it received the moment startup FINISHED,
and killed the process it had just started. The symptom is a startup that looks entirely healthy — the
full log, "listening on :8005", Tomcat reporting itself started — followed by a clean shutdown two
seconds later with no error anywhere. Only a non-zero result exits now; Tomcat's own non-daemon thread
keeps the JVM alive, and the shutdown hook stops it.

## J-16 — J-02, amended by measurement: preloading the system CRT does NOT help

Documented above as a Windows-only deployment note, repeated here because it is the single most
expensive failure in this port and it recurs: `C:\Program Files\Java\jdk-21\bin` ships `msvcp140.dll`
at 14.31 (early 2022), the directory of `java.exe` is searched BEFORE `System32`, and `onnxruntime.dll`
therefore fails `DllMain` with Windows error 1114 for every process started by that `java`.

Measured, not assumed: preloading `System32`'s `msvcp140.dll`, `vcruntime140.dll` and
`vcruntime140_1.dll` by absolute path before ORT loads does **not** help — the JDK's copy is already in
the process before any of this code runs. What works is a JDK whose bundled runtime is current, or a
COPY of the JDK with those three files replaced, which is what this machine uses
(`D:\Grant\jdk21-crtpatched`) rather than modifying a system installation. Cannot occur on Linux or in
Docker.

## J-17 — `ExecutorService`, not coroutines, for the two parallel groups

The plan named `coroutineScope` + `awaitAll`. This port uses a fixed `ExecutorService` with `Future`s
launched in source order and collected BY INDEX instead, and the reason is what the work is: every
task blocks inside native code — ONNX Runtime or OpenCV — for tens of milliseconds. A coroutine that
blocks its thread gains nothing over a thread, `Dispatchers.IO` would add a scheduler with nothing to
schedule, and `kotlinx-coroutines` would be a dependency carried for syntax.

What must NOT change is the shape all four ports share: one launch per member in the Python source's
order, one join, one deterministic collect by index. Collecting as results arrive reorders boxes,
words and the joined field string under load — a failure of exact comparison with no float in it.

The same reasoning applies to the worker's drain loop and the abandoned-work reaper: plain threads,
because they block for seconds in native calls.

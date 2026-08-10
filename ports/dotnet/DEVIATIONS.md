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

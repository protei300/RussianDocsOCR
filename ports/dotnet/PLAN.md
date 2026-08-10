# The .NET port — plan

Status: **PREPARATION ONLY. No code written. Awaiting approval to start.**

Decisions taken by the user, 2026-08-05 — these are settled, not open:

| | |
|---|---|
| Target framework | **`net8.0`, single-target.** Chasing the newest .NET buys nothing here; the machine has SDK 8.0.101 and that is enough |
| Tolerance policy | **unchanged** — floats `abs <= 1e-3`, discrete outputs exact |
| Dependencies | **public only.** No internal company packages, and nothing committed may name an internal host or feed |
| Naming | **`RussianDocs.*`** — neutral. The project is open; the company name must not appear in it |
| Kotlin | planned **now**, together with .NET, so `CONVENTIONS.md` absorbs both languages in one pass — see [`../java/PLAN.md`](../java/PLAN.md) |

The measure of success is stated in the project plan and repeated here because it governs every
decision below: **each milestone must be a mechanical re-typing of the Go port with zero design
decisions.** If a milestone forces a decision, that decision belonged in
[`../CONVENTIONS.md`](../CONVENTIONS.md) — add it there, back-apply it to Go, and only then
continue. A decision made here and not there is a bug in the design, not progress.

Read before starting: [`../PORTING-LESSONS.md`](../PORTING-LESSONS.md), then the four Go documents
it lists.

---

## 0. Hard constraint: public dependencies only

**No internal company packages.** The library is MIT and public; a dependency on a private feed would
make it unbuildable for everyone outside, so every dependency here is public: OpenCvSharp4,
Microsoft.ML.OnnxRuntime, and the framework. This rules out the internal imaging, ONNX-hosting and RPC
stack the sibling projects use — which is a real cost, since it would have supplied pooling and hosting
for free, and it is paid deliberately.

What IS taken from `D:\ML\Faces` is style, which costs nothing and carries no dependency:

| borrowed | why |
|---|---|
| Central Package Management (`Directory.Packages.props`, `ManagePackageVersionsCentrally`) | one place for versions, which is the .NET analogue of a lockfile |
| `Directory.Build.props` for shared properties | `LangVersion`, doc generation, version stamping |
| Project naming `<Product>.<Component>`, with `.Abstractions` / `.Tests` suffixes | matches company reading habits |
| NUnit + `NUnit3TestAdapter` + `Microsoft.NET.Test.Sdk` | their test stack; xUnit would be a gratuitous difference |
| `tests.runsettings` serialising test assemblies | **directly relevant**: this port also has a process-wide singleton (the pipeline pool). Parallel test assemblies would contend on it |

## 1. Dependencies — versions verified against nuget.org

| package | version | note |
|---|---|---|
| `Microsoft.ML.OnnxRuntime` | **1.21.0** | CPU. 1.21.1 does not exist for CPU — same gap that bit the Go image |
| `Microsoft.ML.OnnxRuntime.Gpu` | **1.21.0** | keep both on one version rather than 1.21.2 for GPU |
| `OpenCvSharp4` | **4.13.0.20260627** | managed wrapper; pin all four parts — 18 distinct `4.13.0.*` builds exist |
| `OpenCvSharp4.official.runtime.linux-x64` | same | **no from-source OpenCV build needed** |
| `OpenCvSharp4.runtime.win` | same | Windows dev |
| `NUnit`, `NUnit3TestAdapter`, `Microsoft.NET.Test.Sdk` | current | tests |

OpenCV **4.13.0** matches what the Go port is verified against, so R-02-level pixel agreement
should hold without a version argument. That is the single biggest piece of luck available to this
port and it should be locked in at M1, not assumed at M6.

Sizes, for a container estimate that is not a guess: ORT CPU 120 MB, ORT GPU linux 87 MB,
OpenCvSharp4 1.7 MB + native 49 MB (linux) / 40 MB (win).

## 2. Solution layout

One solution, two runnable projects, mirroring the Go module's two binaries. The `docproc` /
`svc` separation is **enforced**, not merely intended — it is what keeps the service testable
without 215 MB of models, and it mirrors rule 10 of `service/ml/runtime.py`.

```
ports/dotnet/
  RussianDocs.sln
  Directory.Build.props
  Directory.Packages.props
  tests.runsettings
  src/
    RussianDocs.DocumentProcessing/        ← port of document_processing/ (internal/docproc)
      Config/  Imaging/  Tensors/  Preprocess/  Postprocess/  Inference/  Models/  Modules/  Pipeline/
    RussianDocs.ViewModel/                 ← D-01: view model sits on the LIBRARY side
    RussianDocs.Service/                   ← port of service/ (internal/svc)
      Config/ Model/ Store/ Repo/ Auth/ Logging/ SettingsSchema/ Runtime/ Api/ Worker/ Seed/
    RussianDocs.Conform/                   ← console: info / recognize / probe  (= rdocs-conform)
    RussianDocs.ServiceHost/               ← console: the web service           (= rdocs-service)
  tests/
    RussianDocs.DocumentProcessing.Tests/
    RussianDocs.Service.Tests/             ← incl. the SPA contract test
```

`RussianDocs.DocumentProcessing` must have **no reference** to `RussianDocs.Service`. Add an
architecture test asserting it (the .NET equivalent of Go's `depguard`), because a project
reference is added by an IDE in one click.

## 3. Milestones

Each ends with `conformance run --port dotnet --upto <stage>` green on all 7 cases. The runner
already supports a new port through `conformance/ports.json` + an executable exposing
`info` / `recognize` / `probe`; **no checker changes should be needed, and needing one is a signal
that the interface was language-specific after all.**

| # | milestone | ends when | first-time risks |
|---|---|---|---|
| **M0** | `ports.json` entry, `RussianDocs.Conform` skeleton with the three subcommands, `info` reporting `stages_implemented: []` | runner reports 7 cases SKIPPED, 0 failed | exit-code contract (0 / 2 / 3 / 1) |
| **M1** | `Imaging`, `Tensors` (+ `.npy`/`.npz` reader), `Config` | `--upto prepare` PASS 7/7 | `FloorDiv`; `Mat[Rect]` throws → `ClampedCrop`; **pin the OpenCV build here** |
| **M2** | `Inference`, `Models` loader (3 switches), `Modules/DocTypeAngles` | `--upto rotate` PASS | `<U64` label decode from `centers.npz`; output order by NAME not index |
| **M3** | quality group + the parallel primitive | `--upto quality` PASS | `Task.WhenAll` positional collection; per-session mutex |
| **M4** | borders (segmentation) + deskew | `--upto deskew.canvas` PASS | `ConvexHull(clockwise:)` — the argument is explicit in OpenCvSharp and default-different from Python; two-pass variance |
| **M5** | text fields + word splitting | `--upto words` PASS | stable sorts (use `OrderBy`); crop bounds; duplicate-field tie keeps the EARLIER detection |
| **M6** | OCR | every `ocr.<Field>.words` exact | rune- not byte-indexed alphabet (C# strings are UTF-16 → beware surrogate assumptions); `-inf` masking; `check_ddmmyyyy` has THREE outcomes |
| **M7** | view model + `recognize` | **full run PASS 7/7, zero skips** | exactly 14 keys, no `omitempty` analogue — `JsonIgnoreCondition.Never`; `timings` KEY NAMES are part of the contract |
| **M8** | GPU + **soak** | GPU profile PASS; soak 500 docs with flat memory | the leak hunt lives here — sample memory BETWEEN rounds; `Mat` is `IDisposable`, so the Go `TakeCanvas` shape becomes a `using`/ownership-transfer question |
| **M9** | service, in order `Errors → Router → handlers → Worker → Runtime → SPA` | seed cross-check 7/7 through HTTP + SPA contract test green | error body shapes captured from the reference; `--workers 1` invariant; bytes-before-row |
| **M10** | Docker, **built and run** | both images start, recognise, serve the SPA | far simpler than Go: no OpenCV build. Reuse `ports/base/Dockerfile.models` unchanged |
| **M10.5** | `ARCHITECTURE.md` | — | same structure as the Go one, with `Task`/`async` and `IDisposable` in place of goroutines and `defer` |

## 4. Things that are genuinely easier than in Go

Worth knowing so the schedule is not padded for them:

- **No cgo.** First-party ONNX binding, no `ORT_API_VERSION` handshake, no `-tags customenv`, no
  four DLLs beside the binary (D-09 does not apply).
- **No OpenCV build.** Prebuilt native runtimes for both platforms.
- `Math.Round` is already round-half-to-even, matching NumPy.
- `System.Text.Json` with explicit `[JsonPropertyName]` on every property behaves like Go's
  hand-written tags. Keep writing every name by hand — the ~60 wire names must match byte for byte.
- `IDisposable` + `using` is a better fit for `Mat` ownership than `defer`, and the analyser will
  warn about a missing dispose. This may well have caught the Go leak.

## 5. Things that are harder or genuinely new

- **The GC hides the leak differently.** Go's `Mat` wrapper leaked because nothing closed it;
  in .NET a `Mat` has a finalizer, so a leak becomes *delayed* rather than permanent, which is
  harder to see and easier to dismiss. The soak in M8 must still be run, and the criterion is the
  same: shape of the curve across rounds, not a single number.
- **`async` all the way up.** ASP.NET Core is async by default while the pipeline is blocking. The
  Go worker is a goroutine; the .NET worker should be a `BackgroundService` with the blocking call
  wrapped so it does not occupy the thread pool. Getting this wrong shows up as latency under load,
  not as a wrong answer.
- **Culture.** `double.ToString()` and `Parse` are culture-sensitive. One `ru-RU` machine and every
  float in the JSON grows a comma. Force `CultureInfo.InvariantCulture` at every boundary, or set
  `InvariantGlobalization`. Go has no equivalent hazard, so nothing in CONVENTIONS warns about it —
  **this is a new §-worthy trap and should be added to CONVENTIONS when confirmed.**
- **SDK version.** This machine has **8.0.101 only**; the sibling solution targets `net10.0`. See
  the open questions.

## 6. Resolved questions

1. **Target framework — `net8.0`, single-target.** No multi-targeting: it costs build time and buys
   reach for a library nobody consumes as a NuGet package. Revisit only if that changes.
2. **Tolerance policy — unchanged.** No need to read the sibling solution's comparison harness.
3. **NuGet — public nuget.org only, and nothing committed may name an internal host.** A user of
   the library must see no trace of our infrastructure. Local development may go through whatever
   mirror this machine is configured for, but that belongs in the developer's own environment, never
   in a committed `NuGet.config`. See §8 — the npm lockfile was already violating exactly this.
4. **Naming — `RussianDocs.*`.** Neutral, because the project is open and the company name must not
   appear in it.
5. **Kotlin — planned now**, in [`../java/PLAN.md`](../java/PLAN.md).

## 7. Related work done during preparation: an actual leak of internal infrastructure

Auditing for §3 found one, in the **committed** `web/package-lock.json`: **132 URLs naming a private
mirror**, in a public MIT repository. It was not merely embarrassing — it was broken for everyone
outside that network, who would get a bare `E401` from a host they cannot resolve. Both Dockerfiles
papered over it with a build-time `sed`, which is evidence of the problem rather than a fix.

Fixed properly: the lockfile was rewritten once, host prefix only. Versions and `integrity` hashes
are unchanged — verified programmatically by comparing every package entry with `resolved` removed —
because those hash tarball CONTENT and are independent of who served it. Then `npm ci` and
`npm run build` were re-run locally, five tarball URLs were confirmed to return 200 from the public
registry, and the frontend stage was rebuilt inside Docker with a plain `npm ci`. The `sed` is gone
from both Dockerfiles.

Remaining, and **left to the user**: `Melnikov/Text_fields/ocr_v2_reference/` names an internal git
host twice. That tree is a personal research workspace, explicitly not part of the library, and its
disposition before going fully public has not been decided — so it is reported, not edited.

## 8. What is explicitly NOT in scope

Same exclusions as the Go port, and for the same reasons: `INTPASSPORTADDR` (no anonymised sample
exists, so it has no golden and cannot be graded), `ocrGpuBatch`, the OpenVINO and CoreML runtimes,
and the SQL store. The dead Python code listed in `MAPPING.md` §"Not ported" stays unported —
copying it into a third language is the most durable way to keep dead code alive.

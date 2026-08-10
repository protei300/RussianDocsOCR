# Deviations

Numbered places where a port cannot, or should not, follow the Python reference.

**The point of this file:** a developer porting to .NET or Kotlin who hits a
decision that is in neither `CONVENTIONS.md` nor here has found a **gap in the
design**, not a language problem. The fix is to add the entry, decide once, and
back-port the decision to the ports that already exist.

Entries are permanent and never renumbered.

---

## D-01 — the view-model transform lives on the LIBRARY side

Python puts it in `service/ml/transform.py`. Every port puts it in the library
(`internal/viewmodel/` in Go).

**Why:** the conformance CLI has to produce a view model, and it must not depend on
the HTTP service — the library is milestones 1–7, the service is milestone 9. If
the transform lived in the service, no port could be graded until it had a web
server.

**Consequence:** `spec/viewmodel.md` is the contract; `transform.py` is merely the
reference implementation of it. When they disagree that is a bug report against
Python.

---

## D-02 — errors versus exceptions

Go returns `(T, error)`. C#, Kotlin and C++ throw.

This cannot be papered over, and every attempt makes all sides worse: `panic` in Go,
or `Result<T>` monads in C#/Kotlin, produce code that is idiomatic nowhere.

**The rule that preserves shape:** one fallible call per statement, checked
immediately. Go's

```go
x, err := f(a)
if err != nil { return nil, fmt.Errorf("stage fields: %w", err) }
```

becomes, at the same position, `var x = F(a);` and nothing else. The invariant is
that **non-error lines appear in the same relative order**, and the C#/Kotlin file
is shorter by exactly the error blocks. Reviewers of the .NET port should expect
that and not read it as missing logic.

Error taxonomy — exactly these seven, one-to-one with exception classes named the
same minus `Err`, plus `Exception`:

`ErrModelLoad`, `ErrUnsupportedImage`, `ErrDecodeFailed`, `ErrUnexpectedResult`,
`ErrRuntimeNotReady`, `ErrPipelineBusy`, `ErrNotImplemented`.

No `errors.Is` sentinel graphs, no wrapping deeper than one level.

**`doc_type == "NONE"` is NOT an error.** It is a normal short-circuit return with a
populated result. Throwing there breaks the "unrecognised document" path, which the
UI renders as a legitimate state.

---

## D-03 — `warmup` reports failures instead of swallowing them

Python's `Pipeline.warmup()` catches everything and `print`s it, so a failed warmup
is indistinguishable from a successful one. That is why the service calls
`process_img` directly instead of using it.

Ports return the error. Observable service behaviour is preserved (it still starts,
and the status page reports `error`), but the mechanism differs and `info` output
will differ.

---

## D-04 — Kotlin method casing

Type names are identical across all ports. Method names are PascalCase in Go and C#
and camelCase in Kotlin. Mechanical, accepted, not worth fighting.

---

## D-05 — JVM float promotion

`Math.exp` on a `Float` promotes to `Double` on the JVM. Since all tensor maths must
stay `float32` (see `CONVENTIONS.md`), the mandated form is
`exp(x.toDouble()).toFloat()` per element. Validate at the `borders.protomask`
boundary, where a sigmoid runs over 160×160×32 values.

---

## D-06 — the model loader returns an error for an unknown tag

Python's `__load_preprocess` / `__load_postprocess` use `match` with no `else`, so an
unrecognised type tag falls through and returns `None`. The nil then dereferences
three stages later, and the traceback points nowhere near the typo.

Ports return `ErrModelLoad` naming the unknown tag. This is a deliberate
improvement, not a transliteration.

Unimplemented-but-known tags (`YOLOOBBDetector`, `OCR`, `OCRFV`, the legacy
`OCRPreprocessing`) are **wired** and return `ErrNotImplemented`. Never omitted: an
omitted case reads as an oversight and gets "helpfully" added differently in each
port.

---

## D-07 — OpenCV must be built HEADLESS

*Found during the Go spike.*

gocv links `highgui` unconditionally (`window.go` exists), and MSYS2's OpenCV is
built `WITH_QT=ON`, so a service that will never draw a window drags Qt6 into the
image — hundreds of megabytes for nothing.

Build OpenCV with `-DWITH_QT=OFF -DWITH_GTK=OFF`. `highgui` then remains as a stub
that pulls nothing. This is exactly what `opencv-python-headless` does, which is
what the Python side already uses, so the two match.

Applies to every port that links OpenCV: .NET (`OpenCvSharp4`) and JVM (the official
`org.opencv` bindings) have the same exposure.

---

## D-08 — build the rotation matrix by hand

*Found during the Go spike.*

`cv2.getRotationMatrix2D` receives a **fractional** centre — `DocDeskewer` passes
`(sw/2.0, sh/2.0)`, e.g. 105.5. gocv's `GetRotationMatrix2D` takes an `image.Point`
and **cannot express** that.

Compute the 2×3 matrix directly from OpenCV's own formula:

```
alpha = scale*cos(a), beta = scale*sin(a)
[  alpha  beta   (1-alpha)*cx - beta*cy ]
[ -beta   alpha   beta*cx + (1-alpha)*cy ]
```

Verified against `cv2` to 1.6e-14. **Measured cost of the naive route:** rounding the
centre to an integer perturbed the deskew variance array by up to **3.8e-3
relative** — above the 1e-3 policy — even though it happened to pick the same angle
on all four test cases. So this is a correctness requirement, not tidiness.

Check the equivalent API in each language before assuming it accepts a float centre.

---

## D-09 — Windows: ship the shadowed runtime DLLs beside the binary

*Found during the Go spike, after four wrong hypotheses.*

`C:\Windows\System32` may contain `libstdc++-6.dll`, `libgcc_s_seh-1.dll`,
`libwinpthread-1.dll` and `zlib1.dll` — MinGW runtime DLLs placed there by unrelated
installers. Windows searches the **application directory first and System32 second,
both before PATH**, so a binary living anywhere else loads the stale System32
`libstdc++` and dies at load with `STATUS_ENTRYPOINT_NOT_FOUND` (0xC0000139), before
`main()`, printing nothing.

Copy those four files next to the executable. Nothing else needs copying — nothing
else is shadowed.

Diagnostic shortcut: if a binary fails but the identical file runs from inside
`C:\msys64\mingw64\bin`, this is why.

---

## D-10 — `model.json` must be read as BOM-free UTF-8

*Found while building the conformance harness.*

The shipped configs are UTF-8 without a BOM, and Python's `json.loads` rejects a BOM
outright ("Unexpected UTF-8 BOM"). Go's `encoding/json` fails the same way; .NET's
`JsonSerializer` tolerates it; Kotlin depends on the library.

Two obligations: never rewrite a `model.json` with a tool that adds a BOM
(PowerShell's `Set-Content -Encoding utf8` does — this cost a debugging detour), and
a port's reader should fail loudly on one rather than silently strip it, so that a
corrupted config is not mistaken for a model problem.

---

## D-11 — the two OCR engines are ONE type with a `script` field

*Decided in M6.*

The reference has two classes, `OCRCyrillic` and `OCRLatin`, in two packages. They
differ in exactly two things: which artifact key they load, and which corrections
`fix_errors` applies. Everything else — `predict`, the decode path, the batch path,
the close — is duplicated verbatim.

The port collapses them into one `OcrEngine` carrying `script`, constructed by
`NewOcrCyrillic` / `NewOcrLatin`. This is the same reasoning as D-03 (the per-class
NMS subclass folded into an `nmsMode` field): a difference expressed as data rather
than as a type, so the two variants sit side by side where they can be compared.

Why this is safe to diverge on, where most of `DEVIATIONS.md` argues the opposite:
the two classes share no state and override no behaviour, so there is no dispatch to
get wrong. Nothing is hidden — the two `fix_errors` tables are the actual difference
and they are adjacent in one function.

**Obligation for the other ports.** Do the same. Two near-identical classes in C# and
Kotlin would be two more places for the field lists to drift apart, and drift there is
silent: a missing entry in `ruNameFields` does not fail, it just stops stripping a
stray leading dot on one field of one document type.

**What must NOT be merged with it:** the routing that CHOOSES an engine
(`pipeline/ocr.go`). That lives on the pipeline side in the reference too, it carries
the SNILS parity rule, and it is where a mistake changes recognised text.

## D-12 — the CUDA provider may overwrite Go's signal handlers (UNVERIFIED)

`yalue/onnxruntime_go#140` reports that enabling the CUDA execution provider replaces
the process's signal handlers, which in Go means `signal.Notify` stops delivering.

**Consequence if true:** a container gets no graceful shutdown. `docker stop` sends
SIGTERM, the handler never fires, the ten-second grace period elapses and SIGKILL
follows — so in-flight recognition is lost and nothing explains why. On Windows this
does not arise, which is exactly why it will not be noticed during development.

**Status: not verified, and deliberately not claimed either way.** It cannot be tested
honestly on this machine — Windows has no SIGTERM, and faking one through a console
control event tests the console handler rather than the thing in the issue. Verifying it
on Windows and reporting a result would be worse than leaving it open.

**Verify in M10, in the Linux image, with this recipe:**

1. Start the GPU container with a document in flight.
2. `docker kill --signal=TERM <id>`.
3. Expect the shutdown log line within a second and exit code 0. A ten-second pause
   followed by exit 137 is the bug.

**If it reproduces**, the mitigation is a supervisor that does not depend on in-process
signal delivery: `docker run --init` (tini reaps and forwards), plus draining on a
health-check flag rather than on SIGTERM. Do NOT respond by reinstalling the handler
after session construction — the provider is initialised lazily on the first `Run()` on
some builds, so "after construction" is not reliably after the overwrite.

The port therefore does not yet install a signal handler at all. Adding one that silently
does not work would be worse than its absence being visible.

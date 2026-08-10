# Conventions

Normative for **every** port, not only Go.

The Go port is not the goal — it is the executable specification for the .NET,
Kotlin and C++ ports. Every rule here is resolved in favour of "all languages write
this the same way", even where that costs idiomatic Go. The measure of success:
a later port should require **zero design decisions**. If one does, that decision
belonged here.

Read `DEVIATIONS.md` alongside this. Anything you hit that is in neither is a gap in
the design, not a language problem.

---

## 1. Go features to avoid, and what to use instead

| Avoid | Why | Use instead |
|---|---|---|
| `panic`/`recover` as control flow | C#/Kotlin would use exceptions and the shapes diverge immediately | explicit error returns (D-02) |
| bare goroutines + `sync.WaitGroup` | invites a rewrite when transliterated | `errgroup.Group` with `SetLimit`, nothing else |
| channels + `select` as a design element | C# reaches for `Task.WhenAll`, Kotlin for `awaitAll` — three different-looking programs | channels **only** for the lease pool, which maps cleanly to `BlockingCollection`/`Channel` |
| `context.Context` in every signature | `CancellationToken` matches, Kotlin's coroutine context does not | `ctx` only at boundaries: CLI entry, HTTP handler, worker job, `ProcessImg`. Never in pure functions |
| functional options (`WithVerbose(…)`) | idiomatic Go, noisy in both others | an `Options` struct + `DefaultOptions()` → C# record with init props, Kotlin data class with defaults |
| variadic `any` option bags | maps to nothing pleasant | explicit typed context structs (`PostContext{PaddingMeta, ImgShape, Resize, Upsample}`) |
| **embedding as virtual dispatch** | Go embedding is **not** virtual: a "subclass" overriding `nmsIndices` silently calls the base method | flatten the hierarchy (§5) |
| generics beyond `[T any]` on containers | Go's inference gaps produce call sites unlike C#'s and Kotlin's | avoid; a few duplicated 5-line loops beat three different generic designs |
| `iota` integer enums | Kotlin wants `enum class`, C# an `enum`; three serialisations | **string constants** for every tag in `model.json`, and for device, OCR mode and stage names |
| `init()` | no analogue; hidden global setup | an explicit `Load()` called from `main` |
| package-level mutable state | fights DI in the other two | one documented exception: the runtime pool singleton, because Python has one too |
| JSON naming policies / converters | Go, `System.Text.Json` and `kotlinx.serialization` have three different defaults, and ~60 wire names must match byte-for-byte | hand-write **every** `json:"…"` / `[JsonPropertyName]` / `@SerialName` |
| `time.Duration` in wire structs | serialises differently in all three | float seconds, 4 dp, as Python does |
| `map` iteration where order matters | Go randomises; this is **correctness, not style** | `map` for lookup only; anything reaching output goes through an explicit ordered key slice |

That last row is load-bearing: the OCR dict feeds both field ordering and the
service's search text.

---

## 2. The `model.json` tag dispatch

The library is config-driven, and that is the single most portable thing about it.
The same fourteen `model.json` files must drive all four languages **unchanged**.

* DTOs carry the **exact** JSON keys. Every optional numeric is **nullable**
  (`*float64` / `double?` / `Double?`) so "absent" is distinguishable from zero —
  `BlankIndex` legitimately *is* 0 and `Threshold` defaults to 0.5.
* Exactly three functions, one `switch` each. No reflection, no attributes, no DI
  container, no self-registering `init()`:
  * `newPreprocessor(in InputInfo)` — `Classification`, `YOLO`, `YOLOOBB`, `OCR`, `OCRv2`
  * `newPostprocessor(out OutputInfo, workDir string)` — `BinaryClassification`,
    `MultiLabelClassification`, `Metric`, `YOLODetector`, `PerClassYOLODetector`,
    `YOLOOBBDetector`, `YOLOSegmentor`, `OCR`, `OCRFV`, `OCRProbs`
  * `newModel(cfg, pre, sess, post)` — `YOLODetection`, `YOLOSegmentation`,
    `YOLOOBBDetection`, default `UnifiedModel`
* One construction expression per case, **cases in Python's `match` order**, recorded
  in `MAPPING.md`. Go `switch` / C# switch expression / Kotlin `when` then diff
  line-for-line.
* Unknown tag → error naming it (D-06). Unimplemented known tags → wired,
  returning `ErrNotImplemented`.
* **Normalise backslashes.** The shipped artifacts contain
  `"Centers": "resources\\centers.npz"` and `models\Borders` in
  `models_path.yaml` — Windows separators inside *data*. On Linux a backslash is an
  ordinary filename character, so `DocTypeAngles` dies at construction: only in a
  container, never on a Windows dev box. Python normalises in code rather than
  re-shipping the models; every port must do the same.
* Read `model.json` as BOM-free UTF-8 (D-10).

---

## 3. The two parallel groups

Both are structured fan-out/join over a fixed set. The mandated statement shape,
identical in all languages: one launch statement per member **in Python's source
order** → one join → one deterministic collection loop indexed by position → the
concurrent-group timing record.

| | Go | C# | Kotlin |
|---|---|---|---|
| group | `errgroup.Group` + `SetLimit(n)` | array of `Task<T>` + `Task.WhenAll` | `coroutineScope { async(Dispatchers.Default) }` + `awaitAll` |
| results | pre-sized slice, written by index | read the task array by index | the `awaitAll` list |
| limit (word group, ≤8) | `g.SetLimit(min(8,n))` | `SemaphoreSlim(min(8,n))` | `Semaphore(min(8,n))` |
| first error | `g.Wait()` | aggregate → rethrow first | `coroutineScope` cancels and rethrows |

**Never** append to a shared slice and **never** collect through a channel. Python's
`futures[i].result()` collection is positional and deterministic; a channel-collected
version reorders under load, changing box order, word order and the joined field
string — an exact-match conformance failure with no float involved.

**Do the CUDA mutex first, not later.** Concurrent `Run()` on one CUDA session must
be serialised: hold a per-session mutex around `Run()` **only**, and read the result
after releasing. Measured in the spike: 8 goroutines × 300 calls on one session took
**6.6 s with the mutex and over 600 s without it** (killed; 657 s of CPU, GPU pinned
at 100 %). Python's symptom was a wedge or `cudaErrorIllegalAddress`; Go's is a ~90×
thrash. From a service's point of view those are the same thing.

Note the asymmetry this creates and do not "fix" it: the 5-way quality group uses
five *different* sessions and keeps its speedup, while the ≤8-way word group shares
one `WordsDetector` session and therefore serialises on GPU.

---

## 4. Lease of one

Expose exactly one function — a higher-order `Use`, never `Acquire`/`Release`.
Python's `@contextmanager` maps onto it, and it makes the rule "transform the result
*before* releasing the lease" **structurally enforced** rather than a comment someone
deletes.

| | primitive | timeout | busy |
|---|---|---|---|
| Go | pre-filled `chan *Pipeline`; `Use(ctx, fn)` selects vs `time.After`, `defer` puts back | `select` | `ErrPipelineBusy` |
| C# | `BlockingCollection<Pipeline>`; `Use(Func<Pipeline,T>)` | `TryTake(out p, timeout)` | `PipelineBusyException` |
| Kotlin | pre-filled `Channel<Pipeline>(cap)`; `inline fun <T> use(block)` | `withTimeoutOrNull { receive() }` | `PipelineBusyException` |

Restate the hazard in each port's comment: `ProcessImg` rebinds `results` and
`ocrOptions` on every call, so two concurrent calls on one instance silently return
each other's fields — no crash, no reproduction in single-user testing, data
corruption under load.

---

## 5. Interface versus abstract class

**One-method interfaces plus free helper functions. No inheritance anywhere.**

* `Preprocessor interface { Apply(img Image) (Tensor, PaddingMeta, error) }`
* `Postprocessor interface { Apply(in []Tensor, ctx PostContext) (Result, error) }`
* `BasePreprocessing.padding` becomes a **free function**. C#: `IPreprocessor` +
  `static class PreprocessOps`. Kotlin: `interface Preprocessor` + top-level
  functions.

Flatten the two real inheritance uses:

* `OBBPreprocessing extends YoloPreprocessing` → its own type that **calls** the
  shared `letterbox` function.
* `PerClassYOLODetectorPostprocessing` overriding `nms_indices` → **one** type with
  an explicit `nmsMode string` field (`"classAgnostic"` | `"perClass"`).

Flattening is cheaper than explaining virtual dispatch three times, and it removes
the Go embedding footgun entirely.

**Heterogeneous returns.** Python's postprocessors return `(label, conf)`,
`(label, dist, threshold)`, `str`, `[]box`, `(masks, segments)`. Using `any` /
`object` / `Any` throughout kills type safety and makes call sites diverge. Instead
use a **closed set** of result structs — `ClassResult`, `MetricResult`, `TextResult`,
`DetectResult`, `SegmentResult` — and let the module layer, which knows what it
asked for, do **one** checked assertion in exactly one place:

```go
r, ok := out.(MetricResult); if !ok { return ErrUnexpectedResult }
```
```csharp
if (out is not MetricResult r) throw new UnexpectedResultException();
```
```kotlin
val r = out as? MetricResult ?: throw UnexpectedResultException()
```

Do **not** parameterise `Model[T]`: the concrete type is unknown until `model.json`
is read, so all three end in a runtime cast anyway, and Kotlin's variance rules make
the generic version ugly in a third distinct way.

---

## 6. Numeric determinism — the traps that actually bite

Every item here produces a silent **exact-match** failure with no float tolerance to
hide behind.

1. **`np.argmax` returns the FIRST maximum.** Strict `>`, never `>=`. On a tie, `>=`
   flips a CTC timestep and changes a character.
2. **Stable sorts everywhere.** Python's `sort(key=…)`, `np.lexsort` and `np.argsort`
   are all stable. Go's `sort.Slice` is **not**; C#'s `List.Sort` is **not**; LINQ's
   `OrderBy` is; Kotlin's `sortedBy` is. Use `sort.SliceStable`. Two equal-x word
   boxes reordering swaps two tokens in the joined field string.
3. **`np.lexsort((a, b))` sorts by `b` primary, `a` secondary** — the argument order
   is the reverse of the intuition.
4. **NMS tie-breaking:** the code takes `order[-1]` after a stable ascending
   `argsort`, i.e. on a confidence tie it keeps the **highest original index**.
5. **`np.round` is half-to-EVEN.** `np.round(0.5) == 0`, `np.round(1.5) == 2`. Go's
   `math.Round` is half-away-from-zero → a different integer box coordinate → a
   different crop → different OCR text. Use `math.RoundToEven`; C#'s `Math.Round`
   already defaults to `ToEven`; JVM uses `Math.rint`.
6. **Crop bounds — the highest-risk divergence in the whole port.** `img[y1:y2,
   x1:x2]` in Python silently clamps an over-large upper bound and treats a
   **negative** start as from-the-end. gocv's `Region`, OpenCvSharp's `Mat[Rect]` and
   the JVM `submat` all **throw**. A port that "works" is one that clamps, and it
   must clamp to `[0, dim]` the way Python's slicing effectively does — not throw,
   and not return a differently sized crop. Hence `imaging.ClampedCrop` is the
   **only** sanctioned crop path, with a unit test per language.
7. **float32 throughout.** Never accumulate in float64 "for accuracy" — it changes
   results. Applies to sigmoid, the proto-mask matmul, IoU and cosine distance. See
   D-05 for the JVM wrinkle. One exception, and it is deliberate: the cosine metric
   head accumulates in float64 because numpy's `dot`/`norm` promote — measured
   agreement 9e-16.
8. **`centers.npz`** is a zip of three `.npy` members: `labels` `<U64` (fixed-width
   UTF-32LE, NUL-padded), `centers` `<f4` (9,1100), `max_distance` `<f4` (9,). ~80
   lines per language; no npz dependency needed.
9. **Cosine "distance" semantics.** sklearn's is `1 - cosine_similarity`, so
   `radius=1` means "only centroids with positive similarity". Reproduce **both**
   gates — the radius filter *and* the per-class `max_distance` threshold — each
   returning `'NONE'`. With nine centroids this is a loop; sklearn is a convenience,
   not an algorithm.
10. **Integer division.** Python `//` floors toward −∞; Go/C#/Kotlin `/` truncates
    toward 0. Audit every `//`.
10b. **Python's FLOAT `//` is not `floor(x / y)`** — and this one was measured, not
    reasoned about. CPython implements it via `fmod`
    (`mod = fmod(x,y); div = (x-mod)/y; floor(div)` plus a half-ulp nudge), and
    subtracting the remainder first removes rounding error that a plain division
    leaves behind. The two disagree in the last bit. Consequence found in M1: a
    2999×1777 image resizes to width **1499** in Python and **1500** with
    `math.Floor(2999/ratio)` — a one-pixel-different canvas, hence every downstream
    box shifted by a pixel. Use `tensor.FloorDiv`, which reproduces CPython's
    algorithm; never the language's floor-division shortcut. The same trap exists in
    C# and on the JVM. Note the counter-intuitive corollary: `_prepare_image` does
    **not** guarantee the longest side equals `img_size`.
11. **`OCRProbsPostprocessing` masks with `-inf`, not by zeroing.** Naive column
    zeroing lets blank win and silently deletes the character.
12. **OCRv2 minimum width is 16**, and a degenerate zero-size crop returns a
    `(32,16,3)` zero tensor. Port both or a rare edge case becomes a crash.
13. **The deskewer's boundary early-out and `min_angle=2.0`** both change the canvas,
    and therefore every box downstream.
14. **Two-pass variance.** `rotated.sum(axis=1)` on uint8 is exact int64; the only
    float-sensitive step is `.var()`, which numpy computes two-pass. A one-pass
    `E[x²]−E[x]²` loses ~7 significant digits at magnitudes of 255·W — enough to flip
    the argmax between adjacent angles. Measured agreement with two-pass: 6e-6 on
    values of order 1e10.
15. **Copy every OpenCV DEFAULT ARGUMENT the reference relies on, especially the
    booleans.** `cv2.convexHull(cnt)` defaults to `clockwise=False`; the Go, C# and
    JVM bindings all make that parameter explicit, so a port picks a value rather than
    inheriting one. This is not cosmetic: the hull's ORIENTATION decides which vertices
    Douglas-Peucker keeps in `ExtractQuad`, so `clockwise=true` produces a different
    four-point quad on some contours. Measured in M4 as a 6 px canvas-width difference
    on an internal-passport spread — with `borders.segments` passing, which is what
    localised it to the quad rather than the mask. Audit `convexHull`, `approxPolyDP`,
    `arcLength`, `findContours` and `threshold` argument-for-argument against the
    Python call site, not against the binding's own defaults.
16. **Format every float with an INVARIANT locale.** Go has no such hazard — `strconv`
    and `fmt` are locale-free — which is exactly why this entry exists: the trap is
    invisible from the port that was written first, and it is present in **two** of the
    three that follow. On .NET, `double.ToString()`/`Parse` and on the JVM
    `String.format`/`toString` are culture-sensitive, so a machine whose default culture
    is `ru-RU` writes `0,904` where the contract requires `0.904` — every float in the
    view model, silently, on that machine only. Force `CultureInfo.InvariantCulture`
    (or `InvariantGlobalization=true`) and `Locale.ROOT` at every formatting and parsing
    boundary. A CI runner with an English locale will never reproduce the failure.
17. **Never widen float32 to float64 mid-computation.** Also D-05, but it belongs here
    because the symptom is a numeric divergence, not a design difference: on the JVM
    `Math.exp` takes and returns `Double`, so `exp(x)` on a `Float` silently promotes,
    accumulates in double precision and returns a value the reference never computed.
    The mandatory form is elementwise `exp(x.toDouble()).toFloat()`. First observable at
    `borders.protomask`. C# has the same promotion in `Math.Exp`; use `MathF`.

---

## 7. Naming

* **Initialisms:** `Ocr`, `Iou`, `Nms`, `Obb`, `Json`, `Http`, `Api`, `Id`, `Cls` —
  **not** `OCR`/`IOU`/`ID`. Go linters push all-caps and C#/Kotlin push Pascal;
  picking Pascal uniformly and configuring `revive`'s initialism exception is the
  smaller cost. Hard rule, linter config committed.
* **Type names identical across all ports:** `OcrProbsPostprocessing`,
  `YoloSegmentorPostprocessing`, `DocDeskewer`, `PipelineResults`,
  `OcrOptionsIntPassport`.
* **Private stage methods keep Python's words and order:**
  `_quality_and_borders_parallel` → `qualityAndBordersParallel`. This is what lets
  someone grep three files side by side.
* **File names are not derived by rule** — the three-column table in `MAPPING.md` is
  the source of truth.
* **Package/namespace last segment identical:** Go `postprocess`, C#
  `RussianDocs.DocumentProcessing.Postprocess`, Kotlin
  `net.russiandocs.docproc.postprocess`.
* **JSON wire names always explicit, snake_case, never a policy.**

---

## 8. Comments are a portability asset

The Python source is unusually heavily commented, and **every load-bearing comment
encodes a bug that has already been paid for**.

Rule: each is copied verbatim into all ports at the same position, tagged with the
Python `file:line` it came from. This is the mechanism by which ports two and three
do not re-introduce fixed bugs.

Non-negotiable list: the CUDA per-session lock rationale; `'intpassportaddr'` must be
tested before `'intpassport'`; the SNILS odd-word-index Cyrillic routing; the `-inf`
masking rationale; OCRv2 minimum width 16; the deskew coarse-boundary early-out; why
the parallel group is conditional on `low_quality`; the `ocr_gpu_batch` accuracy
measurement; "merge, never assign" on the OCR dict; and the reason the duplicate
`Licence_number` dedup exists.

---

## 9. Resource lifetime

OpenCV `Mat` is unmanaged in every binding — gocv, OpenCvSharp and the JVM alike.
Python's GC hides this completely, which is **the** way a port that passes
conformance still dies in production after 500 documents.

Mandate an `Image` type that owns its `Mat`, and `defer img.Close()` / `using` /
`use {}` at **every** allocation site. Add a leak test per port: process one sample
200 times and assert RSS is flat.

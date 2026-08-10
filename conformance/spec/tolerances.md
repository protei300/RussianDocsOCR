# Comparison rules

Normative. Exactly one implementation of these rules exists —
`conformance/runner/compare.py` — and every verdict in the project comes from it.

## The two-line summary

* **Numeric values** must agree with the Python reference within **absolute
  1e-3**, inclusive.
* **Discrete values** must agree **exactly**: no tolerance at all.

The 1e-3 figure is the company's established practice from the existing .NET work,
not a number chosen here. Typical observed divergence is at the 5th–7th decimal.

## What counts as discrete

Exact, always, with no leniency:

* document type label, and `'NONE'`
* every OCR string, at every stage
* quality verdicts — and note the vocabulary is **not uniform**: `Glare` and
  `Blur` are `'good'`/`'bad'`, while `PrintSpoofing` and `LCDSpoofing` are
  `'REAL'`/`'FAKE'`. All four are *strings*. `DocConf`, in the same dict, is a
  float and is compared numerically.
* detection class labels
* **counts**: number of boxes, number of fields, number of address lines, array
  lengths
* **the set of JSON keys**. A missing or extra key is a failure, not a tolerance
  question: the SPA reads ~60 named fields from `web/src/types/index.ts`, and one
  absent key is a real bug.

## What is never compared by value

Present-and-well-formed only:

* `timings.*` — wall clock, meaningless across languages
* `device` — a real view-model key, but it records environment, not behaviour
* `ocr_device`, `providers` — same, where an implementation reports them
* `processing_ms`, `created_at`, `started_at`, `finished_at` — service-level, not
  part of the view model at all

Everything else **is** compared, including literal strings such as
`coord_space_note`.

## Byte-diffing JSON is forbidden

Always parse, then compare parsed values.

`round(x, 4)` in Python, `math.Round` in Go, `Math.Round(…, ToEven)` in C# and
Kotlin's formatting disagree on ties, so the *text* of a float legitimately
differs between languages while the value does not. A naive `diff viewmodel.json`
will fail forever and cost somebody a day.

## The tolerance is compared with a relative nudge, and it has to be

The implementation tests `delta > tol * (1 + 2⁻⁴⁰)`, not `delta > tol`.

Box confidences are already quantised to three decimals upstream
(`np.round(..., 3)` in `postprocessing.py`), so a tolerance of `1e-3` on them means
"at most one quantisation step". But `0.904 - 0.903` evaluates to
`1.0000000000000009e-3` in binary floating point, so a bare `>` rejects exactly the
difference the tolerance was chosen to admit — measured on the `Sex_ru` confidence of
`INTPASSPORT_2011`, where every other column of that box matched. The nudge is far too
small to admit a second step (that would take +100%) and large enough to absorb
representation error at these magnitudes.

## Confidences downstream of an R-02 stage inherit its noise

A one-step confidence difference in `fields.bbox` is an EXPECTED consequence of the
canvas relaxation below, not a defect in the detector.

`deskew.canvas` is explicitly allowed to differ by up to one grey level (R-02,
`warpPerspective` interpolation across OpenCV minor versions), and the field detector
runs on that canvas. Proven rather than assumed: running the *Python* detector on the
*Go* canvas reproduces Go's `0.904`, and on the reference canvas reproduces `0.903` —
same code, different input, so the detector port is exact.

What this does **not** excuse: a different number of boxes, a different class, a
different label, or a coordinate off by more than the profile allows. Those remain hard
failures, because the confidence is the only quantity fine-grained enough for one grey
level to move it.

## Documented relaxations

These are not the checker being lenient. They are cases where bit-parity is not a
meaningful goal, each with a rationale and a numeric rule.

### R-01 `borders.segments` — contour point lists

`cv2.findContours` returns a different number of points across OpenCV minor
versions, and a CUDA-produced mask differs from the CPU one in the last float bits,
which the binary threshold turns into whole pixels of outline. Compare instead:

| quantity | CPU profile | GPU profile |
|---|---|---|
| polygon **area**, relative | 1e-3 | 1e-3 |
| **centroid**, absolute px | 1e-3 | 2.0 |
| **Hausdorff distance**, px | 1.0 | 8.0 |

Not the point list.

**The centroid is the AREA-WEIGHTED polygon centroid, not the mean of the vertices.**
This is the whole point of R-01 and it is easy to get wrong: the first implementation
used the vertex mean and was measurably broken. Once the point COUNT differs — the very
thing R-01 exists to tolerate — the vertices are no longer uniformly distributed along
the outline, so their mean shifts toward wherever the extra points landed. Measured on
`INTPASSPORT_2011`, GPU against CPU: 173 points versus 176 moved the vertex mean by
**4.8 px** while the shape itself was unchanged. The shoelace-moment centroid is
invariant to how the outline is sampled, which is the property the rule needs.

### R-02 warped images (`prepare`, `rotate`, `borders.canvas`, `deskew.canvas`)

**Committed goldens hold a digest, not pixels.** Four image stages per case at
~2 MB each across seven cases is 50+ MB of binary in git, growing with every new
document type — a bad trade for an open-source reference project. The golden stores
a sha256 over the **array bytes** (not the file: a `.npy` header is padded to a
64-byte boundary, so identical arrays can produce different files) plus dtype and
shape.

A digest still delivers the headline result — *which stage diverged first* — but
not the magnitude, and R-02 needs the magnitude. So when a digest mismatches,
regenerate reference pixels locally:

```bash
python -m conformance.refcli regen --with-pixels --case <slug>
```

Those `.npy` files are gitignored. Localisation is committed; forensics are
reproducible on demand.

When pixels *are* present, the rule is:

`warpPerspective` and `warpAffine` may differ on a small fraction of pixels across
bindings. Rule:

* ≥ 99.5 % of pixels within ±1 LSB
* mean absolute difference < 0.5
* maximum difference ≤ 3

**Measured note:** in practice the spike found `prepare` and the letterbox to be
**bit-identical** between gocv and `cv2`, across OpenCV 4.12 vs 4.13, on ~14
million pixel values. So R-02 is headroom, not an expectation — a port that needs
it should be examined before it is granted.

### R-03 crop digests

Compare a crop's **shape** always, and a sha256 of its pixels only when the
upstream canvas matched exactly. Otherwise shape alone, and rely on the downstream
`ocr.<Field>.words` exact match to catch a wrong crop.

## GPU is a separate profile, and absolute 1e-3 is the WRONG metric there

Goldens are **CPU-generated**. CPU-to-CPU must be green under the rules above.

A GPU run is compared under a different numeric profile, because this was measured
and the naive rule fails on every document. From the spike, `Words` at 640² on the
same ONNX Runtime, CUDA versus CPU:

| quantity | value range | max abs difference | verdict under abs 1e-3 |
|---|---|---|---|
| box coordinates | up to 635 | **0.50** | 28 705 / 42 000 values "fail" |
| confidence score | 0…1 | **5.3e-3** | fails |
| boxes above the 0.2 gate | — | **167 vs 167** | identical |

A 0.5 difference on a coordinate of 635 is 0.08 % — sub-pixel. Nothing is wrong;
the metric is.

**GPU profile:**

* box and polygon coordinates: within **1.0 px** absolute
* confidences and other 0…1 scores: within **1e-2** absolute
* contours: see the R-01 table above
* **counts, labels and all OCR strings: still exact**
* per-case exceptions, if ever needed, go in `cases/<case>/gpu-deviations.json`
  as an explicit allowlist with a reason — never as a global loosening

If a GPU run cannot meet *that*, it is a real bug.

**Positional bbox rows get the same tolerances as named coordinates.** `fields.bbox` and
`words.<Field>.bbox` carry rows as ARRAYS — `[x1, y1, x2, y2, conf, cls, label]` — so a
path like `fields.bbox[11][2]` has no leaf NAME, and a tolerance table keyed by name
silently does not apply to it. The comparator therefore translates the trailing index into
its column name. Without that translation a pixel coordinate is graded with the profile's
*default* allowance (1e-2 on GPU), which is how the first GPU run failed
`fields.bbox[11][2]` at exactly 1.0 while the identical number passed as `boxes[3].x2` in
the view model. Any port-agnostic checker needs the same mapping.

**Result of the first GPU run under this profile:** PASS on all seven cases with zero
skips, and every discrete outcome — document type, box counts, class labels and every OCR
string — identical to the CPU goldens. The numeric allowances above were the only thing
GPU needed.

## Reporting

The headline number is the **first divergent stage per case**, not a pass count.
Full reports land in `conformance/report/<port>-<utc>.json`.

# Stage vocabulary

Normative. This is the fixed, ordered set of names under which an implementation
may dump an intermediate value.

**Why this file is the most valuable thing in the harness.** A port that fails
only on the final JSON tells you *that* it diverged. On a twelve-model pipeline
that is a week of bisection. "First divergence at `fields.bbox`" is an hour. Every
other document here exists to support this one.

## Rules

1. **Additive only.** Names may be added; they may never be renamed or removed.
   Golden files are keyed by them, and a rename silently invalidates history.
2. **Order is part of the contract.** The table below is pipeline order. A
   checker may rely on it to report the *first* divergence rather than a set.
3. **A stage is optional.** Not every document reaches every stage — `'NONE'`
   short-circuits after `doctype.label`, `address.*` exists only for
   `INTPASSPORTADDR`, and `--upto` stops early on purpose. A missing stage is not
   a failure; a *different* stage is.
4. **Names are lowercase, dot-separated, and group by pipeline phase**
   (`borders.canvas`, not `canvas_after_borders`). Per-field stages interpolate
   the field name verbatim as it appears in the library's OCR dict
   (`ocr.Last_name_ru.words`), because that name is itself part of the contract.
5. **Emitting must not change behaviour.** See
   `document_processing/pipeline/probe.py`: payloads are references to values the
   pipeline already computed. Nothing may be computed for the probe's benefit,
   and nothing may be threaded through a return type — otherwise the ports would
   be transliterating the instrumentation instead of the algorithm.

## The stages

| # | Stage | Payload | Type |
|---|---|---|---|
| 1 | `prepare` | decoded, RGB, resized to `img_size` | `.npy` `uint8 (H,W,3)` |
| 2 | `doctype.label` | `{doc_type, doc_type_confidence, angle, angle_confidence}` | `.json` |
| 3 | `rotate` | image rotated upright by `angle // 90` | `.npy` `uint8` |
| 4 | `quality` | `{Glare, Blur, PrintSpoofing, LCDSpoofing, DocConf}` | `.json` |
| 5 | `borders.segments` | the SELECTED document contours, `[[[x,y], ...], ...]` | `.json` |
| 6 | `borders.canvas` | perspective-corrected canvas | `.npy` `uint8` |
| 7 | `deskew.canvas` | canvas after residual-tilt correction | `.npy` `uint8` |
| 8 | `fields.bbox` | text-field detections, `[x1,y1,x2,y2,conf,cls,label]` | `.json` |
| 9 | `address.lines` | per-line address metadata (INTPASSPORTADDR only) | `.json` |
| 10 | `words.<Field>.bbox` | word boxes within each detection of one field | `.json` |
| 11 | `ocr.<Field>.words` | the per-word strings of one field, after `fix_errors` | `.json` |
| 12 | `join` | the assembled OCR dict, after field joining | `.json` |
| 13 | `viewmodel` | the final client-facing JSON (== `recognize` output) | `.json` |

`borders.segments` comes BEFORE `borders.canvas` because the contours are upstream of
the warp: when both diverge, that ordering tells the reader which one to blame. It
earned its place immediately — on the two internal-passport spreads it passed while
`borders.canvas` differed by 6 px, which localised the defect to the quadrilateral
approximation rather than to the mask. Compared under relaxation R-01.

`words.<Field>.bbox` is a LIST PER DETECTION of the field, not a flat box list, because
a field can be detected more than once (the internal passport prints its series and
number twice) and the pipeline concatenates their words. An entry of `null` means that
field needs no splitting, so its whole patch is the single word — which is a different
thing from a detector that found exactly one word, and a port that split a field it
should not have would otherwise look like agreement.

`viewmodel` is produced by the conformance CLI rather than by the library: the
view-model transform lives on the library side of a port (see `DEVIATIONS.md`
D-01) but in Python it sits in `service/ml/transform.py`.

**Image stages are committed as digests, not pixels** — `prepare`, `rotate`,
`borders.canvas` and `deskew.canvas` would be 50+ MB of binary in git across seven
cases. A digest still answers "which stage diverged first"; for the magnitude,
regenerate pixels locally with `regen --with-pixels`. See `tolerances.md` R-02.

## Deliberately absent, and why

* **Raw model output tensors** (`doctype.raw`, `borders.protomask`, …). Emitting
  them would mean reaching inside `pipeline_modules/*` rather than instrumenting
  `Pipeline`, multiplying the emission sites and the risk of perturbing the code
  under test. The spike established that the ONNX layer contributes *zero*
  divergence when fed an identical tensor — Go and Python were bit-identical on
  Glare, both OCR engines and DocTypeAngles — so raw model outputs are the least
  likely place for a port to differ. Add them only if a divergence is ever
  localised to a stage boundary and needs splitting further.
* **`deskew.angle`.** The pipeline does not retain it; `DocDeskewer` returns only
  the rotated image. Storing it would be a library change made purely for the
  harness's convenience. `deskew.canvas` catches any difference anyway, because a
  different angle produces a different canvas.
* **`words.patches`.** The payload is a dict of image crops — megabytes of pixels
  per document, and JSON-encoding them is absurd. Their BOXES are a stage
  (`words.<Field>.bbox`, added in M5 exactly as this note anticipated) because they
  are a handful of integers and they localise a wrong crop to the split rather than
  to the OCR three stages later; the pixels stay out.
* **Per-word OCR stages** (`ocr.<Field>.<i>.probs`). The word list is emitted
  whole, which shows exactly which word of which field differs — the same
  information for one call site instead of three (the routing has three branches).
  The `[1,T,C]` probability matrices would be large and are, again, the part the
  spike proved identical.

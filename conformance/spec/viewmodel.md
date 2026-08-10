# The view model

Normative. This is the JSON that `rdocs-conform recognize` emits and that the
`viewmodel` stage compares.

**This file is the contract; `service/ml/transform.py` is the reference
implementation.** When they disagree, that is a bug report against Python, not
against a port. Derived by hand from `web/src/types/index.ts` (the consumer) and
`service/ml/transform.py` (the producer).

Note the split of responsibility: in Python the transform lives in `service/`, but
in a port it belongs on the **library** side, because the conformance CLI needs it
and must not depend on the HTTP service. See `DEVIATIONS.md` D-01.

## Shape

Exactly fourteen top-level keys. Verified against `build_viewmodel` rather than
transcribed from the frontend types, which describe the *service's* response and
include fields the service adds:

```json
{
  "doc_type": "DL_2011",
  "doc_type_base": "DL",
  "doc_type_era": "2011",
  "recognised": true,
  "device": "cpu",
  "canvas":  {"width": 1021, "height": 637, "is_fallback": false},
  "coord_space": "canvas",
  "coord_space_note": "Boxes are canvas-space only; …",
  "boxes":  [ /* Box */ ],
  "fields": [ /* Field */ ],
  "ocr":     {"Last_name_ru": "ИВАНОВ"},
  "quality": {"DocConf": 0.96, "Glare": "good", "Blur": "good",
              "PrintSpoofing": "REAL", "LCDSpoofing": "REAL"},
  "timings": {"total": 0.53},
  "address": null
}
```

Three things that are easy to get wrong, and that an earlier draft of this file did
get wrong:

* **There is no top-level `doc_conf`.** The document-type confidence is
  `quality.DocConf`. The service's `_row()` lifts it to a column; the view model
  does not.
* **There is no `original` block.** Original dimensions and content type come from
  the uploaded bytes, which the library never sees. Service concern.
* **`device` IS part of the view model**, and is in the never-compared list of
  `tolerances.md` — it records environment, not behaviour.

The service adds `id`, `filename`, `status`, image URLs, timestamps, retry counters
and `doc_conf` on top. A port's **library** layer produces exactly the object above
and nothing more.

## Field-level rules

### `boxes[]` — canvas-space, axis-aligned

`{id, label, display, kind, x1, y1, x2, y2, conf, cls, text, ambiguous}`

* `id` is a stable synthetic handle (`"b3"`), referenced by `fields[].box_ids`.
* `kind` is `"text"` or `"visual"`. `Face` and `Signature` are detected but never
  OCR'd, so they are `visual` and carry no `text`.
* `ambiguous` is `true` when another box shares the label and owns the recognised
  text. This exists because the correspondence cannot be recovered from the
  library's output: `Licence_number` is detected twice on internal passports (the
  pipeline deduplicates the *field*, not the boxes), and split fields such as
  `Birth_place_ru` legitimately produce several boxes.

### `fields[]` — an ORDERED ARRAY, not a dictionary

`{name, display, value, script, conf, box_ids}`

Three problems are solved by this being an array with explicit `box_ids`, and all
three return if a port "simplifies" it into a map:

1. **Association.** Matching a field to a box by string equality of `label` is
   ambiguous (see `ambiguous` above).
2. **Order.** A dictionary has none, and insertion order is not document reading
   order.
3. **Rendering.** `script` (`ru` | `en` | `num`) selects proportional versus
   monospace type; the UI cannot infer it.

### `coord_space` and `coord_space_note`

`coord_space` is `"canvas"`. Boxes are in the pixel space of the corrected canvas
and match it exactly.

`coord_space_note` is a literal string and **is compared by value**. It records a
real limitation: the library does not retain the deskew angle, so boxes
**cannot** be mapped back onto the original upload. `DocDetector`'s `segm` contour
lives in pre-warp space and, if ever exposed, must be tagged `"prewarp"` so nobody
draws it on the canvas.

### `address` — `null` except for `INTPASSPORTADDR`

`{aligned, lines[{id, kind, text, p_handwritten, obbox}]}`, with `obbox` an
oriented box `{cx, cy, w, h, angle_rad, conf, label}`.

`aligned` is `false` when the geometry and text lists desynchronised (it happens
on an empty patch). A consumer must then not draw boxes rather than draw them
against the wrong lines.

`angle_rad` is **radians**, and the sign convention must be verified against one
real `INTPASSPORTADDR` before this path is declared done — no anonymised sample
exists in the repository, so it has no golden.

## Cross-language requirements

These exist so a port's output can be compared at all:

1. **Round every float to 4 decimals on the producing side.** Not the checker's
   job. Without it, float formatting differs between languages and the golden text
   diverges even when the values agree. Box confidences arrive already quantised to
   3 decimals by `postprocessing.py`, so 1e-3 permits at most one LSB there.
   OBB angles use 6 decimals.
2. **Dates are ISO-8601 UTC with an explicit `Z`.**
3. **Never depend on dictionary insertion order.** `fields` is an array for this
   reason among others; anything else that reaches output goes through an explicit
   ordered key list. Go randomises map iteration, and the OCR dict feeds both field
   ordering and the service's search text.
4. **Emit `null` rather than omitting a key.** A port with non-nullable struct
   fields would otherwise disagree with a Python dict that simply lacks the key,
   and the key-set comparison would fail for a reason nobody intended.
5. **NaN becomes `null`.** It is not valid JSON.

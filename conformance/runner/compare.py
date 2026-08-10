"""The single implementation of the comparison rules in spec/tolerances.md.

Every verdict in the project comes from this file. There is deliberately no second
comparator anywhere — not in a port, not in a test helper — because two
implementations of a tolerance rule eventually disagree, and then a real
divergence can hide behind whichever one is more forgiving.

Imports numpy only. Never document_processing, never a port.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #

#: Values that are compared for PRESENCE and TYPE but never for content.
#: Wall-clock timings and environment reporting are not behaviour.
#:
#: Each pattern is anchored to a path SEGMENT boundary, `(^|\.)`, not to the start
#: of the path. Callers pass a root label (`compare_json(..., path="viewmodel")`)
#: so real paths look like `viewmodel.timings._blur`; anchoring on `^` silently
#: matched nothing and every timing was compared as a value. Caught by running the
#: reference against its own goldens — which is exactly what that check is for.
IGNORED_PATHS = (
    re.compile(r"(^|\.)timings(\.|$)"),
    re.compile(r"(^|\.)device$"),
    re.compile(r"(^|\.)ocr_device$"),
    re.compile(r"(^|\.)providers(\[|$)"),
    re.compile(r"(^|\.)processing_ms$"),
    re.compile(r"(^|\.)(created|started|finished)_at$"),
)

#: Leaf names that are geometric coordinates. Under the GPU profile these get a
#: sub-pixel allowance instead of an absolute 1e-3 one -- measured necessity, see
#: spec/tolerances.md: CUDA differs from CPU by up to 0.50 on values reaching 635,
#: which is 0.08 % and changes no discrete outcome.
COORDINATE_LEAVES = frozenset({"x1", "y1", "x2", "y2", "cx", "cy", "w", "h"})

#: Leaf names that are 0..1 confidences.
SCORE_LEAVES = frozenset({"conf", "doc_conf", "DocConf", "p_handwritten"})

#: Leaf names that are integers identifying something, not measuring it. These stay
#: EXACT on every profile: a different class index is a different class.
DISCRETE_NUMERIC_LEAVES = frozenset({"cls", "width", "height", "field_count",
                                     "retry_count", "size_bytes", "id"})


@dataclass(frozen=True)
class Profile:
    """Numeric allowances. Discrete comparisons are exact on every profile."""

    name: str
    default_abs: float
    coordinate_abs: float
    score_abs: float

    def tolerance_for(self, leaf: str) -> float:
        if leaf in COORDINATE_LEAVES:
            return self.coordinate_abs
        if leaf in SCORE_LEAVES:
            return self.score_abs
        return self.default_abs


#: CPU-to-CPU. Goldens are generated here, so this is the strict profile.
CPU = Profile(name="cpu", default_abs=1e-3, coordinate_abs=1e-3, score_abs=1e-3)

#: A GPU run. Not a loosening for convenience -- an absolute 1e-3 on box
#: coordinates fails on every document while every discrete outcome is identical.
GPU = Profile(name="gpu", default_abs=1e-2, coordinate_abs=1.0, score_abs=1e-2)

PROFILES = {"cpu": CPU, "gpu": GPU}


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class Diff:
    path: str
    kind: str          # 'missing' | 'extra' | 'type' | 'value' | 'length' | 'shape'
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.kind}: {self.detail}"


@dataclass
class StageResult:
    stage: str
    ok: bool
    diffs: list[Diff] = field(default_factory=list)
    skipped: str | None = None   # reason, when an implementation does not claim it
    warn: str | None = None      # passed, but the comparison proved little

    @property
    def status(self) -> str:
        if self.skipped:
            return "skip"
        return "pass" if self.ok else "FAIL"


def vacuous_reason(payload: Any) -> str | None:
    """Report why a golden payload proves nothing, or None if it is substantive.

    A stage whose golden is ``null`` or empty compares equal to almost anything and
    shows up green while checking nothing. That is not hypothetical: the first version
    of the ``doctype.label`` emission read a key that does not exist, emitted ``None``,
    and every port "passed" it — the defect only surfaced when a port produced a real
    value and there was nothing to compare it against.

    So a vacuous golden is surfaced as a warning rather than trusted. It is not a
    failure: some stages are legitimately absent for some documents (``address.lines``
    outside INTPASSPORTADDR), and the ordinary way to fix a genuinely wrong one is to
    correct the emission and regenerate.
    """
    if payload is None:
        return "golden is null — this stage compares nothing"
    if isinstance(payload, (dict, list, str)) and len(payload) == 0:
        return f"golden is an empty {type(payload).__name__} — this stage compares nothing"
    if isinstance(payload, dict) and all(v is None for v in payload.values()):
        return "every value in the golden is null — this stage compares nothing"
    return None


def _ignored(path: str) -> bool:
    return any(p.search(path) for p in IGNORED_PATHS)


#: Column names for the stages whose payload is a list of POSITIONAL bbox rows.
#:
#: Without this the profile's tolerances silently do not apply to those stages. The rows
#: are arrays, so `_leaf("fields.bbox[11][2]")` is the empty string, no entry in
#: COORDINATE_LEAVES matches, and a pixel coordinate gets graded with `default_abs` --
#: 1e-2 on the GPU profile, where a one-pixel difference is expected and allowed for the
#: very same number when it appears as `boxes[3].x2` in the view model. Found by the first
#: GPU run, which failed `fields.bbox[11][2]` at exactly 1.0 while the view model's copy
#: of that coordinate passed.
BBOX_ROW_COLUMNS = ("x1", "y1", "x2", "y2", "conf", "cls", "label")

#: Paths that denote a bbox ROW, so the index that follows one is a COLUMN.
#:
#:   fields.bbox[i]           -> a row; [i][j] is column j
#:   words.<Field>.bbox[i][j] -> a row; [i] is the detection, [j] the box within it
_BBOX_ROW_PATHS = (
    re.compile(r"^fields\.bbox\[\d+\]$"),
    re.compile(r"^words\.[^.]+\.bbox\[\d+\]\[\d+\]$"),
)


def _leaf(path: str) -> str:
    """Name of the value at `path`, for tolerance lookup.

    'boxes[3].x1' -> 'x1'. For a positional bbox row the trailing INDEX is translated into
    the column's name ('fields.bbox[11][2]' -> 'x2'), so a coordinate gets the same
    tolerance whether it is reached by name or by position.
    """
    if path.endswith("]"):
        open_bracket = path.rfind("[")
        if any(p.match(path[:open_bracket]) for p in _BBOX_ROW_PATHS):
            try:
                idx = int(path[open_bracket + 1:-1])
            except ValueError:
                idx = -1
            if 0 <= idx < len(BBOX_ROW_COLUMNS):
                return BBOX_ROW_COLUMNS[idx]
    tail = path.rsplit(".", 1)[-1]
    return tail.split("[", 1)[0]


def _is_number(v: Any) -> bool:
    # bool is an int in Python; treat it as discrete, never numeric.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- #
# JSON comparison
# --------------------------------------------------------------------------- #

def compare_json(golden: Any, actual: Any, profile: Profile = CPU,
                 path: str = "") -> list[Diff]:
    """Recursively compare two parsed JSON values.

    Never compares serialised text: round() breaks ties differently in Python, Go,
    C# and Kotlin, so the *text* of a float legitimately differs between languages
    while the value does not. See spec/tolerances.md.
    """
    diffs: list[Diff] = []

    if isinstance(golden, dict):
        if not isinstance(actual, dict):
            return [Diff(path or "$", "type", f"expected object, got {type(actual).__name__}")]
        # The key SET is discrete: a missing or extra key is a real bug, because
        # the SPA reads ~60 named fields and one absent key breaks a page.
        for key in sorted(set(golden) - set(actual)):
            diffs.append(Diff(_join(path, key), "missing", "absent from the implementation's output"))
        for key in sorted(set(actual) - set(golden)):
            diffs.append(Diff(_join(path, key), "extra", "not present in the golden"))
        for key in sorted(set(golden) & set(actual)):
            diffs.extend(compare_json(golden[key], actual[key], profile, _join(path, key)))
        return diffs

    if isinstance(golden, list):
        if not isinstance(actual, list):
            return [Diff(path or "$", "type", f"expected array, got {type(actual).__name__}")]
        if len(golden) != len(actual):
            # Length is a count, and counts are discrete on every profile.
            return [Diff(path or "$", "length", f"{len(golden)} vs {len(actual)}")]
        for i, (g, a) in enumerate(zip(golden, actual)):
            diffs.extend(compare_json(g, a, profile, f"{path}[{i}]"))
        return diffs

    if _ignored(path):
        # Presence and rough type only: both must be "the same kind of thing".
        if (golden is None) != (actual is None):
            diffs.append(Diff(path, "type", f"{golden!r} vs {actual!r} (ignored by value, but nullness differs)"))
        return diffs

    leaf = _leaf(path)

    if _is_number(golden) and _is_number(actual):
        if leaf in DISCRETE_NUMERIC_LEAVES:
            if golden != actual:
                diffs.append(Diff(path, "value", f"{golden} vs {actual} (discrete identifier)"))
            return diffs
        if math.isnan(float(golden)) or math.isnan(float(actual)):
            # NaN is not valid JSON and must have become null upstream.
            diffs.append(Diff(path, "value", "NaN reached the wire; it must be null"))
            return diffs
        tol = profile.tolerance_for(leaf)
        delta = abs(float(golden) - float(actual))
        # The comparison is `delta > tol * (1 + eps)`, not `delta > tol`, and the slack
        # is not sloppiness -- without it the stated policy cannot be met at all.
        #
        # The policy is "abs <= 1e-3", and box confidences are ALREADY quantised to three
        # decimals upstream (`np.round(..., 3)` in postprocessing.py), so 1e-3 is meant to
        # allow exactly one quantisation step. But 0.904 - 0.903 evaluates to
        # 1.0000000000000009e-3 in binary floating point, so a bare `>` rejects the very
        # difference the tolerance was chosen to admit. Measured, not hypothetical: it
        # failed the Sex_ru confidence on INTPASSPORT_2011 while every other column of
        # that box matched exactly.
        #
        # A relative nudge of 2^-40 is far too small to admit a second step (that would
        # need +100%) and large enough to absorb representation error at any magnitude
        # these payloads carry.
        if delta > tol * (1 + 2 ** -40):
            diffs.append(Diff(path, "value", f"{golden} vs {actual} (|d|={delta:.3e} > {tol:g})"))
        return diffs

    # Strings, bools, None and any type mismatch: exact.
    if type(golden) is not type(actual) and not (golden is None and actual is None):
        diffs.append(Diff(path, "type", f"{type(golden).__name__} vs {type(actual).__name__}"))
        return diffs
    if golden != actual:
        diffs.append(Diff(path, "value", f"{golden!r} vs {actual!r}"))
    return diffs


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


# --------------------------------------------------------------------------- #
# .npy comparison
# --------------------------------------------------------------------------- #

def compare_npy(golden: Path, actual: Path, profile: Profile = CPU,
                relaxed_image: bool = False) -> list[Diff]:
    """Compare two array payloads.

    ``relaxed_image`` applies R-02 from spec/tolerances.md, for images produced by
    warpPerspective/warpAffine. Note the spike measured these as bit-identical
    between gocv and cv2 across OpenCV 4.12 and 4.13, so R-02 is headroom rather
    than an expectation -- a port that needs it deserves a look before it is
    granted.
    """
    name = golden.name
    a = np.load(golden, allow_pickle=False)
    b = np.load(actual, allow_pickle=False)

    if a.shape != b.shape:
        return [Diff(name, "shape", f"{a.shape} vs {b.shape}")]
    if a.dtype.kind in ("U", "S"):
        bad = [(i, x, y) for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())) if x != y]
        return [Diff(f"{name}[{i}]", "value", f"{x!r} vs {y!r}") for i, x, y in bad[:10]]

    af, bf = a.astype(np.float64), b.astype(np.float64)
    nan_a, nan_b = np.isnan(af), np.isnan(bf)
    if not np.array_equal(nan_a, nan_b):
        return [Diff(name, "value", f"NaN positions differ ({nan_a.sum()} vs {nan_b.sum()})")]
    delta = np.abs(af - bf)
    delta[nan_a] = 0.0

    if relaxed_image:
        within1 = float((delta <= 1).mean())
        mad = float(delta.mean())
        worst = float(delta.max())
        problems = []
        if within1 < 0.995:
            problems.append(f"only {within1 * 100:.2f}% of pixels within 1 LSB (need 99.5%)")
        if mad >= 0.5:
            problems.append(f"mean abs diff {mad:.3f} (need < 0.5)")
        if worst > 3:
            problems.append(f"max diff {worst:.0f} (need <= 3)")
        return [Diff(name, "value", "; ".join(problems))] if problems else []

    tol = profile.default_abs
    over = int((delta > tol).sum())
    if over:
        idx = np.unravel_index(int(np.argmax(delta)), delta.shape)
        return [Diff(name, "value",
                     f"{over}/{delta.size} values exceed {tol:g}; "
                     f"max |d|={delta.max():.3e} at {tuple(int(i) for i in idx)}")]
    return []


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# R-01: contour comparison
# --------------------------------------------------------------------------- #

def compare_contours(golden: Any, actual: Any, profile: Profile = CPU,
                     path: str = "borders.segments") -> list[Diff]:
    """Compare polygon outlines under relaxation R-01 (spec/tolerances.md).

    The POINT LIST is deliberately not compared. cv2.findContours returns a different
    number of points across OpenCV minor versions, so an exact list comparison fails
    for a reason that has nothing to do with the port. What is compared instead:

      * the NUMBER of contours -- exactly, because that is how many pages were found;
      * each polygon's AREA, within a relative tolerance;
      * each polygon's AREA-WEIGHTED CENTROID;
      * the HAUSDORFF distance between the two point sets.

    Area and centroid catch a wrong or shifted polygon; Hausdorff catches a polygon
    that has the right area and centre but the wrong shape, which the first two would
    miss. Together they are stricter about what matters than a point-by-point diff, and
    silent about what does not.

    The centroid is the POLYGON centroid (area-weighted, via the shoelace moments), NOT
    the mean of the vertices. This is the whole point and it is easy to get wrong -- an
    earlier version of this function used the vertex mean and was measurably broken. If
    the point COUNT differs, which R-01 exists precisely to tolerate, the vertices are no
    longer uniformly distributed along the outline, so their mean shifts toward wherever
    the extra points landed. Measured on INTPASSPORT_2011 GPU-versus-CPU: 173 points
    against 176 moved the vertex mean by 4.8 px while the shape was unchanged. The
    area-weighted centroid is invariant to how the outline is sampled, which is the
    property R-01 needs.
    """
    diffs: list[Diff] = []
    if golden is None and actual is None:
        return diffs
    if not isinstance(golden, list) or not isinstance(actual, list):
        return [Diff(path, "type", f"expected two arrays, got {type(golden).__name__} "
                                   f"and {type(actual).__name__}")]
    if len(golden) != len(actual):
        return [Diff(path, "length", f"{len(golden)} contour(s) vs {len(actual)} — "
                                     f"a different number of pages was detected")]

    for i, (g, a) in enumerate(zip(golden, actual)):
        here = f"{path}[{i}]"
        gp, ap = np.asarray(g, dtype=np.float64), np.asarray(a, dtype=np.float64)
        if gp.size == 0 and ap.size == 0:
            continue
        if gp.ndim != 2 or ap.ndim != 2 or gp.shape[1] != 2 or ap.shape[1] != 2:
            diffs.append(Diff(here, "type", f"expected [[x,y],...], got {gp.shape} and {ap.shape}"))
            continue

        area_rel, centroid_abs, hausdorff_px = _contour_tolerances(profile)

        ga, aa = _polygon_area(gp), _polygon_area(ap)
        if ga > 0:
            rel = abs(ga - aa) / ga
            if rel > area_rel:
                diffs.append(Diff(here + ".area", "value",
                                  f"{ga:.1f} vs {aa:.1f} (relative {rel:.3e} > {area_rel:g})"))
        elif aa > 0:
            diffs.append(Diff(here + ".area", "value", f"golden area is 0 but actual is {aa:.1f}"))

        gc, ac = _polygon_centroid(gp), _polygon_centroid(ap)
        cdiff = float(np.abs(np.array(gc) - np.array(ac)).max())
        if cdiff > centroid_abs:
            diffs.append(Diff(here + ".centroid", "value",
                              f"({gc[0]:.3f},{gc[1]:.3f}) vs ({ac[0]:.3f},{ac[1]:.3f}) "
                              f"(max |d| {cdiff:.3e} > {centroid_abs:g})"))

        hd = _hausdorff(gp, ap)
        if hd > hausdorff_px:
            diffs.append(Diff(here + ".hausdorff", "value", f"{hd:.3f} px > {hausdorff_px:g} px"))

    return diffs


def _contour_tolerances(profile: Profile) -> tuple[float, float, float]:
    """Per-profile contour allowances: (area relative, centroid abs px, Hausdorff px).

    The CPU numbers are R-01 as specified. The GPU numbers are wider for the same reason
    box coordinates are: a mask produced on CUDA differs from the CPU one in the last
    float bits, and the binary threshold turns that into whole PIXELS of outline. The
    allowance is expressed in pixels because that is the unit the difference arrives in.

    Area and centroid stay tight relative to the object's own size -- a 500x700 page has
    an area around 3.5e5, so 1e-3 relative is ~350 px^2, roughly a one-pixel band along a
    third of the perimeter. A genuinely wrong polygon fails these by orders of magnitude,
    which is what keeps the GPU profile a measurement allowance rather than a blindfold.
    """
    if profile.name == "gpu":
        return 1e-3, 2.0, 8.0
    return 1e-3, 1e-3, 1.0


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace area, absolute — matching cv2.contourArea for a simple polygon."""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _polygon_centroid(pts: np.ndarray) -> tuple[float, float]:
    """Area-weighted centroid via the shoelace moments.

    Invariant to how densely the outline is sampled, which the vertex mean is not -- see
    the note in compare_contours. Degenerate polygons (fewer than 3 points, or zero
    signed area, e.g. all points collinear) fall back to the vertex mean, which is the
    only thing defined for them.
    """
    if len(pts) < 3:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    signed = cross.sum() / 2.0
    if signed == 0:
        return float(x.mean()), float(y.mean())
    cx = ((x + x1) * cross).sum() / (6.0 * signed)
    cy = ((y + y1) * cross).sum() / (6.0 * signed)
    return float(cx), float(cy)


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two point sets.

    Brute force over the full distance matrix. Contours here run to a few hundred
    points, so this is microseconds; a spatial index would be more code for no gain.
    """
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def first_divergence(results: Iterable[StageResult]) -> str | None:
    """The headline number for a case: WHERE it first went wrong.

    A pass count tells you how bad things are; this tells you what to fix. Stage
    order is the pipeline order defined in spec/stages.md, so 'first' is meaningful.
    """
    for r in results:
        if not r.ok and not r.skipped:
            return r.stage
    return None

"""Layer B document suitability quality gate.

Evaluates the cropped/rectified document (never the raw background frame)
after ``DocumentDetector``, ``DocTypeAngles``, ``Quality`` and ``Borders``,
and before ``Fields`` / ``Words`` / ``OCR``.

Thresholds are empirically calibrated on 297 real documents (``samples/``,
``Passports_Damir``, real migration cards, ``INTPASSPORTADDR`` issue #27) plus
432 controlled synthetic degradation runs (2026-11-24 measurement):

* **Reject-level rules** (5 visual rules + ``low_doc_conf < 0.35``):
  combined false-reject rate ``2/283 (0.71 %)`` on readable real documents
  (``0/148`` on perfect-OCR documents) and ``122/127 (96.1 %)`` recall on
  failing synthetic degradations (``exact < 0.5``).
* **Warn-only rules** (``glare``, ``extreme_perspective``, plus earlier
  advisory thresholds for size/blur/photometry/confidence): surface actionable
  hints without blocking OCR.
* **Info-only rules** (``doc_clipped``): reported with ``severity='info'`` and
  never elevate ``status`` to ``'warn'`` or ``'reject'`` because tight-cropped
  scans routinely touch the frame border (21.9 % of readable real documents).

In ``mode='warn'`` (default) the gate computes and attaches ``quality_gate``
while allowing the pipeline to continue through OCR. In ``mode='reject'`` the
pipeline stops before ``Fields`` / ``OCR`` whenever ``status == 'reject'`` and
sets ``stopped=True``. In ``mode='off'`` the gate is skipped completely.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..pipeline_modules.doc_detector.image_transformation import extract_quad, order_points

VALID_QUALITY_GATE_MODES: Tuple[str, ...] = ("off", "warn", "reject")
DEFAULT_QUALITY_GATE_MODE: str = "warn"


@dataclass(frozen=True)
class QualityGateThresholds:
    """Calibrated thresholds for Layer B document quality evaluation."""

    # Document size in original-frame pixels (before pipeline resize)
    doc_min_side_reject: float = 250.0
    doc_max_side_reject: float = 350.0
    doc_min_side_warn: float = 380.0
    doc_max_side_warn: float = 550.0

    # Blur (neural Blur coef: 1.0=sharp, 0.0=blurred; Laplacian var on max_side=1000)
    blur_coef_reject: float = 0.20
    blur_lap_reject: float = 10.0
    blur_coef_warn: float = 0.50
    blur_lap_warn: float = 25.0
    blur_lap_min_std: float = 12.0

    # Photometry on inner 94% of rectified document grayscale (0..255)
    dark_median_reject: float = 40.0
    dark_p95_reject: float = 65.0
    dark_median_warn: float = 85.0
    dark_p95_warn: float = 110.0

    bright250_frac_reject: float = 0.75
    bright_dyn_range_reject: float = 40.0
    bright250_frac_warn: float = 0.65
    bright_dyn_range_warn: float = 65.0

    low_contrast_dyn_reject: float = 35.0
    low_contrast_std_reject: float = 12.0
    low_contrast_dyn_warn: float = 50.0
    low_contrast_std_warn: float = 16.0

    # Document type confidence
    doc_conf_reject: float = 0.35
    doc_conf_warn: float = 0.75

    # Glare (warn-only; fraction of 28 tiles flagged as glare)
    glare_coef_warn: float = 0.07

    # Perspective distortion of Borders quad (warn-only)
    persp_angle_dev_warn: float = 15.0
    persp_opp_edge_ratio_warn: float = 1.35

    # Edge clipping tolerance on original frame (info-only)
    clipped_tol_px: float = 2.0


DEFAULT_THRESHOLDS = QualityGateThresholds()


def normalize_quality_gate_mode(mode: Optional[str]) -> str:
    """Validate and normalize a quality-gate mode string."""
    if mode is None:
        return DEFAULT_QUALITY_GATE_MODE
    norm = str(mode).strip().lower()
    if norm not in VALID_QUALITY_GATE_MODES:
        raise ValueError(
            f"quality_gate must be one of {VALID_QUALITY_GATE_MODES}, got {mode!r}"
        )
    return norm


def _quad_perspective_metrics(quad: np.ndarray) -> Tuple[float, float]:
    """Return ``(max_angle_dev_deg, max_opposite_edge_ratio)`` for an ordered (4, 2) quad."""
    q = order_points(np.asarray(quad, dtype=np.float32).reshape(4, 2)).astype(np.float64)
    edges = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    eps = 1e-6
    r02 = max(edges[0] / max(edges[2], eps), edges[2] / max(edges[0], eps))
    r13 = max(edges[1] / max(edges[3], eps), edges[3] / max(edges[1], eps))
    opp_edge_ratio = max(r02, r13)

    max_angle_dev = 0.0
    for i in range(4):
        u = q[(i - 1) % 4] - q[i]
        v = q[(i + 1) % 4] - q[i]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < eps or nv < eps:
            angle = 0.0
        else:
            cos_a = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
            angle = math.degrees(math.acos(cos_a))
        max_angle_dev = max(max_angle_dev, abs(angle - 90.0))

    return max_angle_dev, opp_edge_ratio


def _clipped_edges_count(quad: np.ndarray, fw: int, fh: int, tol_px: float) -> int:
    """Count full quad edges whose endpoints both lie within ``tol_px`` of the same border."""
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        return 0
    n_edges = 0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        same_border = (
            (x1 <= tol_px and x2 <= tol_px)
            or (y1 <= tol_px and y2 <= tol_px)
            or ((fw - x1) <= tol_px and (fw - x2) <= tol_px)
            or ((fh - y1) <= tol_px and (fh - y2) <= tol_px)
        )
        if same_border:
            n_edges += 1
    return n_edges


def compute_document_photometry(doc_img: Optional[np.ndarray]) -> Dict[str, float]:
    """Compute inner-94 % grayscale photometry and normalized Laplacian sharpness."""
    if doc_img is None or doc_img.size == 0 or doc_img.ndim < 2:
        return {}
    if doc_img.ndim == 2:
        gray = doc_img
    elif doc_img.shape[2] == 4:
        gray = cv2.cvtColor(doc_img, cv2.COLOR_RGBA2GRAY)
    elif doc_img.shape[2] == 1:
        gray = doc_img[:, :, 0]
    else:
        gray = cv2.cvtColor(doc_img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]
    mh, mw = (int(h * 0.03), int(w * 0.03)) if (h > 40 and w > 40) else (0, 0)
    inner = gray[mh : h - mh, mw : w - mw] if (mh > 0 and mw > 0) else gray
    if inner.size == 0:
        inner = gray

    p5, p50, p95 = [float(x) for x in np.percentile(inner, [5, 50, 95])]
    std_v = float(inner.std())
    bright250 = float((inner >= 250).mean())

    ih, iw = inner.shape[:2]
    scale1000 = 1000.0 / max(ih, iw, 1)
    nw = max(2, int(round(iw * scale1000)))
    nh = max(2, int(round(ih * scale1000)))
    interp = cv2.INTER_AREA if scale1000 < 1.0 else cv2.INTER_LINEAR
    g1000 = cv2.resize(inner, (nw, nh), interpolation=interp)
    lap_norm1000 = float(cv2.Laplacian(g1000, cv2.CV_64F).var())

    return {
        "photo_median": round(p50, 2),
        "photo_p5": round(p5, 2),
        "photo_p95": round(p95, 2),
        "photo_dyn_range": round(p95 - p5, 2),
        "photo_std": round(std_v, 2),
        "photo_bright250_frac": round(bright250, 4),
        "lap_var_norm1000": round(lap_norm1000, 2),
    }


def _extract_quads(
    segm: Optional[Sequence[Any]],
    pre_borders_geom: Any = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return ``(quads_rot, quads_orig)`` ordered 4-point quads from Borders contours."""
    quads_rot: List[np.ndarray] = []
    quads_orig: List[np.ndarray] = []
    if not segm:
        return quads_rot, quads_orig
    for contour in segm:
        if contour is None or len(contour) < 4:
            continue
        q = extract_quad(contour)
        if q is None:
            continue
        q_ord = order_points(q)
        quads_rot.append(q_ord)
        if pre_borders_geom is not None:
            q_o = pre_borders_geom.to_input(q_ord)
            if q_o is not None:
                quads_orig.append(order_points(q_o))
    return quads_rot, quads_orig


def _reason(code: str, severity: str, value: float | int, threshold: float | int) -> Dict[str, Any]:
    val: float | int = int(value) if isinstance(value, (int, np.integer)) and not isinstance(value, bool) else round(float(value), 4)
    thr: float | int = int(threshold) if isinstance(threshold, (int, np.integer)) and not isinstance(threshold, bool) else round(float(threshold), 4)
    return {
        "code": code,
        "severity": severity,
        "value": val,
        "threshold": thr,
    }


def evaluate_quality_gate(
    *,
    doc_img: Optional[np.ndarray],
    frame_shape: Optional[Tuple[int, int]] = None,
    document_box: Optional[Sequence[float]] = None,
    segm: Optional[Sequence[Any]] = None,
    pre_borders_geom: Any = None,
    doc_conf: Optional[float] = None,
    quality_scores: Optional[Dict[str, float]] = None,
    mode: str = DEFAULT_QUALITY_GATE_MODE,
    thresholds: QualityGateThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Evaluate Layer B suitability signals for one document.

    Args:
        doc_img: Rectified (or rotated) RGB document image.
        frame_shape: ``(height, width)`` of the original input frame before resize.
        document_box: ``[x1, y1, x2, y2]`` of the detected document on the input frame,
            or ``None`` when whole-frame fallback was used.
        segm: Raw ``DocDetector['segm']`` contours on the pre-borders rotated image.
        pre_borders_geom: ``Chain`` geometry mapping pre-borders rotated coordinates
            back to the input frame.
        doc_conf: Metric classifier confidence (``Quality['DocConf']``).
        quality_scores: Numeric model scores (``Blur``, ``Glare``) from ``QualityScores``.
        mode: ``'warn'`` or ``'reject'`` (``'off'`` should skip calling this).
        thresholds: Threshold configuration.

    Returns:
        Structured verdict dict: ``status`` (``'pass' | 'warn' | 'reject'``),
        ``mode``, ``stopped`` (``True`` only when ``mode == 'reject'`` and
        ``status == 'reject'``), ``reasons``, and ``metrics``.
    """
    norm_mode = normalize_quality_gate_mode(mode)
    scores = quality_scores or {}
    blur_coef = float(scores["Blur"]) if scores.get("Blur") is not None else None
    glare_coef = float(scores["Glare"]) if scores.get("Glare") is not None else None
    doc_conf_f = float(doc_conf) if doc_conf is not None else None

    quads_rot, quads_orig = _extract_quads(segm, pre_borders_geom)

    # 1. Effective document dimensions in original-frame pixels (before resize)
    doc_w: Optional[float] = None
    doc_h: Optional[float] = None
    if document_box is not None and len(document_box) == 4:
        x1, y1, x2, y2 = [float(v) for v in document_box]
        doc_w = max(0.0, x2 - x1)
        doc_h = max(0.0, y2 - y1)
    elif quads_orig:
        all_pts = np.vstack(quads_orig)
        doc_w = float(all_pts[:, 0].max() - all_pts[:, 0].min())
        doc_h = float(all_pts[:, 1].max() - all_pts[:, 1].min())
    elif frame_shape is not None and len(frame_shape) >= 2:
        doc_h, doc_w = float(frame_shape[0]), float(frame_shape[1])
    elif doc_img is not None and doc_img.ndim >= 2:
        doc_h, doc_w = float(doc_img.shape[0]), float(doc_img.shape[1])

    doc_min_side = round(min(doc_w, doc_h), 1) if (doc_w is not None and doc_h is not None) else None
    doc_max_side = round(max(doc_w, doc_h), 1) if (doc_w is not None and doc_h is not None) else None

    # 2. Perspective metrics from Borders quads
    persp_quads = quads_orig if quads_orig else quads_rot
    persp_max_angle_dev: Optional[float] = None
    persp_opp_edge_ratio: Optional[float] = None
    if persp_quads:
        pms = [_quad_perspective_metrics(q) for q in persp_quads]
        persp_max_angle_dev = round(max(a for a, _ in pms), 2)
        persp_opp_edge_ratio = round(max(r for _, r in pms), 4)

    # 3. Clipped edges on original frame (info-only)
    clipped_edges: Optional[int] = None
    if quads_orig and frame_shape is not None and len(frame_shape) >= 2:
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        if fh > 0 and fw > 0:
            clipped_edges = sum(
                _clipped_edges_count(q, fw, fh, thresholds.clipped_tol_px)
                for q in quads_orig
            )

    # 4. Photometry + Laplacian sharpness on inner 94% of document image
    photo = compute_document_photometry(doc_img)

    metrics: Dict[str, Any] = {
        "doc_min_side": doc_min_side,
        "doc_max_side": doc_max_side,
        "doc_conf": round(doc_conf_f, 4) if doc_conf_f is not None else None,
        "blur_coef": round(blur_coef, 4) if blur_coef is not None else None,
        "glare_coef": round(glare_coef, 4) if glare_coef is not None else None,
        "lap_var_norm1000": photo.get("lap_var_norm1000"),
        "photo_median": photo.get("photo_median"),
        "photo_p95": photo.get("photo_p95"),
        "photo_std": photo.get("photo_std"),
        "photo_dyn_range": photo.get("photo_dyn_range"),
        "photo_bright250_frac": photo.get("photo_bright250_frac"),
        "persp_max_angle_dev": persp_max_angle_dev,
        "persp_opp_edge_ratio": persp_opp_edge_ratio,
        "quad_clipped_edges": clipped_edges,
    }

    reasons: List[Dict[str, Any]] = []

    # --- Rule 1: doc_too_small ---
    if doc_min_side is not None and doc_max_side is not None:
        if doc_min_side < thresholds.doc_min_side_reject:
            reasons.append(_reason("doc_too_small", "reject", doc_min_side, thresholds.doc_min_side_reject))
        elif doc_max_side < thresholds.doc_max_side_reject:
            reasons.append(_reason("doc_too_small", "reject", doc_max_side, thresholds.doc_max_side_reject))
        elif doc_min_side < thresholds.doc_min_side_warn:
            reasons.append(_reason("doc_too_small", "warn", doc_min_side, thresholds.doc_min_side_warn))
        elif doc_max_side < thresholds.doc_max_side_warn:
            reasons.append(_reason("doc_too_small", "warn", doc_max_side, thresholds.doc_max_side_warn))

    # --- Rule 2: blur (recalibrated Blur coef + Laplacian fallback) ---
    lap = photo.get("lap_var_norm1000")
    p_std = photo.get("photo_std")
    lap_reliable = lap is not None and p_std is not None and p_std >= thresholds.blur_lap_min_std
    if blur_coef is not None and blur_coef < thresholds.blur_coef_reject:
        reasons.append(_reason("blur", "reject", blur_coef, thresholds.blur_coef_reject))
    elif lap_reliable and lap < thresholds.blur_lap_reject:
        reasons.append(_reason("blur", "reject", lap, thresholds.blur_lap_reject))
    elif blur_coef is not None and blur_coef < thresholds.blur_coef_warn:
        reasons.append(_reason("blur", "warn", blur_coef, thresholds.blur_coef_warn))
    elif lap_reliable and lap < thresholds.blur_lap_warn:
        reasons.append(_reason("blur", "warn", lap, thresholds.blur_lap_warn))

    # --- Rule 3: too_dark ---
    p_med = photo.get("photo_median")
    p_95 = photo.get("photo_p95")
    if p_med is not None and p_95 is not None:
        if p_med < thresholds.dark_median_reject and p_95 < thresholds.dark_p95_reject:
            reasons.append(_reason("too_dark", "reject", p_med, thresholds.dark_median_reject))
        elif p_med < thresholds.dark_median_warn:
            reasons.append(_reason("too_dark", "warn", p_med, thresholds.dark_median_warn))
        elif p_95 < thresholds.dark_p95_warn:
            reasons.append(_reason("too_dark", "warn", p_95, thresholds.dark_p95_warn))

    # --- Rule 4: too_bright (gated by dynamic range to protect white-paper scans) ---
    b250 = photo.get("photo_bright250_frac")
    dyn = photo.get("photo_dyn_range")
    if b250 is not None and dyn is not None:
        if b250 > thresholds.bright250_frac_reject and dyn < thresholds.bright_dyn_range_reject:
            reasons.append(_reason("too_bright", "reject", b250, thresholds.bright250_frac_reject))
        elif b250 > thresholds.bright250_frac_warn and dyn < thresholds.bright_dyn_range_warn:
            reasons.append(_reason("too_bright", "warn", b250, thresholds.bright250_frac_warn))

    # --- Rule 5: low_contrast ---
    if dyn is not None and p_std is not None:
        if dyn < thresholds.low_contrast_dyn_reject:
            reasons.append(_reason("low_contrast", "reject", dyn, thresholds.low_contrast_dyn_reject))
        elif p_std < thresholds.low_contrast_std_reject:
            reasons.append(_reason("low_contrast", "reject", p_std, thresholds.low_contrast_std_reject))
        elif dyn < thresholds.low_contrast_dyn_warn:
            reasons.append(_reason("low_contrast", "warn", dyn, thresholds.low_contrast_dyn_warn))
        elif p_std < thresholds.low_contrast_std_warn:
            reasons.append(_reason("low_contrast", "warn", p_std, thresholds.low_contrast_std_warn))

    # --- Rule 6: low_doc_conf (reject < 0.35, warn < 0.75) ---
    if doc_conf_f is not None:
        if doc_conf_f < thresholds.doc_conf_reject:
            reasons.append(_reason("low_doc_conf", "reject", doc_conf_f, thresholds.doc_conf_reject))
        elif doc_conf_f < thresholds.doc_conf_warn:
            reasons.append(_reason("low_doc_conf", "warn", doc_conf_f, thresholds.doc_conf_warn))

    # --- Rule 7: glare (warn-only) ---
    if glare_coef is not None and glare_coef >= thresholds.glare_coef_warn:
        reasons.append(_reason("glare", "warn", glare_coef, thresholds.glare_coef_warn))

    # --- Rule 8: extreme_perspective (warn-only) ---
    if persp_max_angle_dev is not None and persp_max_angle_dev > thresholds.persp_angle_dev_warn:
        reasons.append(_reason("extreme_perspective", "warn", persp_max_angle_dev, thresholds.persp_angle_dev_warn))
    elif persp_opp_edge_ratio is not None and persp_opp_edge_ratio > thresholds.persp_opp_edge_ratio_warn:
        reasons.append(_reason("extreme_perspective", "warn", persp_opp_edge_ratio, thresholds.persp_opp_edge_ratio_warn))

    # --- Rule 9: doc_clipped (info-only, never elevates status to warn/reject) ---
    if clipped_edges is not None and clipped_edges > 0:
        reasons.append(_reason("doc_clipped", "info", clipped_edges, 1))

    # Determine overall status (only 'reject' and 'warn' severities elevate status)
    severities = {r["severity"] for r in reasons}
    if "reject" in severities:
        status = "reject"
    elif "warn" in severities:
        status = "warn"
    else:
        status = "pass"

    return {
        "status": status,
        "mode": norm_mode,
        "stopped": bool(norm_mode == "reject" and status == "reject"),
        "reasons": reasons,
        "metrics": metrics,
    }

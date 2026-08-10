"""Batched OCR inference for the v2 engines (padding-only width bucketing).

EXPERIMENTAL / opt-in (``Pipeline(..., ocr_gpu_batch=True)``) - see the
tradeoff explained below before enabling in production.

Motivation: the v2 OCR models take a dynamic-width input `[1, 32, W, 3]`. On
ONNX Runtime's CUDA provider, every distinct W forces a graph
recompile/replan (extra Memcpy nodes, no CUDA Graph reuse) - measured at
~500-740 ms per single-patch call on an RTX 4070 Ti Super, vs ~1-4 ms/patch
when all patches in a call share one shape (400-3700x). On CPU the dynamic
per-patch path is already cheap and batching does not help (measured slower),
so this is GPU-only - see docs/progress-log.md for the full writeup.

`predict_batch_padded` quantizes both axes of the batch tensor to small FIXED
ladders (`_WIDTH_LADDER` for the width axis, `_COUNT_LADDER` for the
item-count/batch axis - same ladders for every call/document in a process),
grouping patches by width rung and padding each group's row count up to a
count rung too, then runs one inference call per (width-rung, count-rung)
pair actually present. Three things were verified empirically here, not
assumed:

- **Both axes must be fixed, together.** ONNX Runtime's CUDA provider appears
  to cache/compile per the FULL tensor shape, including the batch dimension -
  not just the spatial (width) dims. A per-document dynamic width-only bucket
  (even from a shared width ladder) still produces a near-unique (N, W) pair
  almost every time, because real documents naturally have different word
  counts per rung - so it never gets to reuse anything and stays slow (~2.5-7s
  per document, confirmed on 10 *different* files of the same doc type, and
  even after explicitly pre-warming every width rung once with N=1). Fixing
  width alone is not enough; fixing count alone is not enough.
- **The ladders must also be COARSE (few rungs).** Every extra rung multiplies
  the number of distinct (count, width) shape *pairs* a real, varying document
  can produce; each distinct pair pays its own one-time compile tax, worth it
  only if that exact pair recurs often. A fine-grained width ladder (~19
  rungs, no count ladder) produced so many distinct pairs that real documents
  rarely repeated one and stayed slow throughout a 10-document run. The
  current ladders (4 width rungs x 3 count rungs = <=12 possible shapes)
  converge to a fast steady state within the first few *different* documents
  of a session and stay there (measured ~100-250ms/engine-call once warm, vs.
  ~2.7-7s cold) - at the cost of coarser bucketing per point below.
- Padding the count axis with dummy (all-zero) rows does not affect the real
  rows' decoded output (verified: identical text with/without 31 dummy filler
  rows) - inference-time BatchNorm/SE-style layers use fixed running
  statistics, not per-call batch statistics, so rows are independent. Only the
  width ladder affects accuracy (next point).

**Known accuracy tradeoff (why this is opt-in, not the default):** padding
does NOT decode as harmlessly as hoped. Truncating each row's output to its
own real-content timestep count (via `_WidthToTime`, see below) removes most
of the padding-influenced tail, but measured against a bit-exact baseline
(CPU-provider serial decode, which is IDENTICAL to GPU-provider serial decode
with no padding/batching - confirming this is a batching effect, not generic
CPU/GPU numeric noise) it does not reach 100%. With the current coarse ladder,
measured on 14 real mixed documents: **5.4% of OCR fields differ for
'accurate', 14.1% for 'fast'**, vs. the CPU-exact baseline (finer, per-batch
granularity measured better in isolation - e.g. 18/276 word patches at a
32px-multiple bucket - but was not fast in practice per the point above, so is
not used). The diffs are not confined to the padded tail - some occur
mid-content - which points to a width-wise global-pooling component in the
mn4/EdgeNext backbones (e.g. a Squeeze-and-Excitation block) whose statistics
shift with any padding, however small. There is no known way to make batched
decode bit-exact for this architecture family; only to bound the drift.

The width(px) -> timesteps mapping used for the truncation step is
architecture-specific (empirically, MobileNetV4 rounds up, EdgeNext rounds
down with an offset) and not worth hardcoding: `_WidthToTime` learns it per
model with a handful of tiny calibration probes on first use (cached on the
model instance), then computes it in O(1) per patch.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Fixed rungs a patch's (post-resize, pre-pad) width is quantized up to, and a
# width-rung group's row count is padded up to. Deliberately COARSE (few
# rungs): every extra rung multiplies the number of distinct (count, width)
# shape pairs a document can produce, and each distinct pair pays its own
# one-time CUDA graph-compile tax that is only worth it if that exact pair
# recurs often. Empirically, a fine-grained ladder (~19 width rungs) produced
# so many distinct pairs that real (varying) documents rarely repeated one,
# staying slow (2.5-7s/doc) even after 10 documents of the same type; this
# coarser ladder recurs reliably within a handful of documents instead - see
# the module docstring and docs/progress-log.md for the measured numbers.
_WIDTH_LADDER = (128, 256, 512, 1024)
_COUNT_LADDER = (8, 32, 128)


def _ladder_value(x: int, ladder) -> int:
    for rung in ladder:
        if x <= rung:
            return rung
    # beyond the largest rung: round up to the next multiple of its last step
    step = ladder[-1] - ladder[-2]
    return ((x + step - 1) // step) * step


def _ladder_width(w: int) -> int:
    return _ladder_value(w, _WIDTH_LADDER)


def _ladder_count(n: int) -> int:
    return _ladder_value(n, _COUNT_LADDER)


class _WidthToTime:
    """Learns a loaded OCR model's input-width -> output-timesteps mapping.

    Assumes (verified on all four v2 OCR models) that T is a stride-periodic
    function of W: within one residue class (W mod stride), T increases by
    exactly 1 per `stride` pixels. Two probes find the stride; one probe per
    residue class then fixes the per-class offset (rounding behavior differs
    per architecture, so this is learned rather than assumed).
    """

    _CALIB_W = 64  # safe reference width for calibration probes

    def __init__(self, model):
        self._model = model
        self._stride: int = 1
        self._bases: Dict[int, Tuple[int, int]] = {}  # residue -> (w, t)

    def _t_for_width(self, w: int) -> int:
        x = np.zeros((1, 32, max(w, 1), 3), dtype=np.uint8)
        return self._model.inference_model.predict([x])[0].shape[1]

    def _ensure_calibrated(self):
        if self._bases:
            return
        w1, w2 = self._CALIB_W, self._CALIB_W * 5
        t1, t2 = self._t_for_width(w1), self._t_for_width(w2)
        self._stride = max(1, round((w2 - w1) / max(1, t2 - t1)))
        self._bases[w1 % self._stride] = (w1, t1)
        for r in range(self._stride):
            if r not in self._bases:
                w = self._CALIB_W + r
                self._bases[r] = (w, self._t_for_width(w))

    def t_of(self, w: int) -> int:
        self._ensure_calibrated()
        r = w % self._stride
        w_base, t_base = self._bases[r]
        return max(1, t_base + (w - w_base) // self._stride)


def _get_width_to_time(model) -> _WidthToTime:
    calib = getattr(model, '_ocr_batch_width_to_time', None)
    if calib is None:
        calib = _WidthToTime(model)
        model._ocr_batch_width_to_time = calib
    return calib


def predict_batch_padded(model, patches: List[np.ndarray]) -> List[str]:
    """Batch-decode a list of RGB word-crop patches through a loaded OCR model.

    EXPERIMENTAL: see the module docstring for the measured accuracy tradeoff
    before using this on GPU in production (``Pipeline(ocr_gpu_batch=True)``).

    Args:
        model: a loaded UnifiedModel-like object exposing
            ``preprocessings[0]`` / ``inference_model`` / ``postprocessings[0]``
            (i.e. ``self.model`` on an OCRCyrillic/OCRLatin module).
        patches: RGB ``np.ndarray`` word crops, arbitrary sizes.

    Returns:
        Decoded strings, one per input patch, same order. Empty list for
        empty input.
    """
    if not patches:
        return []

    pre = model.preprocessings[0]
    # pre(patch) -> [1, H, w_i, 3] uint8 (dynamic width); squeeze the batch dim.
    tensors = [pre(p)[0] for p in patches]
    h = tensors[0].shape[0]

    groups: Dict[int, List[int]] = defaultdict(list)
    for i, t in enumerate(tensors):
        groups[_ladder_width(t.shape[1])].append(i)

    w2t = _get_width_to_time(model)
    post = model.postprocessings[0]
    results: List[str] = [None] * len(tensors)  # type: ignore[list-item]

    for w_bucket, idxs in groups.items():
        n_real = len(idxs)
        n_bucket = _ladder_count(n_real)  # pad row count too - see module docstring
        batch = np.zeros((n_bucket, h, w_bucket, 3), dtype=tensors[0].dtype)
        for bi, i in enumerate(idxs):
            t = tensors[i]
            w = t.shape[1]
            batch[bi, :, :w] = t
            if w < w_bucket:
                batch[bi, :, w:] = t[:, -1:, :]  # edge replication (cv2.BORDER_REPLICATE equivalent)

        probs = model.inference_model.predict([batch])[0]  # [n_bucket, T_full, C]
        for bi, i in enumerate(idxs):
            t_real = min(w2t.t_of(tensors[i].shape[1]), probs.shape[1])
            results[i] = post(probs[bi, :t_real])

    return results


def warmup_ladder(model) -> None:
    """Attempt to pre-warm every (width, count) shape ``predict_batch_padded``
    can produce for this model, instead of letting real documents pay the
    compile cost organically, one newly-encountered pair at a time.

    **EXPERIMENTAL / NOT called automatically - measured inconclusive.**
    The idea (matching the ladder design's own logic: ONNX Runtime's CUDA
    provider compiles/caches per the FULL shape including the batch/count
    dimension, so warming every rung up front should mean no real document
    ever hits an uncompiled shape) does not hold up under measurement:

    - Per-combo cost scales sharply with BOTH width and count, not just
      width - the largest rung (count=128, width=1024) alone took ~7s;
      warming the full 4x3 ladder for one engine measured 15-90+ seconds
      total (varied across runs - see docs/progress-log.md), an unacceptable
      construction-time cost.
    - Worse: even after paying that cost, real documents were observed still
      slow (multi-second OCR stage) on shapes that HAD just been warmed
      moments earlier via this exact function - so the premise that a
      compiled shape stays "hot" for later, unrelated calls was not confirmed
      to hold in practice on this session's test machine. Root cause
      unresolved (possibly CUDA memory-arena churn across many large warmup
      shapes invalidating earlier cached plans, possibly this session's
      separately-documented system-level noise - not conclusively isolated).

    Left available for callers who want to experiment with it themselves
    (e.g. on a different machine, or with a narrower custom ladder), but do
    not assume it delivers the intended effect without measuring on your own
    setup first. Not useful on CPU (no per-shape compile cost there).
    """
    h = 32  # OCRv2Preprocessing's fixed target height
    for w in _WIDTH_LADDER:
        for n in _COUNT_LADDER:
            dummy = np.zeros((n, h, w, 3), dtype=np.uint8)
            try:
                model.inference_model.predict([dummy])
            except Exception as e:
                # best-effort (mirrors ModelInference's own warmup policy), but
                # never silent - a skipped rung means that shape will still
                # pay its compile cost on the first real document that hits it.
                print(f"[!] OCR ladder warmup skipped for shape (n={n}, w={w}): {e!r}")

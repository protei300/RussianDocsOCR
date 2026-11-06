"""Stamp OCR v3: a line patch in, a CTC probability matrix out.

The network reads a 32-px-high BGR uint8 line of any width and returns
``[T, C]`` softmax rows over ``blank + alphabet``. Everything after - greedy
reading, dates, numbers, streets, codes - works on that matrix, so the reader
hands it out rather than a string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CTCAlphabet:
    alphabet: str                       # without blank: column i > 0 -> alphabet[i - 1]
    index: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {ch: i + 1 for i, ch in enumerate(self.alphabet)}


def greedy(probs: np.ndarray, abc: CTCAlphabet) -> str:
    idx = probs.argmax(axis=1)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            out.append(abc.alphabet[i - 1])
        prev = i
    return ''.join(out)


def line_order(q, horizontal: bool = False) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left of the line: the long
    side first, left to right (ultralytics does not start an OBB from its long side)."""
    q = np.asarray(q, np.float64)
    first, second = q[1] - q[0], q[2] - q[1]
    # A narrow upright digit is taller than it is wide. Choosing its long
    # side would rotate it sideways; stamps reaching this stage are upright.
    swap = (abs(first[0]) / max(1e-12, np.linalg.norm(first)) <
            abs(second[0]) / max(1e-12, np.linalg.norm(second))) if horizontal else (
            np.linalg.norm(first) < np.linalg.norm(second))
    if swap:
        q = q[[1, 2, 3, 0]]
    if q[1][0] < q[0][0]:
        q = q[[2, 3, 0, 1]]
    return q


def crop(img: np.ndarray, q, px: float, py: float, horizontal: bool = False) -> np.ndarray | None:
    """The upright patch of a line box, grown by ``px`` of its height along the line
    and ``py`` of its height across it."""
    from document_processing.pipeline_modules import AddressLinesDetector
    q = line_order(q, horizontal=horizontal)
    w, h = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
    if w < 2 or h < 2:
        return None
    ang = np.arctan2(q[1][1] - q[0][1], q[1][0] - q[0][0])
    c = q.mean(0)
    p = AddressLinesDetector.crop_rotated(img, (c[0], c[1], w + 2 * px * h, h * (1 + 2 * py), ang))
    return p if p.size else None


class StampOCR:
    def __init__(self, model_dir: Path, providers=None):
        import onnxruntime as ort
        meta = json.loads((model_dir / 'model.json').read_text('utf-8'))['Outputs'][0]
        self.abc = CTCAlphabet(meta['Alphabet'])
        self.sess = ort.InferenceSession(str(model_dir / 'model.onnx'),
                                         providers=providers or ['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name

    def probs(self, patch: np.ndarray) -> np.ndarray:
        """Resize exactly as in the v3 training: PIL bilinear to 32 px high, width <= 1024."""
        from PIL import Image
        h, w = patch.shape[:2]
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((min(1024, max(1, int(w * 32 / max(1, h)))), 32), Image.BILINEAR)
        x = np.asarray(img, dtype=np.uint8)[..., ::-1][None].copy()
        return self.sess.run(None, {self.inp: x})[0][0]

    def join(self, mats: list[np.ndarray]) -> np.ndarray:
        """Matrices of several boxes of one field in a row, a space between them."""
        if len(mats) == 1:
            return mats[0]
        sep = np.zeros((2, mats[0].shape[1]), np.float32)
        sep[0, self.abc.index[' ']] = 1
        sep[1, 0] = 1
        out = [mats[0]]
        for m in mats[1:]:
            out += [sep, m]
        return np.vstack(out)

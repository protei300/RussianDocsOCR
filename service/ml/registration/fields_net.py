"""Five-class YOLO11s-OBB detector for printed and handwritten stamp fields.

Input is an upright stamp in BGR. The graph has an NCHW float32 input and
a raw [1, 10, anchors] output, without embedded NMS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = ('date', 'addr_line', 'house', 'korp', 'flat')


def _probabilistic_iou(boxes: np.ndarray) -> np.ndarray:
    """Pairwise Gaussian OBB similarity, matching the training predictor."""
    eps = 1e-7
    x, y, w, h, r = boxes.T
    a, b = w ** 2 / 12, h ** 2 / 12
    c, s = np.cos(r), np.sin(r)
    aa, bb, cc = a * c ** 2 + b * s ** 2, a * s ** 2 + b * c ** 2, (a - b) * c * s
    a_sum, b_sum, c_sum = aa[:, None] + aa, bb[:, None] + bb, cc[:, None] + cc
    dx, dy = x[:, None] - x, y[:, None] - y
    determinant = a_sum * b_sum - c_sum ** 2
    t1 = (a_sum * dy ** 2 + b_sum * dx ** 2) / (determinant + eps) * .25
    t2 = -c_sum * dx * dy / (determinant + eps) * .5
    det = np.maximum(aa * bb - cc ** 2, 0)
    t3 = np.log(determinant / (4 * np.sqrt(det[:, None] * det) + eps) + eps) * .5
    distance = np.clip(t1 + t2 + t3, eps, 100)
    return 1 - np.sqrt(1 - np.exp(-distance) + eps)


def _nms(boxes, scores, threshold=.7):
    """Fast rotated NMS used by the accepted checkpoint's predictor."""
    order = np.argsort(-scores, kind='stable')
    similarity = np.triu(_probabilistic_iou(boxes[order]), k=1)
    return order[~(similarity >= threshold).any(axis=0)]


def _xywhr_to_corners(cx, cy, w, h, r) -> np.ndarray:
    c, s = np.cos(r), np.sin(r)
    v1 = np.array([w / 2 * c, w / 2 * s])
    v2 = np.array([-h / 2 * s, h / 2 * c])
    ctr = np.array([cx, cy])
    return np.stack([ctr + v1 + v2, ctr + v1 - v2, ctr - v1 - v2, ctr - v1 + v2])


class FieldsDetector:
    def __init__(self, model_path: Path, conf: float = .4, size: int = 1024, providers=None):
        import onnxruntime as ort
        from document_processing.processing.preprocessing import OBBPreprocessing
        config = model_path.with_name('model.json')
        if config.is_file():
            labels = json.loads(config.read_text('utf-8'))['Outputs'][0]['Labels']
            if tuple(labels) != CLASSES:
                raise ValueError(f'Stamp field classes must be {CLASSES}, got {labels}')
        self.sess = ort.InferenceSession(str(model_path), providers=providers or ['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        self.conf = conf
        self.pre = OBBPreprocessing(image_size=(size, size, 3))

    def predict(self, img_bgr: np.ndarray) -> list[dict]:
        """Return class, confidence and four corners on the original stamp."""
        x, ratio, extra, pad, _ = self.pre(img_bgr[..., ::-1])
        raw = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4 + len(CLASSES) + 1:
            raise ValueError(f'Expected five-class raw OBB output [1,10,N], got {raw.shape}')
        rows = raw[0].T
        scores = rows[:, 4:4 + len(CLASSES)]
        classes, confidence = scores.argmax(axis=1), scores.max(axis=1)
        valid = confidence > self.conf
        rows, classes, confidence = rows[valid], classes[valid], confidence[valid]
        out = []
        # NMS is per class: an address line and a short number may overlap.
        for cls in range(len(CLASSES)):
            mask = classes == cls
            selected, confs = rows[mask], confidence[mask]
            if not len(selected):
                continue
            for i in _nms(selected[:, [0, 1, 2, 3, 9]], confs):
                cx, cy, w, h = selected[i, :4]
                # Match scale_boxes: undo the integer left/top actually added,
                # not the half-pixel pad when the total padding is odd.
                cx = (cx - round(pad[0] - .1)) / ratio[0] - extra[0]
                cy = (cy - round(pad[1] - .1)) / ratio[1] - extra[1]
                w, h = w / ratio[0], h / ratio[1]
                angle, conf = float(selected[i, -1]), float(confs[i])
                out.append({'field': CLASSES[cls], 'conf': conf,
                            'quad': _xywhr_to_corners(cx, cy, w, h, angle)})
        return sorted(out, key=lambda b: (float(b['quad'][:, 1].mean()), float(b['quad'][:, 0].mean())))

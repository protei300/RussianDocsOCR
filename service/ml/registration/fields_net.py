"""YOLO-OBB detector for printed and handwritten stamp fields.

Input is an upright stamp in BGR. The graph has an NCHW float32 input and
a raw [1, 4 + classes + 1, anchors] output, without embedded NMS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2

CLASSES = ('date', 'addr_line', 'house', 'korp', 'flat', 'code', 'issuer_line')
LEGACY_CLASSES = CLASSES[:5]


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
    def __init__(self, model_path: Path, conf: float | None = None, size: int | None = None,
                 providers=None, overlap: float | None = None):
        import onnxruntime as ort
        config = model_path.with_name('model.json')
        self.classes = CLASSES
        configured_size = 640
        configured_conf = 0.4
        configured_overlap = 0.7
        if config.is_file():
            cfg = json.loads(config.read_text('utf-8'))
            out_cfg = cfg['Outputs'][0]
            labels = tuple(out_cfg['Labels'])
            if labels not in (CLASSES, LEGACY_CLASSES):
                raise ValueError(f'Stamp field classes must be {CLASSES} or {LEGACY_CLASSES}, got {labels}')
            self.classes = labels
            configured_size = int(cfg.get('Inputs', [{'Shape': [640]}])[0]['Shape'][0])
            for key in ('Confidence', 'CLS'):
                if out_cfg.get(key) is not None:
                    configured_conf = float(out_cfg[key])
                    break
            for key in ('Overlap', 'IOU'):
                if out_cfg.get(key) is not None:
                    configured_overlap = float(out_cfg[key])
                    break
        self.sess = ort.InferenceSession(str(model_path), providers=providers or ['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        self.conf = configured_conf if conf is None else float(conf)
        self.overlap = configured_overlap if overlap is None else float(overlap)
        self.size = size or configured_size

    def prepare(self, img_bgr):
        """RGB/NCHW float32; cap size, preserving smaller images exactly."""
        h,w = img_bgr.shape[:2]; ratio = min(1.,self.size/max(h,w))
        nw,nh = round(w*ratio),round(h*ratio)
        rgb = img_bgr[..., ::-1]
        if (nw,nh)!=(w,h): rgb = cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
        left,top = round((self.size-nw)/2-.1),round((self.size-nh)/2-.1)
        padded = cv2.copyMakeBorder(rgb,top,self.size-nh-top,left,self.size-nw-left,
                                    cv2.BORDER_CONSTANT,value=(114,114,114))
        return np.ascontiguousarray(padded.transpose(2,0,1)[None],dtype=np.float32)/255.,ratio,(left,top)

    def predict(self, img_bgr: np.ndarray) -> list[dict]:
        """Return class, confidence and four corners on the original stamp."""
        x, ratio, pad = self.prepare(img_bgr)
        raw = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4 + len(self.classes) + 1:
            raise ValueError(f'Expected {len(self.classes)}-class raw OBB output, got {raw.shape}')
        rows = raw[0].T
        scores = rows[:, 4:4 + len(self.classes)]
        classes, confidence = scores.argmax(axis=1), scores.max(axis=1)
        valid = confidence > self.conf
        rows, classes, confidence = rows[valid], classes[valid], confidence[valid]
        out = []
        # NMS is per class: an address line and a short number may overlap.
        for cls in range(len(self.classes)):
            mask = classes == cls
            selected, confs = rows[mask], confidence[mask]
            if not len(selected):
                continue
            for i in _nms(selected[:, [0, 1, 2, 3, 4+len(self.classes)]], confs, threshold=self.overlap):
                cx, cy, w, h = selected[i, :4]
                # Match scale_boxes: undo the integer left/top actually added,
                # not the half-pixel pad when the total padding is odd.
                cx = (cx - pad[0]) / ratio
                cy = (cy - pad[1]) / ratio
                w, h = w / ratio, h / ratio
                angle, conf = float(selected[i, -1]), float(confs[i])
                out.append({'field': self.classes[cls], 'conf': conf,
                            'quad': _xywhr_to_corners(cx, cy, w, h, angle)})
        return sorted(out, key=lambda b: (float(b['quad'][:, 1].mean()), float(b['quad'][:, 0].mean())))

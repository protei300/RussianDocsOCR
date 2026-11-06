"""Read five stamp fields without dictionary substitutions.

Address lines stay address lines. Splitting them into region, street or house
requires a separately validated parser. Each box retains raw OCR and geometry.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import RegistrationConfig
from .fields_net import CLASSES, FieldsDetector
from .ocr import StampOCR, crop, greedy

FIELDS = CLASSES


class StampReader:
    def __init__(self, cfg: RegistrationConfig | None = None):
        self.cfg = cfg or RegistrationConfig()
        self.det = FieldsDetector(self.cfg.fields_model, conf=self.cfg.fields_conf)
        self.ocr = StampOCR(self.cfg.ocr_dir)

    def read(self, img: np.ndarray) -> dict:
        """Return raw field text and the boxes read, top to bottom."""
        fields = defaultdict(list)
        boxes = []
        for b in self.det.predict(img):
            q = b['quad']
            patch = crop(img, q, .3, .15, horizontal=b['field'] in ('house', 'korp', 'flat'))
            if patch is None:
                continue
            try:
                text = greedy(self.ocr.probs(patch), self.ocr.abc)
            finally:
                patch.fill(0)
            fields[b['field']].append(text)
            boxes.append({'field': b['field'], 'conf': round(b['conf'], 4),
                          'quad': np.asarray(q).round(1).tolist(), 'text': text})
        return {'fields': {f: '\n'.join(v for v in fields[f] if v) for f in FIELDS if fields[f]},
                'boxes': boxes}

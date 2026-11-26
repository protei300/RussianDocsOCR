"""Read 7-class stamp fields, with optional ``ocr_postprocessing`` refinement.

Each detected box is cropped and scored by the stamp OCR CTC model. When the
optional dictionary bundle is installed and ``cfg.postprocess`` is enabled,
``ocr_postprocessing.OCRPostprocessor.process_crops`` refines the field texts
jointly across the whole stamp and returns structured ``parsed`` fields
alongside the per-class ``fields`` and ``boxes``. When dictionaries are absent
(e.g. minimal installs or unit tests), the reader falls back cleanly to raw
greedy CTC decoding.
"""
from __future__ import annotations

from collections import defaultdict
import re

import numpy as np

from document_processing.pipeline_modules.registration_stamps.alignment import align_stamp

from .config import RegistrationConfig
from .fields_net import CLASSES, FieldsDetector
from .ocr import StampOCR, crop, greedy

FIELDS = CLASSES
_RE_DATE_YEAR_SUFFIX = re.compile(r'\s*Г\.?\s*$', re.IGNORECASE)


def _clean_date_text(text: str) -> str:
    """Drop trailing 'г.' / 'г' from stamp date text (blank-border artifact)."""
    return _RE_DATE_YEAR_SUFFIX.sub('', text.strip()).strip()


def _load_postprocessor(cfg: RegistrationConfig):
    """Build ``OCRPostprocessor`` when enabled and dictionaries are present; else ``None``."""
    if not getattr(cfg, 'postprocess', True):
        return None
    try:
        from ocr_postprocessing import OCRPostprocessor
        from ocr_postprocessing.postprocess import References
    except ImportError:
        return None
    addressru, fms_csv, street_names = cfg.ref_paths()
    refs = References(addressru=addressru, fms_csv=fms_csv, street_names=street_names)
    if not refs.available():
        return None
    return OCRPostprocessor(cfg.ocr_dir, refs=refs)


class StampReader:
    def __init__(self, cfg: RegistrationConfig | None = None):
        self.cfg = cfg or RegistrationConfig()
        self.det = FieldsDetector(self.cfg.fields_model, conf=self.cfg.fields_conf)
        self.ocr = StampOCR(self.cfg.ocr_dir)
        self.postprocessor = _load_postprocessor(self.cfg)

    def read(self, img: np.ndarray) -> dict:
        """Return field text, boxes (top to bottom) and optional structured ``parsed`` dict."""
        if getattr(getattr(self, 'cfg', None), 'stamp_deskew', True):
            aligned, to_input, _ = align_stamp(img)
        else:
            aligned, to_input = img, np.eye(3)
        try:
            return self._read_fields(aligned, to_input)
        finally:
            if aligned is not img:
                aligned.fill(0)

    def _read_fields(self, img: np.ndarray, to_input: np.ndarray) -> dict:
        """Read the aligned stamp, returning boxes on the caller's original crop."""
        entries: list[dict] = []
        probs_list: list[np.ndarray] = []
        try:
            for b in self.det.predict(img):
                q = b['quad']
                patch = crop(img, q, .3, .15,
                             horizontal=b['field'] in ('house', 'korp', 'flat', 'code'))
                if patch is None:
                    continue
                try:
                    pr = self.ocr.probs(patch)
                finally:
                    patch.fill(0)
                probs_list.append(pr)
                quad = np.asarray(q, dtype=float)
                quad = np.c_[quad, np.ones(len(quad))] @ to_input.T
                entries.append({
                    'field': b['field'],
                    'conf': round(b['conf'], 4),
                    'quad': (quad[:, :2] / quad[:, 2:]).round(1).tolist(),
                    'probs': pr,
                })

            postprocessor = getattr(self, 'postprocessor', None)
            parsed = None
            if postprocessor is not None and entries:
                pp_res = postprocessor.process_crops([
                    {'label': e['field'], 'probs': e['probs'], 'idx': i}
                    for i, e in enumerate(entries)
                ])
                texts_by_idx = {f['idx']: f['text'] for f in pp_res['fields']}
                parsed = dict(pp_res.get('stamp') or {})
                texts = [texts_by_idx.get(i, '') for i in range(len(entries))]
            else:
                texts = [greedy(e['probs'], self.ocr.abc) for e in entries]
            texts = [
                _clean_date_text(t) if e['field'] == 'date' else t
                for e, t in zip(entries, texts)
            ]
        finally:
            for pr in probs_list:
                if isinstance(pr, np.ndarray) and pr.flags.writeable and pr.size:
                    pr.fill(0)

        fields = defaultdict(list)
        boxes = []
        for e, text in zip(entries, texts):
            fields[e['field']].append(text)
            boxes.append({'field': e['field'], 'conf': e['conf'], 'quad': e['quad'], 'text': text})
        out = {
            'fields': {f: '\n'.join(v for v in fields[f] if v) for f in FIELDS if fields[f]},
            'boxes': boxes,
        }
        if parsed is not None:
            out['parsed'] = parsed
        return out

from typing import List, Union
from pathlib import Path

import numpy as np

from ..base_module import BaseModule


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inside(inner, outer, share: float = 0.7) -> bool:
    """True when at least `share` of `inner` lies within `outer`."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return False
    return w * h >= share * max(_area(inner), 1e-9)


def group_documents(bbox: list) -> List[dict]:
    """Detector boxes -> documents, each with the pages that lie inside it.

    The detector has two classes: 'document' - whatever lies in the frame as one
    piece (a passport spread, a card, a single visible page) - and 'page', every
    visible page of an internal passport. A page belongs to the document it lies
    in; a document with no page inside is a single sheet. Documents come out
    largest first, which is the one process_img reads.

    Box layout in and out: [x1, y1, x2, y2, conf, cls, label].
    """
    docs = [b for b in bbox if str(b[-1]) == 'document']
    pages = [b for b in bbox if str(b[-1]) == 'page']
    docs.sort(key=_area, reverse=True)
    out = [{'box': list(d[:4]), 'conf': float(d[4]), 'pages': []} for d in docs]
    for p in pages:
        owner = next((d for d in out if _inside(p[:4], d['box'])), None)
        if owner is not None:
            owner['pages'].append({'box': list(p[:4]), 'conf': float(p[4])})
    for d in out:
        d['pages'].sort(key=lambda p: (p['box'][1], p['box'][0]))
    return out


class DocumentDetector(BaseModule):
    """Finds every document lying in a frame, and the pages of a passport spread.

    The first stage of the pipeline: the type classifier, the border detector and
    everything after them then work on the crop of one document instead of the
    whole frame. A whole frame misleads the classifier whenever the document is a
    small part of it - a licence on an A4 scan was read as a passport from the
    white sheet around it (measured 2026-09-29: type by frame 50 % on licences,
    by crop 99 %) - and it can hold several documents (issue #26).
    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False, runtime: str = None):
        self.model_name = 'DocumentDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose,
                         runtime=runtime)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects documents and passport pages.

        Returns:
            {model_name: {'bbox': raw boxes, 'documents': group_documents(bbox)}}
        """
        img = self.load_img(img)
        bbox = self.model.predict(img)
        return {self.model_name: {'bbox': bbox, 'documents': group_documents(bbox)}}

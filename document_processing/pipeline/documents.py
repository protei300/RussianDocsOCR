"""Finding the documents in a frame, cutting one out, its type and angle,
and pairing the two sides of a document read from one frame."""
import re
from typing import Optional

import cv2
import numpy as np

from .results import PipelineResults


#: Front type -> back type of a two-sided document whose sides carry the same
#: number. The vehicle registration certificate prints its series and number on
#: both sides (read as Licence_number on each), so a front and a back lying in one
#: frame - one sheet of a two-sided scan - pair up by it (issue #26). A licence back
#: carries no number the pipeline reads, so it is not here.
PAIRED_SIDES = {'STS': 'STSBACK'}


def _document_number(results: 'PipelineResults') -> str:
    """Digits of the series and number read on a side; '' when nothing was read."""
    return re.sub(r'\D', '', (results.ocr or {}).get('Licence_number', ''))


def pair_sides(documents: list) -> None:
    """Set ``paired_with`` on the two sides of one document read from one frame.

    A front and a back pair when their types belong together (PAIRED_SIDES), they
    read the same number, and that number is unique among the sides of that type
    in the frame - two certificates scanned on one sheet must not cross-pair, and
    an ambiguous number pairs nothing rather than guessing.
    """
    family = lambda r: (r.doctype or 'NONE').rsplit('_', 1)[0]
    for front, back in PAIRED_SIDES.items():
        fronts, backs = {}, {}
        for i, r in enumerate(documents):
            number = _document_number(r)
            if len(number) < 6:
                continue
            side = fronts if family(r) == front else backs if family(r) == back else None
            if side is not None:
                side.setdefault(number, []).append(i)
        for number, f in fronts.items():
            b = backs.get(number, [])
            if len(f) == 1 and len(b) == 1:
                documents[f[0]].paired_with = b[0]
                documents[b[0]].paired_with = f[0]


class DocumentsStage:
    """Pipeline methods: finding the documents in a frame, cutting one out, its type and angle,
    and pairing the two sides of a document read from one frame.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    #: Share of the document box's longer side added around it when it is cut from
    #: the frame: the border detector still needs a strip of background to find the
    #: edges, and the box itself can sit a few pixels inside the paper (handoff of the
    #: detector, 2026-10-01: ~3 %).
    DOCUMENT_CROP_MARGIN = 0.03

    def _start_document(self, image_path: Optional[str] = None):
        """Fresh results for one document. Per-document scratch state is reset
        alongside ``self.results`` so nothing leaks across calls."""
        from .options import OCROptionsClass
        self.results = PipelineResults()
        self._mrz_zone = None
        self._date_lines = {}
        self._field_lines = {}
        self._address_line_obboxes = []
        self._doc_year = None
        self._input_frame_shape = None
        self.ocr_options = OCROptionsClass()
        if image_path:
            self.results._meta_results['image_path'] = image_path

    def _find_documents(self, frame: np.ndarray, img_size: int) -> list:
        """Documents in the frame, largest first, boxes on the input image.

        The detector reads the frame at the processing size; its boxes are scaled
        back to the input image so the crop can be cut at full resolution - a
        licence on an A4 scan keeps its pixels instead of the ~300 left to it once
        the whole sheet is shrunk to img_size.
        """
        if self.document_detector is None:
            return []
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []
        ratio = max(max(h, w) / img_size, 1)
        small = cv2.resize(frame, dsize=(max(1, int(w // ratio)), max(1, int(h // ratio))),
                           interpolation=cv2.INTER_LINEAR)
        sx, sy = w / small.shape[1], h / small.shape[0]
        documents = self._model_call(self._document_detector, small)
        for doc in documents:
            for item in [doc] + doc['pages']:
                x1, y1, x2, y2 = item['box']
                item['box'] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        self._emit('documents', [
            {'box': [round(v, 1) for v in d['box']], 'conf': round(d['conf'], 3),
             'pages': [[round(v, 1) for v in p['box']] for p in d['pages']]} for d in documents])
        return documents

    def _document_detector(self, img):
        return self.document_detector.predict(img)[self.document_detector.model_name]['documents']

    def _document_crop(self, frame: np.ndarray, document: dict) -> Optional[tuple]:
        """(x0, y0, x1, y1) on the input image: the document box plus its margin, or None if degenerate."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = document['box']
        m = self.DOCUMENT_CROP_MARGIN * max(x2 - x1, y2 - y1, 0.0)
        x0 = int(max(0, np.floor(x1 - m)))
        y0 = int(max(0, np.floor(y1 - m)))
        x1_i = int(min(w, np.ceil(x2 + m)))
        y1_i = int(min(h, np.ceil(y2 + m)))
        if x1_i <= x0 or y1_i <= y0:
            return None
        return (x0, y0, x1_i, y1_i)

    def _read_document(self, frame, documents, index, img_size, options) -> PipelineResults:
        """Prepare the crop of documents[index] (the whole frame when index is None) and read it."""
        self._input_frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
        crop = self._document_crop(frame, documents[index]) if index is not None else None
        img = self._prepare_image(frame, img_size=img_size, crop=crop)
        self.results._meta_results['Documents'] = documents
        self.results._meta_results['DocumentIndex'] = index
        return self._process_document(img, **options)

    def _doctype_angle(self, img):
        """Classify document type and its angle, and rotate the image upright.

        The module returns the standard {model_name: payload} dict; the
        pipeline-level meta_results keys ('DocType' string, 'Angle90' dict,
        Quality.DocConf) are kept as-is - they are the public PipelineResults
        contract."""
        meta = self.doctype_angles.predict_transform(img)[self.doctype_angles.model_name]
        self.results._meta_results['DocType'] = meta['doc_type']
        self.results._meta_results['Quality']['DocConf'] = meta['doc_type_confidence']
        self.results._meta_results['Angle90'] = {
            'angle': meta['angle'],
            'confidence': meta['angle_confidence'],
            'warped_img': meta['warped_img'],
            'geometry': meta['geometry'],
        }

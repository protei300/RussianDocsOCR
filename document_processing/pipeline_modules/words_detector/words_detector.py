from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np

# Word patches of SMALL text are cut with a 2 px margin instead of on the detector
# box itself. The box is tight around the ink, and a glyph whose stroke touches it
# loses its edge column: АНАСТАСИЯ->МНАСТАСИЯ, АЛЕКСЕЙ->4ЛЕКСЕЙ, УПРАВЛЕНИЯ->
# ПРАВЛЕНИЯ, ОБЛАСТИ->ОБЛАСТ, Г.->2, and «Саха (Якутия)» losing its closing
# bracket (issues #14 and #15).
#
# GATED ON HEIGHT, and the gate is the whole design. The tempting version - always
# pad - is measurably WORSE overall: over samples/ it moves exact matches
# 1456/1583 -> 1435/1583, and driver's licences alone lose 24 fields (DL_2020
# 266->242), because on large text the margin drags in the neighbouring word.
# With the gate: 1458/1583, birth certificates 6/12 -> 8/12, and no doc type
# regresses.
#
# Why small text specifically, measured rather than assumed: it is NOT that the
# certificate labels are tighter. Ink touches the box edge in 44% of certificate
# patches against 72% on DL_2011 - licences are cut tighter and do not suffer.
# The difference is scale. Median word height is 18 px on a certificate against
# 41-50 px on a licence, so one clipped column is a far larger share of a stroke;
# and since the engine resizes patches to height 32, the certificate patch is then
# magnified ~1.8x, the licence patch shrunk. The threshold sits above the
# certificate's 14-18 px range and well below the other types.
WORD_MARGIN_PX = 2
WORD_MARGIN_MAX_HEIGHT = 20


class WordsDetector(BaseModule):
    """Detects and segments words in document text fields.

    Identifies individual words within text fields and
    returns bounding boxes and image patches for each word.

    Useful for cropping words to prepare for OCR.

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool=False, runtime: str = None):
        """Initializes the words detection model."""
        self.model_name = 'WordsDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose, runtime=runtime)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects words, returns bounding boxes.

        Args:
            img: Image containing text field

        Returns:
            List of detected word bounding boxes
        """
        img = self.load_img(img)

        bbox = self.model.predict(img)
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                }
        }
        return meta

    @staticmethod
    def _reading_order(boxes):
        """Sort word boxes into reading order: cluster into lines by vertical
        center proximity (within half a word height), lines top-to-bottom,
        words left-to-right inside a line.

        A plain x-sort interleaves the lines of multi-line fields (measured
        on the birth-certificate Birth_place/ZAGS fields: word salad with
        CER ~0.65); for single-line fields the result is exactly the old
        x-sorted order."""
        lines = []  # each: [mean_cy, mean_h, [boxes...]]
        for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
            cy, h = (box[1] + box[3]) / 2, box[3] - box[1]
            for line in lines:
                if abs(cy - line[0]) < 0.5 * max(h, line[1]):
                    n = len(line[2])
                    line[0] = (line[0] * n + cy) / (n + 1)
                    line[1] = (line[1] * n + h) / (n + 1)
                    line[2].append(box)
                    break
            else:
                lines.append([cy, h, [box]])
        ordered = []
        for _, _, line_boxes in lines:  # already top-to-bottom
            ordered += sorted(line_boxes, key=lambda b: b[0])
        return ordered

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects words and extracts image patches (in reading order).

        Args:
            img: Image containing text field

        Returns:
            Bounding boxes, List of extracted word image patches
        """
        img = self.load_img(img)
        bbox = self.model.predict(img)
        img_patches = []
        bbox = self._reading_order(bbox)
        h, w = img.shape[:2]
        for box in bbox:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            m = WORD_MARGIN_PX if (y2 - y1) < WORD_MARGIN_MAX_HEIGHT else 0
            img_patches.append(img[max(0, y1 - m):min(h, y2 + m),
                                   max(0, x1 - m):min(w, x2 + m)])
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'warped_img': img_patches,
                }
        }
        return meta

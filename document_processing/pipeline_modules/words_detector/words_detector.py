from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np

# NO crop-time margin here, on purpose. Word patches are cut on the detector box
# exactly as predicted.
#
# Until 2026-08-22 this module widened small-text crops by 2 px (1cc8468), because
# the boxes were labelled tight around the ink and a glyph whose stroke touched the
# edge lost its edge column: АНАСТАСИЯ->МНАСТАСИЯ, УПРАВЛЕНИЯ->ПРАВЛЕНИЯ, ОБЛАСТИ->
# ОБЛАСТ, «Саха (Якутия)» losing its bracket (issues #14 and #15). That was a
# compensation for the labelling, not a fix, and it could only ever be approximate:
# the gate fired below 20 px of box height, so a 21 px word missed it by one pixel -
# which is exactly what kept the apostrophe of "OBLAST'" clipped.
#
# The labelling was redone instead (words_yolo11_v3): every box now contains the
# word's ink in full, with the background margin built into the label where it is
# needed. With that model the compensation became measurably idle - over samples/
# the run WITH it and the run WITHOUT it agree field for field on every document
# type (1488/1583 both ways), so it was removed rather than left as a no-op that
# future readers would have to reason about. The ports never had it.


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
            img_patches.append(img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)])
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'warped_img': img_patches,
                }
        }
        return meta

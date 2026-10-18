"""PipelineResults: everything one run of the pipeline produced."""
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..geometry import Chain


class PipelineResults:
    """Stores results and metadata from a model pipeline.

    Attributes:
        _meta_results (dict): Metadata from pipeline stages. Private, and read
            from inside this module only. Callers get ``meta_results`` -- see
            the property below for why the distinction is load-bearing.
        _timings (dict): Timing measurements for stages

    """
    def __init__(self):
        """Initializes empty result storage."""

        self._meta_results = dict(Quality={})
        self._timings = dict()
        # stage keys that ran inside a concurrent group: kept in the report for
        # visibility, but excluded from the 'total' sum (see add_concurrent_group)
        self._concurrent_members = set()
        # index of the other side of this document in Pipeline.process_frame's list
        # (see pair_sides); None - no pair found, or a single-document call
        self.paired_with = None

    @property
    def meta_results(self) -> dict:
        """Metadata from every stage, as a COPY of the state rather than the state.

        Handing out the live dict meant a caller could write into the pipeline's
        own memory -- adding a key or overwriting a quality verdict -- and every
        later reader saw the tampered value as if a model had produced it. That
        was not theoretical: it is what ``tests/test_meta_results_encapsulation``
        demonstrates, red before this property existed.

        Two things this deliberately does NOT do, both worth knowing before
        anyone extends it:

        * **It does not make the pipeline safe to share between threads.**
          ``process_img`` rebinds ``self.results`` wholesale, so a concurrent
          call replaces the object this copy was taken from; copying on read
          protects the contents, not the identity. The service still has to
          serialise calls (``service/ml/runtime.py``, rule 1). Encapsulation and
          concurrency are different problems and fixing one does not touch the
          other.
        * **It does not copy the images.** ``rotated_image``,
          ``img_with_fixed_perspective`` and ``text_fields`` hand out the stored
          arrays live, because callers want the array and duplicating a
          full-size image on every attribute access would cost more than the
          protection is worth. The dictionary structure is protected; the pixel
          payload is shared on purpose, and
          ``TestTheBoundaryWeDoNotProtect`` marks that edge so it stays a
          decision rather than a hole.

        Cost, measured on a driving licence at 1364 ms end to end: one copy is
        4.0 ms over 13.94 MB of arrays, i.e. 0.3 % of a document. Internal code
        never pays it -- everything inside this module reads ``_meta_results``
        directly, and that is the reason those 59 accesses were rewritten rather
        than left to go through here.
        """
        return deepcopy(self._meta_results)

    @property
    def ocr(self) -> Union[Dict, None]:
        """Gets OCR extraction results dict, if available."""
        if self._meta_results.get('OCR'):
            return deepcopy(self._meta_results.get('OCR'))
        else:
            return None

    @property
    def ocr_normalized(self) -> Dict:
        """Canonical ``dd.mm.yyyy`` view of the date fields, alongside the reading.

        A SEPARATE map, deliberately: ``ocr`` holds what is printed on the
        document, which is what the ground truth describes and what the accuracy
        measurement compares against, and it is also the key set the service
        builds its field list and box links from. Writing a canonical value over
        the reading would quietly change both. Only fields that converted appear
        here; a date the converter would have to guess at is simply absent.
        """
        return self._meta_results.get('OCR_normalized') or {}

    @property
    def leasing(self) -> Optional[Dict]:
        """The leasing record read from the STS special marks, or None.

        ``{'leasing': True}`` when the marks say the vehicle is leased; None
        when they do not, or the document is not an STS. Only the flag for
        now (Pipeline.LEASING_REPORTED says why); the lessor and the contract
        are parsed (sts_marks.parse_leasing) but not reported until the reading
        of the marks can carry them. A SEPARATE view, like ocr_normalized:
        ``ocr['Special_marks']`` keeps the text as read.
        """
        return self._meta_results.get('Leasing')

    @property
    def words_fallback(self) -> list:
        """Lines that were read WHOLE because the word split lost most of them.

        Part of the contract, not debug output. A measurement that corrects for
        the missing spaces (a line read whole comes back glued: «Тракторозаводский
        район» -> one token) must apply that correction ONLY to these lines,
        otherwise it becomes a blanket amnesty and hides word-boundary defects
        that have nothing to do with the guard.

        Each entry is ``{'field': str, 'line': int, 'gap': float | None}``,
        where ``gap`` is the widest empty stretch on the line in typical word
        widths and ``None`` means no words were found at all - the line was one
        hole, so the ratio has no denominator.
        """
        return self._meta_results.get('WordsFallback') or []

    @property
    def words_no_ink(self) -> list:
        """Lines the guard declined to re-read because they carry no strokes.

        Kept apart from ``words_fallback`` on purpose: those are lines the guard
        acted on, these are lines it deliberately left alone. Without this, a
        refusal is indistinguishable from the guard never having looked - and
        "the check was never asked" and "the check said no" are different facts.

        Each entry is ``{'field': str, 'line': int, 'ink': float}``.
        """
        return self._meta_results.get('WordsNoInk') or []

    @property
    def validation(self) -> dict:
        """Field checks over the OCR result (docs/validation-checks.md).

        Deliberately a separate accessor rather than a key inside
        ``full_report``: the report is what the service serialises and what the
        conformance goldens of every port are compared against, so putting
        verdicts there would make this a cross-port contract change. Callers
        that want the checks ask for them.

        Empty dict when the document carries nothing checkable.
        """
        from ..validation import validate
        return validate(self.ocr or {})

    @property
    def doctype(self) -> Union[str, None]:
        """Gets detected document type, if available."""
        doctype = self._meta_results.get('DocType')
        return doctype

    @property
    def quality(self) -> dict:
        """Gets image quality measurements, as a copy of the verdicts."""
        return deepcopy(self._meta_results['Quality'])

    @property
    def rotated_image(self) -> np.ndarray:
        """Gets image rotated by the Angle90 stage."""
        return self._meta_results['Angle90']['warped_img']

    @property
    def angle(self):
        """Gets image angle by the Angle90 stage."""
        return self._meta_results['Angle90']['angle']

    @property
    def img_with_fixed_perspective(self) -> Union[list, None]:
        """Get result from doc detection net"""
        if self._meta_results.get('DocDetector'):
            return self._meta_results['DocDetector']['warped_img']
        else:
            return self.rotated_image

    @property
    def text_fields(self) -> Union[Tuple[list, list], None]:
        """Get text field patches with their meta"""
        if self._meta_results.get('TextFieldsDetector'):
            return self._meta_results['TextFieldsDetector']['bbox'], self._meta_results['TextFieldsDetector']['warped_img']
        else:
            return None

    @property
    def text_fields_meta(self) -> Union[Dict, None]:
        """Get text field meta"""
        if self._meta_results.get('TextFieldsDetector'):
            return self._meta_results['TextFieldsDetector']
        else:
            return None

    @property
    def words_patches(self) -> Union[Dict, None]:
        """Get split words patches"""
        if self._meta_results.get('WordsDetector'):
            return self._meta_results['WordsDetector']
        else:
            return None

    @property
    def documents(self) -> list:
        """Every document the detector found in the frame, largest first.

        ``{'box': [x1, y1, x2, y2], 'conf': float, 'pages': [{'box', 'conf'}, ...]}``
        on the image passed to process_img; ``pages`` are the pages of a passport
        spread lying inside the document. Empty when the detector found nothing
        (or is switched off) and the whole frame was read.
        """
        return deepcopy(self._meta_results.get('Documents') or [])

    @property
    def document_index(self) -> Optional[int]:
        """Which entry of ``documents`` this result reads; None - the whole frame."""
        return self._meta_results.get('DocumentIndex')

    @property
    def document_box(self) -> Optional[list]:
        """[x1, y1, x2, y2] of the document read, on the input image; None - the whole frame."""
        index = self.document_index
        documents = self._meta_results.get('Documents') or []
        return list(documents[index]['box']) if index is not None and index < len(documents) else None

    @property
    def geometry(self) -> Union[Chain, None]:
        """Map from the canvas later stages read back to the image passed to process_img.

        None before the image is prepared (see geometry.py).
        """
        return self._meta_results.get('Geometry')

    def to_input(self, points) -> Optional[np.ndarray]:
        """Points of the canvas, (N, 2), on the image passed to process_img.

        None when a stage of this run changed the image in a way no point map
        expresses (see geometry.Unknown) - the canvas and what was read are fine,
        the way back is not known.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return points if self.geometry is None else self.geometry.to_input(points)

    @property
    def field_quads(self) -> Optional[Dict[str, list]]:
        """Where the read text fields lie on the input image (see geometry.py).

        label -> one (4, 2) quadrilateral per detection, in the order the fields
        were read (top to bottom); corners follow the canvas box from its top-left.

        None - the way back to the input image is not known for this run. A caller
        draws these over a photo, so a quadrilateral that silently landed elsewhere
        would be worse than no quadrilateral at all.
        """
        return (self._meta_results.get('Quads') or {}).get('fields', {})

    @property
    def word_quads(self) -> Optional[Dict[str, list]]:
        """label -> quadrilateral of each patch of words_patches[label]['patches'], same order.

        None on the same terms as field_quads.
        """
        return (self._meta_results.get('Quads') or {}).get('words', {})

    @property
    def address_line_quads(self) -> Optional[list]:
        """Registration address lines on the input image, in the order of meta_results['Address_lines'].

        None on the same terms as field_quads.
        """
        return (self._meta_results.get('Quads') or {}).get('address_lines', [])

    @property
    def registration_stamps(self) -> list:
        """Registration stamps of the residence registration page (INTPASSPORTADDR), freshest first.

        Each ``{'kind': 'registered' | 'deregistered' | 'unclear', 'confidence',
        'quad': [[x, y] x4] on the input image (None when the way back is not known),
        'lines': indices into meta_results['Address_lines'], 'address': the stamp's own
        lines joined by a newline, 'cancels': index of the registration a 'deregistered'
        stamp cancels, or None}``. Empty when the page is
        not a registration page or the weight set has no stamp detector.
        """
        return self._meta_results.get('RegistrationStamps', [])

    @property
    def full_report(self) -> dict:
        """Returns full report in dict format"""
        summary_dict = {}
        summary_dict['DocType'] = self.doctype
        summary_dict['OCR'] = self.ocr
        summary_dict['Quality'] = self.quality
        summary_dict['Timings'] = self.timings
        return summary_dict

    @property
    def timings(self) -> dict:
        """Gets per stage timings and total time.

        Stages that ran concurrently (see Pipeline._quality_and_borders_parallel)
        keep their individual wall times in the report, but 'total' counts the
        group's own elapsed time once instead of summing overlapping members.
        Summing them would put 'total' above the real processing time and, worse,
        would keep it flat when parallelisation actually saves time.

        'total' covers timed stages only - image loading/resizing (_prepare_image)
        is not a timed stage, so wall-clock time is slightly higher.
        """
        timings = self._timings.copy()
        total_time = sum(v for k, v in timings.items() if k not in self._concurrent_members)
        timings['total'] = round(total_time, 4)
        return timings

    @timings.setter
    def timings(self, value):
        """Sets updated timings."""
        self._timings = self._timings | value

    def add_concurrent_group(self, name: str, wall_time: float, members: dict):
        """Records a set of stages that ran concurrently with each other.

        Args:
            name: key for the group's own elapsed time (this is what counts
                towards 'total').
            wall_time: elapsed time of the whole group.
            members: per-stage times inside the group; reported as usual but
                not summed into 'total'.
        """
        self._timings[name] = round(wall_time, 4)
        self._timings = self._timings | members
        self._concurrent_members |= set(members)

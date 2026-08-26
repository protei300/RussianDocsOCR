import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Union, Dict, Tuple

import cv2
import numpy as np

from ..pipeline_modules import *
from .dates import canonical_dates


def _segments_payload(meta_results: dict):
    """Normalise DocDetector's contours into plain ``[[x, y], ...]`` lists.

    Only for the conformance probe (``borders.segments``). The stored contours are
    numpy arrays of shape ``(N, 1, 2)`` -- the layout ``cv2.findContours`` returns --
    which no other language reproduces, so a raw dump could not be compared. The
    reshape is O(number of points), a few hundred per document, and it is what makes
    the stage language-neutral rather than a convenience.

    Returns ``None`` when border detection did not run, which the checker treats as a
    stage that legitimately did not happen.
    """
    detector = (meta_results or {}).get('DocDetector') or {}
    segments = detector.get('segm')
    if not segments:
        return None
    out = []
    for contour in segments:
        if contour is None:
            out.append([])
            continue
        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        out.append([[float(x), float(y)] for x, y in pts])
    return out


def _resolve_device(device):
    """Resolve the inference device.

    ``None`` -> auto: 'gpu' when onnxruntime reports a CUDA provider, else 'cpu'.
    An explicit 'cpu'/'gpu' is returned unchanged.
    """
    if device is not None:
        return device
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            return 'gpu'
    except Exception:
        pass
    return 'cpu'


@dataclass(init=False)
class OCROptionsClass:
    """Class for storing OCR options for different document types.

    Holds common OCR options like fields needed, split preferences, etc.
    Sub-classes implement options specific to document types.
    """

    """list: Fields that need to be split for this doc type."""
    needed_split = []

    """list: Fields read by the LATIN engine for this doc type."""
    en_fields = []

    """list: Fields read by the CYRILLIC engine for this doc type.

    The two lists select an ENGINE, not a language, and the distinction became
    load-bearing with issue #12: a passport series/number is pure digits, both
    engines carry the digit classes, and they do not agree on them. Membership
    here is decided by measurement, not by what language the field is written
    in - see OCROptionsINTPassport.
    """
    ru_fields = []

    """bool: Whether this doc type needs license number rotation."""
    needs_licence_rotation = False

    """bool: Whether this doc type has a multi-line registration address to OCR."""
    has_address = False


    @classmethod
    def make_options(cls, doc_type):
        """Factory method to make OCR options for a document type.

        Args:
            doc_type (str): Document type string

        Returns:
            OCROptionsClass instance with options for the document type.
        """
        # NOTE: 'intpassportaddr' contains 'intpassport', so it MUST be checked first.
        if 'intpassportaddr' in doc_type.lower():
            return OCROptionsINTPASSPORTADDR()
        elif 'intpassport' in doc_type.lower():
            return OCROptionsINTPassport()
        elif 'extpassport' in doc_type.lower():
            return OCROptionsEXTPassport()
        elif 'dl' in doc_type.lower():
            return OCROptionsDL()
        elif 'snils' in doc_type.lower():
            return OCROptionsSNILS()
        elif 'birthcert' in doc_type.lower():
            return OCROptionsBIRTHCERT()
        # Unknown type: return the empty base options instead of None, so the
        # caller's `.needs_licence_rotation`/`.ru_fields` access doesn't crash
        # (the pipeline then simply produces no OCR fields for it).
        return OCROptionsClass()

class OCROptionsINTPassport(OCROptionsClass):
    """OCR options for internal Russian passports."""

    needed_split = ["Licence_number",
                    "Birth_place_ru", "Issue_organization_ru",
                    ]

    # MRZ is deliberately NOT in needed_split: it is detected one box per line
    # and each line must reach the OCR engine whole - splitting it into "words"
    # at the filler runs would destroy the fixed 44-character layout the check
    # digits are computed over.
    en_fields = ["Issue_date", "Expiration_date", "Birth_date",
                 "Issue_organisation_code", "MRZ", ]
    # Licence_number is CYRILLIC-routed, and that is not a typo: the series and
    # number are digits only, but the Latin engine reads the passport's red '3'
    # as '8' - confidently, at p=0.94..1.00 with '3' as the runner-up at 0.004,
    # so no threshold, alphabet mask or upscaling can recover it (issue #12).
    # The Cyrillic engine reads the SAME crops correctly. Measured over
    # samples/: Latin 17/24 exact, Cyrillic 24/24; across every doc type,
    # Latin 103/112, Cyrillic 111/112, and Cyrillic is never worse except on
    # BIRTHCERT, where NEITHER engine reads the Roman-numeral series correctly.
    # That type is not touched here: it was already Cyrillic-routed, as the
    # lesser evil - see OCROptionsBIRTHCERT.
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru", "Licence_number"]
    needs_licence_rotation = True

class OCROptionsINTPASSPORTADDR(OCROptionsClass):
    """OCR options for the registration ('place of residence') page of an
    internal Russian passport. Only the multi-line registration address is
    recognized (printed text), via the Address_line detections produced by
    the TextFields detector."""

    needed_split = []
    en_fields = []
    ru_fields = []
    has_address = True
    needs_licence_rotation = False

class OCROptionsBIRTHCERT(OCROptionsClass):
    """OCR options for RF birth certificates (BIRTHCERT_1998, BIRTHCERT_2018).

    Field names follow the shared TextFields class vocabulary: the child's
    name maps onto Last_name_ru/First_name_ru, the ZAGS office onto
    Issue_organization_ru, the series/number line onto Licence_number, and
    the parents get their own *_ru classes (added with the 2026-08 detector
    retrain). Dates on this form spell the month out in Cyrillic
    («16 декабря 2001»), so they go through the Cyrillic engine.
    Licence_number mixes a Roman-numeral series with Cyrillic letters and
    «№» — routed to the Cyrillic engine as the lesser evil (the Roman-digit
    series is the expected CER cost; revisit if the eval says otherwise).

    Shared by both blank generations: BIRTHCERT_1998 (digit birth date
    DD.MM.YYYY) and BIRTHCERT_2018 (order 167/2018: worded birth date, parent
    birth dates, place of issue, 21-digit act number). The options cannot see
    the era (make_options gets the type without the year suffix), so
    Birth_date is routed to the Cyrillic engine for BOTH: it must read the
    worded 2018 form, and the Cyrillic engine reads digit-only crops fine —
    the same precedent as the passport Licence_number (issue #12)."""

    needed_split = ["First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                    "Issue_date", "Licence_number",
                    "Father_first_middle_ru", "Mother_first_middle_ru",
                    "Birth_date", "Father_birth_date", "Mother_birth_date",
                    "Issue_place_ru"]
    en_fields = []
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru",
                 "Issue_organization_ru", "Issue_date", "Licence_number",
                 "Father_last_name_ru", "Father_first_middle_ru",
                 "Mother_last_name_ru", "Mother_first_middle_ru",
                 "Birth_date", "Father_birth_date", "Mother_birth_date",
                 "Issue_place_ru", "Act_number"]


class OCROptionsEXTPassport(OCROptionsClass):
    """OCR options for external Russian passports."""

    needed_split = ["Licence_number", "Birth_place_ru", "Birth_place_en", ]

    en_fields = ["Last_name_en", "First_name_en", "Issue_date",
                 "Expiration_date", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en", "Sex_en",
                 "Issue_organisation_code", "Middle_name_en", "MRZ"]
    # Licence_number: Cyrillic-routed for the reason given on OCROptionsINTPassport.
    # Smaller effect here (Latin 41/42 exact, Cyrillic 42/42 over samples/) because
    # the number is printed larger, but it is the same digit confusion.
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru", "Licence_number"]


class OCROptionsDL(OCROptionsClass):
    """OCR options for Russian driver's licenses."""

    needed_split = ["Licence_number", "Driver_class", "Birth_place_ru", "Birth_place_en",
                    "Living_region_ru", "Living_region_en", ]
    en_fields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                 "Expiration_date", "Driver_class", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en",  "Issue_organisation_code", "Middle_name_en"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", ]

class OCROptionsSNILS(OCROptionsClass):
    """OCR options for Russian SNILS documents."""

    needed_split = ["Last_name_ru", "First_name_ru", "Licence_number", "Issue_date",
                    "Birth_date", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]
    en_fields = ["Licence_number", "Issue_date", "Birth_date"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]

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




class Pipeline:
    """Pipeline for OCR processing of documents.

    Performs steps of pre-processing, text detection and OCR
    to extract text from documents.
    """

    def __init__(self, model_format='ONNX', device=None, ocr='accurate', verbose=False,
                 ocr_gpu_batch=False):
        """
        Initialize pipeline.

        Args:
            model_format (str): Format of models to use - ONNX, OpenVINO etc.
            device (str): Device for model inference - 'cpu' or 'gpu'. When None
                (default) it is auto-detected: 'gpu' if a CUDA provider is
                available, otherwise 'cpu'. Applies to the detector models;
                see `ocr_gpu_batch` for how it affects the OCR engines.
            ocr (str): OCR engine selection.
                'accurate' (default) - v2 MobileNetV4 models (best quality);
                'fast'               - v2 EdgeNext models (faster, slightly lower).
            verbose (bool): Whether to print debug information.
            ocr_gpu_batch (bool): EXPERIMENTAL, default False. The OCR engines
                are dynamic-width models; on a CUDA provider, running them one
                word at a time (as the pipeline naturally produces them) forces
                a graph recompile per distinct width and is measured 400-3700x
                SLOWER than CPU - not a viable combination. So by default, when
                device=='gpu', the OCR engines run on CPU regardless of
                `device` (detectors still use GPU) - this default combination is
                bit-exact and, on real documents, measured *faster* overall
                than pure CPU (GPU detectors + CPU OCR).
                Passing ocr_gpu_batch=True instead batches a document's words
                into a small number of fixed-shape padded batches per engine
                call (see pipeline_modules/ocr_batch.py for why the shapes
                must be few and fixed, not dynamic). After a brief per-process
                warmup (the first few *different* documents - cold calls can
                take seconds), this reaches ~100-250ms per engine call, faster
                than the CPU-only default - but at a measured, non-zero
                accuracy cost from a width-wise global-pooling component in
                the v2 backbones (Squeeze-and-Excitation-like): on 14 real
                mixed documents, 5.4% of OCR fields differ from the CPU-exact
                baseline for 'accurate', 14.1% for 'fast'. Only enable after
                validating this tradeoff on your own documents/traffic.
        """
        device = _resolve_device(device)
        self.device = device
        self.model_format = model_format
        if ocr == 'legacy':
            raise ValueError(
                "ocr='legacy' was removed: the original rus/eng+nums models and the "
                "CTC beam-search/regex path are gone. Use 'accurate' (default) or 'fast'."
            )
        if ocr not in ('accurate', 'fast'):
            raise ValueError("ocr must be one of 'accurate', 'fast'")
        self.ocr_mode = ocr
        self.ocr_gpu_batch = bool(ocr_gpu_batch)

        # Viable device for the OCR engines specifically (see ocr_gpu_batch
        # docstring above): on GPU they are only viable batched (opt-in), so
        # they fall back to CPU here unless the user explicitly asked for the
        # batched GPU path. Detectors below always use the requested `device`.
        if device == 'gpu' and not self.ocr_gpu_batch:
            self.ocr_device = 'cpu'
        else:
            self.ocr_device = device

        # `model_format` is the public knob and answers two separate questions:
        # which artifact to load, and which runtime to run it on. Every model
        # ships exactly one artifact (.onnx) - OpenVINO reads it directly - so
        # 'OpenVINO' selects a runtime, not a different set of weights. Keeping
        # these apart is what stops a per-format config from drifting out of
        # sync with the deployed model (that bug hit Borders, TextFields and
        # DocTypeAngles at once; see docs/progress-log.md).
        if model_format == 'OpenVINO':
            artifact, runtime = 'ONNX', 'OpenVINO'
        else:
            artifact, runtime = model_format, None

        self.doctype_angles = DocTypeAngles(model_format=artifact, device=device, verbose=verbose,
                                            runtime=runtime)
        self.doc_detector = DocDetector(model_format=artifact, device=device, verbose=verbose,
                                        runtime=runtime)
        self.text_fields = TextFieldsDetector(model_format=artifact, device=device, verbose=verbose,
                                              runtime=runtime)
        # The OCR family never compiled under OpenVINO, so it stays on
        # onnxruntime regardless of the requested runtime.
        ocr_format = artifact
        # oriented (rotated) address-line detector for the INTPASSPORTADDR page
        self.address_lines = AddressLinesDetector(model_format=ocr_format, device=device, verbose=verbose)
        # printed-vs-handwritten classifier for address lines (OCR is printed-only)
        self.address_textkind = AddressTextKindClassifier(model_format=ocr_format, device=device, verbose=verbose)
        self.words_detector = WordsDetector(model_format=artifact, device=device, verbose=verbose,
                                            runtime=runtime)

        # OCR engines - the Cyrillic and Latin engines used by _ocr/_route_word_ocr.
        # Note: these load on self.ocr_device, not necessarily `device` (see above).
        self.ocr_cyr = OCRCyrillic(tier=ocr, model_format=ocr_format, device=self.ocr_device,
                                   verbose=verbose)
        self.ocr_lat = OCRLatin(tier=ocr, model_format=ocr_format, device=self.ocr_device,
                                verbose=verbose)
        # NOTE: pipeline_modules/ocr_batch.py::warmup_ladder() exists to
        # pre-warm every (width, count) shape the batched path could ever
        # produce, but is NOT called automatically here - measured to cost
        # 15-90+ seconds (large batch x width combos take up to ~7s EACH)
        # and, even after paying that cost, real documents were still
        # observed slow on already-warmed shapes. Left as an opt-in
        # experiment for callers who want to try it themselves; see its
        # docstring and docs/progress-log.md before using it.

        self.lcd_spoofing = LCDSpoofing(model_format=artifact, device=device, verbose=verbose,
                                        runtime=runtime)
        self.print_spoofing = PrintSpoofing(model_format=artifact, device=device, verbose=verbose,
                                            runtime=runtime)
        self.glare = Glare(model_format=artifact, device=device, verbose=verbose, runtime=runtime)
        self.blur = Blur(model_format=artifact, device=device, verbose=verbose, runtime=runtime)
        # residual-tilt correction after perspective fix (projection-profile based).
        # min_angle=2.0: skip small/noisy estimates (handwriting has irregular
        # baselines that yield spurious ~1-2deg) and only fix real tilts.
        self.deskewer = DocDeskewer(angle_range=10.0, angle_steps=101, min_angle=2.0, scale=0.4)
        self.ocr_options = OCROptionsClass

        # Optional per-stage instrumentation, off by default. Used only by the
        # cross-language conformance harness (conformance/) to localise where a
        # port of this library first diverges. See pipeline/probe.py; when None
        # every emission site costs a single attribute test.
        self.probe = None

    def _emit(self, name: str, payload) -> None:
        """Hand one intermediate value to the probe, if one is attached.

        Payloads are passed by REFERENCE and must already exist — never compute
        anything for the probe's benefit, or the instrumentation would change the
        cost of the code it is instrumenting. A sink that keeps a payload is
        responsible for copying it.
        """
        if self.probe is not None:
            self.probe.emit(name, payload)

    def warmup(self, img_path: Union[Path, str, np.ndarray] = None) -> None:
        """Explicitly warm up the pipeline before serving real traffic/benchmarks.

        Each model's ONNX session is already warmed once at construction time
        (see ``ModelInference.__load_onnx``) with a synthetic tensor at its
        declared input shape - this primes CUDA graph/kernel selection so the
        first call on GPU isn't the one paying that cost (verified: the first
        ``process_img()`` call right after ``Pipeline()`` construction is not
        meaningfully slower than later ones).

        This method additionally exercises the FULL call chain end-to-end
        (Python-level pre/postprocessing, OpenCV, deskew, word-splitting, OCR
        routing) on a real image, which the per-model synthetic warmup above
        cannot reach on its own. Call this explicitly before starting a timed
        benchmark or a serving loop, instead of relying on the first few
        documents of real traffic to happen to warm things up (or - as this
        codebase's own benchmark scripts used to do inconsistently - an ad-hoc
        "process a couple of documents and discard the result" loop before
        every timed run).

        Args:
            img_path: A real document image to run through ``process_img()``
                once (result discarded). If omitted, only the doctype/quality
                stages get exercised - a synthetic non-document image
                classifies as ``'NONE'`` and short-circuits before the
                border/field/OCR stages, so there is no synthetic default
                that warms everything; pass a real sample for a full warmup.
        """
        if img_path is None:
            img_path = np.full((900, 1400, 3), 200, dtype=np.uint8)
        try:
            self.process_img(img_path)
        except Exception as e:
            print(f"[!] Pipeline.warmup() encountered an error (non-fatal): {e!r}")

    def process_img(
            self,
            img_path: Union[Path, str, np.ndarray],
            ocr=True,
            get_doc_borders=True,
            find_text_fields=True,
            check_quality=True,
            low_quality=True,
            docconf=0.5,
            img_size=1500,

        ) -> PipelineResults:
        """
        Main pipeline processing method.

        Args:
            img_path: Path to input image.
            ocr: Whether to perform OCR.
            get_doc_borders: Whether to detect document borders.
            find_text_fields: Whether to detect text fields.
            check_quality: Whether to check image quality.
            low_quality: Whether to process low quality images.
            docconf: Minimum doc confidence threshold.
            img_size: Resize image to this size for processing.

        Returns:
            PipelineResults with extracted information.
        """

        self.results = PipelineResults()
        # Per-image state for the MRZ length self-check (see _note_mrz_zone). Reset here
        # for the same reason self.results is: process_img may be called again.
        self._mrz_zone = None

        img = self._prepare_image(img_path, img_size=img_size)
        self._emit('prepare', img)

        # unified doctype + angle classification, then rotate upright
        self._model_call(self._doctype_angle, img)
        img = self.results.rotated_image
        # Assembled from the three places the pipeline actually stores this, not read
        # from a single key: _doctype_angle spreads the module's payload across
        # meta_results['DocType'], meta_results['Quality']['DocConf'] and
        # meta_results['Angle90'] (see its docstring - those key names are the public
        # PipelineResults contract). An earlier version of this emission looked for a
        # non-existent 'DocTypeAngles' key and silently emitted None; the golden
        # recorded null, and null == null passed. Caught when the Go port produced a
        # real value and had nothing to be compared against.
        _angle90 = self.results._meta_results.get('Angle90') or {}
        self._emit('doctype.label', {
            'doc_type': self.results._meta_results.get('DocType'),
            'doc_type_confidence': self.results._meta_results.get('Quality', {}).get('DocConf'),
            'angle': _angle90.get('angle'),
            'angle_confidence': _angle90.get('confidence'),
        })
        self._emit('rotate', img)

        doc_type = self.results.doctype
        if doc_type == 'NONE' and get_doc_borders:
            # Borders-first fallback: a hard but legitimate shot (strong
            # perspective, small document scale, occluded header) can push the
            # raw-frame embedding past the metric NONE-threshold while the
            # same classifier reads the border-cropped document confidently.
            # Costs one extra DocDetector run on NONE frames only. Measured
            # (docs/progress-log.md, 2026-08-02): recovered 6/6 synthetic
            # NONE cases at conf 0.98+, zero new false-accepts on a
            # no-document negative set.
            self._model_call(self._doc_detector, img)
            if self.results._meta_results['DocDetector']['segm']:
                self._model_call(self._doctype_angle,
                                 self.results.img_with_fixed_perspective)
                doc_type = self.results.doctype
                if 'intpassport' in doc_type.lower():
                    # The fallback crop above ran while the type was still
                    # unknown (max_pages=1), so a two-page passport spread
                    # lost its second page before OCR could see it. Now that
                    # the type is known, redo border detection on the same
                    # pre-crop frame: _doc_detector reads the recovered type
                    # and allows the two-page stitch. Type and angle stay as
                    # classified on the single page (measured reliable there);
                    # only the canvas is rebuilt.
                    angle = self.results.angle
                    self._model_call(self._doc_detector, img)
                    img = self.results.img_with_fixed_perspective
                    for _ in range(angle // 90):
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    img = self.results.rotated_image
        if doc_type == 'NONE':
            print("[!] The document on picture has unknown type")
            return self.results
        # labels are '<TYPE>_<YEAR>'; tolerate a missing year suffix
        parts = doc_type.rsplit('_', maxsplit=1)
        doc_type, year = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], None)
        self.ocr_options = OCROptionsClass.make_options(doc_type)

        # Quality checks (Glare/Blur/PrintSpoofing/LCDSpoofing) and border
        # detection all take the same rotated image and are mutually
        # independent, so when low_quality=True (default) - meaning the
        # quality verdict never gates whether border detection runs - they
        # run concurrently (see _quality_and_borders_parallel). When
        # low_quality=False, the quality verdict decides whether to run
        # (skip) the heavier border detector at all, so that early-exit
        # optimization requires the original sequential order.
        if check_quality and get_doc_borders and low_quality:
            self._quality_and_borders_parallel(img)
            img = self.results.img_with_fixed_perspective
            self._emit('quality', self.results.quality)
            self._emit('borders.segments', _segments_payload(self.results._meta_results))
            self._emit('borders.canvas', img)
            self._model_call(self._deskew, img)
            img = self.results.img_with_fixed_perspective
            self._emit('deskew.canvas', img)
        else:
            #getting quality
            if check_quality:
                self._model_call(self._glare, img)
                self._model_call(self._blur, img)
                self._model_call(self._print_spoofing, img)
                self._model_call(self._lcd_spoofing, img)

            # checking quality of doc
            if not low_quality:
                quality = self.results.quality
                # DocConf is a confidence (1.0 = exact centroid hit, <=0 = past the
                # metric threshold), so the gate is a MINIMUM: reject below docconf.
                if quality.get('Glare', False) == 'bad' or quality.get('Blur', False) == 'bad' or quality['DocConf'] < docconf:
                    print("[!] Doc quality is too low. You can check using results.quality, "
                          "or bypass using low_quality=True")
                    return self.results


            self._emit('quality', self.results.quality)

            #detecting doc
            if get_doc_borders:
                self._model_call(self._doc_detector, img)
                img = self.results.img_with_fixed_perspective
                self._emit('borders.segments', _segments_payload(self.results._meta_results))
                self._emit('borders.canvas', img)
                # correct residual tilt so text lines are horizontal (helps field
                # detection, line/word splitting and OCR; train==inference)
                self._model_call(self._deskew, img)
                img = self.results.img_with_fixed_perspective
                self._emit('deskew.canvas', img)

        # detecting fields
        if find_text_fields:
            #Intpassport has licence number rotated 90 deg
            rotate_licence = self.ocr_options.needs_licence_rotation
            self._model_call(self._fields_detector, img, rotate_licence=rotate_licence)
            text_fields = self.results.text_fields_meta
            self._emit('fields.bbox', text_fields.get('bbox') if text_fields else None)
        else:
            return self.results

        # registration address (INTPASSPORTADDR): detect address lines, order by Y,
        # split each line into words and OCR (printed text)
        if ocr and getattr(self.ocr_options, 'has_address', False):
            self._model_call(self._address_lines, img)
            if not self.results._meta_results.get('Address_lines'):
                # Paper-texture gate: the registration page is mostly bare
                # paper, so the metric classifier accepts document-free paper
                # textures as INTPASSPORTADDR (measured 2026-08-02, see
                # docs/progress-log.md). A real registration page virtually
                # always has detectable address lines (49/50 on the test set);
                # zero lines on an accepted "address page" marks a false
                # accept, so reject it the same way the doctype stage does.
                print("[!] The document on picture has unknown type")
                self.results._meta_results['DocType'] = 'NONE'
                self.results._meta_results['Quality']['DocConf'] = 0.0
                return self.results
            self._emit('address.lines', self.results._meta_results.get('Address_lines'))

        #splitting words
        if text_fields:
            self._model_call(self._split_words, text_fields.copy(), doc_type)
            words_splitted = self.results.words_patches

            #OCR words
            if ocr and words_splitted:
                self._model_call(self._ocr, words_splitted, doc_type)
                self._normalize_dates()

        return self.results


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
        }



    def _glare(self, img):
        """Check for glare quality"""
        qual, coef = self.glare.predict(img)[self.glare.model_name]
        self.results._meta_results['Quality']['Glare'] = qual
        return qual

    def _blur(self, img):
        """Check for blur quality."""
        qual, coef = self.blur.predict(img)[self.blur.model_name]
        self.results._meta_results['Quality']['Blur'] = qual
        return qual

    def _print_spoofing(self, img):
        """Check for print spoofing."""
        qual, coef = self.print_spoofing.predict(img)[self.print_spoofing.model_name]
        self.results._meta_results['Quality']['PrintSpoofing'] = qual
        return qual

    def _lcd_spoofing(self, img):
        """Check for LCD spoofing."""
        qual, coef = self.lcd_spoofing.predict(img)[self.lcd_spoofing.model_name]
        self.results._meta_results['Quality']['LCDSpoofing'] = qual
        return qual

    def _quality_and_borders_parallel(self, img):
        """Run the 4 quality checks and border detection concurrently.

        All five take the same (rotated) image and are mutually independent
        (different models/sessions, no shared mutable state in the
        preprocessing/inference/postprocessing layers - verified by code
        review, see docs/progress-log.md). Only used when low_quality=True:
        with low_quality=False the quality verdict must be known BEFORE
        deciding whether to run the heavier border detector at all, which
        needs the original sequential order (see process_img).

        Every `self.results` write happens here, on the main thread, after
        gathering all futures - concurrent writes into the shared
        meta_results dict (particularly _doc_detector's top-level `|` rebind)
        are not safe to do from worker threads.
        """
        doc_type = (self.results.doctype or '')
        is_spread = 'intpassport' in doc_type.lower()
        max_pages = 2 if is_spread else 1

        def timed(fn, *args, **kwargs):
            t0 = time()
            r = fn(*args, **kwargs)
            return r, round(time() - t0, 4)

        group_start = time()
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_glare = ex.submit(timed, self.glare.predict, img)
            f_blur = ex.submit(timed, self.blur.predict, img)
            f_print = ex.submit(timed, self.print_spoofing.predict, img)
            f_lcd = ex.submit(timed, self.lcd_spoofing.predict, img)
            f_det = ex.submit(timed, self.doc_detector.predict_transform, img,
                              stack='auto', max_pages=max_pages)

            glare_res, t_glare = f_glare.result()
            blur_res, t_blur = f_blur.result()
            print_res, t_print = f_print.result()
            lcd_res, t_lcd = f_lcd.result()
            det_res, t_det = f_det.result()
        group_elapsed = time() - group_start

        self.results._meta_results['Quality']['Glare'] = glare_res[self.glare.model_name][0]
        self.results._meta_results['Quality']['Blur'] = blur_res[self.blur.model_name][0]
        self.results._meta_results['Quality']['PrintSpoofing'] = print_res[self.print_spoofing.model_name][0]
        self.results._meta_results['Quality']['LCDSpoofing'] = lcd_res[self.lcd_spoofing.model_name][0]
        self.results._meta_results = self.results._meta_results | det_res

        # these five overlap in time, so only the group's own elapsed time may
        # count towards timings['total'] - the per-stage numbers stay for detail
        self.results.add_concurrent_group(
            '_quality_and_borders', group_elapsed,
            {
                '_glare': t_glare, '_blur': t_blur, '_print_spoofing': t_print,
                '_lcd_spoofing': t_lcd, '_doc_detector': t_det,
            },
        )

    def _doc_detector(self, img):
        """
        Detect document borders and fix perspective.

        Args:
            img: Input image

        Returns:
            np.ndarray: Image with fixed perspective
        """
        # Only internal passports are photographed as a two-page spread; all
        # other doc types are single cards/pages, so keep just the largest
        # segment (prevents background blobs from being stitched in). The
        # stitch direction is chosen automatically from the pages' layout.
        doc_type = (self.results.doctype or '')
        is_spread = 'intpassport' in doc_type.lower()
        max_pages = 2 if is_spread else 1
        result = self.doc_detector.predict_transform(img, stack='auto', max_pages=max_pages)
        self.results._meta_results = self.results._meta_results | result
        # img = result[self.doc_detector.model_name]['warped_img']
        # return img

    def _deskew(self, img):
        """Correct residual tilt of the perspective-fixed canvas and store it
        back so img_with_fixed_perspective returns the deskewed image."""
        desk = self.deskewer.deskew(img)
        if self.results._meta_results.get('DocDetector'):
            self.results._meta_results['DocDetector']['warped_img'] = desk
        else:
            # no doc_detector result (fallback path) -> stash under Angle90
            self.results._meta_results.setdefault('Angle90', {})['warped_img'] = desk
        return desk

    def _fields_detector(self, img, rotate_licence=False):
        """
        Detect text fields in document.

        Args:
            img: Input image
            rotate_licence: Whether to rotate license field

        Returns:
            dict: Detected text fields and patches
        """
        result = self.text_fields.predict_transform(img)
        text_fields = result[self.text_fields.model_name]

        self._note_mrz_zone(text_fields, img)

        if rotate_licence:
            for i, field in enumerate(text_fields['bbox']):
                if field[-1] == 'Licence_number':
                    text_fields['warped_img'][i] = cv2.rotate(text_fields['warped_img'][i],
                                                              cv2.ROTATE_90_COUNTERCLOCKWISE)


        self.results._meta_results = self.results._meta_results | result


    #: An empty stretch on a line wider than this many typical words means the
    #: split dropped a word, and the line is read whole instead.
    #:
    #: Measured over 157 documents (all of samples/ plus 40 generator canons,
    #: where the defect lives), 815 fields: the widest gap on a line that really
    #: lost a long word has a median of 2.74 typical word widths, against 0.06 on
    #: intact lines - a factor of forty. The obvious alternative, the SHARE of the
    #: line covered by word boxes, was measured on the same run and rejected: it
    #: reads print DENSITY, not the integrity of the split, so the lowest coverage
    #: over samples/ belongs to a correctly read internal-passport Licence_number
    #: (0.64-0.75, three digit groups with wide gaps) and no threshold separates
    #: it from real damage. By the gap the same 24 fields sit at a median of 0.92
    #: and a maximum of 1.20, i.e. silent with room to spare.
    #:
    #: 3.0 rather than the 3.056 the selection rule produced: four significant
    #: digits pretend to a precision that 19 target fields cannot support. The
    #: rounding costs one extra false fire on the canons and none on samples/.
    #: Chosen on one half of the canons and reported on the other, so the number
    #: is not fitted to what it is measured on: 3 of 7 and 5 of 10 caught on the
    #: held-out half, which held the catch rate but raised the false count from 1
    #: to 4 - so the cost side of this number is an upper bound, not an
    #: expectation. Full report: notes/report_words_signal_choice_2026-08-25.md.
    #:
    #: Known limit, not a threshold to tune: when the missing word left NO gap
    #: because a neighbour's box swallowed it, geometry cannot see it at all.
    #: That is 8 of 19 measured cases - this guard covers about half of the
    #: complaint, the rest belongs to retraining the word detector.
    WORDS_MAX_GAP = 3.0

    #: Below this much fine-grained detail, a line carries no strokes and the
    #: fallback above must NOT re-read it.
    #:
    #: The guard treats "no word boxes" as "the split lost the text", but that
    #: observation has a second cause: there IS no text. On a real web sample
    #: (f2018_02) the name line is a blurred strip - anonymised at the source,
    #: no glyphs at all - and re-reading it whole produced «ЛВН»: an honest empty
    #: field turned into something that looks like an answer. For an identifier a
    #: plausible wrong value is worse than none.
    #:
    #: The measure is the variance of the Laplacian, i.e. how much fine detail
    #: the strip carries - strokes, not darkness. Three cheaper-looking
    #: alternatives were measured on the same run and rejected because they
    #: overlap: spread of brightness gives 18.8 on the blurred strip against a
    #: minimum of 18.1 on strips that do carry text, and the share of dark pixels
    #: 0.074 against 0.082. They measure how DARK a strip is; the question is
    #: whether it has strokes.
    #:
    #: The threshold comes from a blur ladder over 612 canon lines whose text is
    #: known to be there, blurred with a growing Gaussian (manufactured on
    #: purpose, so the number does not rest on the single real case): sharp text
    #: has a median of 1590, mild blur 249, and blur WIDER THAN THE STROKE 27 -
    #: an order-of-magnitude trough, because a stroke one or two pixels wide
    #: smeared over five stops being a stroke. 100 sits in that trough: it drops
    #: 0% of sharp lines, 1% of mildly blurred ones and all of the smeared ones,
    #: where 150 would cost 10% of the mildly blurred for nothing extra.
    #: Physics gives the trough; the point inside it is chosen by the cost of
    #: being wrong, which is asymmetric - refusing wrongly loses a repair and
    #: restores the pre-guard behaviour, accepting wrongly MANUFACTURES a value.
    #:
    #: Those percentages are measured on MANUFACTURED blur, which is easier than
    #: the real thing: a Gaussian is even, while a real blurred strip still
    #: carries compression, paper texture and sensor noise. On real documents
    #: this will therefore reject LESS than the ladder promises - it errs towards
    #: letting a blurred line through rather than dropping a line with text.
    #: Known limit, named before the measurement: an empty strip and a strip with
    #: VERY FAINT text look the same to this measure. Full report:
    #: notes/proposal_ink_check_2026-08-25.md.
    LINE_MIN_INK = 100.0

    #: A line of a machine-readable zone is exactly 44 characters, and that is a
    #: rare luxury: the pipeline can tell that it read the line WRONG without being
    #: told, and try again. Anything shorter means the crop lost part of the line.
    MRZ_LINE_LEN = 44

    #: The MRZ is printed as ONE rectangle holding two lines, so both lines share
    #: the same horizontal span - but the detector does not know that. Measured
    #: over samples/: the two boxes of a zone start within 10 px of each other
    #: when the zone reads correctly and 90-182 px apart when it does not, and the
    #: characters outside the narrower box never reach the engine. That accounted
    #: for 23 of the 34 damaged lines; the engine was innocent (its CTC output had
    #: 117-552 time steps against the 57-71 a line needs).
    #:
    #: So the zone's own span is the FIRST retry candidate, then the ladder widens
    #: further. Deliberately a candidate and not a rewrite: forcing every MRZ box
    #: to the union span fixed the external passports and damaged an internal one
    #: (both its lines got shorter), because a wider crop can also pull in the page
    #: edge. Reading the detector's own crop first and widening only on a wrong
    #: length cannot lose a line that was already right.
    #: The ladder reaches 34% of the span on each side because that is what the
    #: worst measured case needed: a zone whose BOTH boxes are inset by nine
    #: characters (8_CR of EXTPASSPORTBIO) recovers its first line at +0.30 and not
    #: before. Wider steps cost nothing on a healthy document - the ladder is only
    #: walked for a line whose length is already wrong - and the crop is clamped to
    #: the canvas, so the last steps saturate instead of running away.
    MRZ_RETRY_GROWTH = (0.0, 0.05, 0.10, 0.16, 0.24, 0.34)

    #: The zone's alphabet is closed: capitals, digits and the filler. A line
    #: cannot begin or end with anything else, so a stray '.' or '_' at an edge is
    #: the page border caught by the crop, not text. Trimming those - and ONLY at
    #: the edges - is what keeps a widened crop from turning a correct
    #: 44-character line into a 45-character one.
    MRZ_ALPHABET = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')

    def _note_mrz_zone(self, text_fields: dict, img):
        """Remember the canvas and the MRZ boxes for the length self-check.

        Nothing is modified here: the boxes the detector produced stay exactly as
        they are, so `fields.bbox` and every other field are untouched. The only
        purpose is that _read_mrz can re-cut a line from the canvas later.

        Boxes are kept top to bottom, which is the order the OCR loop walks the
        patches in - the retry has to know which box a patch came from.
        """
        self._mrz_zone = None
        bboxes = text_fields.get('bbox') or []
        idx = [i for i, box in enumerate(bboxes) if box[-1] == 'MRZ']
        if not idx:
            return
        idx.sort(key=lambda i: (bboxes[i][1] + bboxes[i][3]) / 2)
        boxes = [list(bboxes[i][:4]) for i in idx]

        # The span of the zone as a whole: for a line whose own box is too narrow
        # this is where the missing characters are. Only from boxes that are
        # line-shaped - a box several line-heights tall is not a line, and its
        # edges say nothing about where the line ends (measured: the one such zone
        # reads worse from a widened crop).
        line_shaped = [b for b in boxes
                       if b[2] - b[0] >= 10 * max(1, b[3] - b[1])]
        span = None
        if len(line_shaped) > 1:
            span = (min(b[0] for b in line_shaped), max(b[2] for b in line_shaped))
        self._mrz_zone = {'canvas': img, 'boxes': boxes, 'span': span}

    @classmethod
    def _trim_to_mrz_alphabet(cls, text: str) -> str:
        """Drop edge characters that cannot occur in a machine-readable zone.

        Only the ends, and only characters outside the zone's closed alphabet: the
        page border sometimes lands inside a crop and comes back as '.' or '_'.
        Anything inside the line is left alone - a wrong character there is a
        reading error, and hiding it would be worse than showing it.
        """
        if not text:
            return text
        return text.strip(''.join(sorted(set(text) - cls.MRZ_ALPHABET)))

    def _read_mrz(self, line_index: int, text: str) -> str:
        """Re-read one MRZ line from a wider crop when it came out too short.

        The reading the detector's own crop produced is kept unless it is the
        wrong length; then the zone's full span is tried, then progressively wider
        crops, and the first result of exactly 44 characters wins. Falls back to
        the longest reading seen, which is still closer than the short one.

        Never invents a line: it only re-reads a box the detector found, so a zone
        whose second line was never detected stays a one-line zone. That matters -
        "read in" a missing line would turn a real miss into a plausible string.
        """
        text = self._trim_to_mrz_alphabet(text)
        if len(text) == self.MRZ_LINE_LEN:
            return text
        zone = self._mrz_zone
        if not zone or line_index >= len(zone['boxes']):
            return text

        canvas = zone['canvas']
        width = canvas.shape[1]
        x1, y1, x2, y2 = zone['boxes'][line_index]
        span = zone['span']
        best = text
        for growth in self.MRZ_RETRY_GROWTH:
            left, right = (span if span else (x1, x2))
            step = int(round((right - left) * growth))
            crop = canvas[y1:y2, max(0, left - step):min(width, right + step)]
            if crop.size == 0:
                continue
            candidate = self.ocr_lat.predict(crop)[self.ocr_lat.model_name]['ocr_output']
            candidate = self.ocr_lat.fix_errors(field_type='MRZ', text=candidate)
            candidate = self._trim_to_mrz_alphabet(candidate)
            if len(candidate) == self.MRZ_LINE_LEN:
                return candidate
            if len(candidate) > len(best):
                best = candidate
        return best

    @staticmethod
    def _widest_gap(word_boxes, line_width: float) -> float:
        """The widest empty stretch on the line, measured in typical word widths.

        A dropped word leaves a hole about as wide as a word; evenly spaced
        printing does not, however wide the spacing. That is the whole reason
        this is a ratio to the line's OWN median word width instead of a share
        of the line: an internal passport's Licence_number is three digit groups
        with wide gaps and is read correctly, and any absolute measure lumps it
        together with real damage.
        Edges count as gaps too - a word lost from the start or the end of a line
        leaves the hole at the border, not between boxes.
        Nothing found at all means the whole line is one hole, so the answer is
        infinity: that is the measured case where a field used to vanish without
        a trace.
        """
        if word_boxes is None or len(word_boxes) == 0:
            return float('inf')
        if not line_width:
            return 0.0
        spans = sorted((float(b[0]), float(b[2])) for b in word_boxes)
        widths = [b - a for a, b in spans if b > a]
        if not widths:
            return float('inf')
        typical = sorted(widths)[len(widths) // 2]
        if typical <= 0:
            return float('inf')
        gaps = [spans[0][0]]                          # empty stretch on the left
        end = spans[0][1]
        for a, b in spans[1:]:
            gaps.append(max(0.0, a - end))
            end = max(end, b)
        gaps.append(max(0.0, float(line_width) - end))  # and on the right
        return max(gaps) / typical

    @staticmethod
    def _line_ink(patch) -> float:
        """How much fine detail the line crop carries - strokes, not darkness.

        Variance of the Laplacian: a printed stroke is a sharp local change, and
        a strip that has none (blank paper, or a strip blurred past the width of
        a stroke) has almost no such change left, whatever its overall
        brightness.
        """
        if patch is None or getattr(patch, 'size', 0) == 0:
            return 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
        return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())

    def _split_words(self, text_fields: dict, doc_type:str):
        """
        Split text fields into words.

        Args:
            text_fields: Detected text fields

        Returns:
            dict: Text fields splitted into words
        """

        bboxes, patches = text_fields.values()

        # The internal passport prints the series+number (and the FMS code)
        # twice, so the detector returns duplicate boxes; keep only the
        # highest-confidence one to avoid OCR'ing the same value twice.
        UNIQUE_FIELDS = ('Licence_number', 'Issue_organisation_code')
        drop = self._duplicate_field_indices(bboxes, UNIQUE_FIELDS)
        if drop:
            bboxes = [b for i, b in enumerate(bboxes) if i not in drop]
            patches = [p for i, p in enumerate(patches) if i not in drop]

        # fields that will actually contribute to `result`. Multi-line fields
        # are labeled one detection per line (dataset convention), and the
        # per-class assembly below concatenates their words in list order -
        # so order the kept boxes top-to-bottom to get reading order (a pure
        # y-sort is safe: classes collect only their own boxes).
        kept = [i for i, bbox in enumerate(bboxes)
               if bbox[-1] in self.ocr_options.en_fields or bbox[-1] in self.ocr_options.ru_fields]
        kept.sort(key=lambda i: (bboxes[i][1] + bboxes[i][3]) / 2)

        # WordsDetector calls are independent (different crop each, same
        # reused session - ONNX Runtime sessions support concurrent run()
        # calls; verified no shared mutable state in pre/postprocessing, see
        # docs/progress-log.md), so dispatch them concurrently instead of one
        # at a time. Fields that don't need splitting need no call at all.
        split_idxs = [i for i in kept if bboxes[i][-1] in self.ocr_options.needed_split]
        words_by_idx = {}
        word_bbox_by_idx = {}
        if split_idxs:
            with ThreadPoolExecutor(max_workers=min(8, len(split_idxs))) as ex:
                futures = {i: ex.submit(self.words_detector.predict_transform, patches[i])
                          for i in split_idxs}
                for i, fut in futures.items():
                    detected = fut.result()[self.words_detector.model_name]
                    words_by_idx[i] = detected['warped_img']
                    word_bbox_by_idx[i] = detected['bbox']

        # Gap guard. A hole on the line wider than a few typical words means the
        # split dropped a word - measured twice: a 9 px crop where the detector
        # returned NO words (the field vanished without a trace), and a line whose
        # longest word («Тракторозаводский», 17 characters) was the one missed.
        # Reading the line whole recovers it; the price is that the engine emits
        # no spaces, so the line comes back glued, which is why this is a fallback
        # on a signal and not the default.
        #
        # SNILS is excluded BY CONSTRUCTION, not by hoping the threshold spares
        # it: there the engine is chosen by word-index parity (see _ocr_serial),
        # and a line read whole destroys the parity the routing depends on.
        fallback = []
        no_ink = []
        if doc_type != 'SNILS':
            # `kept` order is top-to-bottom, and a multi-line field collects its
            # lines in that same order below - so counting per label here gives
            # the line's ordinal WITHIN its field, which is what a reader of the
            # flag can act on. The raw box index would be meaningless outside
            # this function.
            seen = {}
            for i in kept:
                label = bboxes[i][-1]
                ordinal = seen.get(label, 0)
                seen[label] = ordinal + 1
                if i not in split_idxs:
                    continue
                boxes = word_bbox_by_idx.get(i)
                gap = self._widest_gap(boxes, patches[i].shape[1])
                if gap > self.WORDS_MAX_GAP and boxes is not None and len(boxes) == 0:
                    # No boxes at all has two causes, and only one of them is a
                    # lost split: the other is a line with nothing on it. Asked
                    # HERE only - where boxes were found the text is there by
                    # definition, and the question would be noise.
                    ink = self._line_ink(patches[i])
                    if ink < self.LINE_MIN_INK:
                        no_ink.append({'field': label, 'line': ordinal,
                                       'ink': round(ink, 2)})
                        continue
                if gap > self.WORDS_MAX_GAP:
                    words_by_idx[i] = [patches[i]]
                    fallback.append({'field': label, 'line': ordinal,
                                     'gap': None if gap == float('inf')
                                     else round(gap, 3)})
        if fallback:
            self.results._meta_results['WordsFallback'] = fallback
        if no_ink:
            self.results._meta_results['WordsNoInk'] = no_ink

        result = {}
        word_bboxes = {}
        for i in kept:
            bbox = bboxes[i]
            words = words_by_idx[i] if i in words_by_idx else [patches[i]]
            # None distinguishes "this field needs no splitting, so the whole patch is
            # the single word" from "the detector found exactly one word". Without it a
            # port that split a field it should not have would look like agreement.
            wb = word_bbox_by_idx[i] if i in word_bbox_by_idx else None

            if result.get(bbox[-1]):
                result[bbox[-1]]['patches'].extend(words)
                word_bboxes[bbox[-1]].append(wb)
            else:
                result[bbox[-1]] = {'patches': words,
                                    'ocr': []}
                word_bboxes[bbox[-1]] = [wb]

        # Probe only. Word PATCHES are deliberately not a stage (megabytes of pixels per
        # document, see conformance/spec/stages.md), but their boxes are a handful of
        # integers and they are what localises a wrong crop to the split rather than to
        # the OCR three stages later.
        for field_name, boxes in word_bboxes.items():
            self._emit(f'words.{field_name}.bbox', boxes)

        self.results._meta_results[self.words_detector.model_name] = result
        return result

    def _address_lines(self, img):
        """
        Recognize the multi-line registration address (INTPASSPORTADDR).

        Runs the oriented-bbox address-line detector directly on the
        perspective-fixed canvas (it handles line tilt itself, so no deskew is
        needed), yielding upright line patches already ordered top-to-bottom.
        Each line is classified printed-vs-handwritten; printed lines are split
        into words and OCR'd with the printed Russian model, handwritten lines
        are flagged (OCR is printed-only) and kept as a placeholder so the line
        order/structure is preserved.

        Result is stored as results.meta_results['OCR']['Address'] — lines
        joined by newline. Per-line kinds are stored under
        meta_results['Address_lines'] and a handwritten flag on the OCR dict.

        Args:
            img: Perspective-fixed canvas.
        """
        result = self.address_lines.predict_transform(img)
        self.results._meta_results = self.results._meta_results | result
        line_patches = result[self.address_lines.model_name]['warped_img']
        # ocr_device is only 'gpu' when ocr_gpu_batch=True was explicitly
        # requested (see __init__) - safe to gate on it directly.
        use_batch = self.ocr_device == 'gpu'

        # ASCII brackets on purpose: the report gets printed to consoles that are
        # still cp1251/cp866 on Windows, and fancier delimiters (U+27E8/U+27E9)
        # are not in those codepages - printing the result raised UnicodeEncodeError.
        HW_PLACEHOLDER = '[рукопись]'
        line_meta = []
        line_slots = []          # 'hw' or ('printed', index into printed_words)
        printed_words = []       # list[list[word_patch]], one list per printed line
        has_handwritten = False

        for patch in line_patches:
            if patch is None or patch.size == 0:
                continue
            kind, prob = self.address_textkind.predict(patch)[self.address_textkind.model_name]
            if kind == 'handwritten':
                has_handwritten = True
                line_meta.append({'kind': 'handwritten', 'p_handwritten': prob, 'text': None})
                line_slots.append(('hw', None))
                continue
            # printed line: split into words (left-to-right); OCR happens below
            words = self.words_detector.predict_transform(patch)[self.words_detector.model_name]['warped_img']
            if not words:
                words = [patch]
            words = [w for w in words if w is not None and w.size > 0]
            line_slots.append(('printed', len(printed_words)))
            printed_words.append(words)
            line_meta.append({'kind': 'printed', 'p_handwritten': prob, 'text': None})

        if use_batch:
            # one padded-batch call per engine across ALL address words (see
            # ocr_batch.py / _ocr_batched for why this matters on GPU).
            flat = [w for words in printed_words for w in words]
            cyr_texts = self.ocr_cyr.predict_batch(flat)
            lat_texts = self.ocr_lat.predict_batch(flat)
            flat_texts = [lat if self._is_number_token(lat) else ru
                          for ru, lat in zip(cyr_texts, lat_texts)]
            texts_by_line = []
            it = iter(flat_texts)
            for words in printed_words:
                texts_by_line.append([next(it) for _ in words])
        else:
            texts_by_line = [[self._route_word_ocr(w) for w in words] for words in printed_words]

        address_lines_text = []
        for slot, meta in zip(line_slots, line_meta):
            kind, idx = slot
            if kind == 'hw':
                address_lines_text.append(HW_PLACEHOLDER)
                continue
            line_words = [t.strip() for t in texts_by_line[idx] if t]
            line_text = ' '.join(line_words)
            meta['text'] = line_text
            address_lines_text.append(line_text)

        self.results._meta_results['Address_lines'] = line_meta
        if address_lines_text:
            ocr_dict = self.results._meta_results.get('OCR') or {}
            ocr_dict['Address'] = '\n'.join(address_lines_text)
            ocr_dict['Address_has_handwritten'] = has_handwritten
            self.results._meta_results['OCR'] = ocr_dict

    def _route_word_ocr(self, word) -> str:
        """Pick the right OCR engine for an address word (serial/per-word path;
        used on CPU - see _address_lines for the batched GPU path).

        Run both the Cyrillic and Latin engines and keep the Latin result only
        when it is digit-dominated (a house/building/flat number), otherwise the
        Cyrillic result. The Latin engine reads digits and Latin letters
        cleanly, so its digit-dominated output is the reliable signal for
        numeric tokens; Cyrillic words are taken from the Cyrillic engine.
        """
        ru = self.ocr_cyr.predict(word)[self.ocr_cyr.model_name]['ocr_output']
        en = self.ocr_lat.predict(word)[self.ocr_lat.model_name]['ocr_output']
        return en if self._is_number_token(en) else ru

    @staticmethod
    def _is_number_token(en_text: str) -> bool:
        """True if the eng+nums OCR output is digit-dominated (a number word
        such as a house/building/flat number), i.e. at least one digit and
        digits not outnumbered by letters among the alphanumeric characters."""
        alnum = [c for c in (en_text or '') if c.isalnum()]
        if not alnum:
            return False
        digits = sum(c.isdigit() for c in alnum)
        letters = len(alnum) - digits
        return digits >= 1 and digits >= letters

    @staticmethod
    def _duplicate_field_indices(bboxes, unique_fields) -> set:
        """For each field that must be unique, return the indices of all but the
        highest-confidence detection (bbox layout: [x1,y1,x2,y2,conf,cls,label]).
        Used to drop the duplicate series/number boxes on internal passports."""
        drop = set()
        for field in unique_fields:
            idxs = [i for i, b in enumerate(bboxes) if b[-1] == field]
            if len(idxs) > 1:
                best = max(idxs, key=lambda i: bboxes[i][4])
                drop.update(i for i in idxs if i != best)
        return drop

    def _ocr(self, words_dict: dict, doc_type: str):
        """
        Perform OCR on splitted words.

        With a v2 engine (accurate/fast) AND ocr_gpu_batch=True (so
        ocr_device=='gpu' - see __init__), routes through the batched path
        (_ocr_batched): padded-batch inference calls per engine instead of one
        call per word, which avoids ONNX Runtime recompiling the CUDA graph
        for every distinct patch width (measured 400-3700x faster on real word
        crops), at the cost of a small measured decode-drift risk (see
        pipeline_modules/ocr_batch.py and docs/progress-log.md). Otherwise
        (CPU) uses the per-word path (_ocr_serial), which is exact.

        Args:
            words: Text fields splitted into words

        Returns:
            dict: OCR text for input words
        """
        # ocr_device is only 'gpu' when ocr_gpu_batch=True was explicitly
        # requested (see __init__) - safe to gate on it directly.
        if self.ocr_device == 'gpu':
            self._ocr_batched(words_dict, doc_type)
        else:
            self._ocr_serial(words_dict, doc_type)
        if 'birthcert' in doc_type.lower():
            self._clean_ruler_artifacts()

    #: Characters the ruler dots come back as. Dots, dashes and underscores were
    #: the obvious ones; commas and quotes were added after looking at what the
    #: engine actually emits on this form - «28., ИЮЛЯ 2010», «08.,.АВГУСТА.,.2008»,
    #: «"""СЕМ","" ПОННИЛОВИЧ». The set is deliberately limited to marks observed
    #: as artifacts: every character added here is one that can no longer survive
    #: on its own in a birth-certificate field.
    _RULER_MARKS = r'.,_\-"'

    def _clean_ruler_artifacts(self):
        """The 1998 birth-certificate form prints dotted ruler lines under
        every field value; they land inside the field crops and OCR emits runs
        of those marks around the real words. They carry no information on this
        form, so collapse them (single in-word dots - the digit birth date,
        abbreviations - are left untouched).

        Only runs of two or more, plus marks standing alone between spaces, are
        removed. That is what keeps real punctuation: the comma in
        «Г. ИРКУТСК, ИРКУТСКАЯ ОБЛАСТЬ» is attached to a word, and the hyphen in
        a series like «II-МЮ» sits between letters, so neither matches.

        Runs only for birth certificates (see _ocr), so no other document type is
        affected by what is in _RULER_MARKS."""
        ocr = self.results._meta_results.get('OCR')
        if not ocr:
            return
        runs = re.compile(f'[{self._RULER_MARKS}]{{2,}}')
        lone = re.compile(f'(?:^|(?<=\\s))[{self._RULER_MARKS}](?=\\s|$)')
        for key, value in list(ocr.items()):
            if not isinstance(value, str):
                continue
            text = runs.sub(' ', value)       # runs of ruler marks
            text = lone.sub(' ', text)        # marks standing alone
            ocr[key] = re.sub(r'\s+', ' ', text).strip()

    def _ocr_serial(self, words_dict: dict, doc_type: str):
        """Per-word OCR calls (one predict() per patch). Used on CPU."""
        ocr_dict = {}
        for field_name, words in words_dict.items():
            ocred_words = []
            for i, word in enumerate(words['patches']):
                # SNILS is the one doc type where a single word-index parity
                # check decides Cyrillic vs Latin, regardless of field name:
                # its date fields ("31 октября 1998") are printed as Russian
                # words interleaved with digits, so odd-indexed words (the
                # month name) must go to the Cyrillic engine even though the
                # field itself is en_fields/date-routed below.
                if doc_type == 'SNILS' and i % 2 == 1 or \
                        field_name in self.ocr_options.ru_fields:
                    result = self.ocr_cyr.predict(word)[self.ocr_cyr.model_name]['ocr_output']
                    # text normalization: Sex_ru -> М/Ж, strip stray leading dots
                    # on names. CER-validated win.
                    result = self.ocr_cyr.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)
                elif 'date' in field_name.lower():
                    result = self.ocr_lat.predict(word)[self.ocr_lat.model_name]['ocr_output']
                    # normalize date text to dd.mm.yyyy
                    result = self.ocr_lat.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)
                elif field_name in self.ocr_options.en_fields:
                    result = self.ocr_lat.predict(word)[self.ocr_lat.model_name]['ocr_output']
                    result = self.ocr_lat.fix_errors(field_type=field_name, text=result)
                    if field_name == 'MRZ':
                        result = self._read_mrz(i, result)
                    words['ocr'].append(result)
                    ocred_words.append(result)

            self._join_field(ocr_dict, field_name, doc_type, ocred_words)
            # Per-field rather than per-word: the word list is already built, so
            # this is one call site instead of three (one per routing branch)
            # while still showing exactly which word of which field differs.
            self._emit(f'ocr.{field_name}.words', ocred_words)

        self._fix_fms(ocr_dict, doc_type)
        self._emit('join', ocr_dict)
        # merge, never assign: _address_lines runs earlier and may already have
        # written OCR['Address'] into the same dict (INTPASSPORTADDR)
        self.results._meta_results.setdefault('OCR', {}).update(ocr_dict)

    def _ocr_batched(self, words_dict: dict, doc_type: str):
        """Same field/word routing and fix_errors as _ocr_serial, but the OCR
        calls are batched: every Cyrillic-routed word and every Latin-routed
        word across ALL fields is collected first, each engine is called ONCE
        via predict_batch, then results are redistributed. Only called when
        device=='gpu' (guaranteed by _ocr).
        """
        items = list(words_dict.items())

        # classify every word without calling predict() yet
        plans = []
        for field_name, words in items:
            plan = []
            for i, _word in enumerate(words['patches']):
                # see the matching comment in _ocr_serial: SNILS dates are
                # written out as Russian words ("26 СЕНТЯБРЯ 1997 ГОДА"), so
                # odd-indexed words need the Cyrillic engine even though the
                # field itself is en_fields/date-routed.
                if doc_type == 'SNILS' and i % 2 == 1 or field_name in self.ocr_options.ru_fields:
                    plan.append('cyr')
                elif 'date' in field_name.lower():
                    plan.append('lat_date')
                elif field_name in self.ocr_options.en_fields:
                    plan.append('lat')
                else:
                    plan.append(None)
            plans.append(plan)

        cyr_patches, lat_patches = [], []
        for (_field_name, words), plan in zip(items, plans):
            for kind, word in zip(plan, words['patches']):
                if kind == 'cyr':
                    cyr_patches.append(word)
                elif kind in ('lat_date', 'lat'):
                    lat_patches.append(word)

        cyr_texts = iter(self.ocr_cyr.predict_batch(cyr_patches))
        lat_texts = iter(self.ocr_lat.predict_batch(lat_patches))

        ocr_dict = {}
        for (field_name, words), plan in zip(items, plans):
            ocred_words = []
            for i, kind in enumerate(plan):
                if kind == 'cyr':
                    result = self.ocr_cyr.fix_errors(field_type=field_name, text=next(cyr_texts))
                elif kind == 'lat_date':
                    result = self.ocr_lat.fix_errors(field_type=field_name, text=next(lat_texts))
                elif kind == 'lat':
                    result = self.ocr_lat.fix_errors(field_type=field_name, text=next(lat_texts))
                    if field_name == 'MRZ':
                        # The retry reads one crop at a time, outside the batch: it
                        # runs only on a line whose length is already wrong, so the
                        # batching it breaks is batching that had failed anyway.
                        result = self._read_mrz(i, result)
                else:
                    continue
                words['ocr'].append(result)
                ocred_words.append(result)

            self._join_field(ocr_dict, field_name, doc_type, ocred_words)
            # Same stage names as the serial path, so a dump is comparable
            # regardless of which path produced it. (The batched path is not
            # bit-exact against serial by design - see ocr_batch.py.)
            self._emit(f'ocr.{field_name}.words', ocred_words)

        self._fix_fms(ocr_dict, doc_type)
        self._emit('join', ocr_dict)
        # merge, never assign - see _ocr_serial
        self.results._meta_results.setdefault('OCR', {}).update(ocr_dict)

    @staticmethod
    def _join_field(ocr_dict: dict, field_name: str, doc_type: str, ocred_words: list):
        """Join per-word OCR results into the field's final string (digit dates
        use '.', worded dates use ' ', MRZ uses a newline, everything else ' ')."""
        # The MRZ arrives as one detection per line, top to bottom. The line
        # boundary is load-bearing - every check digit lives at a fixed offset
        # in line 2 - so the lines are joined with a newline and nothing else
        # is done to the text (a space would be outside the MRZ alphabet).
        if field_name == 'MRZ':
            ocr_dict[field_name] = '\n'.join(w for w in ocred_words if w)
            return
        if 'date' in field_name.lower():
            # The separator follows the CONTENT of the date, not the doc type:
            # a digit date (DD.MM.YYYY) is written with dots, a date spelled out
            # in words is not. SNILS is worded by definition ('26 СЕНТЯБРЯ 1997
            # ГОДА') and stays hard-coded; birth certificates need both - the
            # 1998 form has a digit Birth_date but a worded Issue_date, and the
            # whole 2018 form is worded ('15 ОКТЯБРЯ 2020 Г.', the parents'
            # birth dates). The 1998 form hid this: its ruler dots merged with
            # the join dot into a run that _clean_ruler_artifacts wiped out; on
            # the cleaner 2018 crops the lone join dot survived, giving
            # '30.ОКТЯБРЯ.2020'. Multi-word dates are the only ones affected -
            # a digit date reaches this point as a single word ('22.06.2010').
            worded = doc_type == 'SNILS' or any(c.isalpha()
                                                for w in ocred_words for c in w)
            ocr_dict[field_name] = (' ' if worded else '.').join(ocred_words)
        else:
            if ocr_dict.get(field_name):
                ocr_dict[field_name] += ' ' + ' '.join(ocred_words)
            else:
                ocr_dict[field_name] = ' '.join(ocred_words)
        ocr_dict[field_name] = ocr_dict[field_name].replace('  ', ' ').strip()

    def _normalize_dates(self):
        """Build the canonical ``dd.mm.yyyy`` view next to the reading.

        Runs once, after OCR, on the finished ``results.ocr`` - that is the whole
        point of the placement: the conversion happens when the RESULT is formed,
        not while decoding or checking OCR, so nothing upstream sees a rewritten
        value. Date fields are recognised by name, the same convention
        ``_join_field`` already uses.
        """
        ocr = self.results._meta_results.get('OCR')
        if not ocr:
            return
        fields = [name for name in ocr if 'date' in name.lower()]
        normalized = canonical_dates(ocr, fields)
        if normalized:
            self.results._meta_results['OCR_normalized'] = normalized

    @staticmethod
    def _fix_fms(ocr_dict: dict, doc_type: str):
        """Disabled: the OCR reading of the authority code/name is returned as is.

        This used to rewrite ``Issue_organisation_code``/``Issue_organization_ru``
        from the FMS dictionary. Two reasons it is off:

        * **Cost.** When the code is read exactly the lookup is O(1) and ~1 ms,
          but a single misread character falls through to a ``difflib`` scan of
          the whole ~16k-entry dictionary — one ``ratio()`` per entry per query
          word, measured at **3.3-5.1 s for one document** (sample
          ``INTPASSPORT_1997/15_CR_INTPASSPORT_2001.jpg``, code read as
          ``123-005``). That is the entire reason that sample took 3.8 s while
          every other one takes ~0.4 s.
        * **Soundness.** On that fall-through the dictionary does not correct the
          code — it *replaces* it with the code of whichever authority name
          scored highest, so a misread digit can silently become a confident,
          well-formed, wrong code. Reporting what OCR actually read is the
          honest answer.

        Kept as a no-op rather than deleted so the reasoning stays next to the
        decision, and so the language ports have a named counterpart to mirror.
        The dictionary itself is not part of this repository.
        """
        return

    def _model_call(self, func, *args, **kwargs):
        """ Wrapper for making timing calculations."""
        time_start = time()
        result = func(*args, **kwargs)
        self.results.timings = {func.__name__: round(time() - time_start, 4)}
        return result

    def _prepare_image(self, img_path: Union[Path, str, np.ndarray], img_size: int = 1500):
        """
        Load image from path, validate it and resize.

        Args:
            img_path: Path to input image.
            img_size: Resize image to this size.

        Returns:
            np.ndarray: Loaded and resized image.
        """

        if isinstance(img_path, (Path, str)):
            p = Path(img_path)
            img = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode image: {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results._meta_results['image_path'] = p.as_posix()
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise Exception("Unsupported image type")

        # check size of image, and resize if above 1500
        h, w = img.shape[:2]
        ratio = max(max(h, w) / img_size, 1)
        new_h, new_w = int(h // ratio), int(w // ratio)
        img = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.results._meta_results['original_img'] = img

        return img











from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Union, Dict, Tuple

import cv2
import numpy as np

from ..pipeline_modules import *


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

    """list: English fields to recognize for this doc type."""
    en_fields = []

    """list: Russian fields to recognize for this doc type."""
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
        # Unknown type: return the empty base options instead of None, so the
        # caller's `.needs_licence_rotation`/`.ru_fields` access doesn't crash
        # (the pipeline then simply produces no OCR fields for it).
        return OCROptionsClass()

class OCROptionsINTPassport(OCROptionsClass):
    """OCR options for internal Russian passports."""

    needed_split = ["Licence_number",
                    "Birth_place_ru", "Issue_organization_ru",
                    ]

    en_fields = ["Licence_number", "Issue_date", "Expiration_date", "Birth_date", "Issue_organisation_code", ]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru"]
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

class OCROptionsEXTPassport(OCROptionsClass):
    """OCR options for external Russian passports."""

    needed_split = ["Licence_number", "Birth_place_ru", "Birth_place_en", ]

    en_fields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                 "Expiration_date", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en", "Sex_en",
                 "Issue_organisation_code", "Middle_name_en"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru"]


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
        meta_results (dict): Metadata from pipeline stages
        _timings (dict): Timing measurements for stages

    """
    def __init__(self):
        """Initializes empty result storage."""

        self.meta_results = dict(Quality={})
        self._timings = dict()
        # stage keys that ran inside a concurrent group: kept in the report for
        # visibility, but excluded from the 'total' sum (see add_concurrent_group)
        self._concurrent_members = set()

    @property
    def ocr(self) -> Union[Dict, None]:
        """Gets OCR extraction results dict, if available."""
        if self.meta_results.get('OCR'):
            return self.meta_results.get('OCR')
        else:
            return None

    @property
    def doctype(self) -> Union[str, None]:
        """Gets detected document type, if available."""
        doctype = self.meta_results.get('DocType')
        return doctype

    @property
    def quality(self) -> dict:
        """Gets image quality measurements."""
        return self.meta_results['Quality']

    @property
    def rotated_image(self) -> np.ndarray:
        """Gets image rotated by the Angle90 stage."""
        return self.meta_results['Angle90']['warped_img']

    @property
    def angle(self):
        """Gets image angle by the Angle90 stage."""
        return self.meta_results['Angle90']['angle']

    @property
    def img_with_fixed_perspective(self) -> Union[list, None]:
        """Get result from doc detection net"""
        if self.meta_results.get('DocDetector'):
            return self.meta_results['DocDetector']['warped_img']
        else:
            return self.rotated_image

    @property
    def text_fields(self) -> Union[Tuple[list, list], None]:
        """Get text field patches with their meta"""
        if self.meta_results.get('TextFieldsDetector'):
            return self.meta_results['TextFieldsDetector']['bbox'], self.meta_results['TextFieldsDetector']['warped_img']
        else:
            return None

    @property
    def text_fields_meta(self) -> Union[Dict, None]:
        """Get text field meta"""
        if self.meta_results.get('TextFieldsDetector'):
            return self.meta_results['TextFieldsDetector']
        else:
            return None

    @property
    def words_patches(self) -> Union[Dict, None]:
        """Get split words patches"""
        if self.meta_results.get('WordsDetector'):
            return self.meta_results['WordsDetector']
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
                'fast'               - v2 EdgeNext models (faster, slightly lower);
                'legacy'             - the original rus / eng+nums models.
            verbose (bool): Whether to print debug information.
            ocr_gpu_batch (bool): EXPERIMENTAL, default False. The v2 engines
                ('accurate'/'fast') are dynamic-width models; on a CUDA
                provider, running them one word at a time (as the pipeline
                naturally produces them) forces a graph recompile per distinct
                width and is measured 400-3700x SLOWER than CPU - not a viable
                combination. So by default, when device=='gpu' and ocr is not
                'legacy', the OCR engines run on CPU regardless of `device`
                (detectors still use GPU) - this default combination is
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
                'legacy' OCR is fixed-size (31x200) and always safe and fast on
                GPU directly - this flag does not affect it.
        """
        device = _resolve_device(device)
        self.device = device
        self.model_format = model_format
        if ocr not in ('accurate', 'fast', 'legacy'):
            raise ValueError("ocr must be one of 'accurate', 'fast', 'legacy'")
        self.ocr_mode = ocr
        self.ocr_gpu_batch = bool(ocr_gpu_batch)

        # Viable device for the OCR engines specifically (see ocr_gpu_batch
        # docstring above): legacy is fixed-size and fine on GPU directly;
        # v2 engines on GPU are only viable batched (opt-in), so they fall
        # back to CPU here unless the user explicitly asked for the batched
        # GPU path. Detectors below always use the requested `device`.
        if device == 'gpu' and ocr != 'legacy' and not self.ocr_gpu_batch:
            self.ocr_device = 'cpu'
        else:
            self.ocr_device = device

        self.doctype_angles = DocTypeAngles(model_format=model_format, device=device, verbose=verbose)
        self.doc_detector = DocDetector(model_format=model_format, device=device, verbose=verbose)
        self.text_fields = TextFieldsDetector(model_format=model_format, device=device, verbose=verbose)
        # OCR-family models are ONNX-only (no OpenVINO IR artifacts), so pin them
        # to ONNX even when the rest of the pipeline runs OpenVINO.
        ocr_format = 'ONNX' if model_format == 'OpenVINO' else model_format
        # oriented (rotated) address-line detector for the INTPASSPORTADDR page
        self.address_lines = AddressLinesDetector(model_format=ocr_format, device=device, verbose=verbose)
        # printed-vs-handwritten classifier for address lines (OCR is printed-only)
        self.address_textkind = AddressTextKindClassifier(model_format=ocr_format, device=device, verbose=verbose)
        self.words_detector = WordsDetector(model_format=model_format, device=device, verbose=verbose)

        # OCR engines. self.ocr_cyr / self.ocr_lat are the ACTIVE Cyrillic and
        # Latin engines used by _ocr/_route_word_ocr regardless of selection.
        # Note: these load on self.ocr_device, not necessarily `device` (see above).
        if ocr == 'legacy':
            self.ocr_cyr = OCRRus(model_format=ocr_format, device=self.ocr_device, verbose=verbose)
            self.ocr_lat = OCREngNums(model_format=ocr_format, device=self.ocr_device, verbose=verbose)
            # backwards-compatible aliases for the historical attribute names
            self.ocr_ru, self.ocr_en = self.ocr_cyr, self.ocr_lat
        else:
            tier = 'accurate' if ocr == 'accurate' else 'fast'
            self.ocr_cyr = OCRCyrillic(tier=tier, model_format=ocr_format, device=self.ocr_device, verbose=verbose)
            self.ocr_lat = OCRLatin(tier=tier, model_format=ocr_format, device=self.ocr_device, verbose=verbose)
            # NOTE: pipeline_modules/ocr_batch.py::warmup_ladder() exists to
            # pre-warm every (width, count) shape the batched path could ever
            # produce, but is NOT called automatically here - measured to cost
            # 15-90+ seconds (large batch x width combos take up to ~7s EACH)
            # and, even after paying that cost, real documents were still
            # observed slow on already-warmed shapes. Left as an opt-in
            # experiment for callers who want to try it themselves; see its
            # docstring before using it.

        self.lcd_spoofing = LCDSpoofing(model_format=model_format, device=device, verbose=verbose)
        self.print_spoofing = PrintSpoofing(model_format=model_format, device=device, verbose=verbose)
        self.glare = Glare(model_format=model_format, device=device, verbose=verbose)
        self.blur = Blur(model_format=model_format, device=device, verbose=verbose)
        # residual-tilt correction after perspective fix (projection-profile based).
        # min_angle=2.0: skip small/noisy estimates (handwriting has irregular
        # baselines that yield spurious ~1-2deg) and only fix real tilts.
        self.deskewer = DocDeskewer(angle_range=10.0, angle_steps=101, min_angle=2.0, scale=0.4)
        self.ocr_options = OCROptionsClass

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

        img = self._prepare_image(img_path, img_size=img_size)

        # unified doctype + angle classification, then rotate upright
        self._model_call(self._doctype_angle, img)
        img = self.results.rotated_image

        doc_type = self.results.doctype
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
            self._model_call(self._deskew, img)
            img = self.results.img_with_fixed_perspective
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
                if quality.get('Glare', False) == 'bad' or quality.get('Blur', False) == 'bad' or quality['DocConf'] < docconf:
                    print("[!] Doc quality is too low. You can check using results.quality, "
                          "or bypass using low_quality=True")
                    return self.results


            #detecting doc
            if get_doc_borders:
                self._model_call(self._doc_detector, img)
                img = self.results.img_with_fixed_perspective
                # correct residual tilt so text lines are horizontal (helps field
                # detection, line/word splitting and OCR; train==inference)
                self._model_call(self._deskew, img)
                img = self.results.img_with_fixed_perspective

        # detecting fields
        if find_text_fields:
            #Intpassport has licence number rotated 90 deg
            rotate_licence = self.ocr_options.needs_licence_rotation
            self._model_call(self._fields_detector, img, rotate_licence=rotate_licence)
            text_fields = self.results.text_fields_meta
        else:
            return self.results

        # registration address (INTPASSPORTADDR): detect address lines, order by Y,
        # split each line into words and OCR (printed text)
        if ocr and getattr(self.ocr_options, 'has_address', False):
            self._model_call(self._address_lines, img)

        #splitting words
        if text_fields:
            self._model_call(self._split_words, text_fields.copy(), doc_type)
            words_splitted = self.results.words_patches

            #OCR words
            if ocr and words_splitted:
                self._model_call(self._ocr, words_splitted, doc_type)

        return self.results

    # Backwards-compatible alias: the published API is a callable pipeline
    # (`pipeline(img_path, ...)`, as used by the scripts and the README);
    # process_img is the canonical name.
    __call__ = process_img


    def _doctype_angle(self, img):
        """Classify document type and its angle, and rotate the image upright.

        The module returns the standard {model_name: payload} dict; the
        pipeline-level meta_results keys ('DocType' string, 'Angle90' dict,
        Quality.DocConf) are kept as-is - they are the public PipelineResults
        contract."""
        meta = self.doctype_angles.predict_transform(img)[self.doctype_angles.model_name]
        self.results.meta_results['DocType'] = meta['doc_type']
        self.results.meta_results['Quality']['DocConf'] = meta['doc_type_confidence']
        self.results.meta_results['Angle90'] = {
            'angle': meta['angle'],
            'confidence': meta['angle_confidence'],
            'warped_img': meta['warped_img'],
        }



    def _glare(self, img):
        """Check for glare quality"""
        qual, coef = self.glare.predict(img)[self.glare.model_name]
        self.results.meta_results['Quality']['Glare'] = qual
        return qual

    def _blur(self, img):
        """Check for blur quality."""
        qual, coef = self.blur.predict(img)[self.blur.model_name]
        self.results.meta_results['Quality']['Blur'] = qual
        return qual

    def _print_spoofing(self, img):
        """Check for print spoofing."""
        qual, coef = self.print_spoofing.predict(img)[self.print_spoofing.model_name]
        self.results.meta_results['Quality']['PrintSpoofing'] = qual
        return qual

    def _lcd_spoofing(self, img):
        """Check for LCD spoofing."""
        qual, coef = self.lcd_spoofing.predict(img)[self.lcd_spoofing.model_name]
        self.results.meta_results['Quality']['LCDSpoofing'] = qual
        return qual

    def _quality_and_borders_parallel(self, img):
        """Run the 4 quality checks and border detection concurrently.

        All five take the same (rotated) image and are mutually independent
        (different models/sessions, no shared mutable state in the
        preprocessing/inference/postprocessing layers - verified by code
        review). Only used when low_quality=True:
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

        self.results.meta_results['Quality']['Glare'] = glare_res[self.glare.model_name][0]
        self.results.meta_results['Quality']['Blur'] = blur_res[self.blur.model_name][0]
        self.results.meta_results['Quality']['PrintSpoofing'] = print_res[self.print_spoofing.model_name][0]
        self.results.meta_results['Quality']['LCDSpoofing'] = lcd_res[self.lcd_spoofing.model_name][0]
        self.results.meta_results = self.results.meta_results | det_res

        group_elapsed = time() - group_start
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
        self.results.meta_results = self.results.meta_results | result
        # img = result[self.doc_detector.model_name]['warped_img']
        # return img

    def _deskew(self, img):
        """Correct residual tilt of the perspective-fixed canvas and store it
        back so img_with_fixed_perspective returns the deskewed image."""
        desk = self.deskewer.deskew(img)
        if self.results.meta_results.get('DocDetector'):
            self.results.meta_results['DocDetector']['warped_img'] = desk
        else:
            # no doc_detector result (fallback path) -> stash under Angle90
            self.results.meta_results.setdefault('Angle90', {})['warped_img'] = desk
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


        if rotate_licence:
            for i, field in enumerate(text_fields['bbox']):
                if field[-1] == 'Licence_number':
                    text_fields['warped_img'][i] = cv2.rotate(text_fields['warped_img'][i],
                                                              cv2.ROTATE_90_COUNTERCLOCKWISE)


        self.results.meta_results = self.results.meta_results | result



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

        # fields that will actually contribute to `result`, in original order
        kept = [i for i, bbox in enumerate(bboxes)
               if bbox[-1] in self.ocr_options.en_fields or bbox[-1] in self.ocr_options.ru_fields]

        # WordsDetector calls are independent (different crop each, same
        # reused session - ONNX Runtime sessions support concurrent run()
        # calls; verified no shared mutable state in pre/postprocessing),
        # so dispatch them concurrently instead of one
        # at a time. Fields that don't need splitting need no call at all.
        split_idxs = [i for i in kept if bboxes[i][-1] in self.ocr_options.needed_split]
        words_by_idx = {}
        if split_idxs:
            with ThreadPoolExecutor(max_workers=min(8, len(split_idxs))) as ex:
                futures = {i: ex.submit(self.words_detector.predict_transform, patches[i])
                          for i in split_idxs}
                for i, fut in futures.items():
                    words_by_idx[i] = fut.result()[self.words_detector.model_name]['warped_img']

        result = {}
        for i in kept:
            bbox = bboxes[i]
            words = words_by_idx[i] if i in words_by_idx else [patches[i]]

            if result.get(bbox[-1]):
                result[bbox[-1]]['patches'].extend(words)
            else:
                result[bbox[-1]] = {'patches': words,
                                    'ocr': []}

        self.results.meta_results[self.words_detector.model_name] = result
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
        self.results.meta_results = self.results.meta_results | result
        line_patches = result[self.address_lines.model_name]['warped_img']
        # ocr_device is only 'gpu' for v2 engines when ocr_gpu_batch=True was
        # explicitly requested (see __init__) - safe to gate on it directly.
        use_batch = self.ocr_device == 'gpu' and self.ocr_mode != 'legacy'

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

        self.results.meta_results['Address_lines'] = line_meta
        if address_lines_text:
            ocr_dict = self.results.meta_results.get('OCR') or {}
            ocr_dict['Address'] = '\n'.join(address_lines_text)
            ocr_dict['Address_has_handwritten'] = has_handwritten
            self.results.meta_results['OCR'] = ocr_dict

    def _route_word_ocr(self, word) -> str:
        """Pick the right OCR engine for an address word (serial/per-word path;
        used on CPU and legacy - see _address_lines for the batched GPU path).

        Run both the Cyrillic and Latin engines and keep the Latin result only
        when it is digit-dominated (a house/building/flat number), otherwise the
        Cyrillic result. The Latin engine reads digits and Latin letters
        cleanly, so its digit-dominated output is the reliable signal for
        numeric tokens; Cyrillic words are taken from the Cyrillic engine.
        (Legacy engines: OCRRus has no digits and OCREngNums no Cyrillic, so the
        same rule holds.)
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
        pipeline_modules/ocr_batch.py). Otherwise
        (CPU, or 'legacy' on either device) uses the per-word path
        (_ocr_serial), which is exact.

        Args:
            words: Text fields splitted into words

        Returns:
            dict: OCR text for input words
        """
        # ocr_device is only 'gpu' for v2 engines when ocr_gpu_batch=True was
        # explicitly requested (see __init__) - safe to gate on it directly.
        if self.ocr_device == 'gpu' and self.ocr_mode != 'legacy':
            self._ocr_batched(words_dict, doc_type)
        else:
            self._ocr_serial(words_dict, doc_type)

    def _ocr_serial(self, words_dict: dict, doc_type: str):
        """Per-word OCR calls (one predict() per patch). Used for CPU and for
        'legacy'."""
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
                    # text-only normalization: Sex_ru -> М/Ж, strip stray leading
                    # dots on names. CER-validated win.
                    result = self.ocr_cyr.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)
                # get formated dates with CTC vector
                elif 'date' in field_name.lower():
                    result = self.ocr_lat.predict(word)[self.ocr_lat.model_name]['ocr_output']
                    if self.ocr_mode != 'legacy':
                        # v2 engines: normalize date text to dd.mm.yyyy. Legacy keeps
                        # its historical behavior (raw greedy output in this branch).
                        result = self.ocr_lat.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)
                elif field_name in self.ocr_options.en_fields:
                    result = self.ocr_lat.predict(word)[self.ocr_lat.model_name]['ocr_output']
                    result = self.ocr_lat.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)

            self._join_field(ocr_dict, field_name, doc_type, ocred_words)

        # merge, never assign: _address_lines runs earlier and may already have
        # written OCR['Address'] into the same dict (INTPASSPORTADDR)
        self.results.meta_results.setdefault('OCR', {}).update(ocr_dict)

    def _ocr_batched(self, words_dict: dict, doc_type: str):
        """Same field/word routing and fix_errors as _ocr_serial, but the OCR
        calls are batched: every Cyrillic-routed word and every Latin-routed
        word across ALL fields is collected first, each engine is called ONCE
        via predict_batch, then results are redistributed. Only called when
        device=='gpu' and ocr_mode != 'legacy' (guaranteed by _ocr).
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
            for kind in plan:
                if kind == 'cyr':
                    result = self.ocr_cyr.fix_errors(field_type=field_name, text=next(cyr_texts))
                elif kind == 'lat_date':
                    result = self.ocr_lat.fix_errors(field_type=field_name, text=next(lat_texts))
                elif kind == 'lat':
                    result = self.ocr_lat.fix_errors(field_type=field_name, text=next(lat_texts))
                else:
                    continue
                words['ocr'].append(result)
                ocred_words.append(result)

            self._join_field(ocr_dict, field_name, doc_type, ocred_words)

        # merge, never assign: _address_lines runs earlier and may already have
        # written OCR['Address'] into the same dict (INTPASSPORTADDR)
        self.results.meta_results.setdefault('OCR', {}).update(ocr_dict)

    @staticmethod
    def _join_field(ocr_dict: dict, field_name: str, doc_type: str, ocred_words: list):
        """Join per-word OCR results into the field's final string (dates use
        '.', SNILS dates use ' ', everything else appends with ' ')."""
        if 'date' in field_name.lower() and doc_type != 'SNILS':
            ocr_dict[field_name] = '.'.join(ocred_words)
        elif 'date' in field_name.lower() and doc_type == 'SNILS':
            ocr_dict[field_name] = ' '.join(ocred_words)
        else:
            if ocr_dict.get(field_name):
                ocr_dict[field_name] += ' ' + ' '.join(ocred_words)
            else:
                ocr_dict[field_name] = ' '.join(ocred_words)
        ocr_dict[field_name] = ocr_dict[field_name].replace('  ', ' ').strip()

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
            self.results.meta_results['image_path'] = p.as_posix()
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise Exception("Unsupported image type")

        # check size of image, and resize if above 1500
        h, w = img.shape[:2]
        ratio = max(max(h, w) / img_size, 1)
        new_h, new_w = int(h // ratio), int(w // ratio)
        img = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.results.meta_results['original_img'] = img

        return img











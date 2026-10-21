from pathlib import Path
from time import time
from typing import List, Optional, Union

import cv2
import numpy as np

from ..geometry import Chain, Offset, QuarterTurns, Scale
from ..pipeline_modules import (AddressLinesDetector, AddressTextKindClassifier, Blur,
                                DocDeskewer, DocDetector, DocTypeAngles, DocumentDetector,
                                Glare, LCDSpoofing, OCRCyrillic, OCRLatin, PageRegistrar,
                                PrintSpoofing, RegistrationStamps, TextFieldsDetector,
                                WordsDetector)
from ..pipeline_modules.page_registration.geometry import PageGeometry
# The stages of Pipeline, one module each; Pipeline below is the orchestrator
# (construction, process_img / process_frame, _process_document) and inherits
# the stage methods from these.
from ..validation.fields import field_checks
from .date_fields import DateFieldsStage
from .documents import DocumentsStage, pair_sides
from .fields import FieldsStage
from .mrz_reading import MrzReadingStage
from .pages import PagesStage, _segments_payload
from .quality import QualityStage
from .reading import ReadingStage
from .stamps import StampsStage
# The options, the results and pair_sides used to live in this module, and the
# tests, the service and the conformance tools import them from here.
from .documents import PAIRED_SIDES  # noqa: F401
from .options import (OCROptionsBIRTHCERT, OCROptionsClass, OCROptionsDL,  # noqa: F401
                      OCROptionsEXTPassport, OCROptionsINTPASSPORTADDR,
                      OCROptionsINTPassport, OCROptionsSNILS, OCROptionsSTS)
from .results import PipelineResults


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


class Pipeline(DocumentsStage, QualityStage, PagesStage, FieldsStage, MrzReadingStage,
               ReadingStage, StampsStage, DateFieldsStage):
    """Pipeline for OCR processing of documents.

    Performs steps of pre-processing, text detection and OCR
    to extract text from documents.
    """

    def __init__(self, model_format='ONNX', device=None, ocr='accurate', verbose=False,
                 ocr_gpu_batch=False, page_registration=True, page_geometry=False,
                 detect_documents=True, card_registration=True):
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
            page_geometry (bool): default False. Template-free page geometry for
                EVERY document type: each Borders page is re-cut from a quad of
                straight lines fitted to its contour (robust to a thumb or a
                side clipped by the frame), straightened by its own lines and
                text-line profiles and unbent by their local tilt
                (pipeline_modules/page_registration/geometry.py). Same output
                convention as the Borders path (page at its own size, 1 %
                cushion). When page_registration also rebuilt the canvas of an
                internal passport, that result stands and this step is skipped.
            page_registration (bool): default True (since 2026-09-17; opt-in
                before that - pass False for the plain Borders canvas). Locate the two pages of an
                internal passport (INTPASSPORT_*) by aligning their printed blank
                to canonical templates (pipeline_modules/page_registration) and
                rebuild the canvas from them; the page geometry comes from the
                agreeing Borders quad where there is one and from the template
                homography where Borders found nothing. Each page
                comes out in the fixed canonical layout (1000x704 plus a 3%
                cushion, scaled down as one when the photo has fewer pixels -
                a page is never upsampled) and the spread is stacked at one
                scale; pages the templates cannot find fall back to their
                Borders quad, and when nothing is found the Borders canvas is
                kept unchanged. Measured on 136 photos (docs/progress-log.md):
                fixes the flat-scan failure where Borders takes the white
                scanner lid for the document, the per-page scale mismatch and
                most residual keystone, at +0.5-0.9 s CPU per document.
                Opt-in because the canvas pixels change: the conformance
                goldens and the language ports are pinned to the Borders
                canvas until they carry the same module.
            detect_documents (bool): default True (decision #142, 2026-10-01).
                Find the documents in the frame first (DocumentDetector), then
                run everything else - type, borders, fields, reading - on the
                crop of one document, cut from the image at its full
                resolution. process_img reads the largest document found;
                process_frame reads every one. The whole frame misleads the
                type classifier whenever the document is a small part of it:
                on a client's scans a driving licence on an A4 sheet was typed
                from the frame in 50 % of cases and from its crop in 99 %
                (docs/progress-log.md, 2026-09-29). With no document found -
                or False here - the whole frame is read as before.
            card_registration (bool): default True. Straighten a vehicle
                registration certificate (STS_*) by its printed blank instead of
                its Borders quad, which on a card in a sleeve is the sleeve's
                edge (see _register_card). Runs under page_registration: False
                there switches this off too.
        """
        self.card_registration = card_registration
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

        self.document_detector = None
        if detect_documents:
            try:
                self.document_detector = DocumentDetector(model_format=artifact, device=device,
                                                          verbose=verbose, runtime=runtime)
            except FileNotFoundError:
                # A weight set of models-v8 or older has no DocDetect: read the whole
                # frame as before rather than refuse to start.
                print("[!] DocumentDetector weights not found (models/DocDetect): reading whole "
                      "frames. Run scripts/fetch_models.py for a weight set that has them.")
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
        # registration stamps on the same page (issue #27): which address lines
        # belong to which stamp
        self.registration_stamps = None
        try:
            self.registration_stamps = RegistrationStamps(model_format=ocr_format, device=device,
                                                          verbose=verbose)
        except FileNotFoundError:
            # A weight set of models-v10 or older has no RegistrationStamps: the
            # address is read as before, without stamps.
            print("[!] RegistrationStamps weights not found (models/RegistrationStamps): the "
                  "registration page is read without stamps. Run scripts/fetch_models.py for a "
                  "weight set that has them.")
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
        # template-based page rectification for internal passports (opt-in, see
        # the page_registration docstring above); loads ~1 MB of templates.
        self.page_registrar = PageRegistrar() if page_registration else None
        # template-free geometry for every document type (opt-in, see the
        # page_geometry docstring above); no weights, nothing to load
        self.page_geometry = PageGeometry() if page_geometry else None
        # An instance from the start - the same kind of object _process_document
        # puts here for each document (OCROptionsClass.make_options).
        self.ocr_options = OCROptionsClass()

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
            PipelineResults with extracted information. With documents found in
            the frame it describes the largest one (``results.documents`` lists
            all of them, ``results.document_index`` is the one read); use
            ``process_frame`` to read every document.
        """
        options = dict(ocr=ocr, get_doc_borders=get_doc_borders, find_text_fields=find_text_fields,
                       check_quality=check_quality, low_quality=low_quality, docconf=docconf)
        self._start_document()
        frame = self._load_image(img_path)
        documents = self._find_documents(frame, img_size)
        return self._read_document(frame, documents, 0 if documents else None, img_size, options)

    def process_frame(
            self,
            img_path: Union[Path, str, np.ndarray],
            ocr=True,
            get_doc_borders=True,
            find_text_fields=True,
            check_quality=True,
            low_quality=True,
            docconf=0.5,
            img_size=1500,
    ) -> List[PipelineResults]:
        """Read EVERY document in the frame (issue #26), not only the largest.

        Same arguments as process_img. Returns one PipelineResults per document,
        largest first; with no document found, a one-element list holding the
        whole-frame reading, exactly what process_img returns. Two sides of one
        document lying in the same frame are paired by the number printed on
        both (see PAIRED_SIDES): ``results.paired_with`` is the index of the
        other side in the returned list.

        Each element is a separate PipelineResults object, so the list stays
        valid after the call - but the pipeline is still not thread-safe (see
        PipelineResults.meta_results).
        """
        options = dict(ocr=ocr, get_doc_borders=get_doc_borders, find_text_fields=find_text_fields,
                       check_quality=check_quality, low_quality=low_quality, docconf=docconf)
        self._start_document()
        frame = self._load_image(img_path)
        documents = self._find_documents(frame, img_size)
        if not documents:
            return [self._read_document(frame, documents, None, img_size, options)]
        out = []
        for index in range(len(documents)):
            if index:
                self._start_document(self.results._meta_results.get('image_path'))
            out.append(self._read_document(frame, documents, index, img_size, options))
        pair_sides(out)
        return out

    def _process_document(self, img, ocr=True, get_doc_borders=True, find_text_fields=True,
                          check_quality=True, low_quality=True, docconf=0.5) -> PipelineResults:
        """Everything after the image is prepared: type, borders, fields, reading."""
        self._emit('prepare', img)

        # unified doctype + angle classification, then rotate upright
        self._model_call(self._doctype_angle, img)
        img = self.results.rotated_image
        self._canvas(self.results._meta_results['Angle90']['geometry'])
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
                    self._canvas(self._borders_geometry())
                    self._canvas(QuarterTurns(img.shape[1], img.shape[0], int(angle // 90)))
                    for _ in range(angle // 90):
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    img = self.results.rotated_image
                    # the crop of the first border detection, then the turn the
                    # re-classification asked for on it
                    self._canvas(self._borders_geometry())
                    self._canvas(self.results._meta_results['Angle90']['geometry'])
        if doc_type == 'NONE':
            print("[!] The document on picture has unknown type")
            return self.results
        # labels are '<TYPE>_<YEAR>'; tolerate a missing year suffix
        parts = doc_type.rsplit('_', maxsplit=1)
        doc_type, year = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], None)
        self.ocr_options = OCROptionsClass.make_options(doc_type)
        self._doc_year = year

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
            if self.page_registrar is not None:
                self._model_call(self._register_pages, img)
            if self.page_geometry is not None:
                self._model_call(self._geometry_pages, img)
            img = self.results.img_with_fixed_perspective
            self._emit('quality', self.results.quality)
            self._emit('borders.segments', _segments_payload(self.results._meta_results))
            self._emit('borders.canvas', img)
            self._model_call(self._deskew, img)
            img = self.results.img_with_fixed_perspective
            self._canvas(self._borders_geometry())
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
                if self.page_registrar is not None:
                    self._model_call(self._register_pages, img)
                if self.page_geometry is not None:
                    self._model_call(self._geometry_pages, img)
                img = self.results.img_with_fixed_perspective
                self._emit('borders.segments', _segments_payload(self.results._meta_results))
                self._emit('borders.canvas', img)
                # correct residual tilt so text lines are horizontal (helps field
                # detection, line/word splitting and OCR; train==inference)
                self._model_call(self._deskew, img)
                img = self.results.img_with_fixed_perspective
                self._canvas(self._borders_geometry())
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
            self._model_call(self._registration_stamps, img)
            self._emit('address.lines', self.results._meta_results.get('Address_lines'))
            self._emit('address.stamps', self.results._meta_results.get('RegistrationStamps'))

        #splitting words
        if text_fields:
            self._model_call(self._split_words, text_fields.copy(), doc_type)
            words_splitted = self.results.words_patches
            # The way back to the input image (geometry.py): field, word and address-line
            # quadrilaterals, or None per key when this run's way back is not known.
            self._emit('quads', self.results._meta_results.get('Quads'))

            #OCR words
            if ocr and words_splitted:
                self._model_call(self._ocr, words_splitted, doc_type)
                self._reread_dates_whole()
                self._normalize_dates()
                self._read_leasing(doc_type)

        # Verdict per field (docs/validation-checks.md): the last step, over the
        # finished reading, which it never changes. After BOTH reading paths - the
        # registration page reads its address lines without text fields, and its
        # fields must get their (unchecked) entries too.
        if ocr and self.results._meta_results.get('OCR'):
            self.results._meta_results['FieldChecks'] = field_checks(
                self.results._meta_results['OCR'], doc_type)
            self._emit('checks', self.results._meta_results['FieldChecks'])

        return self.results

    def _model_call(self, func, *args, **kwargs):
        """ Wrapper for making timing calculations."""
        time_start = time()
        result = func(*args, **kwargs)
        self.results.timings = {func.__name__: round(time() - time_start, 4)}
        return result

    def _load_image(self, img_path: Union[Path, str, np.ndarray]) -> np.ndarray:
        """Decode the input (a path, or an RGB array passed as is) at its full size."""
        if isinstance(img_path, (Path, str)):
            p = Path(img_path)
            img = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode image: {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results._meta_results['image_path'] = p.as_posix()
            return img
        if isinstance(img_path, np.ndarray):
            return img_path
        raise Exception("Unsupported image type")

    def _prepare_image(self, img_path: Union[Path, str, np.ndarray], img_size: int = 1500,
                       crop: Optional[tuple] = None):
        """
        Load image, cut the document out of it (crop), and resize.

        Args:
            img_path: Path to input image, or the image already loaded.
            img_size: Resize image to this size.
            crop: (x0, y0, x1, y1) on the input image - the document to read; None
                reads the whole image.

        Returns:
            np.ndarray: Loaded and resized image.
        """
        img = self._load_image(img_path)
        maps = ()
        if crop is not None:
            x0, y0, x1, y1 = crop
            img = img[y0:y1, x0:x1]
            maps = (Offset(-x0, -y0),)

        # check size of image, and resize if above 1500
        h, w = img.shape[:2]
        ratio = max(max(h, w) / img_size, 1)
        new_h, new_w = int(h // ratio), int(w // ratio)
        img = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.results._meta_results['original_img'] = img
        # The first maps of the canvas every later stage reads (geometry.py): the
        # crop of the document, then the resize
        self.results._meta_results['Geometry'] = Chain(maps + (Scale(new_w / w, new_h / h),))

        return img

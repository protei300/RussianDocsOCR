package pipeline

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Recognizer owns the twelve model sessions and runs the recognition sequence.
// Port of Pipeline (pipeline/pipeline.py).
//
// It exists so the conformance CLI and the service share ONE implementation. Two walks of
// the pipeline would be free to drift, and then a golden could disagree with a live run for
// a reason that is not a behaviour change.
//
// **Construction is expensive**: twelve sessions, 215 MB of weights, and on GPU a CUDA
// context. Build one and keep it. The service pools instances for exactly this reason.
//
// **Run holds no state on the Recognizer.** This is a deliberate departure from the
// reference, where `process_img` rebinds `self.results` and `self.ocr_options` and two
// concurrent calls therefore return each other's fields. Here every intermediate lives in a
// local and leaves in the returned Results, so Run is safe to call concurrently as far as
// this type is concerned — the remaining constraint is the per-session CUDA mutex inside
// inference, which is a different problem (see svc/runtime rule 2).
type Recognizer struct {
	opts RecognizerOptions

	// docs is nil when the weight set has no DocDetect (models-v8 or older): the whole
	// frame is then read, as before decision #142 - the reference does the same instead
	// of refusing to start.
	docs       *modules.DocumentDetector
	doctype    *modules.DocTypeAngles
	glare      *modules.Glare
	blur       *modules.Blur
	printSpoof *modules.Spoofing
	lcdSpoof   *modules.Spoofing
	borders    *modules.DocDetector
	deskewer   *modules.DocDeskewer
	fields     *modules.TextFieldsDetector
	// pageRegistrar is nil only if templates failed to load - matching the
	// reference's PageRegistrar(), which the Pipeline constructs unconditionally
	// whenever page_registration=True (the default). A missing/broken template set
	// is therefore a construction-time failure, not a silent feature disable.
	pageRegistrar *modules.PageRegistrar
	words      *modules.WordsDetector
	cyr        *modules.OcrEngine
	lat        *modules.OcrEngine

	// cardRegs are the template registrars of the STS types, built on first use (nil value:
	// the type has no templates). See cardRegistrar.
	cardMu   sync.Mutex
	cardRegs map[string]*modules.PageRegistrar
}

// RecognizerOptions are the construction-time choices — the ones baked in, which is why the
// settings schema marks their service-level equivalents restart_required.
type RecognizerOptions struct {
	Root        string
	ModelFormat string
	Device      inference.Device
	// OcrDevice is SEPARATE from Device and defaults to CPU even on GPU. Measured: OCR on
	// CUDA is 13.7x slower end-to-end, because per-word dynamic widths make the runtime
	// recompile the graph for every distinct width (M8).
	OcrDevice inference.Device
	OcrTier   string
	// Threads applies to CPU sessions. The conformance harness pins it to 1 on both sides:
	// ORT's CPU reductions partition by thread, so a differing count shifts results by
	// ~1e-6 — inside the float tolerance, but enough to flip an argmax on a near-tie.
	Threads int
	// NoCardRegistration switches off the straightening of an STS by its printed blank
	// (Pipeline(card_registration=False)): the Borders canvas is kept.
	NoCardRegistration bool
}

// RunOptions are the per-document knobs.
type RunOptions struct {
	Docconf float64
	ImgSize int
	// Sink receives per-stage payloads. Nil means no instrumentation, which is the
	// production case and costs one nil check per stage.
	Sink StageSink
	// UpTo stops AFTER the named stage, leaving Results partial. Used only by the
	// conformance CLI, which is what makes a half-finished port gradeable.
	UpTo string
}

// Results is everything one run produced.
//
// The images are OWNED BY THE CALLER, who must Close. Python's GC hid this entirely, and it
// is exactly how a port that passes conformance dies after five hundred documents.
type Results struct {
	DocType         string
	DocConfidence   float64
	Angle           int
	AngleConfidence float64

	Quality  map[string]any
	Timings  map[string]float64
	Segments [][]imaging.Point
	Boxes    []postprocess.Box
	// Ocr maps field name to the joined value; Words keeps the per-field word lists that
	// localise a single bad word.
	Ocr   map[string]string
	Words []FieldText
	// OcrNormalized is the canonical dd.mm.yyyy view of the date fields, ALONGSIDE the
	// reading - a separate map, deliberately: Ocr holds what is printed on the document,
	// which is what the ground truth describes and the key set the service builds its
	// field list from. Only fields that converted appear; nil when none did
	// (PipelineResults.ocr_normalized, pipeline.py:170-181).
	OcrNormalized map[string]string
	// DatesReadWhole lists the date fields whose reading was replaced by a re-read of the
	// whole line (PipelineResults meta_results['DatesReadWhole']); nil when none was.
	DatesReadWhole []DateReread
	// Documents is every document the detector found in the frame, largest first, boxes on
	// the image passed to Run; empty when it found nothing or there is no detector.
	// DocumentIndex is the one this result reads (the largest, 0, for Run; the i-th for
	// RunFrame); -1 means the whole frame was read.
	Documents     []modules.Document
	DocumentIndex int
	// PairedWith is the index of the other side of this document in RunFrame's list (see
	// PairSides); -1 - no pair found, or a single-document call. (Python: None.)
	PairedWith int
	// Leasing is the record read from the STS special marks (PipelineResults.leasing):
	// {"leasing": true} when the marks say the vehicle is leased; nil when they do not, or
	// the document is not an STS. Only the flag (LeasingReported says why). A SEPARATE view,
	// like OcrNormalized: Ocr["Special_marks"] keeps the text as read.
	Leasing map[string]any
	// Geometry maps a point of Canvas to the image passed to Run (PipelineResults.geometry);
	// ToInput applies it. Quads are the read fields and their word patches on that image.
	Geometry geometry.Geometry
	Quads    *Quads
	// SplitFlags reports the lines the gap guard re-read whole, and the ones it declined
	// to (PipelineResults.words_fallback / words_no_ink).
	SplitFlags SplitFlags

	// Canvas is the deskewed, perspective-corrected image in RGB. HasCanvas is false when
	// the run short-circuited before producing one.
	Canvas    imaging.Image
	HasCanvas bool

	// owned are the intermediates Close must release. Kept as a list rather than named
	// fields because the count varies with how far the run got.
	owned []imaging.Image
}

// Close releases every image the run allocated, including Canvas.
func (r *Results) Close() {
	for i := range r.owned {
		_ = r.owned[i].Close()
	}
	r.owned = nil
	if r.HasCanvas {
		_ = r.Canvas.Close()
		r.HasCanvas = false
	}
}

// TakeCanvas hands the canvas to the caller and releases everything else.
//
// This exists because the service needs exactly one image to outlive the run — the canvas
// it stores as a PNG — while every intermediate must go back immediately. Without it the
// only options are Close (which frees the canvas the caller still needs) or not calling
// Close at all, and the second is what the service did: **measured at ~16 MB retained per
// document, 663 MB -> 2556 MB over 115 documents**, unbounded. The conformance CLI never
// showed it because it processes one document per process and defers Close.
//
// After this returns, Close is a no-op, so a defer left in place stays safe.
func (r *Results) TakeCanvas() (imaging.Image, bool) {
	canvas, has := r.Canvas, r.HasCanvas
	// Cleared BEFORE Close so Close skips the canvas: the caller owns it now. The canvas is
	// deliberately not in `owned` (see processImage), so this cannot double free.
	r.HasCanvas = false
	r.Close()
	return canvas, has
}

// NewRecognizer builds every module. Slow; call once.
func NewRecognizer(opts RecognizerOptions) (*Recognizer, error) {
	if opts.ModelFormat == "" {
		opts.ModelFormat = "ONNX"
	}
	if opts.Device == "" {
		opts.Device = inference.CPU
	}
	if opts.OcrDevice == "" {
		opts.OcrDevice = inference.CPU
	}
	if opts.OcrTier == "" {
		opts.OcrTier = "accurate"
	}

	root := opts.Root
	if root == "" {
		resolved, err := config.ModelsRoot()
		if err != nil {
			return nil, err
		}
		root = resolved
		opts.Root = resolved
	}
	paths, err := config.LoadModelPaths(root)
	if err != nil {
		return nil, err
	}

	r := &Recognizer{opts: opts}
	// Built in the reference's own order, and on ANY failure everything already built is
	// released: a partial construction that holds a CUDA context would compete with the
	// CPU fallback attempt the caller makes next.
	var buildErr error
	step := func(fn func() error) {
		if buildErr == nil {
			buildErr = fn()
		}
	}
	f, dev, th := opts.ModelFormat, opts.Device, opts.Threads
	tier := modules.OcrTier(opts.OcrTier)

	// The document detector is optional: a weight set of models-v8 or older has no DocDetect.
	if modules.DocumentDetectorAvailable(paths, f) {
		step(func() (e error) { r.docs, e = modules.NewDocumentDetector(root, paths, f, dev, th); return })
	} else {
		fmt.Fprintln(os.Stderr, "[!] DocumentDetector weights not found (models/DocDetect): reading whole "+
			"frames. Run scripts/fetch_models.py for a weight set that has them.")
	}
	step(func() (e error) { r.doctype, e = modules.NewDocTypeAngles(root, paths, f, dev, th); return })
	step(func() (e error) { r.glare, e = modules.NewGlare(root, paths, f, dev, th); return })
	step(func() (e error) { r.blur, e = modules.NewBlur(root, paths, f, dev, th); return })
	step(func() (e error) { r.printSpoof, e = modules.NewPrintSpoofing(root, paths, f, dev, th); return })
	step(func() (e error) { r.lcdSpoof, e = modules.NewLCDSpoofing(root, paths, f, dev, th); return })
	step(func() (e error) { r.borders, e = modules.NewDocDetector(root, paths, f, dev, th); return })
	step(func() (e error) { r.fields, e = modules.NewTextFieldsDetector(root, paths, f, dev, th); return })
	step(func() (e error) { r.words, e = modules.NewWordsDetector(root, paths, f, dev, th); return })
	step(func() (e error) { r.cyr, e = modules.NewOcrCyrillic(root, paths, f, opts.OcrDevice, th, tier); return })
	step(func() (e error) { r.lat, e = modules.NewOcrLatin(root, paths, f, opts.OcrDevice, th, tier); return })
	step(func() (e error) { r.pageRegistrar, e = modules.NewPageRegistrar(root, "INTPASSPORT", 6000); return })
	if buildErr != nil {
		_ = r.Close()
		return nil, buildErr
	}
	r.deskewer = modules.NewPipelineDeskewer()
	return r, nil
}

// Close releases every session. Safe on a partially-built Recognizer.
func (r *Recognizer) Close() error {
	closers := []func() error{}
	if r.docs != nil {
		closers = append(closers, r.docs.Close)
	}
	if r.doctype != nil {
		closers = append(closers, r.doctype.Close)
	}
	if r.glare != nil {
		closers = append(closers, r.glare.Close)
	}
	if r.blur != nil {
		closers = append(closers, r.blur.Close)
	}
	if r.printSpoof != nil {
		closers = append(closers, r.printSpoof.Close)
	}
	if r.lcdSpoof != nil {
		closers = append(closers, r.lcdSpoof.Close)
	}
	if r.borders != nil {
		closers = append(closers, r.borders.Close)
	}
	if r.fields != nil {
		closers = append(closers, r.fields.Close)
	}
	if r.words != nil {
		closers = append(closers, r.words.Close)
	}
	if r.cyr != nil {
		closers = append(closers, r.cyr.Close)
	}
	if r.lat != nil {
		closers = append(closers, r.lat.Close)
	}
	if r.pageRegistrar != nil {
		closers = append(closers, r.pageRegistrar.Close)
	}
	r.cardMu.Lock()
	for _, reg := range r.cardRegs {
		if reg != nil {
			closers = append(closers, reg.Close)
		}
	}
	r.cardRegs = nil
	r.cardMu.Unlock()
	var first error
	for _, c := range closers {
		// Every closer runs even after one fails: a session left open holds GPU memory,
		// and stopping at the first error would leak the rest.
		if err := c(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Device and OcrDevice report what this instance actually uses.
func (r *Recognizer) Device() inference.Device    { return r.opts.Device }
func (r *Recognizer) OcrDevice() inference.Device { return r.opts.OcrDevice }

// Run recognises one document: the LARGEST one in the frame (process_img). With no document
// found the whole frame is read.
//
// The sequence and every branch in it are the reference's. Two that look like details and
// are not: the quality group runs CONCURRENTLY because low_quality defaults to true (so the
// verdict never gates border detection), and the OCR options are resolved from the BARE
// document type because the routing compares it with == "SNILS", which the '<TYPE>_<YEAR>'
// label never equals.
func (r *Recognizer) Run(imagePath string, opts RunOptions) (*Results, error) {
	list, err := r.run(imagePath, opts, false)
	if err != nil {
		return nil, err
	}
	return list[0], nil
}

// RunFrame reads EVERY document in the frame (Pipeline.process_frame, issue #26), not only
// the largest: one Results per document, largest first; with no document found, one element
// holding the whole-frame reading, exactly what Run returns. Two sides of one document lying
// in the same frame are paired by the number printed on both (PairSides): Results.PairedWith
// is the index of the other side in the returned list, -1 when there is none.
//
// Every element owns its images, so the list stays valid after the call; the caller closes
// each. The Recognizer holds no per-run state, so concurrent calls are safe as far as this
// type goes (see Recognizer).
func (r *Recognizer) RunFrame(imagePath string, opts RunOptions) ([]*Results, error) {
	list, err := r.run(imagePath, opts, true)
	if err != nil {
		return nil, err
	}
	PairSides(list)
	return list, nil
}

// run is the shared body of process_img and process_frame: load, find the documents once,
// then read the first (all == false) or each of them.
func (r *Recognizer) run(imagePath string, opts RunOptions, all bool) ([]*Results, error) {
	sink := opts.Sink
	if sink == nil {
		sink = NullStageSink{}
	}
	if opts.ImgSize <= 0 {
		opts.ImgSize = 1500
	}
	timings := NewTimings()

	// ---- stage: documents -------------------------------------------------
	// Documents first (decision #142): find what lies in the frame, then read the crop of
	// the LARGEST document, cut from the frame at its full resolution. No document found,
	// or no detector in this weight set - the whole frame, as before.
	src, err := imaging.LoadRGB(imagePath)
	if err != nil {
		return nil, err
	}
	// The frame is only ever read to cut a document out of it: every crop is a new image.
	defer src.Close()

	documents, small, err := r.findDocuments(src, opts.ImgSize, timings, sink)
	if err != nil {
		return nil, err
	}
	defer small.Close()
	if opts.UpTo == "documents" {
		return []*Results{{Quality: map[string]any{}, Ocr: map[string]string{},
			Documents: documents, DocumentIndex: -1, PairedWith: -1,
			Timings: timings.Report()}}, nil
	}

	indexes := []int{-1}
	if n := len(documents); n > 0 {
		indexes = []int{0}
		if all {
			indexes = indexes[:0]
			for i := 0; i < n; i++ {
				indexes = append(indexes, i)
			}
		}
	}
	var list []*Results
	for k, index := range indexes {
		t := timings
		if k > 0 {
			t = NewTimings()
		}
		res, err := r.readDocument(src, documents, small, index, opts, t, sink)
		if err != nil {
			for _, done := range list {
				done.Close()
			}
			return nil, err
		}
		list = append(list, res)
	}
	return list, nil
}

// readDocument is Pipeline._read_document + _process_document: prepare the crop of
// documents[index] (the whole frame when index is -1) and read it. small is the frame at
// the processing size and stays the caller's.
func (r *Recognizer) readDocument(src imaging.Image, documents []modules.Document, small imaging.Image,
	index int, opts RunOptions, timings *Timings, sink StageSink) (*Results, error) {

	out := &Results{Quality: map[string]any{}, Ocr: map[string]string{},
		Documents: documents, DocumentIndex: -1, PairedWith: -1}

	// Any early return past this point must release what has been allocated, so the
	// intermediates are registered with the Results as they are created and `fail` closes
	// them. Without this an error path leaks a canvas per failed document.
	fail := func(err error) (*Results, error) {
		out.Close()
		return nil, err
	}

	// ---- stage: prepare ---------------------------------------------------
	var prepared imaging.Image
	havePrepared := false
	// The first maps of the canvas every later stage reads (geometry.py): the crop of the
	// document, then the resize. cutW x cutH is the size of what was resized.
	var geo geometry.Chain
	cutW, cutH := src.Width(), src.Height()
	if index >= 0 && index < len(documents) {
		x0, y0, x1, y1 := documentCrop(src.Width(), src.Height(), documents[index])
		if x1 > x0 && y1 > y0 {
			out.DocumentIndex = index
			cut, cerr := imaging.ClampedCrop(src, x0, y0, x1, y1)
			if cerr != nil {
				return fail(cerr)
			}
			cutW, cutH = cut.Width(), cut.Height()
			geo = geo.Then(geometry.Offset{DX: float64(-x0), DY: float64(-y0)})
			prepared = imaging.FitToLongestSide(cut, opts.ImgSize)
			_ = cut.Close()
			havePrepared = true
		}
	}
	if !havePrepared {
		prepared = small.Clone()
	}
	out.owned = append(out.owned, prepared)
	geo = geo.Then(geometry.Scale{SX: float64(prepared.Width()) / float64(cutW),
		SY: float64(prepared.Height()) / float64(cutH)})

	if err := emitImage(sink, "prepare", prepared); err != nil {
		return fail(err)
	}
	if opts.UpTo == "prepare" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stages: doctype.label, rotate ------------------------------------
	var meta modules.DocTypeResult
	var upright imaging.Image
	if err := timings.Time(StageDocTypeAngle, func() (e error) {
		meta, upright, e = r.doctype.PredictTransform(prepared)
		return
	}); err != nil {
		return fail(err)
	}
	out.owned = append(out.owned, upright)
	out.DocType = meta.DocType
	out.DocConfidence = meta.DocTypeConfidence
	out.Angle = meta.Angle
	out.AngleConfidence = meta.AngleConfidence
	geo = geo.Then(geometry.QuarterTurns{Width: prepared.Width(), Height: prepared.Height(), Turns: meta.Angle / 90})

	if err := sink.Emit("doctype.label", meta); err != nil {
		return fail(err)
	}
	if err := emitImage(sink, "rotate", upright); err != nil {
		return fail(err)
	}
	if opts.UpTo == "rotate" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stage: quality ---------------------------------------------------
	groupStart := time.Now()
	quality, qualityTimes, err := r.runQuality(upright, meta.DocTypeConfidence)
	if err != nil {
		return fail(err)
	}
	out.Quality = quality
	if err := sink.Emit("quality", quality); err != nil {
		return fail(err)
	}
	if opts.UpTo == "quality" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stages: borders.segments, borders.canvas -------------------------
	// max_pages is 2 only for the internal-passport spread; every other type passes 1 so a
	// background blob can never be stitched in as a second page. A SUBSTRING test of the
	// raw label, matching the reference.
	lowerType := strings.ToLower(meta.DocType)
	isSpread := strings.Contains(lowerType, "intpassport")
	// INTPASSPORTADDR is out of scope for this port (no anonymised sample -> no
	// golden -> nothing to verify against - see MAPPING.md "Not ported"), but it
	// still matches the "intpassport" substring above (is_spread, so maxPages=2,
	// matching the reference's own _doc_detector). _register_pages additionally
	// excludes it (pipeline.py: `'addr' in doc_type: return`), and so does the
	// registration branch below.
	canRegister := isSpread && !strings.Contains(lowerType, "addr")
	maxPages := 1
	if isSpread {
		maxPages = 2
	}
	detectStart := time.Now()
	det, err := r.borders.PredictTransform(upright, maxPages)
	if err != nil {
		return fail(err)
	}
	canvas := det.Canvas
	out.Segments = det.Segments
	out.owned = append(out.owned, canvas)
	pages, pageQuads, placements := det.Pages, det.PageQuads, det.Placements
	// The map from the canvas the later stages read back to the image the border stage received
	// (DocDetector.geometry): REPLACED whenever a stage rebuilds the pages or the canvas.
	detGeo, pageGeos := det.Geometry, det.PageGeometries
	out.owned = append(out.owned, pages...)

	qualityTimes[StageDocDetector] = time.Since(detectStart)
	timings.RecordGroup(StageQualityAndBorders, time.Since(groupStart), qualityTimes)

	// _register_pages: rebuild the internal-passport canvas from template-registered
	// pages when possible. Runs on EVERY document (the reference times it
	// unconditionally too, returning at once for anything but a registrable
	// passport), which is why the timings KEY SET always has it regardless of doc
	// type - see DEVIATIONS for the measured canvas divergence this introduces on
	// the two internal-passport conformance cases.
	registeredWithLineRefine := false
	if err := timings.Time(StageRegisterPages, func() error {
		// A vehicle registration certificate is straightened by its OWN printed blank
		// instead (_register_card): a card in a sleeve gets the sleeve's edge from the
		// border detector. The canvas is the straightened page; the pages' own lines were
		// already followed (line_refine), so the deskew below is skipped, as the reference
		// does (info['sources'] filled).
		if isCardType(meta.DocType) && r.pageRegistrar != nil {
			if r.opts.NoCardRegistration {
				return nil
			}
			cardCanvas, cardGeo, ok, cerr := r.registerCard(upright, meta.DocType, out.Segments)
			if cerr != nil || !ok {
				return cerr
			}
			canvas = cardCanvas
			detGeo, pageGeos = cardGeo, nil
			pages, pageQuads, placements = nil, nil, nil
			out.owned = append(out.owned, canvas)
			registeredWithLineRefine = r.pageRegistrar.LineRefine
			return nil
		}
		if !canRegister || r.pageRegistrar == nil {
			return nil
		}
		regResult, ok := r.registerPages(upright, out.Segments)
		if !ok {
			return nil
		}
		canvas = regResult.Canvas
		detGeo, pageGeos = regResult.Geometry, regResult.PageGeometries
		pages, pageQuads, placements = regResult.Pages, regResult.PageQuads, regResult.Placements
		out.owned = append(out.owned, canvas)
		out.owned = append(out.owned, pages...)
		registeredWithLineRefine = r.pageRegistrar.LineRefine
		return nil
	}); err != nil {
		return fail(err)
	}

	// Emitted before the canvas: the contours are upstream of the warp, so when both
	// diverge this ordering tells the reader which one to blame.
	if err := sink.Emit("borders.segments", SegmentsPayload(out.Segments)); err != nil {
		return fail(err)
	}
	if err := emitImage(sink, "borders.canvas", canvas); err != nil {
		return fail(err)
	}
	if opts.UpTo == "borders.canvas" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stage: deskew.canvas ---------------------------------------------
	// Deskew stays off when the registrar already straightened these pages by their
	// OWN lines (line_refine/line_dewarp, blur-tolerant, template-free): the
	// whole-canvas projection-profile scan can mistake a small canvas's dark cushion
	// for text and rotate an already-straight page (pipeline.py's _deskew docstring,
	// measured on 12_CR_INTPASSPORT_2011). Otherwise, with 2+ pages (a spread that
	// went through the plain Borders path), each page is deskewed ON ITS OWN and the
	// canvas re-stitched - the two pages of an open passport rarely share one angle.
	var deskewed imaging.Image
	if err := timings.Time(StageDeskew, func() (e error) {
		if registeredWithLineRefine {
			deskewed = canvas.Clone()
			return nil
		}
		var turn geometry.Geometry
		if len(pages) >= 2 {
			dp, ok := r.deskewPages(pages, pageQuads, pageGeos)
			if ok {
				deskewed, placements = dp.Canvas, dp.Placements
				detGeo, pageGeos = dp.Geometry, dp.PageGeometries
				pages = dp.Pages
				out.owned = append(out.owned, dp.Pages...)
				return nil
			}
		}
		deskewed, _, turn, e = r.deskewer.DeskewWithGeometry(canvas)
		// the canvas so far, then the turn (nil - handed on unchanged, no map)
		detGeo = geometry.Chain{Maps: []geometry.Geometry{detGeo}}.Then(turn)
		return
	}); err != nil {
		return fail(err)
	}
	// Where the canvas lies on the image passed to Run: the chain so far, then the border stage'"'"'s
	// own map (Pipeline._canvas(self._borders_geometry())).
	geo = geo.Then(detGeo)
	out.Geometry = geo

	// The canvas the client is served, and the space every box below is in. Held on
	// Results rather than in `owned` so the caller can keep it after Close of the rest --
	// which is why Close handles it separately.
	out.Canvas = deskewed
	out.HasCanvas = true

	if err := emitImage(sink, "deskew.canvas", deskewed); err != nil {
		return fail(err)
	}
	if opts.UpTo == "deskew.canvas" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stage: fields.bbox -----------------------------------------------
	bareType, docYear := SplitDocType(meta.DocType)
	ocrOpts := MakeOcrOptions(bareType)

	var fields []modules.Field
	// Where each patch lies on the canvas it was cut from (FieldFrames).
	var frames []fieldFrame
	if err := timings.Time(StageFieldsDetector, func() (e error) {
		// _fields_from_pages: with 2+ pages, the field detector runs on EACH PAGE
		// separately (its 640x640 input gives a stitched spread only half of itself
		// per page) and the boxes are moved onto the canvas by that page's placement.
		if len(pages) >= 2 && len(placements) == len(pages) {
			fields, frames, e = r.fieldsFromPages(pages, placements, ocrOpts.NeedsLicenceRotation)
			return
		}
		fields, e = r.fields.PredictTransform(deskewed, ocrOpts.NeedsLicenceRotation)
		if e == nil {
			frames = make([]fieldFrame, len(fields))
			for i, f := range fields {
				frames[i] = singleCanvasFrame(f, ocrOpts.NeedsLicenceRotation && f.Box.Label == "Licence_number")
			}
			// Single-canvas path only (_read_margins): a field labelled tight to its
			// letters is read from a taller crop; the box itself is not changed.
			if e = readMargins(fields, ocrOpts, deskewed, frames); e != nil {
				modules.FieldsClose(fields)
				fields = nil
			}
		}
		return
	}); err != nil {
		return fail(err)
	}
	defer modules.FieldsClose(fields)
	out.Boxes = boxesOf(fields)

	if err := sink.Emit("fields.bbox", BoxesPayload(out.Boxes)); err != nil {
		return fail(err)
	}
	if opts.UpTo == "fields.bbox" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stages: words.<Field>.bbox ---------------------------------------
	// The address path (INTPASSPORTADDR) is out of scope for this port, so no
	// address.lines stage is emitted and the checker skips it.
	// Remembered BEFORE the words are split, from the boxes as detected: the MRZ retry
	// re-cuts a line from this canvas when its length comes out wrong (_note_mrz_zone).
	mrzZone := NoteMrzZone(out.Boxes, deskewed)

	var fieldWords []FieldWords
	var records []SplitRecord
	if err := timings.Time(StageSplitWords, func() (e error) {
		fieldWords, out.SplitFlags, records, e = SplitWords(fields, ocrOpts, r.words, bareType)
		return
	}); err != nil {
		return fail(err)
	}
	defer FieldWordsClose(fieldWords)

	for _, fw := range fieldWords {
		if err := sink.Emit("words."+fw.Label+".bbox", WordBoxesPayload(fw.WordBoxes)); err != nil {
			return fail(err)
		}
	}
	// The way back to the input image (geometry.go): field and word quadrilaterals, or none when
	// this run'"'"'s way back is not known. Right after the split, before the OCR, like the reference.
	out.Quads = buildQuads(records, frames, out.ToInput, ocrOpts.NeedsLicenceRotation)
	if err := sink.Emit("quads", QuadsPayload(out.Quads)); err != nil {
		return fail(err)
	}
	if opts.UpTo == "words" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stages: ocr.<Field>.words, join ----------------------------------
	var texts []FieldText
	if err := timings.Time(StageOcr, func() (e error) {
		texts, e = OcrFields(fieldWords, bareType, docYear, ocrOpts, r.cyr, r.lat, mrzZone)
		return
	}); err != nil {
		return fail(err)
	}
	FixFms(texts, bareType)
	out.Words = texts

	// Ruler cleanup applies to the FINAL per-field values only: the reference
	// emits `join` from the raw dict and cleans meta_results['OCR'] afterwards
	// (pipeline.py:1058), so the conformance payload stays raw here too.
	cleanRulers := strings.Contains(strings.ToLower(meta.DocType), "birthcert")

	joined := make(map[string]any, len(texts))
	for _, ft := range texts {
		if err := sink.Emit("ocr."+ft.Label+".words", ft.Words); err != nil {
			return fail(err)
		}
		joined[ft.Label] = ft.Value
		value := ft.Value
		if cleanRulers {
			value = CleanRulerArtifacts(value)
		}
		out.Ocr[ft.Label] = value
	}
	if err := sink.Emit("join", joined); err != nil {
		return fail(err)
	}

	// A date read word by word that does not convert is re-read as whole lines, when that
	// does (_reread_dates_whole). After the stages above are emitted - they keep the split
	// reading - and before the canonical dates, which must see the final reading.
	if out.DatesReadWhole, err = rereadDatesWhole(fieldWords, out.Ocr, ocrOpts, r.cyr, r.lat); err != nil {
		return fail(err)
	}

	// Canonical dates are built once, on the FINISHED dict, after the ruler cleanup:
	// the reference's _normalize_dates runs after _ocr, which is where the cleanup
	// happens too, so nothing upstream sees a rewritten value (pipeline.py:1144-1159).
	order := make([]string, 0, len(texts))
	for _, ft := range texts {
		order = append(order, ft.Label)
	}
	out.OcrNormalized = NormalizeDates(out.Ocr, order)

	// The leasing flag from the STS special marks, alongside the reading (_read_leasing); the
	// special marks themselves stay as read. Every STS back - the side with the special marks
	// - emits the stage, null when there is no leasing: a port that misses a leasing record
	// must differ from the reference, not be skipped. The front has no marks to read.
	if len(fieldWords) > 0 && strings.HasPrefix(strings.ToUpper(bareType), "STS") {
		if rec := ParseLeasing(out.Ocr["Special_marks"]); rec != nil {
			out.Leasing = reportedLeasing(rec)
		}
		if strings.HasPrefix(strings.ToUpper(bareType), "STSBACK") {
			var payload any
			if out.Leasing != nil {
				payload = out.Leasing
			}
			if err := sink.Emit("leasing", payload); err != nil {
				return fail(err)
			}
		}
	}

	out.Timings = timings.Report()
	return out, nil
}

// runQuality runs the four quality classifiers and assembles the Quality dict.
//
// The four run CONCURRENTLY through RunGroup, in the reference's source order, with results
// collected positionally. They use four DIFFERENT sessions, so on GPU the per-session mutex
// does not serialise them and the parallelism is real — unlike the word-splitting group,
// which shares one session by design.
//
// DocConf is not computed here: it comes from DocTypeAngles, which the reference also writes
// into this same dict. The key set is part of the contract, so it is assembled in one place
// rather than accumulated.
func (r *Recognizer) runQuality(img imaging.Image, docConf float64) (
	map[string]any, map[string]time.Duration, error) {

	type verdict struct {
		key, stage, label string
		took              time.Duration
	}
	timed := func(key, stage string, predict func() (string, float64, error)) func() (verdict, error) {
		return func() (verdict, error) {
			start := time.Now()
			label, _, err := predict()
			return verdict{key: key, stage: stage, label: label, took: time.Since(start)}, err
		}
	}
	labels, err := RunGroup(0, []func() (verdict, error){
		timed("Glare", StageGlare, func() (string, float64, error) { return r.glare.Predict(img) }),
		timed("Blur", StageBlur, func() (string, float64, error) { return r.blur.Predict(img) }),
		timed("PrintSpoofing", StagePrintSpoofing,
			func() (string, float64, error) { return r.printSpoof.Predict(img) }),
		timed("LCDSpoofing", StageLcdSpoofing,
			func() (string, float64, error) { return r.lcdSpoof.Predict(img) }),
	})
	if err != nil {
		return nil, nil, err
	}

	// Only the LABELS reach the dict; the per-detector scores are not stored, matching the
	// reference. DocConf is the one numeric member.
	out := map[string]any{"DocConf": docConf}
	took := make(map[string]time.Duration, len(labels))
	for _, v := range labels {
		out[v.key] = v.label
		took[v.stage] = v.took
	}
	return out, took, nil
}

func boxesOf(fields []modules.Field) []postprocess.Box {
	out := make([]postprocess.Box, 0, len(fields))
	for i := range fields {
		out = append(out, fields[i].Box)
	}
	return out
}

func emitImage(sink StageSink, name string, img imaging.Image) error {
	arr, err := imaging.ToArray(img)
	if err != nil {
		return fmt.Errorf("pipeline: stage %s: %w", name, err)
	}
	return sink.Emit(name, ArrayPayload{Array: arr})
}

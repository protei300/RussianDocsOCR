package pipeline

import (
	"fmt"
	"strings"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
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

	doctype    *modules.DocTypeAngles
	glare      *modules.Glare
	blur       *modules.Blur
	printSpoof *modules.Spoofing
	lcdSpoof   *modules.Spoofing
	borders    *modules.DocDetector
	deskewer   *modules.DocDeskewer
	fields     *modules.TextFieldsDetector
	words      *modules.WordsDetector
	cyr        *modules.OcrEngine
	lat        *modules.OcrEngine
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

// Run recognises one document.
//
// The sequence and every branch in it are the reference's. Two that look like details and
// are not: the quality group runs CONCURRENTLY because low_quality defaults to true (so the
// verdict never gates border detection), and the OCR options are resolved from the BARE
// document type because the routing compares it with == "SNILS", which the '<TYPE>_<YEAR>'
// label never equals.
func (r *Recognizer) Run(imagePath string, opts RunOptions) (*Results, error) {
	sink := opts.Sink
	if sink == nil {
		sink = NullStageSink{}
	}
	if opts.ImgSize <= 0 {
		opts.ImgSize = 1500
	}
	timings := NewTimings()
	out := &Results{Quality: map[string]any{}, Ocr: map[string]string{}}

	// Any early return past this point must release what has been allocated, so the
	// intermediates are registered with the Results as they are created and `fail` closes
	// them. Without this an error path leaks a canvas per failed document.
	fail := func(err error) (*Results, error) {
		out.Close()
		return nil, err
	}

	// ---- stage: prepare ---------------------------------------------------
	src, err := imaging.LoadRGB(imagePath)
	if err != nil {
		return fail(err)
	}
	out.owned = append(out.owned, src)

	prepared := imaging.FitToLongestSide(src, opts.ImgSize)
	out.owned = append(out.owned, prepared)

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
	maxPages := 1
	if strings.Contains(strings.ToLower(meta.DocType), "intpassport") {
		maxPages = 2
	}
	var canvas imaging.Image
	detectStart := time.Now()
	if canvas, out.Segments, err = r.borders.PredictTransform(upright, maxPages); err != nil {
		return fail(err)
	}
	out.owned = append(out.owned, canvas)

	qualityTimes[StageDocDetector] = time.Since(detectStart)
	timings.RecordGroup(StageQualityAndBorders, time.Since(groupStart), qualityTimes)

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
	var deskewed imaging.Image
	if err := timings.Time(StageDeskew, func() (e error) {
		deskewed, _, e = r.deskewer.Deskew(canvas)
		return
	}); err != nil {
		return fail(err)
	}
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
	bareType, _ := SplitDocType(meta.DocType)
	ocrOpts := MakeOcrOptions(bareType)

	var fields []modules.Field
	if err := timings.Time(StageFieldsDetector, func() (e error) {
		fields, e = r.fields.PredictTransform(deskewed, ocrOpts.NeedsLicenceRotation)
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
	var fieldWords []FieldWords
	if err := timings.Time(StageSplitWords, func() (e error) {
		fieldWords, e = SplitWords(fields, ocrOpts, r.words)
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
	if opts.UpTo == "words" {
		out.Timings = timings.Report()
		return out, nil
	}

	// ---- stages: ocr.<Field>.words, join ----------------------------------
	var texts []FieldText
	if err := timings.Time(StageOcr, func() (e error) {
		texts, e = OcrFields(fieldWords, bareType, ocrOpts, r.cyr, r.lat)
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

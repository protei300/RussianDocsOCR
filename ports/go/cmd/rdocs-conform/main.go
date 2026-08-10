// Command rdocs-conform is the Go port's conformance CLI.
//
// Contract: conformance/spec/cli.md. Four subcommands — info, recognize, probe,
// regen — of which this milestone implements info and the `prepare` stage of probe.
//
// stdout carries ONLY the payload; everything else goes to stderr. A port that logs
// to stdout produces output the checker cannot parse, and the failure looks like a
// serialisation bug rather than a logging mistake.
//
// Exit codes: 0 ran, 2 not implemented (the checker SKIPS rather than fails), 3 input
// error, 1 crash. The 2 is what lets a partial port be graded honestly: M1 can be
// green on `prepare` while nothing else exists yet.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/pipeline"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/viewmodel"
)

const (
	exitOK             = 0
	exitCrash          = 1
	exitNotImplemented = 2
	exitInput          = 3
)

// stagesImplemented is what this build can emit. The checker skips the difference
// against the golden, so this list is the honest statement of progress — keep it
// exact, because overstating it turns a skip into a failure.
var stagesImplemented = []string{
	"prepare", "doctype.label", "rotate", "quality",
	"borders.segments", "borders.canvas", "deskew.canvas",
	// `words.<Field>.bbox` is a PATTERN, expanded by the checker: which fields exist
	// depends on the document, so a port claims the shape rather than the names.
	"fields.bbox", "words.<Field>.bbox", "ocr.<Field>.words", "join", "viewmodel",
}

// errNotImplemented signals exit code 2.
var errNotImplemented = errors.New("not implemented in this build")

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(exitInput)
	}

	var err error
	switch os.Args[1] {
	case "info":
		err = cmdInfo(os.Args[2:])
	case "probe":
		err = cmdProbe(os.Args[2:])
	case "recognize":
		err = cmdRecognize(os.Args[2:])
	case "regen":
		err = fmt.Errorf("regen: %w (goldens are produced by the Python reference)", errNotImplemented)
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n", os.Args[1])
		usage()
		os.Exit(exitInput)
	}

	switch {
	case err == nil:
	case errors.Is(err, errNotImplemented):
		fmt.Fprintln(os.Stderr, err)
		os.Exit(exitNotImplemented)
	case errors.Is(err, os.ErrNotExist):
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
		os.Exit(exitInput)
	default:
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(exitCrash)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `rdocs-conform — Go conformance CLI

  info
  probe     --image <path> --dump-dir <dir> [--upto <stage>]
  recognize --image <path>                        (M7)

Common flags: --device cpu|gpu --ocr accurate|fast --img-size N --docconf F
See conformance/spec/cli.md.
`)
}

// commonFlags mirrors the reference CLI's options so the checker can pass the same
// arguments to every port without special-casing.
type commonFlags struct {
	device      string
	ocr         string
	imgSize     int
	docconf     float64
	modelFormat string
	includeDbg  bool
	ocrDevice   string
}

func addCommon(fs *flag.FlagSet) *commonFlags {
	c := &commonFlags{}
	fs.StringVar(&c.device, "device", "cpu", "cpu|gpu")
	fs.StringVar(&c.ocr, "ocr", "accurate", "accurate|fast")
	fs.IntVar(&c.imgSize, "img-size", 1500, "longest side; only ever shrinks")
	fs.Float64Var(&c.docconf, "docconf", 0.5, "minimum document confidence")
	fs.StringVar(&c.modelFormat, "model-format", "ONNX", "artifact folder")
	fs.BoolVar(&c.includeDbg, "include-debug", false, "include debug payloads")
	// Separate from --device on purpose: the reference pins OCR to CPU even when the
	// detectors are on GPU, and that rule needs to be MEASURABLE rather than assumed.
	fs.StringVar(&c.ocrDevice, "ocr-device", "cpu", "cpu|gpu — OCR engines only")
	return c
}

func cmdInfo(argv []string) error {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	c := addCommon(fs)
	if err := fs.Parse(argv); err != nil {
		return err
	}

	root, rootErr := config.ModelsRoot()
	// Initialising the runtime here costs a shared-library load, which is worth it:
	// `info` is what the checker records alongside a verdict, and "which ONNX Runtime
	// produced these numbers" is the first question asked when two ports disagree.
	ortVersion := any(nil)
	if err := inference.InitEnvironment(); err == nil {
		ortVersion = inference.Version()
	} else {
		ortVersion = "unavailable: " + err.Error()
	}
	versions := map[string]any{
		"runtime":     runtime.Version(),
		"opencv":      imaging.OpenCVVersion(),
		"gocv":        imaging.BindingVersion(),
		"onnxruntime": ortVersion,
	}
	payload := map[string]any{
		"port":               "go",
		"language":           "Go " + runtime.Version(),
		"versions":           versions,
		"device":             c.device,
		"ocr_device":         nil,
		"providers":          []string{},
		"model_format":       c.modelFormat,
		"ocr_mode":           c.ocr,
		"stages_implemented": stagesImplemented,
		"commit":             os.Getenv("RDOCS_COMMIT"),
		"platform":           runtime.GOOS + "/" + runtime.GOARCH,
		"models_root":        root,
		"models_root_error":  errString(rootErr),
	}
	return emitJSON(payload)
}

func cmdProbe(argv []string) error {
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	image := fs.String("image", "", "path to the input image")
	dumpDir := fs.String("dump-dir", "", "where to write stage payloads")
	upto := fs.String("upto", "", "stop AFTER this stage (inclusive)")
	c := addCommon(fs)
	if err := fs.Parse(argv); err != nil {
		return err
	}
	if *image == "" || *dumpDir == "" {
		return fmt.Errorf("%w: --image and --dump-dir are required", os.ErrNotExist)
	}
	if _, err := os.Stat(*image); err != nil {
		return fmt.Errorf("%w", err)
	}

	sink, err := pipeline.NewDirectoryStageSink(*dumpDir, *upto)
	if err != nil {
		return err
	}
	if _, err := run(*image, c, sink, *upto); err != nil {
		return err
	}
	if err := sink.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %d stage(s) to %s\n", sink.Count(), *dumpDir)
	return nil
}

// cmdRecognize emits the view model on stdout and nothing else.
//
// It shares `run` with probe rather than walking the pipeline again. Two code paths would
// be free to drift, and then a golden could disagree with a live run for a reason that is
// not a behaviour change — the same argument that makes the reference's `regen` reuse
// `probe` and `recognize` instead of reimplementing them.
func cmdRecognize(argv []string) error {
	fs := flag.NewFlagSet("recognize", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	image := fs.String("image", "", "path to the input image")
	c := addCommon(fs)
	if err := fs.Parse(argv); err != nil {
		return err
	}
	if *image == "" {
		return fmt.Errorf("%w: --image is required", os.ErrNotExist)
	}
	if _, err := os.Stat(*image); err != nil {
		return fmt.Errorf("%w", err)
	}

	// No stage dump: the null sink costs nothing and keeps `run` single-shaped.
	in, err := run(*image, c, pipeline.NullStageSink{}, "")
	if err != nil {
		return err
	}
	return emitJSON(viewmodel.Build(*in, c.includeDbg))
}

// run walks the pipeline through the shared Recognizer and returns the view-model input.
//
// The walk itself moved into internal/docproc/pipeline in M9 so the CLI and the service share
// ONE implementation. Two walks would be free to drift, and then a golden could disagree with
// a live run for a reason that is not a behaviour change.
//
// `upto` stops AFTER the named stage; the returned Input is then partial, which is fine
// because only `probe` passes it.
func run(image string, c *commonFlags, sink pipeline.StageSink, upto string) (*viewmodel.Input, error) {
	device := inference.CPU
	if c.device == "gpu" {
		device = inference.GPU
	}
	// OCR DEFAULTS TO CPU even when the detectors are on GPU, matching the reference. The
	// flag exists because M8 had to settle whether that rule applies to this port: measured,
	// GPU OCR is 13.7x slower end to end, so it does.
	ocrDevice := inference.CPU
	if c.ocrDevice == "gpu" {
		ocrDevice = inference.GPU
	}

	rec, err := pipeline.NewRecognizer(pipeline.RecognizerOptions{
		ModelFormat: c.modelFormat,
		Device:      device,
		OcrDevice:   ocrDevice,
		OcrTier:     c.ocr,
		// One thread on CPU, matching what the harness pins on the Python side: ORT's CPU
		// reductions partition by thread, so a differing count shifts results by ~1e-6 —
		// inside the float tolerance, but enough to flip an argmax on a near-tie.
		Threads: 1,
	})
	if err != nil {
		return nil, err
	}
	defer rec.Close()

	res, err := rec.Run(image, pipeline.RunOptions{
		Docconf: c.docconf,
		ImgSize: c.imgSize,
		Sink:    sink,
		UpTo:    upto,
	})
	if err != nil {
		return nil, err
	}
	defer res.Close()

	in := &viewmodel.Input{
		DocType:       res.DocType,
		Device:        c.device,
		CanvasMissing: !res.HasCanvas,
		Boxes:         res.Boxes,
		Ocr:           res.Ocr,
		Quality:       res.Quality,
		Timings:       res.Timings,
		Segments:      res.Segments,
	}
	if res.HasCanvas {
		in.CanvasW, in.CanvasH = res.Canvas.Width(), res.Canvas.Height()
	}
	return in, nil
}

func emitJSON(payload any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	// No HTML escaping: the values include Cyrillic and '&', and escaping would make
	// every golden unreadable while changing nothing about the parsed value.
	enc.SetEscapeHTML(false)
	return enc.Encode(payload)
}

func errString(err error) any {
	if err == nil {
		return nil
	}
	return err.Error()
}

var _ = filepath.Join // reserved for the stages that follow

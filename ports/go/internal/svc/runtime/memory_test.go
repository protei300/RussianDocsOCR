package runtime

import (
	"bytes"
	"errors"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/pipeline"
)

// fakeRunner stands for a pipeline: it produces a Results holding a canvas, keeps its own handle
// on that canvas so the test can look at it after the wipe, and holds "weights" that must survive.
type fakeRunner struct {
	weights  imaging.Image
	produced imaging.Image // shared handle on the canvas it returned
	fail     error
}

func (f *fakeRunner) RunImage(src imaging.Image, opts pipeline.RunOptions) (*pipeline.Results, error) {
	if !opts.Scrub {
		return nil, errors.New("RecogniseImage must run with Scrub")
	}
	canvas := imaging.NewFilled(40, 50, 5, 5, 5)
	f.produced = canvas.Share()
	res := &pipeline.Results{DocType: "DL_2011", Canvas: canvas, HasCanvas: true,
		Quality: map[string]any{}, Ocr: map[string]string{}}
	// A run that reports an error alongside a partial result: it must be wiped all the same.
	return res, f.fail
}

// §7.3 — after RecogniseImage (and after it fails) every image of the run is zero, the canvas
// is released, and the weights are untouched.
func TestRecogniseInLeaseWipesTheRunAndNotTheWeights(t *testing.T) {
	for _, fail := range []error{nil, errors.New("boom")} {
		f := &fakeRunner{weights: imaging.NewFilled(4, 4, 3, 3, 3), fail: fail}
		rgb := imaging.NewFilled(10, 10, 1, 1, 1)
		vm, png, err := recogniseInLease(f, "cpu", rgb, ImageOptions{WithCanvas: true})
		if fail != nil {
			if !errors.Is(err, fail) {
				t.Errorf("error %v, want %v", err, fail)
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			if vm.DocType == nil || *vm.DocType != "DL_2011" {
				t.Errorf("view model %+v", vm)
			}
			if !bytes.HasPrefix(png, []byte("\x89PNG")) {
				t.Errorf("png %q", png[:min(8, len(png))])
			}
		}
		if !f.produced.IsAllZero() {
			t.Errorf("fail=%v: the canvas went back to the pool unzeroed", fail)
		}
		weights, _ := f.weights.Bytes()
		for _, b := range weights {
			if b != 3 {
				t.Fatalf("fail=%v: the weights were wiped", fail)
			}
		}
		if rgbBytes, _ := rgb.Bytes(); rgbBytes[0] != 1 {
			t.Errorf("the caller's image was modified inside the lease")
		}
		_ = f.produced.Close()
		_ = f.weights.Close()
		_ = rgb.Close()
	}
}

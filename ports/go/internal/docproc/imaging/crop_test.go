package imaging

import (
	"image"
	"testing"

	"gocv.io/x/gocv"
)

// ClampedCrop is the highest-risk divergence in the whole port (CONVENTIONS §6.6):
// Python crops with a numpy slice, which silently clamps an over-large upper bound,
// while every OpenCV binding throws instead. A port that "works" is one that clamps —
// and it must clamp the same way, or the OCR input changes size and the text changes
// with it.
//
// These cases lock the semantics down. The equivalent test is mandatory in every port.
func TestClampedCrop(t *testing.T) {
	src := Wrap(gocv.NewMatWithSize(60, 80, gocv.MatTypeCV8UC3)) // h=60, w=80
	defer src.Close()

	tests := []struct {
		name           string
		x1, y1, x2, y2 int
		wantW, wantH   int
	}{
		{"inside", 10, 5, 30, 25, 20, 20},
		{"whole image", 0, 0, 80, 60, 80, 60},

		// Python: img[5:999, 10:999] returns rows 5..60 and cols 10..80. gocv's
		// Region would throw. Clamping is what makes the two agree.
		{"upper bound past the edge", 10, 5, 999, 999, 70, 55},

		// The detector clamps negatives to zero only on the resize path, so negative
		// starts DO reach here. Clamped to 0 — deliberately NOT reproduced as numpy's
		// from-the-end indexing, which would crop the opposite side of the document.
		{"negative start", -20, -10, 30, 25, 30, 25},

		// Zero area must be representable, not fatal: OCRv2Preprocessing has an
		// explicit degenerate-crop path that returns a 32x16 black patch.
		{"zero width", 40, 5, 40, 25, 0, 0},
		{"zero height", 10, 30, 30, 30, 0, 0},
		{"entirely outside", 200, 200, 300, 300, 0, 0},

		// A reversed box is meaningless; numpy would yield an empty slice, and
		// ordering the corners is the more useful reading of the caller's intent.
		{"reversed corners", 30, 25, 10, 5, 20, 20},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := ClampedCrop(src, tc.x1, tc.y1, tc.x2, tc.y2)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			defer out.Close()
			if out.Width() != tc.wantW || out.Height() != tc.wantH {
				t.Errorf("got %dx%d, want %dx%d",
					out.Width(), out.Height(), tc.wantW, tc.wantH)
			}
		})
	}
}

// A crop must be an independent copy, not a view. gocv's Region returns a view into
// the parent, and the pipeline reuses and rebinds its canvas between stages — a view
// would change under the caller. This is the same class of bug the service hit by
// reading PipelineResults after releasing the lease.
func TestClampedCropIsIndependentOfItsParent(t *testing.T) {
	parent := gocv.NewMatWithSize(10, 10, gocv.MatTypeCV8UC3)
	defer parent.Close()
	src := Wrap(parent.Clone())
	defer src.Close()

	crop, err := ClampedCrop(src, 2, 2, 6, 6)
	if err != nil {
		t.Fatal(err)
	}
	defer crop.Close()

	before, err := crop.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	original := append([]byte(nil), before...)

	// Scribble over the parent region the crop came from.
	// Mat() returns a value and Region is a pointer method, so it needs an
	// addressable local.
	parentMat := src.Mat()
	region := parentMat.Region(image.Rect(2, 2, 6, 6))
	region.SetTo(gocv.NewScalar(255, 255, 255, 0))
	region.Close()

	after, err := crop.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	for i := range original {
		if original[i] != after[i] {
			t.Fatalf("crop changed when the parent was modified at byte %d: "+
				"ClampedCrop returned a view, not a copy", i)
		}
	}
}

// FitToLongestSide only ever shrinks, and uses FLOOR — `int(h // ratio)` in Python.
// Rounding instead produces an off-by-one canvas on some inputs, and therefore
// off-by-one boxes on every downstream stage.
//
// The expectations below were taken from running the reference arithmetic, not from
// reasoning about it, and they record a non-obvious property: **the longest side is
// NOT guaranteed to equal imgSize.** For 1789x1083 the result is 1499, not 1500,
// because `1789 / (1789/1500)` is 1499.999... in float64 and the floor takes it down.
// Same for 2999x1777 -> 1499x888. A port that "corrects" this to produce exactly
// imgSize would diverge from the reference on those inputs — which is precisely why
// the values here are pinned rather than computed.
func TestFitToLongestSide(t *testing.T) {
	tests := []struct {
		name         string
		w, h, target int
		wantW, wantH int
	}{
		{"already smaller: untouched", 800, 600, 1500, 800, 600},
		{"exactly at the limit", 1500, 1000, 1500, 1500, 1000},
		{"landscape shrink", 3000, 2000, 1500, 1500, 1000},
		{"portrait shrink", 2000, 3000, 1500, 1000, 1500},
		{"float32 slack loses a pixel", 1789, 1083, 1500, 1499, 908},
		{"float32 slack loses a pixel, again", 2999, 1777, 1500, 1499, 888},
		{"barely over the limit", 1501, 1000, 1500, 1500, 999},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := Wrap(gocv.NewMatWithSize(tc.h, tc.w, gocv.MatTypeCV8UC3))
			defer src.Close()
			out := FitToLongestSide(src, tc.target)
			defer out.Close()
			if out.Width() != tc.wantW || out.Height() != tc.wantH {
				t.Errorf("%dx%d -> got %dx%d, want %dx%d",
					tc.w, tc.h, out.Width(), out.Height(), tc.wantW, tc.wantH)
			}
		})
	}
}

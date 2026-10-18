package modules

import (
	"math"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Mirrors the check of AddressLinesDetector.crop_geometry in the reference: a mark drawn on the
// image where the map says a point of the patch lies must be found at that point of the patch
// CropRotated cuts. The two functions are written apart (a warp and a sub-pixel read, and a chain
// of two maps), so agreement is the evidence that the way back is the way the patch was cut.
func TestAnAddressLinePatchRoundTripsTheMark(t *testing.T) {
	cases := []struct {
		name string
		ob   postprocess.ObbBox
		u, v float64 // a point of the patch, continuous coordinates
	}{
		{"upright", postprocess.ObbBox{Cx: 300, Cy: 200, W: 160, H: 40, Angle: 0}, 120.5, 12.5},
		{"tilted", postprocess.ObbBox{Cx: 300.37, Cy: 200.61, W: 160.4, H: 39.6, Angle: 0.21}, 130.5, 10.5},
		{"tilted the other way", postprocess.ObbBox{Cx: 280.25, Cy: 190.5, W: 121, H: 31, Angle: -0.34}, 40.5, 20.5},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			geo := CropGeometry(c.ob)
			where, ok := geo.ToInput([]geometry.Point{{X: c.u, Y: c.v}})
			if !ok {
				t.Fatal("the way back is not known")
			}
			m := newCanvas(600, 400, background)
			defer m.Close()
			mark(&m, int(math.Floor(where[0].X)), int(math.Floor(where[0].Y)), 3)
			img := imaging.Wrap(m.Clone())
			defer img.Close()

			patch, err := CropRotated(img, c.ob)
			if err != nil {
				t.Fatal(err)
			}
			defer patch.Close()
			wi, hi := obbSize(c.ob)
			if patch.Width() != wi || patch.Height() != hi {
				t.Fatalf("the patch is %dx%d, want %dx%d", patch.Width(), patch.Height(), wi, hi)
			}
			found := markCentres(t, patch, 1, 1)[0]
			if d := math.Hypot(found.X-c.u, found.Y-c.v); d > 1.5 {
				t.Errorf("the mark is at (%.2f, %.2f) of the patch, the map says (%.1f, %.1f): %.2f px apart",
					found.X, found.Y, c.u, c.v, d)
			}
		})
	}
}

// A box of less than half a pixel still gives a patch of one pixel: the size is max(1, round()).
func TestATinyBoxStillGivesAOnePixelPatch(t *testing.T) {
	wi, hi := obbSize(postprocess.ObbBox{W: 0.3, H: 0.49})
	if wi != 1 || hi != 1 {
		t.Fatalf("got %dx%d", wi, hi)
	}
	// round() is half-to-even: 2.5 -> 2, 3.5 -> 4
	wi, hi = obbSize(postprocess.ObbBox{W: 2.5, H: 3.5})
	if wi != 2 || hi != 4 {
		t.Fatalf("got %dx%d, want 2x4", wi, hi)
	}
}

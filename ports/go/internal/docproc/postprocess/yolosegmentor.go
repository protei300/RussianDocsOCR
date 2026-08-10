package postprocess

import (
	"fmt"
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// SegmentResult carries the polygon outline of each detected instance.
//
// The binary masks themselves are not returned: the only consumer in the pipeline is
// FixPerspective, which needs the contours, and keeping N full-resolution masks alive
// per document costs megabytes for nothing.
type SegmentResult struct {
	Segments [][]imaging.Point
}

func (SegmentResult) isResult() {}

// YoloSegmentor turns proto-masks plus per-instance coefficients into contours.
// Port of YOLOSegmentorPostprocessing (postprocessing.py:621-748). Borders only.
type YoloSegmentor struct {
	maskFilter float64
}

func NewYoloSegmentor(maskFilter float64) (*YoloSegmentor, error) {
	if maskFilter < 0 || maskFilter > 1 {
		return nil, fmt.Errorf("postprocess: MaskFilter must be within [0,1], got %g", maskFilter)
	}
	return &YoloSegmentor{maskFilter: maskFilter}, nil
}

// Apply is not used: the segmentor needs the detector's output as well as the proto
// tensor, so the model wrapper calls Segment directly. Present to satisfy the
// Postprocessor interface, which keeps the loader's dispatch uniform.
func (s *YoloSegmentor) Apply(*tensor.Array, Context) (Result, error) {
	return nil, fmt.Errorf("postprocess: YOLOSegmentor requires Segment(), not Apply()")
}

// Segment computes the instance masks and their outlines.
//
// proto is the [imh, imw, chn] proto-mask tensor (HWC, per this project's export
// convention). boxes carry both the mask coefficients and the already-mapped-back
// bounding boxes. origH/origW are the dimensions AFTER the extra padding and BEFORE the
// letterbox, and extraPad is that padding.
func (s *YoloSegmentor) Segment(proto *tensor.Array, boxes []Box,
	extraPad [2]int, origH, origW int) ([][]imaging.Point, error) {

	if len(boxes) == 0 {
		return nil, nil
	}
	shape := proto.Shape
	if len(shape) == 4 && shape[0] == 1 {
		shape = shape[1:]
	}
	if len(shape) != 3 {
		return nil, fmt.Errorf("postprocess: proto masks expect [h,w,chn], got %v", proto.Shape)
	}
	imh, imw, chn := shape[0], shape[1], shape[2]
	protoData, err := proto.AsFloat32()
	if err != nil {
		return nil, err
	}

	segments := make([][]imaging.Point, 0, len(boxes))
	for _, b := range boxes {
		if len(b.Seg) != chn {
			return nil, fmt.Errorf("postprocess: %d mask coefficients for %d proto channels",
				len(b.Seg), chn)
		}

		// masks @ proto.transpose(-1,0,1).reshape(chn,-1), then sigmoid. Written as a
		// direct loop over pixels: the transpose exists in the reference only to make
		// numpy's matmul line up, and materialising it here would be pure copying.
		//
		// Accumulated in float32 to match the reference's dtype. Widening "for
		// accuracy" would change the mask boundary and therefore the contour.
		mask := make([]float32, imh*imw)
		for y := 0; y < imh; y++ {
			for x := 0; x < imw; x++ {
				base := (y*imw + x) * chn
				var acc float32
				for c := 0; c < chn; c++ {
					acc += float32(b.Seg[c]) * protoData[base+c]
				}
				mask[y*imw+x] = sigmoid32(acc)
			}
		}

		// Undo the letterbox INSIDE the proto resolution, before upscaling. gain is
		// old/new, and the padding is halved because it was split across both sides.
		gain := math.Min(float64(imh)/float64(origH), float64(imw)/float64(origW))
		padX := (float64(imw) - float64(origW)*gain) / 2
		padY := (float64(imh) - float64(origH)*gain) / 2
		top, left := int(padY), int(padX)
		// TRUNCATE THE DIFFERENCE, do not subtract the truncated padding. The
		// reference writes `int(imh - pad[1])`, and `imh - int(pad[1])` is a different
		// number whenever the padding is fractional: for imh=160 and pad=20.5 the two
		// give 139 and 140. One row at proto resolution becomes ~9 rows after the
		// upscale, which is exactly how this was caught — the conformance run reported
		// borders.canvas at 868 rows against the golden's 877, with the width matching
		// to the pixel.
		bottom, right := int(float64(imh)-padY), int(float64(imw)-padX)
		if top < 0 || left < 0 || bottom > imh || right > imw || bottom <= top || right <= left {
			return nil, fmt.Errorf("postprocess: degenerate mask crop [%d:%d, %d:%d] in %dx%d",
				top, bottom, left, right, imh, imw)
		}

		cropped := imaging.NewFloatMask(mask, imh, imw).Crop(top, bottom, left, right)
		// Upscale to the pre-letterbox size, then strip the extra padding — the same
		// order the reference uses, and it matters because the extra padding is
		// expressed in ORIGINAL pixels.
		full := cropped.Resize(origW, origH)
		full = full.Crop(extraPad[1], origH-extraPad[1], extraPad[0], origW-extraPad[0])

		// Zero everything outside the instance's own box, so two adjacent documents
		// cannot bleed into each other's contour. Note the comparisons are STRICT
		// (`r > x1`, `r < x2`), so the boundary column and row are excluded — matching
		// clip_boxes exactly.
		full.ZeroOutsideBox(b.X1-float64(extraPad[0]), b.Y1-float64(extraPad[1]),
			b.X2-float64(extraPad[0]), b.Y2-float64(extraPad[1]))

		binary := full.Threshold(s.maskFilter)
		defer binary.Close()

		contours := imaging.FindExternalContours(binary)
		// "Largest" here means MOST POINTS, not greatest area — the reference selects
		// with argmax over contour lengths. An empty result yields an empty polygon
		// rather than being dropped, so segments stay index-aligned with boxes.
		largest := imaging.LargestContour(contours)
		if largest == nil {
			largest = []imaging.Point{}
		}
		segments = append(segments, largest)
	}
	return segments, nil
}

// sigmoid32 stays in float32 deliberately (CONVENTIONS §6.7). On the JVM the equivalent
// must be written exp(x.toDouble()).toFloat() per element, because Math.exp promotes —
// see DEVIATIONS D-05.
func sigmoid32(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

package postprocess

import (
	"fmt"
	"math"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// ObbBox is one oriented detection in the coordinates of the image the detector read:
// [cx, cy, w, h, angle, conf, cls, label] of the reference. Angle is in RADIANS, the
// ultralytics convention (clockwise-positive in image space).
//
// Cx, Cy, W, H, Angle and Conf hold float32 values widened to float64, exactly as the
// reference's float(...) of a numpy float32 does: the arithmetic that maps a box back out of
// the letterbox is float32 there (numpy 2 keeps a float32 scalar float32 against a Python
// float), and the patch cut about the centre is sensitive to the last digits.
type ObbBox struct {
	Cx, Cy, W, H float64
	Angle        float64
	Conf         float64
	Cls          int
	Label        string
}

// obbCand is one anchor that passed the confidence filter, in the head's own float32.
type obbCand struct {
	cx, cy, w, h, angle, conf float32
	cls                       int
}

// ObbResult is a decoded, suppressed and ordered set of oriented detections.
type ObbResult struct {
	Boxes []ObbBox
}

func (ObbResult) isResult() {}

// YoloObb decodes the raw head of an ultralytics OBB export. Port of
// YOLOOBBDetectorPostprocessing (postprocessing.py:601-690).
//
// Per anchor the columns are [cx, cy, w, h, cls_0 .. cls_{n-1}, angle]. The confidence filter,
// a greedy NMS by rotated IoU WITHOUT regard to the class, the mapping back out of the
// letterbox and a reading-order sort - rows of 10 px top to bottom, then left to right.
type YoloObb struct {
	labels []string
	iou    float64
	cls    float64
}

func NewYoloObb(labels []string, iou, cls float64) (*YoloObb, error) {
	if len(labels) == 0 {
		return nil, fmt.Errorf("postprocess: YOLOOBBDetector needs Labels")
	}
	return &YoloObb{labels: labels, iou: iou, cls: cls}, nil
}

func (y *YoloObb) Apply(out *tensor.Array, ctx Context) (Result, error) {
	boxes, err := y.decode(out, ctx)
	if err != nil {
		return nil, err
	}
	return ObbResult{Boxes: boxes}, nil
}

func (y *YoloObb) decode(out *tensor.Array, ctx Context) ([]ObbBox, error) {
	data, err := out.AsFloat32()
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	// np.squeeze: every axis of length 1 goes.
	var shape []int
	for _, d := range out.Shape {
		if d != 1 {
			shape = append(shape, d)
		}
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("postprocess: OBB detector expects a 2-D head after the squeeze, got %v", out.Shape)
	}
	// "ensure [n_anchors, features]": the smaller axis is the features.
	transposed := shape[0] < shape[1]
	anchors, feats := shape[0], shape[1]
	if transposed {
		anchors, feats = shape[1], shape[0]
	}
	at := func(a, f int) float32 {
		if transposed {
			return data[f*anchors+a]
		}
		return data[a*feats+f]
	}
	nc := len(y.labels)
	if feats < 4+nc+1 { // 4 box + nc classes + angle
		return nil, fmt.Errorf("postprocess: OBB row width %d is too small for %d classes", feats, nc)
	}

	clsThr := float32(y.cls)
	var cands []obbCand
	for a := 0; a < anchors; a++ {
		best, bestScore := 0, at(a, 4)
		for c := 1; c < nc; c++ {
			if v := at(a, 4+c); v > bestScore { // first maximum on a tie, like np.argmax
				best, bestScore = c, v
			}
		}
		if !(bestScore > clsThr) {
			continue
		}
		cands = append(cands, obbCand{at(a, 0), at(a, 1), at(a, 2), at(a, 3), at(a, 4+nc), bestScore, best})
	}
	if len(cands) == 0 {
		return nil, nil
	}

	// Greedy NMS on rotated IoU. np.argsort(-conf) is not stable in numpy, but equal float32
	// confidences of two different anchors are not expected; a stable sort breaks such a tie by
	// anchor order.
	order := make([]int, len(cands))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return cands[order[a]].conf > cands[order[b]].conf })
	var kept []int
	for len(order) > 0 {
		i := order[0]
		order = order[1:]
		kept = append(kept, i)
		rest := order[:0:0]
		for _, j := range order {
			v, err := rotatedIoU(cands[i], cands[j])
			if err != nil {
				return nil, err
			}
			if v < float32(y.iou) {
				rest = append(rest, j)
			}
		}
		order = rest
	}

	// Back out of the letterbox, all in float32 (see ObbBox).
	dw, dh := float32(ctx.PadLetter[0]), float32(ctx.PadLetter[1])
	ratio := float32(ctx.Ratio)
	ex, ey := float32(ctx.PadExtra[0]), float32(ctx.PadExtra[1])
	results := make([]ObbBox, 0, len(kept))
	for _, i := range kept {
		c := cands[i]
		cx := float32(float32(float32(c.cx-dw)/ratio) - ex)
		cy := float32(float32(float32(c.cy-dh)/ratio) - ey)
		w := float32(c.w / ratio)
		h := float32(c.h / ratio)
		results = append(results, ObbBox{
			Cx: float64(cx), Cy: float64(cy), W: float64(w), H: float64(h),
			Angle: float64(c.angle), Conf: float64(c.conf), Cls: c.cls, Label: y.labels[c.cls],
		})
	}
	// Reading order: top to bottom in rows of 10 px (round() is half-to-even), then left to right.
	sort.SliceStable(results, func(a, b int) bool {
		ra, rb := math.RoundToEven(results[a].Cy/10), math.RoundToEven(results[b].Cy/10)
		if ra != rb {
			return ra < rb
		}
		return results[a].Cx < results[b].Cx
	})
	return results, nil
}

// rotatedIoU is rotated_iou: the intersection through OpenCV, the union and the quotient in the
// float32 arithmetic numpy applies to the float32 box values (np.degrees of a float32 is a
// float32 too).
func rotatedIoU(a, b obbCand) (float32, error) {
	pi := float32(math.Pi)
	toDeg := float32(180) / pi
	ra := [5]float32{a.cx, a.cy, a.w, a.h, float32(a.angle * toDeg)}
	rb := [5]float32{b.cx, b.cy, b.w, b.h, float32(b.angle * toDeg)}
	area, err := imaging.RotatedIntersectionArea(ra, rb)
	if err != nil {
		return 0, err
	}
	inter := float32(area)
	union := float32(float32(float32(a.w*a.h)+float32(b.w*b.h)) - inter)
	if union > 0 {
		return float32(inter / union), nil
	}
	return 0, nil
}

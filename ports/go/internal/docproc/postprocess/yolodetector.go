package postprocess

import (
	"fmt"
	"math"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Box is one detection. Field order matches the reference's row layout
// [x1,y1,x2,y2,conf,cls,label] so the two can be read side by side.
type Box struct {
	X1, Y1, X2, Y2 float64
	Conf           float64
	Cls            int
	Label          string
	// Seg carries the mask coefficients for a segmentation model, empty otherwise.
	Seg []float64
}

// DetectResult is a decoded, suppressed and ordered set of detections.
type DetectResult struct {
	Boxes []Box
}

func (DetectResult) isResult() {}

// NmsMode selects how suppression treats classes.
//
// A field on ONE type rather than a subclass overriding a method: Go embedding is not
// virtual dispatch, so a "PerClassYOLODetector" type embedding this one would silently
// call the base suppression (CONVENTIONS §5). Flattened here on purpose.
type NmsMode string

const (
	// NmsClassAgnostic lets any box suppress any other. Words and Borders.
	NmsClassAgnostic NmsMode = "classAgnostic"
	// NmsPerClass suppresses only within a class. Required for TextFields, where the
	// ru/en field pairs on external passports legitimately overlap at IOU 0.2-0.3 and
	// cross-class suppression would silently drop one field of each pair.
	NmsPerClass NmsMode = "perClass"
)

// YoloDetector decodes an anchors-first detection tensor.
// Port of YOLODetectorPostprocessing (postprocessing.py:294-495) and its per-class
// subclass.
type YoloDetector struct {
	labels []string
	iou    float64
	cls    float64
	mode   NmsMode
	// numpyOnly skips label attachment and integer coercion, matching the
	// `numpy=True` call the segmentation wrapper makes.
	numpyOnly bool
}

func NewYoloDetector(labels []string, iou, cls float64, mode NmsMode) (*YoloDetector, error) {
	if len(labels) == 0 {
		return nil, fmt.Errorf("postprocess: YOLODetector needs Labels")
	}
	return &YoloDetector{labels: labels, iou: iou, cls: cls, mode: mode}, nil
}

// WithNumpyOnly returns a copy that behaves like the reference's `numpy=True` call:
// no label lookup, no integer coercion of coordinates.
func (y *YoloDetector) WithNumpyOnly() *YoloDetector {
	c := *y
	c.numpyOnly = true
	return &c
}

func (y *YoloDetector) Apply(out *tensor.Array, ctx Context) (Result, error) {
	boxes, err := y.decode(out, ctx)
	if err != nil {
		return nil, err
	}
	return DetectResult{Boxes: boxes}, nil
}

// decode is the whole pipeline: confidence filter, xywh->xyxy, argmax class, NMS,
// reading-order sort, then the coordinate mapping back to the original image.
func (y *YoloDetector) decode(out *tensor.Array, ctx Context) ([]Box, error) {
	data, err := out.AsFloat32()
	if err != nil {
		return nil, err
	}
	// The tensor is anchors-first: [anchors, 4 + nc (+ 32 seg)] after the squeeze the
	// model wrapper applies. A [1, anchors, k] shape is accepted too.
	shape := out.Shape
	if len(shape) == 3 && shape[0] == 1 {
		shape = shape[1:]
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("postprocess: detector expects [anchors, 4+nc(+seg)], got %v", out.Shape)
	}
	anchors, stride := shape[0], shape[1]
	nc := len(y.labels)
	if stride < 4+nc {
		return nil, fmt.Errorf("postprocess: row width %d is too small for %d classes", stride, nc)
	}
	segLen := stride - 4 - nc

	type cand struct {
		box  [4]float64
		conf float64
		cls  int
		seg  []float64
	}
	cands := make([]cand, 0, 64)

	for a := 0; a < anchors; a++ {
		row := data[a*stride : (a+1)*stride]

		// Strict `>`, matching `.max(axis=1) > self.cls`. A box exactly at the
		// threshold is dropped.
		best, bestScore := 0, float64(math.Inf(-1))
		for c := 0; c < nc; c++ {
			if v := float64(row[4+c]); v > bestScore {
				best, bestScore = c, v
			}
		}
		if !(bestScore > y.cls) {
			continue
		}

		// xywh -> xyxy, centre-based.
		cx, cy, w, h := float64(row[0]), float64(row[1]), float64(row[2]), float64(row[3])
		c := cand{
			box:  [4]float64{cx - w/2, cy - h/2, cx + w/2, cy + h/2},
			conf: bestScore,
			cls:  best,
		}
		if segLen > 0 {
			c.seg = make([]float64, segLen)
			for i := 0; i < segLen; i++ {
				c.seg[i] = float64(row[4+nc+i])
			}
		}
		cands = append(cands, c)
	}
	if len(cands) == 0 {
		return nil, nil
	}

	boxesXY := make([][4]float64, len(cands))
	confs := make([]float64, len(cands))
	classes := make([]int, len(cands))
	for i, c := range cands {
		boxesXY[i], confs[i], classes[i] = c.box, c.conf, c.cls
	}

	var keep []int
	if y.mode == NmsPerClass {
		keep = nmsPerClass(boxesXY, confs, classes, y.iou)
	} else {
		keep = nms(boxesXY, confs, y.iou)
	}

	kept := make([]Box, 0, len(keep))
	for _, i := range keep {
		kept = append(kept, Box{
			X1: cands[i].box[0], Y1: cands[i].box[1],
			X2: cands[i].box[2], Y2: cands[i].box[3],
			Conf: cands[i].conf, Cls: cands[i].cls, Seg: cands[i].seg,
		})
	}

	// Reading order. `np.lexsort((x1, y1))` sorts by the LAST key first, so y1 is the
	// primary key and x1 the tie-breaker: top-to-bottom, then left-to-right. The
	// argument order is the reverse of the intuition (CONVENTIONS §6.3).
	sort.SliceStable(kept, func(a, b int) bool {
		if kept[a].Y1 != kept[b].Y1 {
			return kept[a].Y1 < kept[b].Y1
		}
		return kept[a].X1 < kept[b].X1
	})

	if ctx.Resize {
		y.mapBack(kept, ctx)
	}

	if !y.numpyOnly {
		for i := range kept {
			// int() after the rounding, exactly as the reference coerces.
			kept[i].X1 = math.Trunc(kept[i].X1)
			kept[i].Y1 = math.Trunc(kept[i].Y1)
			kept[i].X2 = math.Trunc(kept[i].X2)
			kept[i].Y2 = math.Trunc(kept[i].Y2)
			if kept[i].Cls >= 0 && kept[i].Cls < len(y.labels) {
				kept[i].Label = y.labels[kept[i].Cls]
			} else {
				// The reference's bare `except` produces this literal string.
				kept[i].Label = "Unsupported class"
			}
		}
	}
	return kept, nil
}

// mapBack undoes the letterbox and the extra padding, then rounds.
//
// Three things here are exact reproductions of the reference and none should be
// "improved":
//
//   - the order of operations: subtract the letterbox padding, divide by the ratio,
//     THEN subtract the extra padding;
//   - `np.round(..., 0)` is half-to-EVEN, so 0.5 rounds down to 0 and 1.5 up to 2.
//     Go's math.Round would round both away from zero and shift boxes by a pixel;
//   - **the negative clamp applies to EVERY column, not just the coordinates.**
//     `detect_res[detect_res < 0] = 0` in the reference zeroes negative values across
//     the whole row — which for a segmentation model includes the 32 MASK
//     COEFFICIENTS that are handed to the segmentor immediately afterwards. That looks
//     like it was meant to clamp coordinates only, but it is shipped behaviour, the
//     goldens encode it, and a port that clamps only the coordinates produces
//     different masks. Reproduced deliberately.
func (y *YoloDetector) mapBack(boxes []Box, ctx Context) {
	for i := range boxes {
		b := &boxes[i]
		b.X1 = (b.X1-ctx.PadLetter[0])/ctx.Ratio - float64(ctx.PadExtra[0])
		b.X2 = (b.X2-ctx.PadLetter[0])/ctx.Ratio - float64(ctx.PadExtra[0])
		b.Y1 = (b.Y1-ctx.PadLetter[1])/ctx.Ratio - float64(ctx.PadExtra[1])
		b.Y2 = (b.Y2-ctx.PadLetter[1])/ctx.Ratio - float64(ctx.PadExtra[1])

		b.X1 = clampNeg(b.X1)
		b.Y1 = clampNeg(b.Y1)
		b.X2 = clampNeg(b.X2)
		b.Y2 = clampNeg(b.Y2)
		b.Conf = clampNeg(b.Conf)
		for j := range b.Seg {
			// See the note above: this is intentional fidelity, not a copy-paste slip.
			b.Seg[j] = clampNeg(b.Seg[j])
		}

		b.X1 = math.RoundToEven(b.X1)
		b.Y1 = math.RoundToEven(b.Y1)
		b.X2 = math.RoundToEven(b.X2)
		b.Y2 = math.RoundToEven(b.Y2)
		b.Conf = tensor.RoundHalfEven(b.Conf, 3)
	}
}

func clampNeg(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

// nms is greedy class-agnostic suppression.
//
// The tie-breaking is a real behaviour, not an accident: the reference sorts ASCENDING
// by score with a stable argsort and then takes `order[-1]` each round, so on equal
// scores it keeps the box with the HIGHEST ORIGINAL INDEX. An unstable sort, or taking
// the first of a descending sort, keeps a different box (CONVENTIONS §6.4).
func nms(boxes [][4]float64, scores []float64, threshold float64) []int {
	if len(boxes) == 0 {
		return nil
	}
	areas := make([]float64, len(boxes))
	for i, b := range boxes {
		areas[i] = (b[2] - b[0]) * (b[3] - b[1])
	}

	// Stable ascending sort by score, exactly like np.argsort.
	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] < scores[order[b]] })

	var picked []int
	for len(order) > 0 {
		last := len(order) - 1
		index := order[last]
		picked = append(picked, index)

		rest := order[:last]
		next := rest[:0]
		for _, j := range rest {
			x1 := math.Max(boxes[index][0], boxes[j][0])
			y1 := math.Max(boxes[index][1], boxes[j][1])
			x2 := math.Min(boxes[index][2], boxes[j][2])
			y2 := math.Min(boxes[index][3], boxes[j][3])
			w := math.Max(0, x2-x1)
			h := math.Max(0, y2-y1)
			inter := w * h
			ratio := inter / (areas[index] + areas[j] - inter)
			// `< threshold` survives, matching np.where(ratio < threshold).
			if ratio < threshold {
				next = append(next, j)
			}
		}
		order = next
	}
	return picked
}

// nmsPerClass runs suppression independently within each class.
//
// Classes are visited in ASCENDING order, matching `np.unique`, because the resulting
// index order feeds the reading-order sort afterwards and a different visit order can
// break ties differently.
func nmsPerClass(boxes [][4]float64, scores []float64, classes []int, threshold float64) []int {
	seen := map[int]bool{}
	var unique []int
	for _, c := range classes {
		if !seen[c] {
			seen[c] = true
			unique = append(unique, c)
		}
	}
	sort.Ints(unique)

	var keep []int
	for _, c := range unique {
		var idx []int
		for i, cc := range classes {
			if cc == c {
				idx = append(idx, i)
			}
		}
		sub := make([][4]float64, len(idx))
		subScores := make([]float64, len(idx))
		for i, j := range idx {
			sub[i], subScores[i] = boxes[j], scores[j]
		}
		for _, k := range nms(sub, subScores, threshold) {
			keep = append(keep, idx[k])
		}
	}
	return keep
}

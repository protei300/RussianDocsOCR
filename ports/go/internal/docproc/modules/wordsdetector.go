package modules

import (
	"fmt"
	"math"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// WordsDetector splits one text-field patch into word crops.
// Port of pipeline_modules/words_detector. Uses the Words artifact (class-agnostic NMS,
// a single "Word" class).
type WordsDetector struct {
	model *models.DetectionModel
}

func NewWordsDetector(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*WordsDetector, error) {

	dir, err := paths.Dir("WordsDetector", format)
	if err != nil {
		return nil, err
	}
	m, err := models.LoadDetection(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: WordsDetector: %w", err)
	}
	return &WordsDetector{model: m}, nil
}

func (d *WordsDetector) Close() error { return d.model.Close() }

// readingOrder sorts word boxes into reading order: cluster into lines by vertical centre
// proximity (within half a word height), lines top-to-bottom, words left-to-right inside
// a line. Port of WordsDetector._reading_order.
//
// A plain x-sort interleaves the lines of a multi-line field — measured on the birth
// certificates' Birth_place/ZAGS fields as word salad — so this is a correctness rule,
// not a tidiness one. On a single-line field it reproduces the old x-sorted order exactly.
//
// Two things are load-bearing for byte-identical results:
//   - Every sort is STABLE (CONVENTIONS §6.3). Python's sorted() is; sort.Slice is not,
//     and two words sharing a centre or an x1 would swap.
//   - The running means are updated per box, in float64, in the reference's order:
//     a box joins the FIRST line it fits, and the line's centre and height are the means
//     over the boxes admitted so far. Comparing against the first box instead would
//     cluster differently on a field whose line drifts.
func readingOrder(boxes []postprocess.Box) []postprocess.Box {
	type line struct {
		cy, h float64
		boxes []postprocess.Box
	}

	byCentre := make([]postprocess.Box, len(boxes))
	copy(byCentre, boxes)
	sort.SliceStable(byCentre, func(a, b int) bool {
		return (byCentre[a].Y1+byCentre[a].Y2)/2 < (byCentre[b].Y1+byCentre[b].Y2)/2
	})

	var lines []*line
	for _, b := range byCentre {
		cy, h := (b.Y1+b.Y2)/2, b.Y2-b.Y1
		placed := false
		for _, ln := range lines {
			if math.Abs(cy-ln.cy) < 0.5*math.Max(h, ln.h) {
				n := float64(len(ln.boxes))
				ln.cy = (ln.cy*n + cy) / (n + 1)
				ln.h = (ln.h*n + h) / (n + 1)
				ln.boxes = append(ln.boxes, b)
				placed = true
				break
			}
		}
		if !placed {
			lines = append(lines, &line{cy: cy, h: h, boxes: []postprocess.Box{b}})
		}
	}

	ordered := make([]postprocess.Box, 0, len(boxes))
	for _, ln := range lines { // already top-to-bottom: lines are created in centre order
		sort.SliceStable(ln.boxes, func(a, b int) bool { return ln.boxes[a].X1 < ln.boxes[b].X1 })
		ordered = append(ordered, ln.boxes...)
	}
	return ordered
}

// PredictTransform returns the word boxes and their crops, in reading order.
//
// The boxes are returned REORDERED, not just the crops: the order is what the conformance
// `words.<field>.bbox` stage records and what the OCR loop walks, so the two must agree.
//
// An empty result is normal: the caller falls back to the whole patch, as the reference
// does.
func (d *WordsDetector) PredictTransform(patch imaging.Image) ([]postprocess.Box, []imaging.Image, error) {
	boxes, err := d.model.Predict(patch)
	if err != nil {
		return nil, nil, err
	}

	boxes = readingOrder(boxes)

	words := make([]imaging.Image, 0, len(boxes))
	for _, b := range boxes {
		// Cut ON the box. Python pads small word boxes by 2 px since 1cc8468, and the ports
		// deliberately do NOT follow yet: the words detector is being retrained with the
		// margin inside the labelled box, which may remove the compensation altogether. The
		// ports are synced to the FINAL Python behaviour in one pass before the goldens are
		// regenerated.
		crop, err := imaging.ClampedCrop(patch, int(b.X1), int(b.Y1), int(b.X2), int(b.Y2))
		if err != nil {
			for i := range words {
				_ = words[i].Close()
			}
			return nil, nil, err
		}
		words = append(words, crop)
	}
	return boxes, words, nil
}

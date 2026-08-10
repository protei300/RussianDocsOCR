package modules

import (
	"fmt"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
)

// secondSegmentAreaFrac is the share of the largest page's area a second segment must
// reach to be kept.
//
// Selection is by AREA and not by confidence, and the reference explains why:
// spurious thin strips routinely score 0.9+ while a faint genuine page can score ~0.65,
// so confidence is not a usable filter here. Real spread pages run ~0.84-1.0 of the
// largest and background blobs stay under ~0.5, which is what 0.6 separates.
const secondSegmentAreaFrac = 0.6

// DocDetector finds the document's borders and rectifies its perspective.
// Port of pipeline_modules/doc_detector. Uses the Borders artifact.
type DocDetector struct {
	model *models.SegmentationModel
}

func NewDocDetector(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*DocDetector, error) {

	dir, err := paths.Dir("DocDetector", format)
	if err != nil {
		return nil, err
	}
	m, err := models.LoadSegmentation(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: DocDetector: %w", err)
	}
	return &DocDetector{model: m}, nil
}

func (d *DocDetector) Close() error { return d.model.Close() }

// PredictTransform returns the perspective-corrected canvas.
//
// maxPages caps how many document segments are kept: 1 for single-page types so a
// background blob can never be stitched in, 2 for an internal-passport spread.
//
// When no usable segment is found the ORIGINAL image is returned. That is not a
// fallback bolted on for safety — it is what the reference does, and a port that
// errored instead would fail every document whose borders the model cannot see.
// Returns the SELECTED contours alongside the canvas, so the conformance harness can
// compare them (the borders.segments stage) and localise a divergence to the mask
// rather than to the warp. Nil when border detection found nothing.
func (d *DocDetector) PredictTransform(img imaging.Image, maxPages int) (imaging.Image, [][]imaging.Point, error) {
	_, segments, err := d.model.Predict(img)
	if err != nil {
		return imaging.Image{}, nil, err
	}
	if len(segments) == 0 {
		out := img.Clone()
		return out, nil, nil
	}

	kept := selectPages(segments, maxPages)
	if len(kept) == 0 {
		out := img.Clone()
		return out, nil, nil
	}

	chosen := make([][]imaging.Point, 0, len(kept))
	for _, i := range kept {
		chosen = append(chosen, segments[i])
	}

	warped, ok := imaging.FixPerspective(img, chosen, imaging.StackAuto, imaging.DocMarginFrac)
	if !ok {
		out := img.Clone()
		return out, chosen, nil
	}
	return warped, chosen, nil
}

// selectPages ranks segments by contour area and applies the area-fraction rule.
//
// Returns indices in ASCENDING order, matching the reference's `sorted(keep)`, because
// the order then decides which page FixPerspective treats as first when stitching a
// spread.
func selectPages(segments [][]imaging.Point, maxPages int) []int {
	areas := make([]float64, len(segments))
	for i, s := range segments {
		if len(s) >= 3 {
			areas[i] = contourArea(s)
		}
	}

	order := make([]int, len(areas))
	for i := range order {
		order[i] = i
	}
	// Descending by area, stable so equal areas keep their original relative order.
	sort.SliceStable(order, func(a, b int) bool { return areas[order[a]] > areas[order[b]] })

	if len(order) == 0 || areas[order[0]] <= 0 {
		return nil
	}
	limit := maxPages
	if limit < 1 {
		limit = 1
	}
	maxArea := areas[order[0]]
	keep := []int{order[0]}
	for _, idx := range order[1:] {
		if len(keep) >= limit {
			break
		}
		if areas[idx] >= secondSegmentAreaFrac*maxArea {
			keep = append(keep, idx)
		}
	}
	sort.Ints(keep)
	return keep
}

// contourArea is the shoelace formula, matching cv2.contourArea's magnitude.
//
// OpenCV returns the absolute area for a simple polygon, so the sign of the traversal
// direction is discarded — which is what the reference's comparisons assume.
func contourArea(pts []imaging.Point) float64 {
	n := len(pts)
	if n < 3 {
		return 0
	}
	var acc float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		acc += pts[i].X*pts[j].Y - pts[j].X*pts[i].Y
	}
	if acc < 0 {
		acc = -acc
	}
	return acc / 2
}

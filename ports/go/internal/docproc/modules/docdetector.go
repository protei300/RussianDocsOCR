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

// blankInk is BLANK_INK (doc_detector.py:15-26). A segment with (almost) no ink inside
// it is not a document page, however confident the model is: the white lid of a flatbed
// scanner next to a passport scores 0.90-0.96 as 'Document' and, being 3x larger than a
// page, used to win the area-based selection and push both real pages out (~20 of 100
// scans in the Damir set came out as an empty canvas). Ink is the mean gradient
// magnitude inside the eroded mask at ~400 px: measured 1.5-5.4 on lids, 19-101 on
// passport pages (19 = a badly blurred one), 24-43 on the mostly bare registration
// page. A blank segment is dropped only when another segment with real ink exists, so a
// lone blank sheet still goes through as before.
const blankInk = 8.0

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

// DocDetectorResult is PredictTransform's output: the stitched canvas plus, for a
// multi-page spread, the individual rectified pages, the (expanded, clipped) quad each
// came from, and where each landed on the canvas - exactly what DocDetector.predict_transform
// hands to the pipeline (pages/page_quads/page_placements), so the caller can run field
// detection PER PAGE (pipeline.py's _fields_from_pages) instead of on the half-size
// letterboxed spread. Pages is nil/empty for a single-page document; the CALLER owns
// every Image here (Canvas and each element of Pages), and Pages Images are separate
// Mats from Canvas even when len(Pages)==1 (StitchPages returns that one page AS the
// canvas in that case, i.e. Pages[0] and Canvas alias the same Mat then - Close only
// once).
type DocDetectorResult struct {
	Canvas      imaging.Image
	Segments    [][]imaging.Point
	Pages       []imaging.Image
	PageQuads   [][]imaging.Point
	Placements  []imaging.PagePlacement
}

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
func (d *DocDetector) PredictTransform(img imaging.Image, maxPages int) (DocDetectorResult, error) {
	_, segments, err := d.model.Predict(img)
	if err != nil {
		return DocDetectorResult{}, err
	}
	if len(segments) == 0 {
		return DocDetectorResult{Canvas: img.Clone()}, nil
	}

	// First drop segments without ink (a scanner lid, a blank sheet next to the
	// document): see blankInk. Then the area rule (doc_detector.py:125-131).
	inked := dropBlankSegments(img, segments)
	if len(inked) < len(segments) {
		filtered := make([][]imaging.Point, 0, len(inked))
		for _, i := range inked {
			filtered = append(filtered, segments[i])
		}
		segments = filtered
	}

	kept := selectPages(segments, maxPages)
	if len(kept) == 0 {
		return DocDetectorResult{Canvas: img.Clone()}, nil
	}

	chosen := make([][]imaging.Point, 0, len(kept))
	for _, i := range kept {
		chosen = append(chosen, segments[i])
	}

	// Pages are rectified separately and stitched afterwards (matching the reference's
	// doc_detector.predict_transform), so a two-page spread can be handed to the field
	// detector one page at a time - a 640x640 detector input gives a stitched spread
	// only half of itself per page.
	pages, quads := imaging.RectifyPages(img, chosen, imaging.DocMarginFrac)
	if len(pages) == 0 {
		return DocDetectorResult{Canvas: img.Clone(), Segments: chosen}, nil
	}
	if len(pages) < 2 {
		// One page: StitchPages returns it AS the canvas (no copy needed - nothing
		// downstream needs Pages for fewer than 2 pages anyway, both consumers gate
		// on len(Pages) >= 2), so hand it the original directly.
		canvas, _, ok := imaging.StitchPages(pages, quads, imaging.StackAuto)
		if !ok {
			return DocDetectorResult{Canvas: img.Clone(), Segments: chosen}, nil
		}
		return DocDetectorResult{Canvas: canvas, Segments: chosen}, nil
	}
	// 2+ pages: StitchPages CONSUMES (Closes) every page it is handed - but the
	// per-page field detector (pipeline._fields_from_pages) needs the pages
	// themselves to survive past this call, so it is handed CLONES, never the
	// originals. Skipping this clone is a double-free: Pages and the canvas each
	// point at freed memory the instant this returns (caught the hard way, see the
	// task log - this exact bug crashed every internal-passport document).
	stitchInput := make([]imaging.Image, len(pages))
	for i, p := range pages {
		stitchInput[i] = p.Clone()
	}
	canvas, placements, ok := imaging.StitchPages(stitchInput, quads, imaging.StackAuto)
	if !ok {
		closeImages(pages)
		return DocDetectorResult{Canvas: img.Clone(), Segments: chosen}, nil
	}
	return DocDetectorResult{
		Canvas: canvas, Segments: chosen,
		Pages: pages, PageQuads: quads, Placements: placements,
	}, nil
}

func closeImages(imgs []imaging.Image) {
	for _, im := range imgs {
		_ = im.Close()
	}
}

// dropBlankSegments returns the indices of the segments to keep: blank ones
// (ink < blankInk) are dropped when at least one inked segment exists.
// Port of doc_detector.drop_blank_segments (doc_detector.py:48-58).
func dropBlankSegments(img imaging.Image, segments [][]imaging.Point) []int {
	ink := make([]float64, len(segments))
	anyInked := false
	for i, s := range segments {
		ink[i] = imaging.SegmentInk(img, s)
		if ink[i] >= blankInk {
			anyInked = true
		}
	}
	keep := make([]int, 0, len(segments))
	for i := range segments {
		if !anyInked || ink[i] >= blankInk {
			keep = append(keep, i)
		}
	}
	return keep
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

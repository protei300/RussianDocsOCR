package modules

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Page is one passport page found inside a Document.
type Page struct {
	X1, Y1, X2, Y2 float64
	Conf           float64
}

// Document is one document the detector found in a frame, with the passport pages that
// lie inside it. Mirrors the dicts of group_documents:
// {'box': [x1, y1, x2, y2], 'conf': float, 'pages': [{'box', 'conf'}, ...]}.
type Document struct {
	X1, Y1, X2, Y2 float64
	Conf           float64
	Pages          []Page
}

// DocumentDetector finds every document lying in a frame, and the pages of a passport
// spread. Port of pipeline_modules/document_detector (decision #142, 2026-10-01). Uses
// the DocDetect artifact: yolo11s with two classes, `document` and `page`.
//
// The first stage of the pipeline: the type classifier, the border detector and
// everything after them then work on the crop of one document instead of the whole
// frame. A whole frame misleads the classifier whenever the document is a small part of
// it - a licence on an A4 scan was read as a passport from the white sheet around it
// (type by frame 50 % on licences, by crop 99 %).
type DocumentDetector struct {
	model *models.DetectionModel
}

// DocumentDetectorAvailable reports whether the weight set has DocDetect at all. A set
// of models-v8 or older does not, and the reference then reads the whole frame as before
// rather than refusing to start (Pipeline.__init__, `except FileNotFoundError`). A
// missing registration in models_path.yaml counts the same: both mean "this set has no
// document detector".
func DocumentDetectorAvailable(paths *config.ModelPaths, format string) bool {
	dir, err := paths.Dir("DocumentDetector", format)
	if err != nil {
		return false
	}
	for _, name := range []string{"model.json", "model.onnx"} {
		if info, err := os.Stat(filepath.Join(dir, name)); err != nil || info.IsDir() {
			return false
		}
	}
	return true
}

// NewDocumentDetector loads the detector. Call DocumentDetectorAvailable first when the
// weight set may predate it: this returns an error for a missing artifact.
func NewDocumentDetector(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*DocumentDetector, error) {

	dir, err := paths.Dir("DocumentDetector", format)
	if err != nil {
		return nil, err
	}
	m, err := models.LoadDetection(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: DocumentDetector: %w", err)
	}
	return &DocumentDetector{model: m}, nil
}

func (d *DocumentDetector) Close() error { return d.model.Close() }

// Predict detects the documents and passport pages in img and groups them. Boxes are in
// the coordinates of img; documents come out largest first.
func (d *DocumentDetector) Predict(img imaging.Image) ([]Document, error) {
	boxes, err := d.model.Predict(img)
	if err != nil {
		return nil, err
	}
	return GroupDocuments(boxes), nil
}

func boxArea(x1, y1, x2, y2 float64) float64 {
	w, h := x2-x1, y2-y1
	if w < 0 {
		w = 0
	}
	if h < 0 {
		h = 0
	}
	return w * h
}

// inside is _inside: true when at least share of the inner box lies within the outer.
func inside(inner, outer [4]float64, share float64) bool {
	w := min(inner[2], outer[2]) - max(inner[0], outer[0])
	h := min(inner[3], outer[3]) - max(inner[1], outer[1])
	if w <= 0 || h <= 0 {
		return false
	}
	a := boxArea(inner[0], inner[1], inner[2], inner[3])
	return w*h >= share*max(a, 1e-9)
}

// GroupDocuments turns detector boxes into documents, each with the pages that lie inside
// it. Port of group_documents.
//
// The detector has two classes: 'document' - whatever lies in the frame as one piece (a
// passport spread, a card, a single visible page) - and 'page', every visible page of an
// internal passport. A page belongs to the document it lies in (the first, largest one
// that holds at least 70 % of it); a document with no page inside is a single sheet.
// Documents come out largest first, which is the one the pipeline reads. Pages of one
// document are ordered by (y1, x1).
func GroupDocuments(boxes []postprocess.Box) []Document {
	var docs, pages []postprocess.Box
	for _, b := range boxes {
		switch b.Label {
		case "document":
			docs = append(docs, b)
		case "page":
			pages = append(pages, b)
		}
	}
	// Descending by area. A stable sort with a strict `>` keeps the original order of
	// equal areas, which is what Python's sort(reverse=True) does as well.
	sort.SliceStable(docs, func(i, j int) bool {
		return boxArea(docs[i].X1, docs[i].Y1, docs[i].X2, docs[i].Y2) >
			boxArea(docs[j].X1, docs[j].Y1, docs[j].X2, docs[j].Y2)
	})
	out := make([]Document, 0, len(docs))
	for _, d := range docs {
		out = append(out, Document{X1: d.X1, Y1: d.Y1, X2: d.X2, Y2: d.Y2, Conf: d.Conf})
	}
	for _, p := range pages {
		pb := [4]float64{p.X1, p.Y1, p.X2, p.Y2}
		for i := range out {
			d := &out[i]
			if inside(pb, [4]float64{d.X1, d.Y1, d.X2, d.Y2}, 0.7) {
				d.Pages = append(d.Pages, Page{X1: p.X1, Y1: p.Y1, X2: p.X2, Y2: p.Y2, Conf: p.Conf})
				break
			}
		}
	}
	for i := range out {
		pg := out[i].Pages
		sort.SliceStable(pg, func(a, b int) bool {
			if pg[a].Y1 != pg[b].Y1 {
				return pg[a].Y1 < pg[b].Y1
			}
			return pg[a].X1 < pg[b].X1
		})
	}
	return out
}

package pipeline

import (
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// DocumentCropMargin is Pipeline.DOCUMENT_CROP_MARGIN: the share of the document box's
// longer side added around it when it is cut from the frame. The border detector still
// needs a strip of background to find the edges, and the box itself can sit a few pixels
// inside the paper (handoff of the detector, 2026-10-01: ~3 %).
const DocumentCropMargin = 0.03

// documentsPayload renders the `documents` stage:
// [{box, conf, pages: [[x1, y1, x2, y2], ...]}], largest first, boxes on the input image
// rounded to 0.1 px. An empty list (JSON []) - not null - when the detector ran and found
// nothing, which is what the reference emits.
func documentsPayload(docs []modules.Document) []any {
	r1 := func(v float64) float64 { return tensor.RoundHalfEven(v, 1) }
	out := make([]any, 0, len(docs))
	for _, d := range docs {
		pages := make([]any, 0, len(d.Pages))
		for _, p := range d.Pages {
			pages = append(pages, []float64{r1(p.X1), r1(p.Y1), r1(p.X2), r1(p.Y2)})
		}
		out = append(out, map[string]any{
			"box":   []float64{r1(d.X1), r1(d.Y1), r1(d.X2), r1(d.Y2)},
			"conf":  tensor.RoundHalfEven(d.Conf, 3),
			"pages": pages,
		})
	}
	return out
}

// findDocuments is Pipeline._find_documents: the documents in the frame, largest first,
// boxes on the INPUT image. ran is false when this Recognizer has no document detector
// (a weight set of models-v8 or older) - the reference then emits nothing and reads the
// whole frame; with a detector that finds nothing it emits an empty list.
//
// The detector reads the frame at the processing size; its boxes are scaled back to the
// input image so the crop can be cut at full resolution - a licence on an A4 scan keeps
// its pixels instead of the ~300 left to it once the whole sheet is shrunk to img_size.
//
// The resized frame is returned too (caller owns it): with no document to cut, it IS the
// image the rest of the pipeline reads, byte for byte (the same FitToLongestSide of the
// same frame), so recomputing it would only cost time.
func (r *Recognizer) findDocuments(frame imaging.Image, imgSize int, timings *Timings,
	sink StageSink) (docs []modules.Document, small imaging.Image, err error) {

	small = imaging.FitToLongestSide(frame, imgSize)
	if r.docs == nil {
		return nil, small, nil
	}
	if err = timings.Time(StageDocumentDetector, func() (e error) {
		docs, e = r.docs.Predict(small)
		return
	}); err != nil {
		_ = small.Close()
		return nil, imaging.Image{}, err
	}
	sx := float64(frame.Width()) / float64(small.Width())
	sy := float64(frame.Height()) / float64(small.Height())
	for i := range docs {
		d := &docs[i]
		d.X1, d.Y1, d.X2, d.Y2 = d.X1*sx, d.Y1*sy, d.X2*sx, d.Y2*sy
		for j := range d.Pages {
			p := &d.Pages[j]
			p.X1, p.Y1, p.X2, p.Y2 = p.X1*sx, p.Y1*sy, p.X2*sx, p.Y2*sy
		}
	}
	if err = sink.Emit("documents", documentsPayload(docs)); err != nil {
		_ = small.Close()
		return nil, imaging.Image{}, err
	}
	return docs, small, nil
}

// documentCrop is Pipeline._document_crop: (x0, y0, x1, y1) on the input image - the
// document box plus its margin, clamped to the frame.
func documentCrop(frameW, frameH int, d modules.Document) (x0, y0, x1, y1 int) {
	m := DocumentCropMargin * math.Max(d.X2-d.X1, d.Y2-d.Y1)
	x0 = int(math.Max(0, math.Floor(d.X1-m)))
	y0 = int(math.Max(0, math.Floor(d.Y1-m)))
	x1 = int(math.Min(float64(frameW), math.Ceil(d.X2+m)))
	y1 = int(math.Min(float64(frameH), math.Ceil(d.Y2+m)))
	return
}

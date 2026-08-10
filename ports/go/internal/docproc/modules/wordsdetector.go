package modules

import (
	"fmt"
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

// PredictTransform returns the word boxes and their crops, left to right.
//
// The ordering is the one trap in this module. The reference sorts with
// `bbox.sort(key=lambda x: x[0])`, and Python's list.sort is STABLE — so words keep the
// reading-order sort the detector already applied (y1 then x1) whenever their x1 ties.
// `sort.Slice` in Go is NOT stable, nor is List.Sort in C#; two words sharing an x1 would
// swap and reorder two tokens of the joined field string. SliceStable, always
// (CONVENTIONS §6.3).
//
// An empty result is normal: the caller falls back to the whole patch, as the reference
// does.
func (d *WordsDetector) PredictTransform(patch imaging.Image) ([]postprocess.Box, []imaging.Image, error) {
	boxes, err := d.model.Predict(patch)
	if err != nil {
		return nil, nil, err
	}

	sort.SliceStable(boxes, func(a, b int) bool { return boxes[a].X1 < boxes[b].X1 })

	words := make([]imaging.Image, 0, len(boxes))
	for _, b := range boxes {
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

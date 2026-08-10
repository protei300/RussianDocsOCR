package pipeline

import (
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Stage keys for the timings report.
//
// These are WIRE NAMES and part of the compared contract: the checker ignores timing
// VALUES but still compares the timings key SET, so a port that invents its own stage
// names fails. In Python they fall out of `func.__name__` on the pipeline's private
// methods, which is why they are snake_case with a leading underscore — a Go-idiomatic
// renaming here would be a breaking change, not a tidy-up (CONVENTIONS §1).
const (
	StageDocTypeAngle      = "_doctype_angle"
	StageQualityAndBorders = "_quality_and_borders"
	StageGlare             = "_glare"
	StageBlur              = "_blur"
	StagePrintSpoofing     = "_print_spoofing"
	StageLcdSpoofing       = "_lcd_spoofing"
	StageDocDetector       = "_doc_detector"
	StageDeskew            = "_deskew"
	StageFieldsDetector    = "_fields_detector"
	StageSplitWords        = "_split_words"
	StageOcr               = "_ocr"
)

// Timings accumulates per-stage wall times.
//
// Port of PipelineResults._timings plus its `timings` property and add_concurrent_group.
type Timings struct {
	stages map[string]float64
	// concurrent names the stages that ran INSIDE a group. They keep their individual
	// times in the report but are excluded from `total`.
	concurrent map[string]bool
}

func NewTimings() *Timings {
	return &Timings{stages: map[string]float64{}, concurrent: map[string]bool{}}
}

// Record stores one stage's elapsed time, rounded to 4 places like the reference.
func (t *Timings) Record(stage string, elapsed time.Duration) {
	t.stages[stage] = tensor.RoundHalfEven(elapsed.Seconds(), 4)
}

// Time runs fn and records how long it took. The equivalent of _model_call, whose only
// job in the reference is exactly this.
func (t *Timings) Time(stage string, fn func() error) error {
	start := time.Now()
	err := fn()
	t.Record(stage, time.Since(start))
	return err
}

// RecordGroup stores a concurrent group: its own elapsed time under `name`, and each
// member's individual time marked as concurrent.
//
// Why the group's own time counts and the members' do not: summing overlapping members
// would put `total` ABOVE the real processing time and, worse, would keep it flat when
// parallelisation actually saves time — hiding the very thing the group exists for.
func (t *Timings) RecordGroup(name string, wall time.Duration, members map[string]time.Duration) {
	t.Record(name, wall)
	for stage, d := range members {
		t.Record(stage, d)
		t.concurrent[stage] = true
	}
}

// Report returns the stage times plus `total`.
//
// `total` covers TIMED STAGES ONLY — image loading and resizing are not a timed stage in
// the reference either, so wall-clock time is slightly higher. Reproduced rather than
// corrected: the number is on the wire.
func (t *Timings) Report() map[string]float64 {
	out := make(map[string]float64, len(t.stages)+1)
	var total float64
	for stage, v := range t.stages {
		out[stage] = v
		if !t.concurrent[stage] {
			total += v
		}
	}
	out["total"] = tensor.RoundHalfEven(total, 4)
	return out
}

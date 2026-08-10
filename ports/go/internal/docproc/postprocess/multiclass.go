package postprocess

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// MultiClass is plain argmax classification — port of MultiClassPostprocessing
// (postprocessing.py:257-291). Used by Blur and by the DocTypeAngles angle head.
type MultiClass struct {
	labels []string
}

func NewMultiClass(labels []string) (*MultiClass, error) {
	if len(labels) == 0 {
		return nil, fmt.Errorf("postprocess: MultiLabelClassification needs Labels")
	}
	return &MultiClass{labels: labels}, nil
}

func (m *MultiClass) Apply(out *tensor.Array, _ Context) (Result, error) {
	v, err := out.AsFloat32()
	if err != nil {
		return nil, fmt.Errorf("postprocess: multiclass input: %w", err)
	}
	if len(v) == 0 {
		return nil, fmt.Errorf("postprocess: empty score vector")
	}
	// tensor.Argmax takes the FIRST maximum, like numpy. `>=` would take the last and
	// change the predicted class on a tie.
	i := tensor.Argmax(v)
	if i >= len(m.labels) {
		return nil, fmt.Errorf("postprocess: class %d has no label (only %d declared)",
			i, len(m.labels))
	}
	// Max mirrors `probability.max(initial=0)`: zero for an empty vector rather than
	// an exception.
	return ClassResult{Label: m.labels[i], Confidence: float64(tensor.Max(v))}, nil
}

// BinaryClass is threshold classification over a single score — port of
// BinaryClassPostprocessing (postprocessing.py:42-75). Glare, LCDSpoofing,
// PrintSpoofing and AddressTextKind all use it.
//
// Note the returned confidence is the RAW score, not the winning class's probability:
// `p[0]` is reported whichever side of the threshold it lands on. Glare's caller then
// aggregates those raw scores across tiles, so "helpfully" flipping it to 1-p for the
// negative class would silently change the quality verdict.
type BinaryClass struct {
	labels    []string
	threshold float64
}

func NewBinaryClass(labels []string, threshold float64) (*BinaryClass, error) {
	if len(labels) != 2 {
		return nil, fmt.Errorf("postprocess: BinaryClassification needs 2 labels, got %d",
			len(labels))
	}
	return &BinaryClass{labels: labels, threshold: threshold}, nil
}

func (b *BinaryClass) Apply(out *tensor.Array, _ Context) (Result, error) {
	v, err := out.AsFloat32()
	if err != nil {
		return nil, fmt.Errorf("postprocess: binary input: %w", err)
	}
	if len(v) == 0 {
		return nil, fmt.Errorf("postprocess: empty score vector")
	}
	score := float64(v[0])
	label := b.labels[0]
	if score >= b.threshold {
		label = b.labels[1]
	}
	return ClassResult{Label: label, Confidence: score}, nil
}

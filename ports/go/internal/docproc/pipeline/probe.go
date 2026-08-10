package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// StageSink receives one named intermediate value per pipeline stage.
//
// Mirrors document_processing/pipeline/probe.py. Off by default: Pipeline holds a nil
// Sink and every emission site costs one nil check. Nothing may be computed for the
// sink's benefit — payloads are references to values the pipeline already produced —
// because instrumentation that changes the cost of the instrumented code is worse
// than none.
//
// Stage names are the fixed vocabulary in conformance/spec/stages.md. Additive only:
// goldens are keyed by them.
type StageSink interface {
	Emit(name string, payload any) error
}

// NullStageSink discards every emission.
//
// Mirrors probe.py's NullStageSink and exists for the same reason: it lets `recognize`
// share one code path with `probe` instead of maintaining a second walk of the pipeline
// that would be free to drift from it.
type NullStageSink struct{}

func (NullStageSink) Emit(string, any) error { return nil }

// ArrayPayload marks a payload that should be written as .npy rather than JSON.
//
// Go has no numpy, so the "is this an array?" question that Python answers with an
// isinstance check has to be explicit here. That is a deviation in mechanism, not in
// behaviour: the files produced are identical.
type ArrayPayload struct {
	Array *tensor.Array
}

// DirectoryStageSink writes each stage to <root>/<name>.npy or <name>.json, plus an
// ordered stages.json index — byte-compatible with the Python sink so the checker
// cannot tell which implementation produced a dump.
type DirectoryStageSink struct {
	root    string
	upto    string
	stopped bool
	index   []stageEntry
}

type stageEntry struct {
	Stage string `json:"stage"`
	File  string `json:"file"`
	Kind  string `json:"kind"`
	Dtype string `json:"dtype,omitempty"`
	Shape []int  `json:"shape,omitempty"`
}

// NewDirectoryStageSink creates the directory. An empty upto means "emit everything".
func NewDirectoryStageSink(root, upto string) (*DirectoryStageSink, error) {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, err
	}
	return &DirectoryStageSink{root: root, upto: upto}, nil
}

func (s *DirectoryStageSink) Emit(name string, payload any) error {
	if s.stopped {
		return nil
	}
	safe := filepath.Base(name) // stage names contain dots, never separators

	var entry stageEntry
	switch p := payload.(type) {
	case ArrayPayload:
		file := safe + ".npy"
		if err := tensor.Save(filepath.Join(s.root, file), p.Array); err != nil {
			return fmt.Errorf("probe: %s: %w", name, err)
		}
		entry = stageEntry{Stage: name, File: file, Kind: "npy",
			Dtype: dtypeName(p.Array.Dtype), Shape: p.Array.Shape}
	default:
		file := safe + ".json"
		blob, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			return fmt.Errorf("probe: %s: %w", name, err)
		}
		if err := os.WriteFile(filepath.Join(s.root, file), append(blob, '\n'), 0o644); err != nil {
			return fmt.Errorf("probe: %s: %w", name, err)
		}
		entry = stageEntry{Stage: name, File: file, Kind: "json"}
	}
	s.index = append(s.index, entry)

	// --upto stops AFTER the named stage, so the stage asked about is included.
	if s.upto != "" && name == s.upto {
		s.stopped = true
	}
	return nil
}

// Count reports how many stages were written.
func (s *DirectoryStageSink) Count() int { return len(s.index) }

// Close writes the ordered index.
func (s *DirectoryStageSink) Close() error {
	payload := map[string]any{
		"upto":          nullableString(s.upto),
		"stopped_early": s.stopped,
		"stages":        s.index,
	}
	blob, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(s.root, "stages.json"), append(blob, '\n'), 0o644)
}

// nullableString keeps the index byte-compatible with Python's, which writes JSON
// null rather than "" for an absent --upto.
func nullableString(v string) any {
	if v == "" {
		return nil
	}
	return v
}

// dtypeName maps our dtype tags to the numpy spellings the index records
// ("uint8", not "|u1"), matching what np.save reports on the Python side.
func dtypeName(d tensor.DType) string {
	switch d {
	case tensor.Uint8:
		return "uint8"
	case tensor.Float32:
		return "float32"
	case tensor.Float64:
		return "float64"
	case tensor.Int64:
		return "int64"
	default:
		return string(d)
	}
}

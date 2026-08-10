// Package models is the config-driven model loader: it reads each artifact's
// model.json and assembles a preprocessor, a session and a postprocessor.
//
// This data-driven design is the single most portable thing in the library. The same
// fourteen model.json files drive Python, Go, .NET, Kotlin and C++ **unchanged** — so
// the dispatch must stay a plain switch on the string tags, with no reflection, no
// attributes, no DI container and no self-registering init(). See CONVENTIONS §2 and
// MAPPING.md for the mandated case order.
package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
)

// Config mirrors model.json.
//
// Every optional numeric is a POINTER so that "absent" is distinguishable from zero.
// This is not defensive style: `BlankIndex` legitimately is 0, and `Threshold`
// defaults to 0.5 when missing — collapsing the two would silently change behaviour.
type Config struct {
	Name      string   `json:"Name"`
	File      string   `json:"File"`
	ModelType string   `json:"ModelType"`
	Runtime   string   `json:"Runtime"`
	Inputs    []Input  `json:"Inputs"`
	Outputs   []Output `json:"Outputs"`

	// Dir is where this config was read from; relative paths inside it resolve
	// against it (notably Output.Centers).
	Dir string `json:"-"`
}

type Input struct {
	Type          string    `json:"Type"`
	Name          string    `json:"Name"`
	Shape         []int     `json:"Shape"`
	Normalization []float64 `json:"Normalization"`
	PaddingSize   []int     `json:"PaddingSize"`
	PaddingColor  []int     `json:"PaddingColor"`
	Height        *int      `json:"Height"`
	ColorOrder    string    `json:"ColorOrder"`
	Dtype         string    `json:"Dtype"`
}

type Output struct {
	Type       string          `json:"Type"`
	Name       string          `json:"Name"`
	Shape      []int           `json:"Shape"`
	Labels     json.RawMessage `json:"Labels"`
	Threshold  *float64        `json:"Threshold"`
	IOU        *float64        `json:"IOU"`
	CLS        *float64        `json:"CLS"`
	MaskFilter *float64        `json:"MaskFilter"`
	Metric     string          `json:"Metric"`
	Centers    string          `json:"Centers"`
	Alphabet   string          `json:"Alphabet"`
	Script     string          `json:"Script"`
	Country    string          `json:"Country"`
	BlankIndex *int            `json:"BlankIndex"`
}

// LabelsAsStrings decodes Labels as strings.
//
// The field is heterogeneous across the shipped configs — Glare has
// ["NO","GLARE"] while the DocTypeAngles angle head has [0,90,180,270] — so it is
// kept raw and decoded on demand by whichever postprocessor knows what it wants.
// Numbers are formatted rather than rejected, so a caller that wants labels as text
// always gets them.
func (o Output) LabelsAsStrings() ([]string, error) {
	if len(o.Labels) == 0 {
		return nil, nil
	}
	var asStrings []string
	if err := json.Unmarshal(o.Labels, &asStrings); err == nil {
		return asStrings, nil
	}
	var asNumbers []float64
	if err := json.Unmarshal(o.Labels, &asNumbers); err != nil {
		return nil, fmt.Errorf("models: Labels is neither strings nor numbers: %s", o.Labels)
	}
	out := make([]string, len(asNumbers))
	for i, v := range asNumbers {
		out[i] = fmt.Sprintf("%g", v)
	}
	return out, nil
}

// LabelsAsInts decodes Labels as integers, for heads whose classes are angles.
func (o Output) LabelsAsInts() ([]int, error) {
	if len(o.Labels) == 0 {
		return nil, nil
	}
	var asNumbers []float64
	if err := json.Unmarshal(o.Labels, &asNumbers); err != nil {
		return nil, fmt.Errorf("models: Labels is not numeric: %s", o.Labels)
	}
	out := make([]int, len(asNumbers))
	for i, v := range asNumbers {
		out[i] = int(v)
	}
	return out, nil
}

// ThresholdOr returns Threshold, or the given default when the key is absent.
func (o Output) ThresholdOr(def float64) float64 {
	if o.Threshold == nil {
		return def
	}
	return *o.Threshold
}

// BlankIndexOr returns BlankIndex, or the given default. Needed because 0 is both the
// shipped value and Go's zero value.
func (o Output) BlankIndexOr(def int) int {
	if o.BlankIndex == nil {
		return def
	}
	return *o.BlankIndex
}

// CentersPath resolves Output.Centers against the config's directory.
//
// The shipped value is `"resources\\centers.npz"` — a WINDOWS separator inside data.
// On Linux a backslash is an ordinary filename character, so without normalisation
// DocTypeAngles fails to load: only inside a container, never on a Windows dev box.
// Python normalises in code rather than re-shipping the artifacts, and every port must
// do the same (CONVENTIONS §2).
func (c Config) CentersPath(o Output) string {
	if o.Centers == "" {
		return ""
	}
	return filepath.Join(c.Dir, config.NormalizeRelPath(o.Centers))
}

// ModelPath resolves Config.File against the config's directory.
func (c Config) ModelPath() string {
	return filepath.Join(c.Dir, config.NormalizeRelPath(c.File))
}

// LoadConfig reads model.json from a model directory.
func LoadConfig(dir string) (*Config, error) {
	path := filepath.Join(dir, "model.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("models: %w", err)
	}
	var cfg Config
	// A BOM is rejected rather than stripped: the shipped configs are BOM-free UTF-8,
	// and one that has acquired a BOM is corrupt rather than merely unusual. Failing
	// here beats mistaking it for a model problem three stages later (D-10).
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("models: %s: %w", path, err)
	}
	if len(cfg.Inputs) == 0 || len(cfg.Outputs) == 0 {
		return nil, fmt.Errorf("models: %s declares no inputs or outputs", path)
	}
	cfg.Dir = dir
	return &cfg, nil
}

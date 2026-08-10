// Package settingsschema is the server-owned schema for runtime-tunable settings.
//
// The server describes its own knobs — type, bounds, choices, help text, group — and the UI
// renders itself from that. The alternative, a hand-written form, means every new pipeline
// knob is a frontend change and the defaults end up duplicated on both sides, where they
// drift.
//
// Two properties matter beyond convenience:
//
//   - RestartRequired marks settings baked into the pipeline's construction. Changing
//     `ocr_mode` in the UI cannot affect a pipeline that is already built, and silently
//     pretending otherwise is worse than saying so.
//   - Values are STORED AS STRINGS (the store is JSON; SQL would use a key/value table).
//     Coercion and validation happen here, in one place, on the way IN.
//
// Port of service/core/settings_schema.py.
package settingsschema

import (
	"fmt"
	"strconv"
	"strings"
)

// Types a setting can have. String constants, because they go to the UI which switches on
// them to pick a widget.
const (
	TypeBool   = "bool"
	TypeInt    = "int"
	TypeFloat  = "float"
	TypeChoice = "choice"
	TypeStr    = "str"
)

// Def describes one setting.
//
// Nullable numeric bounds, so "no minimum" is distinguishable from "minimum zero" — a
// distinction that matters for docconf, whose valid range genuinely starts at 0.
type Def struct {
	Key             string   `json:"key"`
	Type            string   `json:"type"`
	Default         string   `json:"default"`
	Label           string   `json:"label"`
	Description     string   `json:"description"`
	Group           string   `json:"group"`
	MinValue        *float64 `json:"min_value"`
	MaxValue        *float64 `json:"max_value"`
	Choices         []string `json:"choices"`
	RestartRequired bool     `json:"restart_required"`
}

func f(v float64) *float64 { return &v }

// Schema is the ordered list. ORDER IS THE UI ORDER — the settings page renders it as given,
// so this is grouping and sequencing, not just a registry.
var Schema = []Def{
	{
		Key: "compute_device", Type: TypeChoice, Default: "auto", Label: "Compute device",
		Description: "GPU is used only when onnxruntime reports a CUDA provider AND the " +
			"pipeline actually builds on it. Applied at startup.",
		Group: "Recognition", Choices: []string{"auto", "cpu", "gpu"}, RestartRequired: true,
	},
	{
		Key: "ocr_mode", Type: TypeChoice, Default: "accurate", Label: "OCR engine",
		Description: "'accurate' is MobileNetV4 (best quality); 'fast' is EdgeNext. " +
			"Baked into the pipeline at construction.",
		Group: "Recognition", Choices: []string{"accurate", "fast"}, RestartRequired: true,
	},
	{
		Key: "docconf", Type: TypeFloat, Default: "0.5",
		Label:       "Document confidence threshold",
		Description: "Minimum confidence for accepting a detected document type.",
		Group:       "Recognition", MinValue: f(0.0), MaxValue: f(1.0),
	},
	{
		Key: "img_size", Type: TypeInt, Default: "1500", Label: "Processing image size",
		Description: "Longest side the image is scaled to before inference. Only ever " +
			"downscales — a smaller upload is not enlarged.",
		Group: "Recognition", MinValue: f(640), MaxValue: f(2560),
	},
	{
		Key: "job_timeout_sec", Type: TypeInt, Default: "120", Label: "Job timeout (seconds)",
		Description: "Typical processing is well under one second; this is a wedge detector, " +
			"not a performance limit.",
		Group: "Queue", MinValue: f(10), MaxValue: f(600),
	},
	{
		Key: "max_retries", Type: TypeInt, Default: "2", Label: "Max retries",
		Description: "Applies to transient failures only. A corrupt image fails immediately " +
			"and is never retried.",
		Group: "Queue", MinValue: f(0), MaxValue: f(5),
	},
	{
		Key: "log_level", Type: TypeChoice, Default: "INFO", Label: "Log level",
		Description: "", Group: "Service",
		Choices: []string{"DEBUG", "INFO", "WARNING", "ERROR"},
	},
}

// ByKey indexes the schema. The write whitelist is DERIVED from it rather than duplicated,
// so a new setting cannot be readable but not writable.
var ByKey = func() map[string]Def {
	m := make(map[string]Def, len(Schema))
	for _, d := range Schema {
		m[d.Key] = d
	}
	return m
}()

// IsUIKey reports whether a key may be written through the settings endpoint.
func IsUIKey(key string) bool { _, ok := ByKey[key]; return ok }

// ValidationError is a rejected value. Its message reaches the UI, so it names the bound
// that was violated rather than saying "invalid".
type ValidationError struct{ msg string }

func (e *ValidationError) Error() string { return e.msg }

func invalid(format string, args ...any) error {
	return &ValidationError{msg: fmt.Sprintf(format, args...)}
}

// Coerce validates against the schema and normalises to the stored string form.
//
// A KNOWN key with a bad value is an ERROR, not a silent drop: a UI that reports "saved"
// while discarding the value is worse than one that shows a message.
func Coerce(key string, value any) (string, error) {
	def, ok := ByKey[key]
	if !ok {
		return "", invalid("unknown setting %q", key)
	}
	raw := strings.TrimSpace(fmt.Sprint(value))

	switch def.Type {
	case TypeBool:
		switch strings.ToLower(raw) {
		case "1", "true", "yes", "on":
			return "1", nil
		default:
			return "0", nil
		}

	case TypeInt, TypeFloat:
		number, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return "", invalid("%s must be a number, got %q", key, raw)
		}
		if def.MinValue != nil && number < *def.MinValue {
			return "", invalid("%s must be >= %v", key, *def.MinValue)
		}
		if def.MaxValue != nil && number > *def.MaxValue {
			return "", invalid("%s must be <= %v", key, *def.MaxValue)
		}
		if def.Type == TypeInt {
			return strconv.Itoa(int(number)), nil
		}
		// 'g' rather than a fixed precision, so 0.5 stores as "0.5" and not "0.500000"
		// — the stored form is compared against the previous value to decide whether a
		// restart is required, and a formatting change would look like an edit.
		return strconv.FormatFloat(number, 'g', -1, 64), nil

	case TypeChoice:
		if len(def.Choices) > 0 {
			for _, c := range def.Choices {
				if c == raw {
					return raw, nil
				}
			}
			return "", invalid("%s must be one of %s", key, strings.Join(def.Choices, ", "))
		}
		return raw, nil
	}
	return raw, nil
}

// Typed converts a stored string to the value the worker wants.
//
// A malformed STORED value must not take the worker down — it falls back to the schema
// default. That is the opposite policy from Coerce, deliberately: bad input is rejected at
// the boundary, but a store that somehow holds a bad value must still yield a running
// service.
func Typed(key, stored string) any {
	def, ok := ByKey[key]
	if !ok {
		return stored
	}
	raw := stored
	if raw == "" {
		raw = def.Default
	}
	switch def.Type {
	case TypeBool:
		switch strings.ToLower(raw) {
		case "1", "true", "yes", "on":
			return true
		default:
			return false
		}
	case TypeInt:
		if n, err := strconv.ParseFloat(raw, 64); err == nil {
			return int(n)
		}
		if n, err := strconv.ParseFloat(def.Default, 64); err == nil {
			return int(n)
		}
		return 0
	case TypeFloat:
		if n, err := strconv.ParseFloat(raw, 64); err == nil {
			return n
		}
		if n, err := strconv.ParseFloat(def.Default, 64); err == nil {
			return n
		}
		return 0.0
	}
	return raw
}

// TypedInt, TypedFloat and TypedString are the worker's accessors.
//
// They exist because `any` at a call site means a type assertion at a call site, and the
// worker reads six settings on every loop iteration. Wrong assertions there would panic in
// the one goroutine that must not.
func TypedInt(key, stored string, fallback int) int {
	if v, ok := Typed(key, stored).(int); ok {
		return v
	}
	return fallback
}

func TypedFloat(key, stored string, fallback float64) float64 {
	if v, ok := Typed(key, stored).(float64); ok {
		return v
	}
	return fallback
}

func TypedString(key, stored, fallback string) string {
	if v, ok := Typed(key, stored).(string); ok && v != "" {
		return v
	}
	return fallback
}

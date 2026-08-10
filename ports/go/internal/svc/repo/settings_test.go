package repo

import (
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/settingsschema"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

func newStore(t *testing.T) store.DocumentStore {
	t.Helper()
	s, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// The precedence rule: STORED VALUE, then ENVIRONMENT, then SCHEMA DEFAULT.
//
// This is tested rather than asserted because the reference had it missing in two different
// ways and both were real: the worker's value ignored the environment, so COMPUTE_DEVICE=cpu
// was logged and then disregarded; and the settings page read the schema default, so it showed
// "auto" for a service actually running on CPU.
func TestSettingsPrecedence(t *testing.T) {
	db := newStore(t)

	// 1. Nothing stored, nothing in the environment: the schema default.
	bare := config.Defaults()
	if got := SettingValue(db, bare, "compute_device"); got != "auto" {
		t.Errorf("schema default = %q, want auto", got)
	}

	// 2. The environment overrides the schema default.
	env := config.Defaults()
	env.ComputeDevice = "cpu"
	if got := SettingValue(db, env, "compute_device"); got != "cpu" {
		t.Errorf("with the environment set, got %q, want cpu", got)
	}

	// 3. A stored value overrides BOTH — an operator's explicit choice wins.
	if _, _, err := BulkUpdateSettings(db, env, map[string]any{"compute_device": "gpu"}); err != nil {
		t.Fatal(err)
	}
	if got := SettingValue(db, env, "compute_device"); got != "gpu" {
		t.Errorf("with a stored value, got %q, want gpu", got)
	}
}

// A bad environment value must not take the service down, but it must not be silent either:
// it falls back to the schema default and logs. Tested for the fallback; the log line is
// visible in the ring buffer.
func TestInvalidEnvironmentValueFallsBack(t *testing.T) {
	db := newStore(t)
	bad := config.Defaults()
	bad.ComputeDevice = "quantum"
	if got := SettingValue(db, bad, "compute_device"); got != "auto" {
		t.Errorf("an invalid env value gave %q, want the schema default auto", got)
	}
}

// A float coming from the environment must compare equal to the same value coming from the
// settings page, or every start would report a spurious restart_required.
func TestFloatFromEnvironmentMatchesStoredForm(t *testing.T) {
	db := newStore(t)
	env := config.Defaults()
	env.Docconf = 0.7

	if got := SettingValue(db, env, "docconf"); got != "0.7" {
		t.Fatalf("docconf from the environment = %q, want 0.7", got)
	}
	_, restart, err := BulkUpdateSettings(db, env, map[string]any{"docconf": 0.7})
	if err != nil {
		t.Fatal(err)
	}
	if len(restart) != 0 {
		t.Errorf("writing the same value reported restart_required=%v", restart)
	}
}

// restart_required fires only for keys that are BAKED INTO the pipeline, and only when the
// value actually changed.
func TestRestartRequiredOnlyForRealChanges(t *testing.T) {
	db := newStore(t)
	cfg := config.Defaults()

	_, restart, err := BulkUpdateSettings(db, cfg, map[string]any{
		"ocr_mode": "fast", // restart_required in the schema
		"docconf":  0.6,    // not
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(restart) != 1 || restart[0] != "ocr_mode" {
		t.Fatalf("restart_required = %v, want [ocr_mode]", restart)
	}

	// Writing the SAME value again is not a change.
	_, restart, err = BulkUpdateSettings(db, cfg, map[string]any{"ocr_mode": "fast"})
	if err != nil {
		t.Fatal(err)
	}
	if len(restart) != 0 {
		t.Fatalf("rewriting an unchanged value reported %v", restart)
	}
}

// Unknown keys are dropped SILENTLY — that is the whitelist doing its job — while a known key
// with a bad value is an ERROR, because a UI reporting "saved" while discarding the value is
// worse than a message.
func TestUnknownKeysDroppedKnownBadValuesRejected(t *testing.T) {
	db := newStore(t)
	cfg := config.Defaults()

	values, _, err := BulkUpdateSettings(db, cfg, map[string]any{"not_a_setting": "x"})
	if err != nil {
		t.Fatalf("an unknown key must be ignored, not an error: %v", err)
	}
	if _, present := values["not_a_setting"]; present {
		t.Error("an unknown key was stored")
	}

	if _, _, err := BulkUpdateSettings(db, cfg, map[string]any{"docconf": 5}); err == nil {
		t.Fatal("an out-of-range value was accepted")
	}
	if _, _, err := BulkUpdateSettings(db, cfg, map[string]any{"ocr_mode": "telepathy"}); err == nil {
		t.Fatal("a value outside the choice list was accepted")
	}
}

func TestCoerceAndTyped(t *testing.T) {
	if _, err := settingsschema.Coerce("img_size", "640"); err != nil {
		t.Errorf("640 is the minimum and must be accepted: %v", err)
	}
	if _, err := settingsschema.Coerce("img_size", "639"); err == nil {
		t.Error("639 is below the minimum and must be rejected")
	}
	// An int setting given a float is truncated rather than rejected: a slider sends 1500.0.
	if got, err := settingsschema.Coerce("img_size", 1500.0); err != nil || got != "1500" {
		t.Errorf("Coerce(1500.0) = %q, %v", got, err)
	}
	// A malformed STORED value falls back to the default rather than taking the worker down.
	if got := settingsschema.TypedInt("img_size", "banana", 0); got != 1500 {
		t.Errorf("TypedInt on a bad stored value = %d, want the schema default 1500", got)
	}
}

// Every schema default must itself be valid. Trivially true today, and cheap insurance
// against a typo in a new entry that would otherwise only surface at runtime.
func TestEverySchemaDefaultIsValid(t *testing.T) {
	for _, def := range settingsschema.Schema {
		if _, err := settingsschema.Coerce(def.Key, def.Default); err != nil {
			t.Errorf("%s: default %q is not valid: %v", def.Key, def.Default, err)
		}
	}
}

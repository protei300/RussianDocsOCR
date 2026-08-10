package repo

import (
	"log/slog"
	"strconv"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/settingsschema"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// Runtime settings are stored as strings and validated against the schema.
//
// The worker reads them fresh on every loop iteration, so an operator change takes effect
// without a restart — except for the ones flagged RestartRequired, which are baked into the
// pipeline's construction.

// EffectiveDefault is the default for a key AFTER the environment has had its say.
//
// Precedence is STORED VALUE → ENVIRONMENT → SCHEMA DEFAULT, resolved here so no caller can
// get it wrong. Every schema key that is also configurable by environment shares its name
// with the config field, so the two tiers line up by construction rather than through a
// hand-maintained table.
//
// The reference had this layering missing in two different ways and both were real: the
// worker's value ignored the environment entirely, so COMPUTE_DEVICE=cpu was logged and then
// disregarded; and the settings page read the schema default, so it displayed "auto" for a
// service actually running on CPU. Bypassing this function reintroduces both.
func EffectiveDefault(cfg config.Settings, key string) string {
	def, ok := settingsschema.ByKey[key]
	if !ok {
		return ""
	}
	envValue := envValueFor(cfg, key)
	if envValue == "" {
		return def.Default
	}
	coerced, err := settingsschema.Coerce(key, envValue)
	if err != nil {
		// A bad environment value must not take the service down, but silence would hide
		// a deployment mistake behind a plausible default.
		slog.Warn("[SETTINGS] ignoring invalid value from the environment",
			"key", strings.ToUpper(key), "value", envValue, "using", def.Default, "err", err)
		return def.Default
	}
	return coerced
}

// envValueFor maps a schema key onto its config field.
//
// An explicit switch rather than reflection: it is seven lines, it is the same shape in C#
// and Kotlin, and a reflective version would silently return nothing the moment somebody
// renames a field.
func envValueFor(cfg config.Settings, key string) string {
	switch key {
	case "compute_device":
		return cfg.ComputeDevice
	case "ocr_mode":
		return cfg.OcrMode
	case "docconf":
		return trimFloat(cfg.Docconf)
	case "img_size":
		return itoa(cfg.ImgSize)
	case "job_timeout_sec":
		return itoa(cfg.JobTimeoutSec)
	case "max_retries":
		return itoa(cfg.MaxRetries)
	case "log_level":
		return cfg.LogLevel
	default:
		return ""
	}
}

func itoa(v int) string { return strconv.Itoa(v) }

// trimFloat renders a float the way Coerce will store it, so a value that came from the
// environment and one that came from the settings page compare equal.
func trimFloat(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

// AllSettings returns current values, with environment-or-schema defaults for anything unset.
func AllSettings(db store.DocumentStore, cfg config.Settings) map[string]string {
	stored := db.AllSettings()
	out := make(map[string]string, len(settingsschema.Schema))
	for _, def := range settingsschema.Schema {
		if v, ok := stored[def.Key]; ok {
			out[def.Key] = v
			continue
		}
		out[def.Key] = EffectiveDefault(cfg, def.Key)
	}
	return out
}

// SettingValue returns the stored string for one key, resolved through the same precedence.
//
// The worker's accessor, paired with settingsschema.TypedInt and friends. There is no
// `fallback` parameter for known keys on purpose: the environment layer above is
// authoritative, precisely so a caller passing the wrong fallback cannot desync the runtime
// from what the settings page displays.
func SettingValue(db store.DocumentStore, cfg config.Settings, key string) string {
	if _, ok := settingsschema.ByKey[key]; !ok {
		return ""
	}
	if v, ok := db.AllSettings()[key]; ok {
		return v
	}
	return EffectiveDefault(cfg, key)
}

// BulkUpdateSettings validates and stores. Returns all values and the keys needing a restart.
//
// UNKNOWN keys are dropped silently — that is the whitelist doing its job. KNOWN keys with
// bad values return an error, because a UI reporting "saved" while discarding the value is
// worse than an error message.
func BulkUpdateSettings(db store.DocumentStore, cfg config.Settings,
	values map[string]any) (map[string]string, []string, error) {

	accepted := map[string]string{}
	var restartRequired []string
	current := db.AllSettings()

	// Iterated in SCHEMA order rather than map order, so the restart_required list and any
	// error message are deterministic across runs. Go randomises map iteration, and a
	// nondeterministic error message is a bad thing to debug from a screenshot.
	for _, def := range settingsschema.Schema {
		value, present := values[def.Key]
		if !present {
			continue
		}
		normalised, err := settingsschema.Coerce(def.Key, value)
		if err != nil {
			return nil, nil, err
		}
		accepted[def.Key] = normalised

		previous, ok := current[def.Key]
		if !ok {
			previous = EffectiveDefault(cfg, def.Key)
		}
		if def.RestartRequired && normalised != previous {
			restartRequired = append(restartRequired, def.Key)
		}
	}

	if len(accepted) > 0 {
		if _, err := db.SetSettings(accepted); err != nil {
			return nil, nil, err
		}
	}
	return AllSettings(db, cfg), restartRequired, nil
}

// SettingsSchema is the schema as the UI receives it.
func SettingsSchema() []settingsschema.Def { return settingsschema.Schema }

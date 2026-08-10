// Package config is the environment tier of configuration.
//
// Two tiers, split by WHO changes them and how often:
//
//   - here (environment / .env) — secrets and anything that cannot change without a
//     restart: the PIN, the JWT secret, the default API key, the data directory;
//   - the SETTINGS STORE (svc/settingsschema + svc/repo) — operator-tunable knobs the
//     worker re-reads every loop, so a change applies without bouncing the service.
//
// Secrets never appear in the settings store and are never returned by any endpoint. That
// split is the whole reason AuthPin, JwtSecret and DefaultApiKey live here and nowhere
// else.
//
// Port of service/core/config.py.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Settings is the environment tier, resolved once at startup.
//
// No struct tags and no reflection-driven binding: the reference uses pydantic-settings,
// but a hand-written Load is what the .NET and Kotlin ports can copy line for line, and it
// keeps the DEFAULTS visible next to the field they belong to instead of in a framework.
type Settings struct {
	// --- Auth --------------------------------------------------------------
	// AuthPin protects the WEBSITE only. API endpoints authenticate with API keys.
	AuthPin          string
	JwtSecret        string
	JwtAlgorithm     string
	JwtExpireMinutes int

	// DefaultApiKey is the bootstrap key: always present, never deletable — without it
	// a restart (which wipes runtime-created keys) would leave the API with no way in.
	//
	// EMPTY BY DEFAULT, on purpose. A hardcoded fallback would mean every deployment
	// that forgot to set this shares one publicly-known key. Empty means a random key is
	// generated at startup and logged; set the variable when an integration needs one
	// that survives restarts.
	DefaultApiKey string

	// --- Storage -----------------------------------------------------------
	// DatabaseConnectionString selects the metadata store. Empty falls back to a
	// temporary on-disk store wiped at every start — fine for a demo, useless for
	// anything real, so startup says so loudly.
	DatabaseConnectionString string

	// DataDir holds uploaded originals, canvases and thumbnails. Artifacts stay on the
	// filesystem in BOTH modes: multi-megabyte PNGs do not belong in a database row.
	DataDir string

	// DataWipeOnStart wipes DataDir at startup. Only meaningful in temporary mode —
	// with a database configured the rows outlive the process, so wiping the images
	// would leave every stored document pointing at a missing file, and
	// storage.ResolveMode forces this off.
	//
	// Note that "no Docker volume" alone does NOT make the directory ephemeral:
	// `docker restart` keeps the writable layer. This flag is what makes the promise
	// true.
	DataWipeOnStart bool
	MaxUploadMB     int

	// SeedSamples queues the anonymised documents from samples/ when the store is empty,
	// so the log demonstrates real results instead of showing nothing. ONLY ever
	// repository samples, never user uploads. Negative disables; 0 means all available.
	SeedSamples int

	// --- Recognition -------------------------------------------------------
	ComputeDevice    string // auto | cpu | gpu
	ModelFormat      string // ONNX | OpenVINO
	OcrMode          string // accurate | fast  ('legacy' was removed in 3.0.0)
	PipelinePoolSize int
	// WarmupImage must be an anonymised repository sample. NEVER a real user document:
	// warmup re-reads this file at every start.
	WarmupImage string

	// --- Worker ------------------------------------------------------------
	JobTimeoutSec int
	MaxRetries    int
	Docconf       float64
	ImgSize       int

	// --- Ops ---------------------------------------------------------------
	LogLevel           string
	CorsAllowedOrigins string
	GitCommit          string
}

// Defaults returns the settings with no environment applied.
//
// Separate from Load so tests and the settings schema can both see the baseline without
// touching the process environment.
func Defaults() Settings {
	return Settings{
		AuthPin:          "1234",
		JwtSecret:        "changeme-in-production",
		JwtAlgorithm:     "HS256",
		JwtExpireMinutes: 480, // 8 h, one working day
		DefaultApiKey:    "",

		DatabaseConnectionString: "",
		DataDir:                  "data",
		DataWipeOnStart:          true,
		MaxUploadMB:              20,
		SeedSamples:              0,

		ComputeDevice:    "auto",
		ModelFormat:      "ONNX",
		OcrMode:          "accurate",
		PipelinePoolSize: 1,
		WarmupImage:      "",

		JobTimeoutSec: 120,
		MaxRetries:    2,
		Docconf:       0.5,
		ImgSize:       1500,

		LogLevel:           "INFO",
		CorsAllowedOrigins: "",
		GitCommit:          "unknown",
	}
}

// Load applies the environment over the defaults.
//
// Variable names are UPPER_SNAKE of the field, which is what pydantic-settings derives on
// the Python side — so a compose file works unchanged against either implementation. The
// one exception is the connection string, which the reference aliases explicitly so it
// reads unambiguously in a compose file; both spellings are accepted here.
func Load() (Settings, error) {
	s := Defaults()
	var errs []string

	str := func(key string, dst *string) {
		if v, ok := os.LookupEnv(key); ok {
			*dst = v
		}
	}
	num := func(key string, dst *int) {
		if v, ok := os.LookupEnv(key); ok {
			n, err := strconv.Atoi(strings.TrimSpace(v))
			if err != nil {
				errs = append(errs, fmt.Sprintf("%s=%q is not an integer", key, v))
				return
			}
			*dst = n
		}
	}
	flt := func(key string, dst *float64) {
		if v, ok := os.LookupEnv(key); ok {
			f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil {
				errs = append(errs, fmt.Sprintf("%s=%q is not a number", key, v))
				return
			}
			*dst = f
		}
	}
	boolean := func(key string, dst *bool) {
		if v, ok := os.LookupEnv(key); ok {
			// Accepts the spellings pydantic accepts, so a compose file is portable.
			switch strings.ToLower(strings.TrimSpace(v)) {
			case "1", "true", "yes", "on":
				*dst = true
			case "0", "false", "no", "off":
				*dst = false
			default:
				errs = append(errs, fmt.Sprintf("%s=%q is not a boolean", key, v))
			}
		}
	}

	str("AUTH_PIN", &s.AuthPin)
	str("JWT_SECRET", &s.JwtSecret)
	str("JWT_ALGORITHM", &s.JwtAlgorithm)
	num("JWT_EXPIRE_MINUTES", &s.JwtExpireMinutes)
	str("DEFAULT_API_KEY", &s.DefaultApiKey)

	str("DATABASE_CONNECTIONSTRING", &s.DatabaseConnectionString)
	str("RUSSIANDOCS_DATABASE_CONNECTIONSTRING", &s.DatabaseConnectionString)
	str("DATA_DIR", &s.DataDir)
	boolean("DATA_WIPE_ON_START", &s.DataWipeOnStart)
	num("MAX_UPLOAD_MB", &s.MaxUploadMB)
	num("SEED_SAMPLES", &s.SeedSamples)

	str("COMPUTE_DEVICE", &s.ComputeDevice)
	str("MODEL_FORMAT", &s.ModelFormat)
	str("OCR_MODE", &s.OcrMode)
	num("PIPELINE_POOL_SIZE", &s.PipelinePoolSize)
	str("WARMUP_IMAGE", &s.WarmupImage)

	num("JOB_TIMEOUT_SEC", &s.JobTimeoutSec)
	num("MAX_RETRIES", &s.MaxRetries)
	flt("DOCCONF", &s.Docconf)
	num("IMG_SIZE", &s.ImgSize)

	str("LOG_LEVEL", &s.LogLevel)
	str("CORS_ALLOWED_ORIGINS", &s.CorsAllowedOrigins)
	str("GIT_COMMIT", &s.GitCommit)

	if len(errs) > 0 {
		return s, fmt.Errorf("config: %s", strings.Join(errs, "; "))
	}
	return s, nil
}

// CorsOrigins splits the comma-separated list, dropping blanks.
func (s Settings) CorsOrigins() []string {
	var out []string
	for _, part := range strings.Split(s.CorsAllowedOrigins, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// MaxUploadBytes is the upload ceiling in bytes.
func (s Settings) MaxUploadBytes() int64 { return int64(s.MaxUploadMB) * 1024 * 1024 }

// RepoRoot locates the repository root, for finding samples/ and web/.
//
// Resolved from the executable's location, then from the working directory, because the
// service runs from three places with three different layouts: a container (/app), a `go run`
// from ports/go, and a built binary in ports/go/bin. Returns "" when nothing matches, and
// callers degrade rather than fail — a missing samples/ costs a cold first document, not a
// broken service.
func (s Settings) RepoRoot() string {
	if v := os.Getenv("RDOCS_REPO_ROOT"); v != "" {
		return v
	}
	candidates := []string{"."}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		candidates = append(candidates, dir, filepath.Join(dir, ".."),
			filepath.Join(dir, "..", "..", "..", ".."))
	}
	candidates = append(candidates, filepath.Join("..", ".."))
	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(c, "document_processing", "models")); err == nil {
			abs, err := filepath.Abs(c)
			if err == nil {
				return abs
			}
		}
	}
	return ""
}

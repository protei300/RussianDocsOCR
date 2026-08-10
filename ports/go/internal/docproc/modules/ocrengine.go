package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// OcrTier selects the backbone: 'accurate' (MobileNetV4) or 'fast' (EdgeNext).
//
// A STRING constant, not an iota enum, because it appears in configuration and on the
// wire and must read identically in all four ports (CONVENTIONS §1).
type OcrTier string

const (
	OcrAccurate OcrTier = "accurate"
	OcrFast     OcrTier = "fast"
)

// cfgKeyByTier maps a tier to the models_path.yaml key, so a module keeps a stable
// model_name while loading a tier-specific artifact.
var (
	cyrillicCfgKey = map[OcrTier]string{
		OcrAccurate: "OCRCyrillicAccurate",
		OcrFast:     "OCRCyrillicFast",
	}
	latinCfgKey = map[OcrTier]string{
		OcrAccurate: "OCRLatinAccurate",
		OcrFast:     "OCRLatinFast",
	}
)

// ruNameFields get stray leading dots stripped. Named exactly as the reference's
// _RU_NAME_FIELDS, because the membership is the behaviour.
var ruNameFields = []string{"Last_name_ru", "First_name_ru", "Birth_place_ru",
	"Living_region_ru", "Middle_name_ru", "Issue_organization_ru"}

// dateFields get date normalisation from the Latin engine.
var dateFields = []string{"Issue_date", "Expiration_date", "Birth_date"}

// OcrEngine is one recognition engine. Port of pipeline_modules/ocr_cyrillic and
// ocr_latin, which differ ONLY in which artifact they load and which corrections
// fix_errors applies — so they are one type with a script field rather than two
// near-identical files.
//
// This is a deliberate divergence from the reference's two classes (D-11). It removes a
// copy of the predict/close plumbing without hiding anything: the two fix_errors tables
// are the actual difference and they are right here, side by side.
type OcrEngine struct {
	script string
	model  *models.Model
}

// NewOcrCyrillic loads the Cyrillic engine: Russian names, places, organisations, plus
// digits and punctuation.
func NewOcrCyrillic(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int, tier OcrTier) (*OcrEngine, error) {
	return newOcrEngine(root, paths, format, device, threads, tier, "cyrillic", cyrillicCfgKey)
}

// NewOcrLatin loads the Latin engine: Latin letters and digits, and every date field.
func NewOcrLatin(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int, tier OcrTier) (*OcrEngine, error) {
	return newOcrEngine(root, paths, format, device, threads, tier, "latin", latinCfgKey)
}

func newOcrEngine(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int, tier OcrTier, script string,
	keys map[OcrTier]string) (*OcrEngine, error) {

	key, ok := keys[tier]
	if !ok {
		return nil, fmt.Errorf("modules: OCR tier %q is not %q or %q",
			tier, OcrAccurate, OcrFast)
	}
	dir, err := paths.Dir(key, format)
	if err != nil {
		return nil, err
	}
	m, err := models.Load(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: %s OCR: %w", script, err)
	}
	return &OcrEngine{script: script, model: m}, nil
}

func (e *OcrEngine) Close() error { return e.model.Close() }

// Predict recognises one word patch.
//
// The decode (greedy CTC plus per-step alphabet masking) happens inside the model's
// postprocessor, so this returns the finished string.
func (e *OcrEngine) Predict(patch imaging.Image) (string, error) {
	results, err := e.model.Predict(patch)
	if err != nil {
		return "", err
	}
	if len(results) != 1 {
		return "", fmt.Errorf("modules: %s OCR returned %d outputs, want 1",
			e.script, len(results))
	}
	text, ok := results[0].(postprocess.TextResult)
	if !ok {
		return "", fmt.Errorf("modules: %s OCR output is %T, want TextResult",
			e.script, results[0])
	}
	return text.Text, nil
}

// FixErrors applies the field-specific corrections for this engine's script.
//
// Split by script and not by field alone: the same field name can reach either engine,
// and Sex_ru must become М/Ж while Sex_en must become M/F.
func (e *OcrEngine) FixErrors(fieldType, text string) string {
	if e.script == "cyrillic" {
		if fieldType == "Sex_ru" {
			return CheckRusSex(text)
		}
		if contains(ruNameFields, fieldType) {
			return StripEdgeDots(text)
		}
		return text
	}

	if contains(dateFields, fieldType) {
		// The reference wraps this in `except ValueError: return text`; the port folds
		// that into CheckDdmmyyyy, which returns the input unchanged rather than raising.
		return CheckDdmmyyyy(text)
	}
	if fieldType == "Sex_en" {
		return CheckEnSex(text)
	}
	if fieldType == "Driver_class" {
		return CheckDriverClass(text)
	}
	return text
}

func contains(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}

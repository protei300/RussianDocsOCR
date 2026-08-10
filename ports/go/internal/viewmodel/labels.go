// Package viewmodel turns recognition results into the client-facing JSON.
//
// D-01: in Python this lives in `service/ml/transform.py`, but in a port it belongs on
// the LIBRARY side — the conformance CLI needs it and must not depend on an HTTP service.
// The normative contract is conformance/spec/viewmodel.md; transform.py is the reference
// IMPLEMENTATION. When the two disagree that is a bug report against Python.
package viewmodel

import "sort"

// fieldLabels are the English UI names for the library's raw labels.
// Port of service/ml/labels.py::FIELD_LABELS.
//
// The UI is English by product decision while the values are Cyrillic. The `_ru`/`_en`
// suffix is a SCRIPT marker, not part of a field's meaning, so it is surfaced separately
// via FieldScript rather than baked into the display string — which is why several
// entries here are deliberately identical.
var fieldLabels = map[string]string{
	"Last_name_ru":            "Last name",
	"Last_name_en":            "Last name",
	"First_name_ru":           "First name",
	"First_name_en":           "First name",
	"Middle_name_ru":          "Middle name",
	"Middle_name_en":          "Middle name",
	"Birth_date":              "Date of birth",
	"Birth_place_ru":          "Place of birth",
	"Birth_place_en":          "Place of birth",
	"Sex_ru":                  "Sex",
	"Sex_en":                  "Sex",
	"Licence_number":          "Document number",
	"Issue_date":              "Date of issue",
	"Expiration_date":         "Valid until",
	"Issue_organization_ru":   "Issuing authority",
	"Issue_organization_en":   "Issuing authority",
	"Issue_organisation_code": "Authority code",
	"Living_region_ru":        "Place of residence",
	"Living_region_en":        "Place of residence",
	"Driver_class":            "Categories",
	"Face":                    "Photo",
	"Signature":               "Signature",
	// Synthetic keys produced by the address branch, not TextFields classes.
	"Address":                 "Registration address",
	"Address_has_handwritten": "Contains handwriting",
}

// nonTextLabels are detected but never OCR'd.
//
// Still worth drawing — "did it find the photo?" is a real diagnostic question — but they
// need different visual treatment, because a UI expecting every box to carry a value
// renders them as broken rows.
var nonTextLabels = map[string]bool{"Face": true, "Signature": true}

// monospaceFields are rendered monospace.
//
// An explicit allowlist of digit/Latin-only fields rather than "monospace anything that
// looks like data": monospace earns its keep on digit runs, but on ALL-CAPS Cyrillic it
// squeezes Ш, Щ, Ж and Ы and legibility drops.
var monospaceFields = map[string]bool{
	"Licence_number":          true,
	"Issue_date":              true,
	"Expiration_date":         true,
	"Birth_date":              true,
	"Issue_organisation_code": true,
}

// Reading order, per base document type.
//
// A map has no order, and even the library's insertion order is not the order a human
// reads a document in. Boxes could be sorted geometrically, but that breaks down for
// fields with no box and for split fields whose parts sit far apart. An explicit order is
// boring and correct.
var (
	passportOrder = []string{
		"Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
		"Birth_date", "Birth_place_ru",
		"Licence_number", "Issue_date", "Expiration_date",
		"Issue_organization_ru", "Issue_organisation_code",
		"Living_region_ru",
	}
	extPassportOrder = []string{
		"Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
		"Middle_name_ru", "Middle_name_en", "Sex_ru", "Sex_en",
		"Birth_date", "Birth_place_ru", "Birth_place_en",
		"Licence_number", "Issue_date", "Expiration_date",
		"Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
		"Living_region_ru", "Living_region_en",
	}
	dlOrder = []string{
		"Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
		"Middle_name_ru", "Middle_name_en",
		"Birth_date", "Birth_place_ru", "Birth_place_en",
		"Licence_number", "Issue_date", "Expiration_date",
		"Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
		"Living_region_ru", "Living_region_en",
		"Driver_class",
	}
	snilsOrder = []string{
		"Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
		"Birth_date", "Birth_place_ru",
		"Licence_number", "Issue_date",
	}
	addrOrder = []string{"Address", "Address_has_handwritten"}
)

// fieldOrder is keyed by the BASE document type — the part before the trailing year,
// which is how the pipeline itself dispatches.
var fieldOrder = map[string][]string{
	"INTPASSPORT":     passportOrder,
	"INTPASSPORTADDR": addrOrder,
	"EXTPASSPORT":     extPassportOrder,
	"EXTPASSPORTBIO":  extPassportOrder,
	"DL":              dlOrder,
	"SNILS":           snilsOrder,
}

// BaseDocType strips the era suffix: "INTPASSPORT_2011" -> "INTPASSPORT".
//
// Unlike the pipeline's own split this never fails on a label with no underscore: here an
// unexpected label just means "no known ordering", which is survivable.
func BaseDocType(docType string) string {
	for i := len(docType) - 1; i >= 0; i-- {
		if docType[i] == '_' {
			return docType[:i]
		}
	}
	return docType
}

// DocTypeEra is the era suffix alone: "INTPASSPORT_2011" -> "2011", and nil when the
// label has no underscore.
//
// "INTPASSPORTADDR_ALL" yields "ALL", which the UI renders as a chip like any other era.
// That is intentional — it is what the model reports.
func DocTypeEra(docType string) *string {
	for i := len(docType) - 1; i >= 0; i-- {
		if docType[i] == '_' {
			era := docType[i+1:]
			return &era
		}
	}
	return nil
}

// FieldDisplay is the English UI label, falling back to the raw name.
func FieldDisplay(name string) string {
	if v, ok := fieldLabels[name]; ok {
		return v
	}
	return name
}

// FieldScript is "ru", "en" or "num" — what the field's VALUE is.
//
// Drives two client decisions that would otherwise be hardcoded per language: the `lang`
// attribute (font matching, spell-check, screen readers) and monospace rendering.
//
// The monospace check comes FIRST: Birth_date has no script suffix, but Licence_number
// would fall through to the "ru" default without it.
func FieldScript(name string) string {
	if monospaceFields[name] {
		return "num"
	}
	if hasSuffix(name, "_ru") {
		return "ru"
	}
	if hasSuffix(name, "_en") {
		return "en"
	}
	return "ru"
}

// OrderFields sorts field names into document reading order: known fields in their
// canonical order, then anything unrecognised alphabetically.
//
// The alphabetical tail is why the API degrades gracefully instead of DROPPING data when
// the library gains a field before this file is updated.
func OrderFields(docType string, names []string) []string {
	canonical := fieldOrder[BaseDocType(docType)]
	rank := make(map[string]int, len(canonical))
	for i, n := range canonical {
		rank[n] = i
	}

	var known, unknown []string
	for _, n := range names {
		if _, ok := rank[n]; ok {
			known = append(known, n)
		} else {
			unknown = append(unknown, n)
		}
	}
	sort.SliceStable(known, func(a, b int) bool { return rank[known[a]] < rank[known[b]] })
	sort.Strings(unknown)
	return append(known, unknown...)
}

func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

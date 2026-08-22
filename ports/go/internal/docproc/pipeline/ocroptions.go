package pipeline

import "strings"

// OcrOptions says which fields a document type has, which of them need splitting into
// words, and which script each is read with.
// Port of OCROptionsClass and its subclasses (pipeline.py:57-161).
//
// Plain data with a factory rather than a class hierarchy: the reference's subclasses
// only override class attributes, so inheritance buys nothing a struct literal does not,
// and a struct maps to a C# record and a Kotlin data class without a decision.
type OcrOptions struct {
	// NeededSplit lists fields handed to the word detector. A field not listed here
	// is OCR'd as one whole patch.
	NeededSplit []string
	// EnFields and RuFields select the OCR engine. A field in NEITHER list is
	// detected but never read — Face and Signature are the obvious cases.
	EnFields []string
	RuFields []string
	// NeedsLicenceRotation rotates the licence-number patch 90 degrees
	// counter-clockwise. Internal passports print it sideways.
	NeedsLicenceRotation bool
	// HasAddress routes through the address-line path instead of text fields.
	HasAddress bool
}

// SplitDocType splits a '<TYPE>_<YEAR>' label into its bare type and year.
//
// The bare type is not cosmetic: the OCR routing compares it with `== "SNILS"`, which the
// raw label never equals. A label with no underscore yields itself and an empty year --
// the reference tolerates that explicitly, and the port must too, because Python's
// rsplit-and-unpack raises ValueError on it (pipeline.py:522).
func SplitDocType(label string) (bare, year string) {
	if i := strings.LastIndex(label, "_"); i >= 0 {
		return label[:i], label[i+1:]
	}
	return label, ""
}

// MakeOcrOptions selects the options for a document-type label.
//
// The case ORDER is load-bearing and must not be sorted: 'intpassportaddr' CONTAINS
// 'intpassport', so checking the shorter one first would route every registration page
// down the ordinary text-field path (pipeline.py:91).
//
// An unknown type yields EMPTY options rather than an error, matching the reference's
// deliberate choice: the pipeline then produces no OCR fields for it instead of crashing
// on the next attribute access.
func MakeOcrOptions(docType string) OcrOptions {
	t := strings.ToLower(docType)
	switch {
	case strings.Contains(t, "intpassportaddr"):
		return OcrOptions{HasAddress: true}
	case strings.Contains(t, "intpassport"):
		return OcrOptions{
			NeededSplit: []string{"Licence_number", "Birth_place_ru", "Issue_organization_ru"},
			// MRZ is read by the Latin engine and is NOT in NeededSplit: the zone is
			// detected one box per LINE, and each line must reach the engine whole -
			// splitting it at its filler runs would destroy the fixed 44-character
			// layout the check digits are computed over (pipeline.py:126).
			EnFields: []string{"Issue_date", "Expiration_date",
				"Birth_date", "Issue_organisation_code", "MRZ"},
			// Licence_number is CYRILLIC-routed although it is digits only: the Latin
			// engine reads the passport's red '3' as '8' at p=0.94..1.00, and the
			// Cyrillic engine reads the same crops correctly (issue #12). Matches the
			// reference, OCROptionsINTPassport in pipeline.py.
			RuFields: []string{"Last_name_ru", "First_name_ru", "Birth_place_ru",
				"Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru",
				"Licence_number"},
			NeedsLicenceRotation: true,
		}
	case strings.Contains(t, "extpassport"):
		return OcrOptions{
			NeededSplit: []string{"Licence_number", "Birth_place_ru", "Birth_place_en"},
			// MRZ: Latin engine, never split - see intpassport above.
			EnFields: []string{"Last_name_en", "First_name_en", "Issue_date",
				"Expiration_date", "Birth_date", "Birth_place_en", "Issue_organization_en",
				"Living_region_en", "Sex_en", "Issue_organisation_code", "Middle_name_en",
				"MRZ"},
			// Licence_number: Cyrillic-routed, same reason as intpassport above.
			RuFields: []string{"Last_name_ru", "First_name_ru", "Birth_place_ru",
				"Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru",
				"Licence_number"},
		}
	case strings.Contains(t, "dl"):
		return OcrOptions{
			NeededSplit: []string{"Licence_number", "Driver_class", "Birth_place_ru",
				"Birth_place_en", "Living_region_ru", "Living_region_en"},
			EnFields: []string{"Last_name_en", "First_name_en", "Licence_number", "Issue_date",
				"Expiration_date", "Driver_class", "Birth_date", "Birth_place_en",
				"Issue_organization_en", "Living_region_en", "Issue_organisation_code",
				"Middle_name_en"},
			RuFields: []string{"Last_name_ru", "First_name_ru", "Birth_place_ru",
				"Issue_organization_ru", "Living_region_ru", "Middle_name_ru"},
		}
	case strings.Contains(t, "snils"):
		return OcrOptions{
			NeededSplit: []string{"Last_name_ru", "First_name_ru", "Licence_number",
				"Issue_date", "Birth_date", "Birth_place_ru", "Middle_name_ru", "Sex_ru"},
			EnFields: []string{"Licence_number", "Issue_date", "Birth_date"},
			RuFields: []string{"Last_name_ru", "First_name_ru", "Birth_place_ru",
				"Middle_name_ru", "Sex_ru"},
		}
	case strings.Contains(t, "birthcert"):
		// Birth certificates (OCROptionsBIRTHCERT, pipeline.py:156). ONE branch for
		// both blank generations - BIRTHCERT_1998 and BIRTHCERT_2018 (order 167/2018:
		// worded birth date, parents' birth dates, place of issue, 21-digit act
		// number) - because the dispatcher never sees the year suffix.
		//
		// EnFields is EMPTY and that is the whole point: every date on these forms is
		// spelled out in Cyrillic («16 декабря 2001», «15 октября 2020 г.»), and the
		// Cyrillic engine reads the 1998 digit-only birth date just as well - the same
		// precedent as the passport Licence_number (issue #12). Licence_number mixes a
		// Roman-numeral series with Cyrillic and «№»; routed Cyrillic as the lesser
		// evil, same as the reference.
		return OcrOptions{
			NeededSplit: []string{"First_name_ru", "Birth_place_ru", "Issue_organization_ru",
				"Issue_date", "Licence_number",
				"Father_first_middle_ru", "Mother_first_middle_ru",
				"Birth_date", "Father_birth_date", "Mother_birth_date",
				"Issue_place_ru"},
			EnFields: []string{},
			RuFields: []string{"Last_name_ru", "First_name_ru", "Birth_place_ru",
				"Issue_organization_ru", "Issue_date", "Licence_number",
				"Father_last_name_ru", "Father_first_middle_ru",
				"Mother_last_name_ru", "Mother_first_middle_ru",
				"Birth_date", "Father_birth_date", "Mother_birth_date",
				"Issue_place_ru", "Act_number"},
		}
	}
	return OcrOptions{}
}

// IsOcrField reports whether a label is read at all, by either engine.
func (o OcrOptions) IsOcrField(label string) bool {
	return contains(o.EnFields, label) || contains(o.RuFields, label)
}

// NeedsSplit reports whether a field is handed to the word detector.
func (o OcrOptions) NeedsSplit(label string) bool { return contains(o.NeededSplit, label) }

func contains(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}

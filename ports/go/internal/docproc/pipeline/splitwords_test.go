package pipeline

import (
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

type spec struct {
	label string
	conf  float64
}

func fieldsOf(specs ...spec) []modules.Field {
	out := make([]modules.Field, len(specs))
	for i, s := range specs {
		out[i] = modules.Field{Box: postprocess.Box{Label: s.label, Conf: s.conf}}
	}
	return out
}

// The internal passport prints its series and number twice, so the detector returns two
// Licence_number boxes and only the more confident one may be read.
func TestDuplicateFieldIndicesKeepsHighestConfidence(t *testing.T) {
	drop := duplicateFieldIndices(fieldsOf(
		spec{"Last_name_ru", 0.9},
		spec{"Licence_number", 0.71},
		spec{"Licence_number", 0.88},
	))
	if len(drop) != 1 || !drop[1] {
		t.Fatalf("expected to drop only index 1, got %v", drop)
	}
}

// On EQUAL confidence Python's max() returns the FIRST maximum, so the earlier detection
// survives. A port using >= would keep the later one and read a different crop.
func TestDuplicateFieldIndicesTieKeepsEarlier(t *testing.T) {
	drop := duplicateFieldIndices(fieldsOf(
		spec{"Licence_number", 0.8},
		spec{"Licence_number", 0.8},
	))
	if len(drop) != 1 || !drop[1] {
		t.Fatalf("a tie must keep the EARLIER detection; dropped %v", drop)
	}
}

// Only fields that must be unique are deduplicated. Two Birth_place_ru boxes are a
// legitimate multi-line field and both contribute words.
func TestDuplicateFieldIndicesIgnoresOtherLabels(t *testing.T) {
	drop := duplicateFieldIndices(fieldsOf(
		spec{"Birth_place_ru", 0.9},
		spec{"Birth_place_ru", 0.5},
	))
	if len(drop) != 0 {
		t.Fatalf("non-unique labels must not be deduplicated, dropped %v", drop)
	}
}

// The dedup covers the FMS code as well as the licence number, and each label is treated
// independently.
func TestDuplicateFieldIndicesPerLabel(t *testing.T) {
	drop := duplicateFieldIndices(fieldsOf(
		spec{"Issue_organisation_code", 0.4},
		spec{"Licence_number", 0.6},
		spec{"Issue_organisation_code", 0.9},
		spec{"Licence_number", 0.5},
	))
	if len(drop) != 2 || !drop[0] || !drop[3] {
		t.Fatalf("expected to drop indices 0 and 3, got %v", drop)
	}
}

// 'intpassportaddr' CONTAINS 'intpassport', so the longer label must be tested first.
// Getting this backwards routes every registration page down the text-field path, which
// then produces no address at all.
func TestMakeOcrOptionsChecksAddrBeforeIntpassport(t *testing.T) {
	addr := MakeOcrOptions("INTPASSPORTADDR_ALL")
	if !addr.HasAddress {
		t.Fatal("INTPASSPORTADDR must take the address path")
	}
	if addr.NeedsLicenceRotation {
		t.Fatal("the address page has no rotated licence number")
	}

	plain := MakeOcrOptions("INTPASSPORT_2011")
	if plain.HasAddress {
		t.Fatal("an ordinary internal passport must NOT take the address path")
	}
	if !plain.NeedsLicenceRotation {
		t.Fatal("the internal passport prints its licence number sideways")
	}
}

// An unrecognised type yields empty options rather than an error, matching the
// reference's deliberate choice: the pipeline produces no OCR fields instead of crashing
// on the next attribute access.
func TestMakeOcrOptionsUnknownTypeIsEmptyNotFatal(t *testing.T) {
	o := MakeOcrOptions("SOMETHING_ELSE")
	if o.HasAddress || o.NeedsLicenceRotation ||
		len(o.EnFields) != 0 || len(o.RuFields) != 0 || len(o.NeededSplit) != 0 {
		t.Fatalf("unknown type must yield empty options, got %+v", o)
	}
	if o.IsOcrField("Last_name_ru") {
		t.Fatal("empty options must claim no OCR fields")
	}
}

// Detected but never read: Face and Signature have no place in either engine's list, and
// a port that treated "detected" as "read" would OCR a photograph.
func TestIsOcrFieldExcludesVisualClasses(t *testing.T) {
	o := MakeOcrOptions("DL_2020")
	for _, label := range []string{"Face", "Signature"} {
		if o.IsOcrField(label) {
			t.Fatalf("%s must not be an OCR field", label)
		}
	}
	if !o.IsOcrField("Last_name_ru") || !o.IsOcrField("Licence_number") {
		t.Fatal("real fields must be claimed")
	}
}

// SNILS splits nearly everything; the external passport splits only three fields. The
// distinction decides whether a field reaches the word detector at all.
func TestNeedsSplit(t *testing.T) {
	snils := MakeOcrOptions("SNILS_1996")
	if !snils.NeedsSplit("Issue_date") {
		t.Fatal("SNILS dates are split: they read as '31 октября 1998'")
	}
	ext := MakeOcrOptions("EXTPASSPORT_2003")
	if ext.NeedsSplit("Issue_date") {
		t.Fatal("the external passport does not split its dates")
	}
	if !ext.NeedsSplit("Birth_place_en") {
		t.Fatal("Birth_place_en is split on the external passport")
	}
}

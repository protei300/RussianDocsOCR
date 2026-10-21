package pipeline

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Mirrors tests/test_field_checks.py of the reference. Every check has a negative control: a
// deliberately wrong value must come out as `mismatch`. The zones are the fictitious documents of
// tests/test_mrz_validation.py; the page values next to them are what those zones say, so a
// consistent document is the starting point and each test changes exactly one thing.

const (
	extMrz = "P<RUSROGOVA<<IRINA<<<<<<<<<<<<<<<<<<<<<<<<<<\n" +
		"7648201572RUS8803148F3105205<<<<<<<<<<<<<<04"
	intMrz = "PNRUSLE4KIN<<PAVEL<PETROVI3<<<<<<<<<<<<<<<<<\n" +
		"4503182940RUS8704234M<<<<<<<7070515770021<46"
)

type ocrMap = map[string]string

// with is a copy of base with the given pairs set (dict(base, **changes)).
func with(base ocrMap, kv ...string) ocrMap {
	out := ocrMap{}
	for k, v := range base {
		out[k] = v
	}
	for i := 0; i < len(kv); i += 2 {
		out[kv[i]] = kv[i+1]
	}
	return out
}

func intOcr() ocrMap {
	return ocrMap{
		"MRZ": intMrz, "Licence_number": "45 07 318294", "Birth_date": "23.04.1987",
		"Issue_date": "15.05.2007", "Issue_organisation_code": "770-021", "Sex_ru": "МУЖ.",
		"Last_name_ru": "ЛЕШКИН", "First_name_ru": "ПАВЕЛ", "Middle_name_ru": "ПЕТРОВИЧ",
		"Birth_place_ru": "ГОР. МОСКВА",
	}
}

// The printed number differs from the valid zone in its last digit (see test_mrz_validation).
func extOcr() ocrMap {
	return ocrMap{
		"MRZ": extMrz, "Licence_number": "76 4820154", "Birth_date": "14.03.1988",
		"Expiration_date": "20.05.2031", "Last_name_en": "ROGOVA", "First_name_en": "IRINA",
		"Sex_en": "F",
	}
}

func extOcrConsistent() ocrMap { return with(extOcr(), "Licence_number", "76 4820157") }

var testToday = time.Date(2026, 10, 19, 0, 0, 0, 0, time.UTC)

func checksOf(ocr ocrMap, docType string) map[string]FieldVerdict {
	return FieldChecks(ocr, docType, testToday)
}

func statuses(ocr ocrMap, docType string) map[string]string {
	out := map[string]string{}
	for k, v := range checksOf(ocr, docType) {
		out[k] = v.Status
	}
	return out
}

func hasCheck(v FieldVerdict, id, result string) bool {
	for _, c := range v.Checks {
		if c.ID == id && c.Result == result {
			return true
		}
	}
	return false
}

func findCheck(t *testing.T, v FieldVerdict, id string) FieldCheck {
	t.Helper()
	for _, c := range v.Checks {
		if c.ID == id {
			return c
		}
	}
	t.Fatalf("no check %s in %+v", id, v.Checks)
	return FieldCheck{}
}

func wantStatus(t *testing.T, got map[string]string, field, want string) {
	t.Helper()
	if got[field] != want {
		t.Errorf("%s: status %q, want %q", field, got[field], want)
	}
}

func allStatuses(t *testing.T, got map[string]string, want string, except ...string) {
	t.Helper()
Next:
	for k, v := range got {
		for _, e := range except {
			if k == e {
				continue Next
			}
		}
		if v != want {
			t.Errorf("%s: status %q, want %q", k, v, want)
		}
	}
}

// --- shape -----------------------------------------------------------------------------

func TestEveryFieldGetsAnEntryEvenEmptyOnes(t *testing.T) {
	ocr := with(intOcr(), "Living_region_ru", "")
	result := checksOf(ocr, "INTPASSPORT_2011")
	if len(result) != len(ocr) {
		t.Fatalf("%d entries for %d fields", len(result), len(ocr))
	}
	got := result["Living_region_ru"]
	if got.Status != FieldUnchecked || got.Checks == nil || len(got.Checks) != 0 ||
		got.Notes == nil || len(got.Notes) != 0 {
		t.Errorf("empty field: %+v", got)
	}
	blob, _ := json.Marshal(got)
	if string(blob) != `{"status":"unchecked","checks":[],"notes":[]}` {
		t.Errorf("wire form of an empty entry: %s", blob)
	}
}

func TestReadingIsNeverChanged(t *testing.T) {
	ocr := with(intOcr(), "Licence_number", "45 07 318295")
	before := with(ocr)
	checksOf(ocr, "INTPASSPORT_2011")
	if !reflect.DeepEqual(ocr, before) {
		t.Error("the reading was modified")
	}
}

func TestUnmeasuredFamiliesAreUncheckedThroughout(t *testing.T) {
	for _, docType := range []string{"INTPASSPORTADDR_ALL", "DLBACK_ALL", "BIRTHCERT_2018", "STS_2019", "NONE", ""} {
		got := statuses(ocrMap{"Birth_date": "31.02.2020", "MRZ": intMrz}, docType)
		allStatuses(t, got, FieldUnchecked)
	}
}

func TestFamilyDoesNotConfuseTheRegistrationPageOrTheLicenceBack(t *testing.T) {
	for docType, want := range map[string]string{
		"INTPASSPORT_2011": "INTPASSPORT", "INTPASSPORTADDR_ALL": "", "DL_2020": "DL",
		"DLBACK_ALL": "", "EXTPASSPORTBIO_2007": "EXTPASSPORT", "SNILS_1996": "SNILS",
	} {
		if got := FieldFamily(docType); got != want {
			t.Errorf("FieldFamily(%q) = %q, want %q", docType, got, want)
		}
	}
}

// --- internal passport zone ------------------------------------------------------------

func TestConsistentInternalPassportIsVerifiedFieldByField(t *testing.T) {
	got := statuses(intOcr(), "INTPASSPORT_2011")
	wantStatus(t, got, "Birth_place_ru", FieldUnchecked) // nothing to compare it with
	allStatuses(t, got, FieldVerified, "Birth_place_ru")
}

func TestInternalNumberIsRebuiltFromTwoPlacesOfTheZone(t *testing.T) {
	// The defect this work found: 10 printed digits against the zone's 9.
	if v := checksOf(intOcr(), "INTPASSPORT_2011")["Licence_number"]; !hasCheck(v, "MRZ-2", "ok") {
		t.Errorf("MRZ-2 not ok: %+v", v.Checks)
	}
}

func TestInternalNumberNegativeControls(t *testing.T) {
	// a digit of the number, and the series' fourth digit (lives in the optional data)
	for _, wrong := range []string{"45 07 318295", "45 08 318294"} {
		v := checksOf(with(intOcr(), "Licence_number", wrong), "INTPASSPORT_2011")["Licence_number"]
		if v.Status != FieldMismatch {
			t.Errorf("%s: %s", wrong, v.Status)
		}
		mrz2 := findCheck(t, v, "MRZ-2")
		if mrz2.Reference == nil || *mrz2.Reference != "4507318294" {
			t.Errorf("%s: reference %v", wrong, mrz2.Reference)
		}
	}
}

func TestInternalIssueDateAndCodeNegativeControls(t *testing.T) {
	got := statuses(with(intOcr(), "Issue_date", "16.05.2007", "Issue_organisation_code", "770-022"), "INTPASSPORT_2011")
	wantStatus(t, got, "Issue_date", FieldMismatch)
	wantStatus(t, got, "Issue_organisation_code", FieldMismatch)
}

func TestInternalNamesThroughTheOrder279Table(t *testing.T) {
	// Ч -> 3, Ш -> 4 follow the order's table; one changed letter must break it
	for name, wrong := range map[string]string{"Last_name_ru": "ЛЕЩКИН", "First_name_ru": "ПАВЕК", "Middle_name_ru": "ПЕТРОВИ"} {
		wantStatus(t, statuses(with(intOcr(), name, wrong), "INTPASSPORT_2011"), name, FieldMismatch)
	}
}

func TestYoAndYeAreTheSameLetterForTheZone(t *testing.T) {
	// Whether a clerk typed Ё is not shared between page and zone.
	ocr := with(intOcr(), "MRZ", strings.Replace(intMrz, "LE4KIN", "L24KIN", 1))
	wantStatus(t, statuses(ocr, "INTPASSPORT_2011"), "Last_name_ru", FieldVerified)
	wantStatus(t, statuses(with(ocr, "Last_name_ru", "ЛЁШКИН"), "INTPASSPORT_2011"), "Last_name_ru", FieldVerified)
}

func TestSexNegativeControl(t *testing.T) {
	wantStatus(t, statuses(with(intOcr(), "Sex_ru", "ЖЕН."), "INTPASSPORT_2011"), "Sex_ru", FieldMismatch)
	// the first letter decides, in either script and either case
	wantStatus(t, statuses(with(intOcr(), "Sex_ru", "муж"), "INTPASSPORT_2011"), "Sex_ru", FieldVerified)
}

func TestFailedCheckDigitTakesTheZonePartOutOfPlay(t *testing.T) {
	// A zone part whose own digit fails is not a reference: the page field loses that comparison
	// (unchecked), it does not become a mismatch.
	zone := strings.Replace(intMrz, "RUS8704234", "RUS8704244", 1) // birth date, digit kept
	result := checksOf(with(intOcr(), "MRZ", zone), "INTPASSPORT_2011")
	if result["MRZ"].Status != FieldMismatch {
		t.Errorf("MRZ: %s", result["MRZ"].Status)
	}
	for _, c := range result["Birth_date"].Checks {
		if c.ID == "MRZ-3" {
			t.Error("MRZ-3 present on a failed digit")
		}
	}
	if result["Birth_date"].Status != FieldUnchecked {
		t.Errorf("Birth_date: %s", result["Birth_date"].Status)
	}
	if result["Licence_number"].Status != FieldVerified { // other parts still hold
		t.Errorf("Licence_number: %s", result["Licence_number"].Status)
	}
}

func TestMalformedZoneIsAMismatchAndNoReference(t *testing.T) {
	result := checksOf(with(intOcr(), "MRZ", intMrz[:len(intMrz)-1]), "INTPASSPORT_2011")
	if result["MRZ"].Status != FieldMismatch || result["Licence_number"].Status != FieldUnchecked {
		t.Errorf("MRZ %s, number %s", result["MRZ"].Status, result["Licence_number"].Status)
	}
}

func TestLine1WithAnImpossibleCharacterIsNoReference(t *testing.T) {
	// Seen on samples/: the letter O of a surname read as the digit 0 in the zone. The page reading is
	// right; it must not be blamed for the zone's misreading.
	zone := strings.Replace(extMrz, "ROGOVA", "R0GOVA", 1)
	result := checksOf(with(extOcrConsistent(), "MRZ", zone), "EXTPASSPORTBIO_2007")
	if result["MRZ"].Status != FieldMismatch || !hasCheck(result["MRZ"], "MRZ-9", "mismatch") {
		t.Errorf("MRZ: %+v", result["MRZ"])
	}
	if result["Last_name_en"].Status != FieldUnchecked {
		t.Errorf("Last_name_en: %s", result["Last_name_en"].Status)
	}
	if result["Licence_number"].Status != FieldVerified { // line 2 still holds
		t.Errorf("Licence_number: %s", result["Licence_number"].Status)
	}
}

func TestInternalZoneDigitsOutsideTheTableAreImpossible(t *testing.T) {
	for _, wrong := range []string{"LE0KIN", "LE1KIN", "LE5KIN"} {
		result := checksOf(with(intOcr(), "MRZ", strings.Replace(intMrz, "LE4KIN", wrong, 1)), "INTPASSPORT_2011")
		if result["MRZ"].Status != FieldMismatch || result["Last_name_ru"].Status != FieldUnchecked {
			t.Errorf("%s: MRZ %s, name %s", wrong, result["MRZ"].Status, result["Last_name_ru"].Status)
		}
	}
}

func TestZoneIsVerifiedOnlyThroughItsNames(t *testing.T) {
	// Line-2 check digits say nothing about line 1.
	noNames := intOcr()
	for _, n := range []string{"Last_name_ru", "First_name_ru", "Middle_name_ru"} {
		delete(noNames, n)
	}
	wantStatus(t, statuses(noNames, "INTPASSPORT_2011"), "MRZ", FieldUnchecked)
	wantStatus(t, statuses(intOcr(), "INTPASSPORT_2011"), "MRZ", FieldVerified)
	// a page name that disagrees withholds the confirmation, it does not blame the zone
	got := statuses(with(intOcr(), "First_name_ru", "ПАВЕК"), "INTPASSPORT_2011")
	wantStatus(t, got, "MRZ", FieldUnchecked)
	wantStatus(t, got, "First_name_ru", FieldMismatch)
}

func TestZoneLayoutFollowsTheClassifierNotTheZoneTypeCharacters(t *testing.T) {
	// 'PH' also fails the line-1 form (MRZ-9), so names are not compared; the optional data of
	// line 2 is still read with the internal layout
	zone := "PH" + intMrz[2:]
	wantStatus(t, statuses(with(intOcr(), "MRZ", zone), "INTPASSPORT_2011"), "Issue_date", FieldVerified)
}

func TestDigitsAgreeingDoNotHideStrayCharacters(t *testing.T) {
	// Damir set: a glare read as quotes inside the number, a doubled hyphen in the code. The
	// zone comparisons look at digits only; the form catches it.
	got := statuses(with(intOcr(), "Licence_number", `88 12 106"436""`, "Issue_organisation_code", "770--021"), "INTPASSPORT_2011")
	wantStatus(t, got, "Licence_number", FieldMismatch)
	wantStatus(t, got, "Issue_organisation_code", FieldMismatch)
	// whitespace is not part of the form
	wantStatus(t, statuses(with(intOcr(), "Licence_number", "45  07 318294"), "INTPASSPORT_2011"), "Licence_number", FieldVerified)
}

func TestExternalNumberForm(t *testing.T) {
	wantStatus(t, statuses(with(extOcrConsistent(), "Licence_number", "764820157"), "EXTPASSPORTBIO_2007"),
		"Licence_number", FieldMismatch)
}

// --- external passport zone ------------------------------------------------------------

func TestConsistentExternalPassport(t *testing.T) {
	allStatuses(t, statuses(with(extOcrConsistent(), "Issue_date", "20.05.2021"), "EXTPASSPORTBIO_2007"), FieldVerified)
}

func TestExternalNegativeControls(t *testing.T) {
	ocr := with(extOcrConsistent(), "Expiration_date", "20.05.2032", "Last_name_en", "ROGOVA", "First_name_en", "IRENA")
	got := statuses(ocr, "EXTPASSPORTBIO_2007")
	wantStatus(t, got, "Expiration_date", FieldMismatch)
	wantStatus(t, got, "First_name_en", FieldMismatch)
}

func TestTheSpecimensAlteredNumberIsAMismatch(t *testing.T) {
	// As on the anonymised samples/: the printed number differs, the zone is valid.
	wantStatus(t, statuses(extOcr(), "EXTPASSPORTBIO_2007"), "Licence_number", FieldMismatch)
}

// td3Line2 builds a TD3 second line with every check digit computed.
func td3Line2(number, birth, sex, expiry, personal string) string {
	pad := func(s string, n int) string { return s + strings.Repeat("<", n-len(s)) }
	digit := func(s string) string {
		d, _ := checkDigit([]rune(s))
		return string(rune('0' + d))
	}
	number, personal = pad(number, 9), pad(personal, 14)
	line := number + digit(number) + "RUS" + birth + digit(birth) + sex + expiry + digit(expiry) + personal + digit(personal)
	return line + digit(line[0:10]+line[13:20]+line[21:43])
}

func TestTd3BuilderReproducesARealZone(t *testing.T) {
	if got := td3Line2("764820157", "880314", "F", "310520", ""); got != strings.Split(extMrz, "\n")[1] {
		t.Errorf("builder: %s", got)
	}
}

func TestSevenDigitZoneNumberOf2003Form(t *testing.T) {
	// Five of 19 references of the 2003 form write the number without its series.
	zone := strings.Split(extMrz, "\n")[0] + "\n" + td3Line2("4820157", "880314", "F", "310520", "")
	ocr := with(extOcrConsistent(), "MRZ", zone)
	result := checksOf(ocr, "EXTPASSPORT_2003")
	if result["MRZ"].Status != FieldVerified || !hasCheck(result["Licence_number"], "MRZ-2", "ok") {
		t.Errorf("seven-digit zone: MRZ %s, number %+v", result["MRZ"].Status, result["Licence_number"].Checks)
	}
	bad := checksOf(with(ocr, "Licence_number", "76 4820158"), "EXTPASSPORT_2003")["Licence_number"]
	if bad.Status != FieldMismatch {
		t.Errorf("wrong number: %s", bad.Status)
	}
}

func TestTruncatedNameComparesByPrefix(t *testing.T) {
	longName := strings.Repeat("KONSTANTINOPOLSKAYA", 3)
	line1 := ("P<RUS" + "ROGOVA<<" + longName)[:44]
	zone := line1 + "\n" + strings.Split(extMrz, "\n")[1]
	ocr := with(extOcrConsistent(), "MRZ", zone, "First_name_en", longName)
	wantStatus(t, statuses(ocr, "EXTPASSPORTBIO_2007"), "First_name_en", FieldVerified)
}

// --- dates -----------------------------------------------------------------------------

func dlOcr() ocrMap {
	return ocrMap{"Birth_date": "01.02.1990", "Issue_date": "15.03.2015",
		"Expiration_date": "15.03.2025", "Licence_number": "99 12 345678"}
}

func TestLicenceValidityConfirmsBothDates(t *testing.T) {
	got := statuses(dlOcr(), "DL_2011")
	wantStatus(t, got, "Issue_date", FieldVerified)
	wantStatus(t, got, "Expiration_date", FieldVerified)
	wantStatus(t, got, "Birth_date", FieldUnchecked) // only refuting checks reach it
	wantStatus(t, got, "Licence_number", FieldUnchecked)
}

func TestLicenceValidityNegativeControlMarksBothWithReferences(t *testing.T) {
	result := checksOf(with(dlOcr(), "Expiration_date", "16.03.2025"), "DL_2020")
	if result["Issue_date"].Status != FieldMismatch || result["Expiration_date"].Status != FieldMismatch {
		t.Fatalf("statuses: %s %s", result["Issue_date"].Status, result["Expiration_date"].Status)
	}
	if r := findCheck(t, result["Expiration_date"], "DL-2").Reference; r == nil || *r != "15.03.2025" {
		t.Errorf("expiry reference %v", r)
	}
	if r := findCheck(t, result["Issue_date"], "DL-2").Reference; r == nil || *r != "16.03.2015" {
		t.Errorf("issue reference %v", r)
	}
}

func TestExternalValidityAcceptsFiveOrTenYears(t *testing.T) {
	for _, expiry := range []string{"01.06.2015", "01.06.2020"} {
		wantStatus(t, statuses(ocrMap{"Issue_date": "01.06.2010", "Expiration_date": expiry}, "EXTPASSPORT_2003"),
			"Issue_date", FieldVerified)
	}
	wantStatus(t, statuses(ocrMap{"Issue_date": "01.06.2010", "Expiration_date": "01.06.2017"}, "EXTPASSPORTBIO_2007"),
		"Issue_date", FieldMismatch)
}

func TestExternalValidityHasNoReferenceForTwoCandidates(t *testing.T) {
	v := checksOf(ocrMap{"Issue_date": "01.06.2010", "Expiration_date": "01.06.2017"}, "EXTPASSPORTBIO_2007")["Issue_date"]
	if c := findCheck(t, v, "EXT-3"); c.Reference != nil {
		t.Errorf("reference %v where the validity is ambiguous", *c.Reference)
	}
}

func TestLeapDayValidity(t *testing.T) {
	wantStatus(t, statuses(ocrMap{"Issue_date": "29.02.2016", "Expiration_date": "28.02.2026"}, "DL_2011"),
		"Expiration_date", FieldVerified)
	// 01.03 is the other accepted day; 02.03 is not
	wantStatus(t, statuses(ocrMap{"Issue_date": "29.02.2016", "Expiration_date": "01.03.2026"}, "DL_2011"),
		"Expiration_date", FieldVerified)
	wantStatus(t, statuses(ocrMap{"Issue_date": "29.02.2016", "Expiration_date": "02.03.2026"}, "DL_2011"),
		"Expiration_date", FieldMismatch)
}

func TestNotACalendarDate(t *testing.T) {
	wantStatus(t, statuses(ocrMap{"Birth_date": "31.02.1990"}, "SNILS_1996"), "Birth_date", FieldMismatch)
	// a date as the SNILS prints it converts, and nothing confirms it
	wantStatus(t, statuses(ocrMap{"Birth_date": "10 ДЕКАБРЯ 1999 ГОДА"}, "SNILS_1996"), "Birth_date", FieldUnchecked)
}

func TestIssueBeforeBirthMarksBoth(t *testing.T) {
	got := statuses(ocrMap{"Birth_date": "01.01.2000", "Issue_date": "01.01.1999"}, "SNILS_1996")
	if !reflect.DeepEqual(got, map[string]string{"Birth_date": FieldMismatch, "Issue_date": FieldMismatch}) {
		t.Errorf("%v", got)
	}
}

func TestFutureDate(t *testing.T) {
	wantStatus(t, statuses(ocrMap{"Issue_date": "01.01.2030"}, "SNILS_1996"), "Issue_date", FieldMismatch)
	// today itself is not the future
	wantStatus(t, statuses(ocrMap{"Issue_date": "19.10.2026"}, "SNILS_1996"), "Issue_date", FieldUnchecked)
	wantStatus(t, statuses(ocrMap{"Issue_date": "20.10.2026"}, "SNILS_1996"), "Issue_date", FieldMismatch)
}

func TestMinimumAgeAtIssue(t *testing.T) {
	young := ocrMap{"Birth_date": "01.01.2000", "Issue_date": "31.12.2013"}
	wantStatus(t, statuses(young, "INTPASSPORT_1997"), "Issue_date", FieldMismatch)
	wantStatus(t, statuses(with(young, "Issue_date", "01.01.2014"), "INTPASSPORT_1997"), "Issue_date", FieldUnchecked)
	wantStatus(t, statuses(ocrMap{"Birth_date": "01.01.2000", "Issue_date": "01.06.2015"}, "DL_2020"), "Birth_date", FieldMismatch)
	// SNILS is issued to children: no age rule
	wantStatus(t, statuses(young, "SNILS_1996"), "Issue_date", FieldUnchecked)
}

// --- SNILS -----------------------------------------------------------------------------

func TestSnilsCheckNumber(t *testing.T) {
	// the worked example of docs/validation-checks.md
	wantStatus(t, statuses(ocrMap{"Licence_number": "112-233-445 95"}, "SNILS_1996"), "Licence_number", FieldVerified)
	v := checksOf(ocrMap{"Licence_number": "112-233-445 96"}, "SNILS_1996")["Licence_number"]
	if v.Status != FieldMismatch {
		t.Errorf("wrong check number: %s", v.Status)
	}
	last := v.Checks[len(v.Checks)-1]
	if last.Reference == nil || *last.Reference != "95" {
		t.Errorf("reference %v", last.Reference)
	}
	// a changed digit in the body
	wantStatus(t, statuses(ocrMap{"Licence_number": "112-233-446 95"}, "SNILS_1996"), "Licence_number", FieldMismatch)
}

func TestSnilsCheckNumberWrapsMod101(t *testing.T) {
	// all nines: 9 * (9+8+...+1) = 405 -> 405 mod 101 = 1
	wantStatus(t, statuses(ocrMap{"Licence_number": "999-999-999 01"}, "SNILS_1996"), "Licence_number", FieldVerified)
	// 9*9 + 2*8 + 3*1 = 100 -> 00, not 100
	wantStatus(t, statuses(ocrMap{"Licence_number": "920-000-003 00"}, "SNILS_1996"), "Licence_number", FieldVerified)
	wantStatus(t, statuses(ocrMap{"Licence_number": "920-000-003 01"}, "SNILS_1996"), "Licence_number", FieldMismatch)
}

func TestSnilsLengthAndSmallNumbers(t *testing.T) {
	wantStatus(t, statuses(ocrMap{"Licence_number": "112-233-445 9"}, "SNILS_1996"), "Licence_number", FieldMismatch)
	wantStatus(t, statuses(ocrMap{"Licence_number": "112-233-445-95"}, "SNILS_1996"), "Licence_number", FieldMismatch)
	// up to 001-001-998 there is no valid check number to verify
	wantStatus(t, statuses(ocrMap{"Licence_number": "001-001-998 00"}, "SNILS_1996"), "Licence_number", FieldUnchecked)
}

// --- soft ------------------------------------------------------------------------------

func TestSeriesYearIsANoteNotAVerdict(t *testing.T) {
	ocr := with(intOcr(), "Licence_number", "45 02 318294")
	result := checksOf(ocr, "INTPASSPORT_1997")
	notes := result["Licence_number"].Notes
	if len(notes) == 0 || notes[0].ID != "PASS-3" {
		t.Fatalf("notes: %+v", notes)
	}
	if want := "series printed in 2002, issued in 2007"; notes[0].Text != want {
		t.Errorf("note text %q, want %q", notes[0].Text, want)
	}
	// the note does not change any verdict by itself
	plain := checksOf(with(ocr, "MRZ", ""), "INTPASSPORT_1997")
	if plain["Licence_number"].Status != FieldUnchecked || plain["Licence_number"].Notes[0].ID != "PASS-3" {
		t.Errorf("plain: %+v", plain["Licence_number"])
	}
	// both fields carry it; a series within the window carries none
	if len(result["Issue_date"].Notes) != 1 {
		t.Errorf("Issue_date notes: %+v", result["Issue_date"].Notes)
	}
	if n := checksOf(intOcr(), "INTPASSPORT_2011")["Licence_number"].Notes; len(n) != 0 {
		t.Errorf("unexpected note: %+v", n)
	}
}

// --- the order of checks and the wire form (compared exactly by the conformance stage) ---

func TestChecksComeInTheOrderTheyAreMade(t *testing.T) {
	v := checksOf(intOcr(), "INTPASSPORT_2011")
	var ids []string
	for _, c := range v["Issue_date"].Checks {
		ids = append(ids, c.ID)
	}
	if got, want := strings.Join(ids, " "), "GEN-1 GEN-2 GEN-2 PASS-7 MRZ-5"; got != want {
		t.Errorf("Issue_date: %s, want %s", got, want)
	}
	ids = nil
	for _, c := range v["MRZ"].Checks {
		ids = append(ids, c.ID)
	}
	if got, want := strings.Join(ids, " "), "MRZ-1 MRZ-9 MRZ-8"; got != want {
		t.Errorf("MRZ: %s, want %s", got, want)
	}
}

func TestReferenceKeyIsAbsentWithoutAValue(t *testing.T) {
	blob, err := json.Marshal(checksOf(dlOcr(), "DL_2011")["Birth_date"].Checks[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(blob) != `{"id":"GEN-1","kind":"refuting","result":"ok"}` {
		t.Errorf("%s", blob)
	}
}

// --- the zone's own arithmetic ---------------------------------------------------------

func TestCheckDigitMatchesIcaoWeights(t *testing.T) {
	for payload, want := range map[string]int{"764820157": 2, "<<<<<<<<<<<<<<": 0, "AB": 3} {
		if got, ok := checkDigit([]rune(payload)); !ok || got != want {
			t.Errorf("checkDigit(%q) = %d, %v; want %d", payload, got, ok, want)
		}
	}
	if _, ok := checkDigit([]rune("76482015!")); ok {
		t.Error("a character outside the alphabet must have no digit")
	}
}

func TestInternalZoneHasNoExpiryDate(t *testing.T) {
	checked := ValidateMrz(intMrz)
	if !checked.OK || checked.Parts["expiry_date"] != "not_applicable" {
		t.Errorf("internal zone: %+v", checked)
	}
	if bad := ValidateMrz(strings.Replace(extMrz, "7648201572", "7648201582", 1)); bad.OK || bad.Parts["document_number"] != "failed" {
		t.Errorf("corrupted number: %+v", bad)
	}
	if short := ValidateMrz(extMrz[:len(extMrz)-1]); short.OK || len(short.Parts) != 0 {
		t.Errorf("43-character line: %+v", short)
	}
	if one := ValidateMrz(strings.Split(extMrz, "\n")[0]); one.OK {
		t.Error("one line")
	}
}

func TestMrzDateCenturyFollowsTheClock(t *testing.T) {
	defer func(f func() time.Time) { clock = f }(clock)
	clock = func() time.Time { return testToday }
	// a birth date is in the past (26 -> 2026 only because it is not later than the year),
	// an expiry date is always 20xx
	if got := mrzDate([]rune("261001"), false); got != "01.10.2026" {
		t.Errorf("past 26: %s", got)
	}
	if got := mrzDate([]rune("271001"), false); got != "01.10.1927" {
		t.Errorf("past 27: %s", got)
	}
	if got := mrzDate([]rune("991001"), true); got != "01.10.2099" {
		t.Errorf("future 99: %s", got)
	}
	if got := mrzDate([]rune("261301"), false); got != "" {
		t.Errorf("month 13: %q", got)
	}
}

func TestPythonWhitespaceRules(t *testing.T) {
	if got := pySplitLines("a\r\nb\vc\x1cd\n\ne"); !reflect.DeepEqual(got, []string{"a", "b", "c", "d", "", "e"}) {
		t.Errorf("splitlines: %q", got)
	}
	if got := strings.Join(pyFields(" 45 07\t318294\x1f"), " "); got != "45 07 318294" {
		t.Errorf("fields: %q", got)
	}
}

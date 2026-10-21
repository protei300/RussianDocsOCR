package pipeline

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// A verdict for every recognised field: verified, mismatch or unchecked.
// Port of validation/fields.py (decision No. 152;
// check ids are the rows of docs/validation-checks.md).
//
// Three verdicts, one per field of the reading:
//
//   - verified - at least one CONFIRMING check agreed and no hard check disagreed;
//   - mismatch - a hard check disagreed: almost certainly a reading error;
//   - unchecked - nothing applicable, or nothing to lean on.
//
// Every check declares what its agreement is worth. A CONFIRMING one (two independent
// readings agree, or a check digit holds) verifies the field: any single changed character
// breaks it. A REFUTING one (a plausibility rule: birth before issue, a real calendar date)
// proves nothing by agreeing, only by disagreeing. A soft one never changes a verdict and
// shows up as a note.
//
// A disagreement between two fields marks BOTH: the check knows that one of them is wrong,
// not which. The reading is never changed - the value the field was compared with travels in
// the check as Reference, and what to do with it is the integrator's call.
//
// Pure functions of the reading and the document type: no image, no model. The order of the
// entries of Checks is part of the contract (the conformance stage `checks` compares lists).
//
// Deliberate difference from the reference: Python's \d and str.isdigit() accept the digits of
// other scripts, these helpers accept ASCII only. The recognisers never produce the others
// (the OCR alphabets have none), so no reading can tell the two apart.

// Verdicts and kinds, spelled as the reference spells them.
const (
	FieldVerified  = "verified"
	FieldMismatch  = "mismatch"
	FieldUnchecked = "unchecked"

	CheckConfirming = "confirming"
	CheckRefuting   = "refuting"

	checkOK = "ok"
)

// FieldCheck is one check of one field: Reference is the value it compared with, absent
// (nil, no key in the JSON) when the check has none.
type FieldCheck struct {
	ID        string  `json:"id"`
	Kind      string  `json:"kind"`
	Result    string  `json:"result"`
	Reference *string `json:"reference,omitempty"`
}

// FieldNote is a soft remark that does not affect Status.
type FieldNote struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// FieldVerdict is the entry of one field. Checks and Notes are never nil (JSON `[]`).
type FieldVerdict struct {
	Status string       `json:"status"`
	Checks []FieldCheck `json:"checks"`
	Notes  []FieldNote  `json:"notes"`
}

// cyrillicToZone is the Cyrillic -> zone characters table of the internal passport's top MRZ
// line, FMS order No. 279 (2011-06-30), the "modernised clear" table: letters with no Latin
// counterpart become digits.
var cyrillicToZone = map[rune]rune{
	'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': '2',
	'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Q', 'К': 'K', 'Л': 'L', 'М': 'M',
	'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
	'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': '3', 'Ш': '4', 'Щ': 'W', 'Ъ': 'X',
	'Ы': 'Y', 'Ь': '9', 'Э': '6', 'Ю': '7', 'Я': '8',
}

// zoneSex is the sex as printed -> the zone's letter. The page prints «МУЖ.»/«ЖЕН.» (the
// ground truth keeps the first letter), the external passport also «M»/«F».
var zoneSex = map[string]string{"М": "M", "Ж": "F", "M": "M", "F": "F"}

// validityYears are the years between the issue date and the expiry date. Driving licences:
// 10. External passports: 10 for the biometric form, 5 for the 2003 one - either is accepted
// for both, so a form the classifier confused costs a little strength instead of a false alarm.
var validityYears = map[string][]int{"DL": {10}, "EXTPASSPORT": {5, 10}}

// minAge is the minimum age at issue. SNILS is issued to children and the external passport
// too, so neither has an age rule.
var minAge = map[string]int{"INTPASSPORT": 14, "DL": 16}

// seriesYearWindow (soft): the second pair of the internal passport's series is the year the
// blank was printed, usually within a few years of the issue date.
const seriesYearWindow = 3

var dateFieldNames = []string{"Birth_date", "Issue_date", "Expiration_date"}

type numberForm struct {
	field, id string
	pattern   *regexp.Regexp
}

// numberForms are the printed forms of numbers, whitespace collapsed. The comparisons with the
// zone look at digits only, so without the form a reading with a stray character was
// 'verified' while not being what is printed. Per family, in the reference's order.
var numberForms = map[string][]numberForm{
	"INTPASSPORT": {
		{"Licence_number", "PASS-1", regexp.MustCompile(`^[0-9]{2} [0-9]{2} [0-9]{6}$`)},
		{"Issue_organisation_code", "PASS-4", regexp.MustCompile(`^[0-9]{3}-[0-9]{3}$`)},
	},
	"EXTPASSPORT": {{"Licence_number", "EXT-1", regexp.MustCompile(`^[0-9]{2} [0-9]{7}$`)}},
	"SNILS":       {{"Licence_number", "SNILS-1", regexp.MustCompile(`^[0-9]{3}-[0-9]{3}-[0-9]{3} [0-9]{2}$`)}},
}

// The top MRZ line: its fixed start and the characters its name area can hold. An external
// passport writes names in Latin letters only; the internal one also uses the digits of the
// order No. 279 table (2 3 4 6 7 8 9) - never 0, 1 or 5.
var line1Prefix = map[string]string{"INTPASSPORT": "PNRUS", "EXTPASSPORT": "P<RUS"}

func nameCharsFor(family string) func(rune) bool {
	return func(r rune) bool {
		if (r >= 'A' && r <= 'Z') || r == '<' {
			return true
		}
		return family == "INTPASSPORT" && strings.ContainsRune("2346789", r)
	}
}

// FieldFamily is family(): the document family the checks are defined for, or "". Order
// matters as in MakeOcrOptions: the registration page and the licence back side contain the
// names of the families they are not. Everything outside the four families gets `unchecked`
// throughout - their rules were not measured.
func FieldFamily(docType string) string {
	label := strings.ToUpper(docType)
	if strings.Contains(label, "INTPASSPORTADDR") || strings.Contains(label, "DLBACK") {
		return ""
	}
	for _, name := range []string{"INTPASSPORT", "EXTPASSPORT", "SNILS", "DL"} {
		if strings.HasPrefix(label, name) {
			return name
		}
	}
	return ""
}

// verdicts collects check results per field and turns them into verdicts.
type verdicts struct {
	ocr     map[string]string
	entries map[string]*FieldVerdict
}

func newVerdicts(ocr map[string]string) *verdicts {
	v := &verdicts{ocr: ocr, entries: make(map[string]*FieldVerdict, len(ocr))}
	for name := range ocr {
		v.entries[name] = &FieldVerdict{Checks: []FieldCheck{}, Notes: []FieldNote{}}
	}
	return v
}

// value is the field's reading, stripped, or "" when absent or empty.
func (v *verdicts) value(name string) string { return pyStrip(v.ocr[name]) }

func ref(s string) *string { return &s }

// add records one check over one or several fields (only those in the reading). refs maps a
// field to the value it was compared with; a field absent from it gets no reference.
func (v *verdicts) add(fields []string, id, kind string, agrees bool, refs map[string]string) {
	for _, name := range fields {
		entry, ok := v.entries[name]
		if !ok {
			continue
		}
		check := FieldCheck{ID: id, Kind: kind, Result: checkOK}
		if !agrees {
			check.Result = FieldMismatch
		}
		if r, has := refs[name]; has {
			check.Reference = ref(r)
		}
		entry.Checks = append(entry.Checks, check)
	}
}

func (v *verdicts) addOne(name, id, kind string, agrees bool) {
	v.add([]string{name}, id, kind, agrees, nil)
}

func (v *verdicts) addRef(name, id, kind string, agrees bool, reference string) {
	v.add([]string{name}, id, kind, agrees, map[string]string{name: reference})
}

func (v *verdicts) note(fields []string, id, text string) {
	for _, name := range fields {
		if entry, ok := v.entries[name]; ok {
			entry.Notes = append(entry.Notes, FieldNote{ID: id, Text: text})
		}
	}
}

func (v *verdicts) result() map[string]FieldVerdict {
	out := make(map[string]FieldVerdict, len(v.entries))
	for name, entry := range v.entries {
		status := FieldUnchecked
		for _, c := range entry.Checks {
			if c.Result == FieldMismatch {
				status = FieldMismatch
				break
			}
		}
		if status != FieldMismatch {
			for _, c := range entry.Checks {
				if c.Kind == CheckConfirming {
					status = FieldVerified
					break
				}
			}
		}
		out[name] = FieldVerdict{Status: status, Checks: entry.Checks, Notes: entry.Notes}
	}
	return out
}

// calDate is a calendar date. Only real dates are made (see asDate / mrzDate).
type calDate struct{ y, m, d int }

func (a calDate) key() int              { return a.y*10000 + a.m*100 + a.d }
func (a calDate) before(b calDate) bool { return a.key() < b.key() }
func (a calDate) after(b calDate) bool  { return a.key() > b.key() }

func fmtDate(y, m, d int) string { return fmt.Sprintf("%02d.%02d.%04d", d, m, y) }
func (a calDate) String() string { return fmtDate(a.y, a.m, a.d) }

func dateOf(t time.Time) calDate { return calDate{t.Year(), int(t.Month()), t.Day()} }

// asCalDate is _as_date: the canonical form of a date field as a date, or ok=false.
func asCalDate(name, text string) (calDate, bool) {
	if text == "" {
		return calDate{}, false
	}
	canonical := CanonicalDate(name, text)
	if canonical == "" {
		return calDate{}, false
	}
	parts := strings.Split(canonical, ".")
	d, _ := strconv.Atoi(parts[0])
	m, _ := strconv.Atoi(parts[1])
	y, _ := strconv.Atoi(parts[2])
	return calDate{y, m, d}, true
}

// addYears is _add_years: the same day `years` later; 29.02 maps to both 28.02 and 01.03.
func addYears(day calDate, years int) []calDate {
	y := day.y + years
	if day.d <= daysIn(day.m, y) {
		return []calDate{{y, day.m, day.d}}
	}
	return []calDate{{y, 2, 28}, {y, 3, 1}}
}

// age is _age: whole years from birth to `at`.
func age(birth, at calDate) int {
	n := at.y - birth.y
	if at.m < birth.m || (at.m == birth.m && at.d < birth.d) {
		n--
	}
	return n
}

// digitsOf is _digits: the ASCII digits of a text.
func digitsOf(text string) string {
	var b strings.Builder
	for _, r := range text {
		if r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// checkDates is GEN-1, GEN-2, PASS-7 / DL-4, DL-2 / EXT-3.
func checkDates(v *verdicts, fam string, today calDate) {
	dates := map[string]calDate{}
	for _, name := range dateFieldNames {
		text := v.value(name)
		if text == "" {
			continue
		}
		parsed, ok := asCalDate(name, text)
		// GEN-1: every reference date in samples/ converts (309 of 309), so a reading that
		// does not is not a date.
		v.addOne(name, "GEN-1", CheckRefuting, ok)
		if ok {
			dates[name] = parsed
		}
	}

	birth, hasBirth := dates["Birth_date"]
	issue, hasIssue := dates["Issue_date"]
	expiry, hasExpiry := dates["Expiration_date"]
	if hasBirth {
		v.addOne("Birth_date", "GEN-2", CheckRefuting, !birth.after(today))
	}
	if hasIssue {
		v.addOne("Issue_date", "GEN-2", CheckRefuting, !issue.after(today))
	}
	both := []string{"Birth_date", "Issue_date"}
	if hasBirth && hasIssue {
		v.add(both, "GEN-2", CheckRefuting, birth.before(issue), nil)
		if limit, ok := minAge[fam]; ok {
			id := "DL-4"
			if fam == "INTPASSPORT" {
				id = "PASS-7"
			}
			v.add(both, id, CheckRefuting, age(birth, issue) >= limit, nil)
		}
	}
	years := validityYears[fam]
	if len(years) > 0 && hasIssue && hasExpiry {
		var expected []calDate
		for _, n := range years {
			expected = append(expected, addYears(issue, n)...)
		}
		var refs map[string]string
		if len(years) == 1 && len(expected) == 1 {
			back := addYears(expiry, -years[0])
			refs = map[string]string{"Expiration_date": expected[0].String()}
			if len(back) == 1 {
				refs["Issue_date"] = back[0].String()
			}
		}
		id := "EXT-3"
		if fam == "DL" {
			id = "DL-2"
		}
		agrees := false
		for _, e := range expected {
			if e == expiry {
				agrees = true
			}
		}
		v.add([]string{"Issue_date", "Expiration_date"}, id, CheckConfirming, agrees, refs)
	}
}

// zoneNames is _zone_names: a name for comparison with the zone - no spaces, hyphens or fillers.
func zoneNames(text string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(text) {
		if pyIsSpace(r) || r == '-' || r == '<' {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// zoneFromCyrillic is _cyrillic_to_zone: a Cyrillic name as the internal passport's zone
// writes it. Ё and Е are made equal on both sides (Ё is '2' in the zone): whether a name was
// typed with Ё is the clerk's choice, not something the page and the zone are guaranteed to share.
func zoneFromCyrillic(text string) string {
	var b strings.Builder
	for _, r := range strings.ReplaceAll(zoneNames(text), "Ё", "Е") {
		if t, ok := cyrillicToZone[r]; ok {
			r = t
		}
		b.WriteRune(r)
	}
	return b.String()
}

// namesAgree is _names_agree: equal, or - when the zone's name area is full - the page starts
// with it.
func namesAgree(zone, page string, truncated bool) bool {
	if zone == "" || page == "" {
		return false
	}
	if zone == page {
		return true
	}
	return truncated && strings.HasPrefix(page, zone)
}

// checkMrz is MRZ-1 .. MRZ-9 (docs/validation-checks.md, section MRZ).
func checkMrz(v *verdicts, fam string) {
	mrzText := v.value("MRZ")
	if mrzText == "" {
		return
	}
	checked := ValidateMrz(mrzText)
	// The check digits cover line 2 only, so for the zone as a whole they can refute but not
	// confirm; the confirmation is the names of line 1 agreeing with the page (MRZ-8 below).
	v.addOne("MRZ", "MRZ-1", CheckRefuting, checked.OK)
	internal := fam == "INTPASSPORT"
	// The layout is chosen by the classifier's type, not by the zone's own first characters,
	// which may be misread.
	zone := ParseMrz(mrzText, internal)

	trusted := func(parts ...string) bool {
		for _, p := range parts {
			if checked.Parts[p] != "ok" {
				return false
			}
		}
		return true
	}

	// MRZ-2: the number, in the form the page prints it.
	pageNumber := digitsOf(v.value("Licence_number"))
	numberParts := []string{"document_number"}
	if internal {
		numberParts = append(numberParts, "personal_number")
	}
	zoneNumber := PrintedNumber(zone, internal)
	if pageNumber != "" && zoneNumber != "" && trusted(numberParts...) {
		v.addRef("Licence_number", "MRZ-2", CheckConfirming, NumbersAgree(zoneNumber, pageNumber), zoneNumber)
	}

	// MRZ-3..MRZ-5: dates, each behind its own check digit.
	type dateCheck struct{ page, zone, part, id string }
	dateParts := []dateCheck{{"Birth_date", zone.BirthDate, "birth_date", "MRZ-3"}}
	if internal {
		dateParts = append(dateParts, dateCheck{"Issue_date", zone.IssueDate, "personal_number", "MRZ-5"})
	} else {
		dateParts = append(dateParts, dateCheck{"Expiration_date", zone.ExpiryDate, "expiry_date", "MRZ-4"})
	}
	for _, dc := range dateParts {
		text := v.value(dc.page)
		if text == "" || dc.zone == "" || !trusted(dc.part) {
			continue
		}
		parsed, ok := asCalDate(dc.page, text)
		v.addRef(dc.page, dc.id, CheckConfirming, ok && parsed.String() == dc.zone, dc.zone)
	}

	// MRZ-6: the subdivision code (internal passport only).
	codePage, codeZone := digitsOf(v.value("Issue_organisation_code")), zone.SubdivisionCode
	if internal && codePage != "" && codeZone != "" && trusted("personal_number") {
		v.addRef("Issue_organisation_code", "MRZ-6", CheckConfirming, codePage == codeZone,
			codeZone[:3]+"-"+codeZone[3:])
	}

	// MRZ-7: sex. No check digit covers it, but two independent readings agreeing is a
	// confirmation all the same.
	if zone.Sex != "" {
		for _, name := range []string{"Sex_ru", "Sex_en"} {
			text := v.value(name)
			if text == "" {
				continue
			}
			first, _ := utf8.DecodeRuneInString(text)
			v.addRef(name, "MRZ-7", CheckConfirming, zoneSex[strings.ToUpper(string(first))] == zone.Sex, zone.Sex)
		}
	}

	// MRZ-8: names from the top line. No check digit covers that line, so a disagreement says
	// "one of the two readings is wrong", not which - and a line of the wrong length may have
	// lost part of a name: not compared.
	lines := mrzSplitLines(mrzText)
	var line1 []rune
	if len(lines) > 0 {
		line1 = lines[0]
	}
	wellFormed := len(line1) == mrzLineLen && strings.HasPrefix(string(line1), line1Prefix[fam]) &&
		strings.Contains(string(line1[5:]), "<<")
	if wellFormed {
		allowed := nameCharsFor(fam)
		for _, r := range line1[5:] {
			if !allowed(r) {
				wellFormed = false
				break
			}
		}
	}
	v.addOne("MRZ", "MRZ-9", CheckRefuting, wellFormed)
	if !wellFormed {
		return
	}
	truncated := line1[len(line1)-1] != '<'
	surname := strings.ReplaceAll(zone.Surname, " ", "")
	given := strings.SplitN(zone.GivenNames, " ", 2)
	zoneFirst := given[0]
	zoneRest := ""
	if len(given) > 1 {
		zoneRest = strings.ReplaceAll(given[1], " ", "")
	}
	var convert func(string) string
	var names []string
	if internal {
		// Ё is '2' in the zone; made equal to Е on both sides (zoneFromCyrillic).
		surname = strings.ReplaceAll(surname, "2", "E")
		zoneFirst = strings.ReplaceAll(zoneFirst, "2", "E")
		zoneRest = strings.ReplaceAll(zoneRest, "2", "E")
		convert = zoneFromCyrillic
		names = []string{"Last_name_ru", "First_name_ru", "Middle_name_ru"}
	} else {
		convert = zoneNames
		names = []string{"Last_name_en", "First_name_en", "Middle_name_en"}
	}
	zoneValues := []string{surname, zoneFirst, zoneRest}
	var agreed []bool
	for i, name := range names {
		text := v.value(name)
		if text == "" || zoneValues[i] == "" {
			continue
		}
		agreed = append(agreed, namesAgree(zoneValues[i], convert(text), truncated))
		v.addRef(name, "MRZ-8", CheckConfirming, agreed[len(agreed)-1], zoneValues[i])
	}
	// The zone itself is confirmed only when every compared name agrees. A disagreement does
	// not mark it: the zone's own misreadings are caught for certain by MRZ-1 and MRZ-9, and
	// which of the two readings of a name is wrong is unknown - the page field already carries
	// that mismatch.
	if len(agreed) > 0 {
		all := true
		for _, a := range agreed {
			all = all && a
		}
		if all {
			v.addOne("MRZ", "MRZ-8", CheckConfirming, true)
		}
	}
}

// snilsControl is the SNILS check number over nine digits (weights 9..1, mod 101).
func snilsControl(body string) int {
	total := 0
	for i := 0; i < 9; i++ {
		total += int(body[i]-'0') * (9 - i)
	}
	for total > 101 {
		total %= 101
	}
	if total == 100 || total == 101 {
		return 0
	}
	return total
}

// checkForms is PASS-1, PASS-4, EXT-1, SNILS-1: the printed form of numbers (refuting).
func checkForms(v *verdicts, fam string) {
	for _, form := range numberForms[fam] {
		if text := v.value(form.field); text != "" {
			v.addOne(form.field, form.id, CheckRefuting,
				form.pattern.MatchString(strings.Join(pyFields(text), " ")))
		}
	}
}

// checkSnils is SNILS-2 (the check number); its form is SNILS-1 in checkForms.
func checkSnils(v *verdicts) {
	text := v.value("Licence_number")
	if text == "" {
		return
	}
	digits := digitsOf(text)
	if len(digits) != 11 {
		return
	}
	body := digits[:9]
	// Numbers up to 001-001-998 are issued without a valid check number.
	if n, _ := strconv.Atoi(body); n <= 1001998 {
		return
	}
	expected := snilsControl(body)
	got, _ := strconv.Atoi(digits[9:])
	v.addRef("Licence_number", "SNILS-2", CheckConfirming, got == expected, fmt.Sprintf("%02d", expected))
}

// noteSeriesYear is PASS-3 (soft): the blank's year in the series against the issue year.
func noteSeriesYear(v *verdicts) {
	digits := digitsOf(v.value("Licence_number"))
	issue, ok := asCalDate("Issue_date", v.value("Issue_date"))
	if len(digits) != 10 || !ok {
		return
	}
	yy, _ := strconv.Atoi(digits[2:4])
	blankYear := 1900 + yy
	if 2000+yy <= issue.y+seriesYearWindow {
		blankYear = 2000 + yy
	}
	gap := issue.y - blankYear
	if gap < 0 {
		gap = -gap
	}
	if gap > seriesYearWindow {
		v.note([]string{"Licence_number", "Issue_date"}, "PASS-3",
			fmt.Sprintf("series printed in %d, issued in %d", blankYear, issue.y))
	}
}

// FieldChecks is field_checks: the verdict per field of the reading, an entry for EVERY key of
// it (so "no entry" never has to be told apart from "unchecked"). `today` is the calendar the
// "not in the future" rule reads; the zero time means the clock, as date.today() in the
// reference. The reading is never changed.
func FieldChecks(ocr map[string]string, docType string, today time.Time) map[string]FieldVerdict {
	v := newVerdicts(ocr)
	fam := FieldFamily(docType)
	if fam == "" {
		return v.result()
	}
	if today.IsZero() {
		today = clock()
	}
	now := dateOf(today)
	checkDates(v, fam, now)
	checkForms(v, fam)
	if fam == "INTPASSPORT" || fam == "EXTPASSPORT" {
		checkMrz(v, fam)
	}
	if fam == "SNILS" {
		checkSnils(v)
	}
	if fam == "INTPASSPORT" {
		noteSeriesYear(v)
	}
	return v.result()
}

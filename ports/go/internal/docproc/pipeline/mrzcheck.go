package pipeline

import (
	"strconv"
	"strings"
	"time"
	"unicode"
)

// The machine-readable-zone arithmetic that fieldchecks.go leans on: the ICAO 9303 check
// digits, the zone's parts, the printed form of its number. Port of the part of
// document_processing/validation/mrz.py that validation/fields.py uses (check_digit,
// validate_mrz, parse_mrz, printed_number, numbers_agree, _mrz_date). The reference's
// cross_check and validate (EXT-5 as a report) are not ported: nothing here reads them.
//
// Two document families use a TD3-shaped (2x44) zone and they are NOT the same: the external
// passport is a full ICAO TD3, the internal one (2011+) is type PN with NO expiry date -
// positions 21-27 are filler where the check digit would be, which is `not_applicable`,
// never a failed digit.

const (
	mrzFiller  = '<'
	mrzLineLen = 44
)

// clock is the calendar the zone's century and the "not in the future" rule read. The
// reference calls date.today() (the local date) in both places; a variable so a test can pin it.
var clock = time.Now

// mrzField is one slice of line 2 and the position of the check digit that covers it.
type mrzField struct {
	name   string
	lo, hi int
	digit  int
}

var mrzLine2Fields = []mrzField{
	{"document_number", 0, 9, 9},
	{"birth_date", 13, 19, 19},
	{"expiry_date", 21, 27, 27},
	{"personal_number", 28, 42, 42},
}

// The composite digit covers the number, the birth date and everything from the expiry date to
// the personal-number check digit, each WITH its own digit.
var mrzCompositeParts = [][2]int{{0, 10}, {13, 20}, {21, 43}}

const mrzCompositePos = 43

// pyIsSpace is Python's str.isspace for one character: Go's unicode.IsSpace plus the
// information separators \x1c-\x1f, which Python counts as whitespace and Go does not.
func pyIsSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

// pyStrip is str.strip().
func pyStrip(s string) string { return strings.TrimFunc(s, pyIsSpace) }

// pyFields is str.split() with no argument.
func pyFields(s string) []string { return strings.FieldsFunc(s, pyIsSpace) }

// pySplitLines is str.splitlines(): it breaks at \n, \r, \r\n, \v, \f, \x1c-\x1e, \x85,
//   and   and keeps no terminators.
func pySplitLines(s string) []string {
	var out []string
	var cur []rune
	rs := []rune(s)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '\r':
			if i+1 < len(rs) && rs[i+1] == '\n' {
				i++
			}
		case r == '\n', r == '\v', r == '\f', r >= 0x1c && r <= 0x1e,
			r == 0x85, r == 0x2028, r == 0x2029:
		default:
			cur = append(cur, r)
			continue
		}
		out = append(out, string(cur))
		cur = cur[:0]
	}
	if len(cur) > 0 {
		out = append(out, string(cur))
	}
	return out
}

// mrzSplitLines is split_lines: the zone as non-empty stripped lines.
func mrzSplitLines(text string) [][]rune {
	var out [][]rune
	if text == "" {
		return out
	}
	for _, line := range pySplitLines(text) {
		if line = pyStrip(line); line != "" {
			out = append(out, []rune(line))
		}
	}
	return out
}

// checkDigit is the ICAO 9303 digit: weights cycle 7-3-1, A-Z are 10-35, '<' is 0. ok is
// false (None in the reference) when the payload holds a character outside the MRZ alphabet.
func checkDigit(payload []rune) (digit int, ok bool) {
	weights := [3]int{7, 3, 1}
	total := 0
	for i, ch := range payload {
		var value int
		switch {
		case ch == mrzFiller:
			value = 0
		case ch >= '0' && ch <= '9':
			value = int(ch - '0')
		case ch >= 'A' && ch <= 'Z':
			value = int(ch) - 55
		default:
			return 0, false
		}
		total += value * weights[i%3]
	}
	return total % 10, true
}

func isMrzChar(r rune) bool {
	return (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == mrzFiller
}

// MrzDigits is validate_mrz reduced to what the verdicts use: the overall flag and the
// per-part result ("ok" / "failed" / "not_applicable"). Parts is empty when the zone is too
// short or its second line is malformed (the reference's `checks` is then {}).
type MrzDigits struct {
	OK    bool
	Parts map[string]string
}

// ValidateMrz is EXT-4: verify every check digit of the zone. OK is false for a malformed
// zone - a 43-character line means characters were lost, which is itself a finding.
func ValidateMrz(text string) MrzDigits {
	res := MrzDigits{Parts: map[string]string{}}
	lines := mrzSplitLines(text)
	if len(lines) < 2 {
		return res
	}
	line2 := lines[1]
	if len(line2) != mrzLineLen {
		return res
	}
	for _, r := range line2 {
		if !isMrzChar(r) {
			return res
		}
	}
	for _, f := range mrzLine2Fields {
		payload, digit := line2[f.lo:f.hi], line2[f.digit]
		if digit == mrzFiller && allFiller(payload) {
			res.Parts[f.name] = "not_applicable"
			continue
		}
		expected, ok := checkDigit(payload)
		res.Parts[f.name] = verdictOf(ok && strconv.Itoa(expected) == string(digit))
	}
	var composite []rune
	for _, part := range mrzCompositeParts {
		composite = append(composite, line2[part[0]:part[1]]...)
	}
	expected, ok := checkDigit(composite)
	res.Parts["composite"] = verdictOf(ok && strconv.Itoa(expected) == string(line2[mrzCompositePos]))

	res.OK = true
	for _, v := range res.Parts {
		if v == "failed" {
			res.OK = false
		}
	}
	return res
}

func allFiller(payload []rune) bool {
	for _, r := range payload {
		if r != mrzFiller {
			return false
		}
	}
	return true
}

func verdictOf(ok bool) string {
	if ok {
		return "ok"
	}
	return "failed"
}

// mrzDate is _mrz_date: YYMMDD -> dd.mm.yyyy, or "" if it is not a real calendar date.
//
// The zone carries no century, so it comes from what the field means, and the two directions
// are opposite. A birth date is in the past: a two-digit year later than the current one
// belongs to the previous century. An expiry date is in the future and is always 20xx -
// reading it by the birth-date rule turns a passport valid until 2029 into one that expired
// in 1929.
func mrzDate(raw []rune, future bool) string {
	if len(raw) != 6 {
		return ""
	}
	for _, r := range raw {
		if r < '0' || r > '9' {
			return ""
		}
	}
	yy, _ := strconv.Atoi(string(raw[0:2]))
	mm, _ := strconv.Atoi(string(raw[2:4]))
	dd, _ := strconv.Atoi(string(raw[4:6]))
	century := 2000
	if !future && yy > clock().Year()%100 {
		century = 1900
	}
	year := century + yy
	if mm < 1 || mm > 12 || dd < 1 || dd > daysIn(mm, year) {
		return ""
	}
	return fmtDate(year, mm, dd)
}

// MrzFields are the values the zone carries. A missing or malformed part is "" (None in the
// reference).
type MrzFields struct {
	Surname, GivenNames string
	DocumentNumber      string
	BirthDate           string
	Sex                 string
	ExpiryDate          string
	PersonalNumber      string
	// The internal passport's optional data (positions 29-42): see internalOptionalData.
	SeriesLastDigit string
	IssueDate       string
	SubdivisionCode string
}

// ParseMrz is parse_mrz(text, internal=...). `internal` says whether this is an internal
// passport's zone; the caller passes the classifier's answer because the zone's own two type
// characters can be misread like any other. (The reference's `None` - take it from the zone's
// own type characters - is not ported: fields.py never asks for it.)
func ParseMrz(text string, internal bool) MrzFields {
	var f MrzFields
	lines := mrzSplitLines(text)
	if len(lines) > 0 {
		rs := lines[0]
		names := ""
		if len(rs) > 5 {
			names = string(rs[5:])
		}
		parts := strings.SplitN(names, "<<", 2)
		f.Surname = pyStrip(strings.ReplaceAll(parts[0], "<", " "))
		if len(parts) > 1 {
			f.GivenNames = pyStrip(strings.ReplaceAll(parts[1], "<", " "))
		}
	}
	if len(lines) > 1 && len(lines[1]) == mrzLineLen {
		line2 := lines[1]
		f.DocumentNumber = strings.ReplaceAll(string(line2[0:9]), "<", "")
		f.BirthDate = mrzDate(line2[13:19], false)
		if line2[20] == 'M' || line2[20] == 'F' {
			f.Sex = string(line2[20])
		}
		f.ExpiryDate = mrzDate(line2[21:27], true)
		f.PersonalNumber = strings.ReplaceAll(string(line2[28:42]), "<", "")
		if internal {
			optional := line2[28:42]
			// FMS order No. 279 (2011-06-30), table 2, positions 29-42: the LAST digit of
			// the series, the issue date as YYMMDD, the issuing subdivision's code as six
			// digits without the hyphen, then a filler. The zone's own number (positions
			// 1-9) holds only the first three digits of the series.
			if d := optional[0]; d >= '0' && d <= '9' {
				f.SeriesLastDigit = string(d)
			}
			f.IssueDate = mrzDate(optional[1:7], false)
			code := optional[7:13]
			if len(code) == 6 && isDigits(string(code)) {
				f.SubdivisionCode = string(code)
			}
		}
	}
	return f
}

// PrintedNumber is printed_number: the zone's document number in the form the page prints it,
// digits only; "" when it cannot be built. External passport: the number field as is. Internal
// passport: the first three digits of the series + the fourth one from the optional data +
// the six digits of the number - the zone splits the ten printed digits in two places.
func PrintedNumber(f MrzFields, internal bool) string {
	number := f.DocumentNumber
	if number == "" {
		return ""
	}
	if !internal {
		return number
	}
	if f.SeriesLastDigit == "" || len(number) != 9 {
		return ""
	}
	return number[:3] + f.SeriesLastDigit + number[3:]
}

// NumbersAgree compares the zone number with the page's digits. Five of the 19 external
// passports of the 2003 form in samples/ write the number WITHOUT its two-digit series (seven
// digits); a seven-digit zone number is therefore compared with the end of the printed one.
func NumbersAgree(zoneNumber, pageDigits string) bool {
	if zoneNumber == "" || pageDigits == "" {
		return false
	}
	if len(zoneNumber) == 7 && len(pageDigits) == 9 {
		return strings.HasSuffix(pageDigits, zoneNumber)
	}
	return zoneNumber == pageDigits
}

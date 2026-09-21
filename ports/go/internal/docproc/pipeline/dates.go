package pipeline

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// Canonical dd.mm.yyyy view of a recognised date.
// Port of pipeline/dates.py.
//
// The pipeline returns dates AS PRINTED - «15 ОКТЯБРЯ 2020 Г.» on a 2018 birth
// certificate, «10 ДЕКАБРЯ 1999 ГОДА» on a SNILS, «03.АВГУСТ.1989» on a 1997 internal
// passport - because that is what the ground truth describes and what the accuracy
// measurement compares against. A consumer usually wants a machine form instead, so the
// canonical view is built ALONGSIDE the reading, never in place of it (Results.Ocr keeps
// the reading; Results.OcrNormalized holds this).
//
// Two rules shape everything here (dates.py:11-17):
//
//   - Never guess. No year in the text -> no canonical value. A month word that does not
//     match -> no canonical value. A date outside the calendar (31.02) -> no canonical
//     value. The caller falls back to the reading instead of receiving an invention.
//   - Never touch the reading. Trailing «Г.» / «ГОДА» stay in the reading; they are
//     printed on the document. They simply have no place in dd.mm.yyyy.
//
// Pure functions of a string: no image, no model, no configuration.

// months maps the month names as the documents print them, nominative and genitive.
var months = map[string]int{
	"ЯНВАРЬ": 1, "ЯНВАРЯ": 1,
	"ФЕВРАЛЬ": 2, "ФЕВРАЛЯ": 2,
	"МАРТ": 3, "МАРТА": 3,
	"АПРЕЛЬ": 4, "АПРЕЛЯ": 4,
	"МАЙ": 5, "МАЯ": 5,
	"ИЮНЬ": 6, "ИЮНЯ": 6,
	"ИЮЛЬ": 7, "ИЮЛЯ": 7,
	"АВГУСТ": 8, "АВГУСТА": 8,
	"СЕНТЯБРЬ": 9, "СЕНТЯБРЯ": 9,
	"ОКТЯБРЬ": 10, "ОКТЯБРЯ": 10,
	"НОЯБРЬ": 11, "НОЯБРЯ": 11,
	"ДЕКАБРЬ": 12, "ДЕКАБРЯ": 12,
}

// dateNoise lists the words a document prints next to a date that carry no date
// information. `Г.` can never be a token of the tokenizer below (it splits at '.'), but
// it is kept so the set reads the same as the reference's.
var dateNoise = map[string]bool{"Г": true, "Г.": true, "ГОД": true, "ГОДА": true, "ГОДУ": true}

// dateToken is `[^\W\d_]+|\d+` with re.UNICODE: runs of letters, or runs of digits.
// Go's RE2 has no \W-minus-\d class, so the letter run is spelled with the Unicode
// property, which is what Python's `[^\W\d_]` denotes (letters of any script).
var dateToken = regexp.MustCompile(`\pL+|\d+`)

// asDate formats dd.mm.yyyy for a real calendar date, else "" (31.02 is not a date).
func asDate(day, month, year int) string {
	if month < 1 || month > 12 || year < 1900 || year > 2100 {
		return ""
	}
	if day < 1 || day > daysIn(month, year) {
		return ""
	}
	return fmt.Sprintf("%02d.%02d.%04d", day, month, year)
}

func daysIn(month, year int) int {
	switch month {
	case 1, 3, 5, 7, 8, 10, 12:
		return 31
	case 4, 6, 9, 11:
		return 30
	}
	if year%4 == 0 && (year%100 != 0 || year%400 == 0) {
		return 29
	}
	return 28
}

func isDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return s != ""
}

// ToDdmmyyyy returns the canonical dd.mm.yyyy, or "" when the text does not yield one.
// Port of dates.to_ddmmyyyy (dates.py:60-104):
//
//	'22.06.2010'           -> '22.06.2010' (already canonical)
//	'15 ОКТЯБРЯ 2020 Г.'   -> '15.10.2020'
//	'10 ДЕКАБРЯ 1999 ГОДА' -> '10.12.1999'
//	'03.АВГУСТ.1989'       -> '03.08.1989'
//	'5 МАЯ'                -> ""  (no year: guessing one would invent data)
//	'31.02.2020'           -> ""  (not a calendar date)
func ToDdmmyyyy(text string) string {
	if text == "" {
		return ""
	}
	var tokens []string
	for _, t := range dateToken.FindAllString(text, -1) {
		t = strings.ToUpper(t)
		if dateNoise[t] || t == "Г" {
			continue
		}
		tokens = append(tokens, t)
	}
	if len(tokens) == 0 {
		return ""
	}

	day, month, year := -1, -1, -1
	for _, token := range tokens {
		if isDigits(token) {
			value, err := strconv.Atoi(token)
			if err != nil {
				return ""
			}
			switch {
			case len(token) == 4 && year < 0:
				year = value
			case day < 0 && value >= 1 && value <= 31:
				day = value
			case month < 0 && value >= 1 && value <= 12:
				month = value
			case year < 0 && len(token) <= 2:
				// A two-digit year is ambiguous (26 -> 1926 or 2026?) and this module
				// does not guess, so it is left unresolved.
				return ""
			}
			continue
		}
		resolved, ok := months[token]
		if !ok || month >= 0 {
			return ""
		}
		month = resolved
	}
	if day < 0 || month < 0 || year < 0 {
		return ""
	}
	return asDate(day, month, year)
}

// CanonicalDates builds the canonical view of every date field that yields one.
// Port of dates.canonical_dates.
//
// Returns a NEW map holding only the fields that converted - a field that did not
// convert is simply absent, so the consumer can tell "no canonical form" from "canonical
// form equals the reading". Never mutates ocr.
func CanonicalDates(ocr map[string]string, fields []string) map[string]string {
	out := map[string]string{}
	for _, name := range fields {
		value, ok := ocr[name]
		if !ok {
			continue
		}
		if canonical := ToDdmmyyyy(value); canonical != "" {
			out[name] = canonical
		}
	}
	return out
}

// NormalizeDates is Pipeline._normalize_dates: the date fields are recognised BY NAME,
// the same `'date' in name.lower()` convention _join_field uses. Runs once, on the
// finished OCR dict, so nothing upstream sees a rewritten value. Returns nil when no
// field converted, matching the absent 'OCR_normalized' key of the reference.
func NormalizeDates(ocr map[string]string, order []string) map[string]string {
	var fields []string
	for _, name := range order {
		if strings.Contains(strings.ToLower(name), "date") {
			fields = append(fields, name)
		}
	}
	out := CanonicalDates(ocr, fields)
	if len(out) == 0 {
		return nil
	}
	return out
}

// keep unicode imported for the letter-class note above; the regexp does the work.
var _ = unicode.IsLetter

package pipeline

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
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
// it is kept so the set reads the same as the reference's. «месяца» and «числа» belong
// to the 1998 birth certificate's record date, printed in reverse order around the
// values: «2010 года июня месяца 15 числа».
var dateNoise = map[string]bool{"Г": true, "Г.": true, "ГОД": true, "ГОДА": true, "ГОДУ": true,
	"МЕСЯЦ": true, "МЕСЯЦА": true, "ЧИСЛО": true, "ЧИСЛА": true}

// dateToken is `[^\W\d_]+|\d+` with re.UNICODE: runs of letters, or runs of digits.
// Go's RE2 has no \W-minus-\d class, so the letter run is spelled with the Unicode
// property, which is what Python's `[^\W\d_]` denotes (letters of any script).
var dateToken = regexp.MustCompile(`\pL+|\d+`)

// dropQuoteLetters drops a lone letter standing right next to the day number. Port of
// dates._drop_quote_letters (issue #23).
//
// The 1998 birth certificate prints the issue date as «10» ЯНВАРЯ 2013 г., and the field
// box starts on the opening quote. «» are not in the Cyrillic engine's alphabet, so the
// engine reads the quote as the nearest letter it knows: «И 10 ЯНВАРЯ 2013». The letter
// carries no date information, but as an unknown word it made the whole date refuse.
//
// Only a SINGLE letter and only ADJACENT to a one- or two-digit number (the day, on either
// side - the closing quote sits after it) is dropped. Anything else - a longer word, a
// letter elsewhere - still refuses: this reads a known misreading of printed punctuation,
// it does not guess. Length is counted in runes, as Python's len counts characters.
func dropQuoteLetters(tokens []string) []string {
	isDay := func(i int) bool {
		return i >= 0 && i < len(tokens) && isDigits(tokens[i]) && utf8.RuneCountInString(tokens[i]) <= 2
	}
	out := make([]string, 0, len(tokens))
	for i, t := range tokens {
		if utf8.RuneCountInString(t) == 1 && !isDigits(t) && months[t] == 0 &&
			(isDay(i-1) || isDay(i+1)) {
			continue
		}
		out = append(out, t)
	}
	return out
}

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
//	'И 10 ЯНВАРЯ 2013'     -> '10.01.2013' (the quote « read as a letter)
//	'2010 ГОДА ИЮНЯ МЕСЯЦА 15 ЧИСЛА' -> '15.06.2010' (record date, 1998 form)
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
	tokens = dropQuoteLetters(tokens)
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

// RecordDateFields are the fields printed as a civil-registry record date: year, month,
// day in a FIXED order with printed words between them - «2010 года июня месяца 15
// числа» on the 1998 birth certificate. The box spans the printed words, the word split
// often loses the gaps («2015ГОДАИЮНЯМЕСЯЦА16») and the printed words come back misread
// («ИЕСЯЦА», «ТОДА»), so the general converter refuses most of them. Mirrors
// RECORD_DATE_FIELDS in dates.py.
var RecordDateFields = []string{"Act_date"}

// genitive holds the month names in the genitive - the only case a record date prints.
// Built from `months` the way dates.py builds _GENITIVE: the names ending in Я or А.
var genitive = func() map[string]int {
	out := map[string]int{}
	for name, n := range months {
		if strings.HasSuffix(name, "Я") || strings.HasSuffix(name, "А") {
			out[name] = n
		}
	}
	return out
}()

// RecordDateToDdmmyyyy is the canonical dd.mm.yyyy of a civil-registry record date, or ""
// when the text does not yield one. Port of dates.record_date_to_ddmmyyyy.
//
// Whatever the general converter accepts is taken as is. Otherwise the parts are found by
// their FORM, which is what the fixed layout allows: exactly one four-digit year, exactly
// one one- or two-digit day, and exactly one genitive month name found INSIDE the letters
// (glued or not), whatever the printed words around it were read as. Any ambiguity - two
// days, two months, no year - refuses, as everywhere in this module:
//
//	'2015ГОДАИЮНЯИЕСЯЦА16'        -> '16.06.2015'
//	'2010 ГОДА ЦЮЛЯ МЕСЯЦА 17'    -> ""  (the month itself is misread)
//	'2020 ГОДА ИЮЛЯ МЕСЯЦА'       -> ""  (no day)
func RecordDateToDdmmyyyy(text string) string {
	canonical := ToDdmmyyyy(text)
	if canonical != "" || text == "" {
		return canonical
	}
	var years, days []string
	stray := false
	var letters strings.Builder
	for _, run := range dateToken.FindAllString(strings.ToUpper(text), -1) {
		switch {
		case !isDigits(run):
			letters.WriteString(run)
		case len(run) == 4:
			years = append(years, run)
		case len(run) <= 2:
			days = append(days, run)
		default:
			stray = true
		}
	}
	if len(years) != 1 || len(days) != 1 || stray {
		return ""
	}
	found := map[int]bool{}
	joined := letters.String()
	for name, n := range genitive {
		if strings.Contains(joined, name) {
			found[n] = true
		}
	}
	if len(found) != 1 {
		return ""
	}
	day, _ := strconv.Atoi(days[0])
	year, _ := strconv.Atoi(years[0])
	for month := range found {
		return asDate(day, month, year)
	}
	return ""
}

// CanonicalDate is the canonical view of one field: by its printed layout. Port of
// dates.canonical_date.
func CanonicalDate(field, text string) string {
	if contains(RecordDateFields, field) {
		return RecordDateToDdmmyyyy(text)
	}
	return ToDdmmyyyy(text)
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
		if canonical := CanonicalDate(name, value); canonical != "" {
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

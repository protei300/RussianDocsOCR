package modules

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Field-level text corrections shared by the two OCR engines.
// Port of pipeline_modules/ocr_corrections.py.
//
// These are semantic fixes applied on top of the raw decoded string, and each one is a
// CER-validated win rather than a tidy-up — reproduce them exactly, including the cases
// where they look careless.

// CheckDdmmyyyy normalises a recognised date to dd.mm.yyyy, best effort.
//
// THREE outcomes, and the difference between the last two is easy to miss because in the
// reference it is spread across two functions:
//
//   - eight digits forming a real date -> reformatted;
//   - NOT eight digits -> the SUBSTITUTED string ("O6-13-1985" -> "06.13.1985"), because
//     check_ddmmyyyy returns its local `date` variable;
//   - eight digits that are NOT a real date -> the UNTOUCHED ORIGINAL, because strptime
//     raises and the except in fix_errors returns the argument it was given, which never
//     saw the substitutions.
//
// Also: the 'O' -> '0' substitution is UPPERCASE ONLY. A lowercase 'o' is left alone, and
// the printed models emit uppercase.
func CheckDdmmyyyy(date string) string {
	s := strings.ReplaceAll(date, "O", "0")
	s = strings.ReplaceAll(s, "-", ".")

	var digits []rune
	for _, r := range s {
		// unicode.IsDigit matches Python's str.isnumeric closely enough for this
		// alphabet, which contains ASCII digits only.
		if unicode.IsDigit(r) {
			digits = append(digits, r)
		}
	}
	if len(digits) != 8 {
		return s
	}
	pure := string(digits)

	// strptime('%d%m%Y') VALIDATES: an impossible date raises, and the reference catches
	// that and returns the text unchanged. A port that only reformats would turn
	// "32.13.1998" into a confident-looking wrong date.
	day, _ := strconv.Atoi(pure[0:2])
	month, _ := strconv.Atoi(pure[2:4])
	year, _ := strconv.Atoi(pure[4:8])
	if !validDate(day, month, year) {
		// The ORIGINAL, not `s`: see the third outcome in the doc comment.
		return date
	}
	return fmt.Sprintf("%02d.%02d.%04d", day, month, year)
}

// validDate mirrors what strptime accepts: a real calendar date, leap years included.
//
// Deliberately not time.Parse: Go's parser accepts a year 0 and has its own normalisation
// rules, and the question here is only whether Python would have raised.
func validDate(day, month, year int) bool {
	if month < 1 || month > 12 || day < 1 {
		return false
	}
	// strptime('%Y') accepts 1..9999; year 0 is rejected.
	if year < 1 || year > 9999 {
		return false
	}
	days := [...]int{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
	limit := days[month-1]
	if month == 2 && (year%4 == 0 && (year%100 != 0 || year%400 == 0)) {
		limit = 29
	}
	return day <= limit
}

// CheckEnSex standardises Latin sex text to M or F.
//
// Note the asymmetry, which is behaviour and not a bug: anything WITHOUT an 'M' becomes
// 'F', including an empty string. The field is binary and the detector only fires on a
// sex box, so guessing F is better than emitting noise.
func CheckEnSex(sex string) string {
	toCheck := strings.ReplaceAll(strings.ToUpper(strings.TrimLeft(sex, ".")), ".", "")
	if strings.Contains(toCheck, "M") {
		return "M"
	}
	return "F"
}

// CheckRusSex standardises Cyrillic sex text to М or Ж. Same asymmetry as CheckEnSex.
//
// The letters here are CYRILLIC М (U+041C) and Ж (U+0416), not Latin M. They look
// identical in most fonts and a Latin M would silently fail every comparison downstream.
func CheckRusSex(sex string) string {
	toCheck := strings.ReplaceAll(strings.ToUpper(strings.TrimLeft(sex, ".")), ".", "")
	if strings.ContainsRune(toCheck, 'М') {
		return "М"
	}
	return "Ж"
}

// CheckDriverClass keeps only valid driver-category characters.
//
// The allowed set is ASCII "ABCDEM1" — Latin letters, because the categories are printed
// in Latin on the licence.
func CheckDriverClass(driverClass string) string {
	const allowed = "ABCDEM1"
	var b strings.Builder
	for _, r := range strings.ReplaceAll(driverClass, " ", "") {
		if strings.ContainsRune(allowed, r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// StripEdgeDots removes stray LEADING dots the detector sometimes prepends to names.
//
// Leading only, despite the name: Python's lstrip. A trailing dot is left in place.
func StripEdgeDots(name string) string { return strings.TrimLeft(name, ".") }

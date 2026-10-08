package pipeline

import "testing"

// The cases below mirror tests/test_date_canon.py of the reference, the ones added with
// the record date (Act_date) of a birth certificate, and the quote-letter cases at the end.

// What the documents print: the 1998 form puts the printed words «года», «месяца»,
// «числа» between the parts, the 2018 form uses the usual order.
func TestToDdmmyyyyRecordDateWords(t *testing.T) {
	for _, c := range []struct{ printed, want string }{
		{"2010 ГОДА ИЮНЯ МЕСЯЦА 15 ЧИСЛА", "15.06.2010"},
		{"2010 года июня месяца 15", "15.06.2010"}, // «числа» left outside the box
		{"2 МАРТА 2025 Г.", "02.03.2025"},          // BIRTHCERT_2018
		{"15 ОКТЯБРЯ 2020 Г.", "15.10.2020"},
		{"22.06.2010", "22.06.2010"},
	} {
		if got := ToDdmmyyyy(c.printed); got != c.want {
			t.Errorf("ToDdmmyyyy(%q) = %q, want %q", c.printed, got, c.want)
		}
	}
}

// Inputs the general converter must REFUSE rather than invent.
func TestToDdmmyyyyRecordDateRefusals(t *testing.T) {
	for _, text := range []string{
		"2010 ГОДА ИЮНЯ МЕСЯЦА 31 ЧИСЛА", // there is no 31 June
		"2010 ГОДА ИЮНЯ МЕСЯЦА ЧИСЛА",    // no day
	} {
		if got := ToDdmmyyyy(text); got != "" {
			t.Errorf("ToDdmmyyyy(%q) = %q, want a refusal", text, got)
		}
	}
}

// The record date finds its parts by FORM: glued words and misread «года»/«месяца» do
// not matter.
func TestRecordDateReadsThroughGlueAndMisreadWords(t *testing.T) {
	for _, c := range []struct{ printed, want string }{
		{"2015ГОДАИЮНЯИЕСЯЦА16", "16.06.2015"}, // glued, «месяца» misread
		{"2026ТОДАМАЯНСЯЦА3", "03.05.2026"},
		{"2002ТОДАИЮНЯ,МЕСЯЦА18", "18.06.2002"},
		{"2003 ДЕКАБРЯ МЕСЯЦА 27", "27.12.2003"},
		{"2 МАРТА 2025 Г.", "02.03.2025"}, // usual order goes through the general reading
	} {
		if got := RecordDateToDdmmyyyy(c.printed); got != c.want {
			t.Errorf("RecordDateToDdmmyyyy(%q) = %q, want %q", c.printed, got, c.want)
		}
	}
}

// ... and still refuses on any ambiguity, as everywhere in this module.
func TestRecordDateStillRefusesRatherThanGuesses(t *testing.T) {
	for _, text := range []string{
		"2010 ГОДА ЦЮЛЯ МЕСЯЦА 17", // the month itself is misread
		"2020 ГОДА ИЮЛЯ МЕСЯЦА",    // no day
		"2020 ЯНВАРЯ 110201114335", // the record number got into the box
		".110266032",
		"2010 ИЮНЯ МАЯ 15",  // two months
		"2010ГОДАИЮНЯ 15 16", // two days
		"",
	} {
		if got := RecordDateToDdmmyyyy(text); got != "" {
			t.Errorf("RecordDateToDdmmyyyy(%q) = %q, want a refusal", text, got)
		}
	}
}

// Only the record date gets the lenient reading; any other field keeps the strict one.
func TestOnlyTheRecordDateGetsTheLenientReading(t *testing.T) {
	ocr := map[string]string{"Act_date": "2015ГОДАИЮНЯИЕСЯЦА16", "Issue_date": "2015ГОДАИЮНЯИЕСЯЦА16"}
	got := CanonicalDates(ocr, []string{"Act_date", "Issue_date"})
	if len(got) != 1 || got["Act_date"] != "16.06.2015" {
		t.Fatalf("got %v, want only Act_date -> 16.06.2015", got)
	}
}

// The fields of a series/number are never looked at: the caller passes the date fields.
func TestNonDateFieldsAreNeverTouched(t *testing.T) {
	ocr := map[string]string{"Licence_number": "62 1483828", "Act_number": "110202778751843181007"}
	if got := CanonicalDates(ocr, []string{"Birth_date"}); len(got) != 0 {
		t.Fatalf("got %v, want nothing", got)
	}
}

// Act_date is a field of the birth certificate, read by the Cyrillic engine with the word
// split; both blanks share one options branch.
func TestBirthcertOptionsCarryActDate(t *testing.T) {
	for _, docType := range []string{"BIRTHCERT_1998", "BIRTHCERT_2018"} {
		o := MakeOcrOptions(docType)
		if !o.NeedsSplit("Act_date") {
			t.Errorf("%s: Act_date is not word-split", docType)
		}
		if !contains(o.RuFields, "Act_date") || contains(o.EnFields, "Act_date") {
			t.Errorf("%s: Act_date must be routed to the Cyrillic engine only", docType)
		}
	}
}

// The quote cases of tests/test_date_canon.py (issue #23): the 1998 blank prints the issue
// date as «10» ЯНВАРЯ 2013 г. and the engine reads the quotes as letters. A single letter
// right next to the day is a read quote; anything else still refuses.
func TestToDdmmyyyyQuoteLetters(t *testing.T) {
	for _, c := range []struct{ printed, want string }{
		{"И 10 ЯНВАРЯ 2013", "10.01.2013"},     // opening quote
		{"И10 ЯНВАРЯ 2013", "10.01.2013"},      // no space
		{"И 10 Н ЯНВАРЯ 2013", "10.01.2013"},   // both quotes
		{"10 П ЯНВАРЯ 2013 Г.", "10.01.2013"},  // closing quote
	} {
		if got := ToDdmmyyyy(c.printed); got != c.want {
			t.Errorf("ToDdmmyyyy(%q) = %q, want %q", c.printed, got, c.want)
		}
	}
}

func TestToDdmmyyyyQuoteLettersElsewhereStillRefuse(t *testing.T) {
	for _, text := range []string{
		"ИЗ 10 ЯНВАРЯ 2013", // a word, not a single letter
		"10 ЯНВАРЯ И 2013",  // the letter is next to the year, not the day
		"И 10 ЯНВАРЯ",       // quote dropped, but there is still no year
	} {
		if got := ToDdmmyyyy(text); got != "" {
			t.Errorf("ToDdmmyyyy(%q) = %q, want a refusal", text, got)
		}
	}
}

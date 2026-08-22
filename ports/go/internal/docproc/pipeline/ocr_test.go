package pipeline

import "testing"

// A digit date joins with '.', but a date spelled out in WORDS ("31 октября 1998") joins
// with spaces -- joining those with '.' would produce "31.октября.1998". SNILS is worded by
// definition; a birth certificate is worded by content (the 2018 blank spells every date
// out, the 1998 blank keeps a digit Birth_date).
func TestJoinFieldDateSeparatorFollowsContent(t *testing.T) {
	if got := joinField(map[string]string{}, "Birth_date", "DL",
		[]string{"06", "01", "1985"}); got != "06.01.1985" {
		t.Fatalf("got %q, want 06.01.1985", got)
	}
	if got := joinField(map[string]string{}, "Birth_date", "SNILS",
		[]string{"26", "СЕНТЯБРЯ", "1997"}); got != "26 СЕНТЯБРЯ 1997" {
		t.Fatalf("got %q, want the space-joined form", got)
	}
	if got := joinField(map[string]string{}, "Birth_date", "BIRTHCERT",
		[]string{"15", "ОКТЯБРЯ", "2020", "Г."}); got != "15 ОКТЯБРЯ 2020 Г." {
		t.Fatalf("got %q, want the space-joined form", got)
	}
	if got := joinField(map[string]string{}, "Birth_date", "BIRTHCERT",
		[]string{"22", "06", "2010"}); got != "22.06.2010" {
		t.Fatalf("got %q, want 22.06.2010 -- the 1998 blank's digit birth date", got)
	}
}

// The date test is a case-insensitive SUBSTRING of the label, so Issue_date,
// Expiration_date and Birth_date all match -- and so would anything else containing it.
func TestJoinFieldDetectsDateBySubstring(t *testing.T) {
	for _, label := range []string{"Issue_date", "Expiration_date", "Birth_date"} {
		if got := joinField(map[string]string{}, label, "DL", []string{"1", "2"}); got != "1.2" {
			t.Errorf("%s: got %q, want the dot-joined form", label, got)
		}
	}
	if got := joinField(map[string]string{}, "Last_name_ru", "DL",
		[]string{"1", "2"}); got != "1 2" {
		t.Fatalf("a non-date field must space-join, got %q", got)
	}
}

// A single pass of "  " -> " ", matching Python's str.replace. THREE spaces leave one
// behind in both implementations; collapsing fully would produce a different string.
func TestJoinFieldCollapsesDoubleSpacesInOnePassOnly(t *testing.T) {
	got := joinField(map[string]string{}, "Last_name_ru", "DL", []string{"A", "", "", "B"})
	// "A" + " " + "" + " " + "" + " " + "B" == "A   B"; one pass removes one pair.
	if got != "A  B" {
		t.Fatalf("got %q, want %q -- a single replace pass, not a full collapse", got, "A  B")
	}
}

// Surrounding whitespace is trimmed, so a field whose words are all empty comes out
// empty rather than as a run of spaces.
func TestJoinFieldTrims(t *testing.T) {
	if got := joinField(map[string]string{}, "Last_name_ru", "DL", []string{"", ""}); got != "" {
		t.Fatalf("got %q, want empty", got)
	}
}

// A non-date field APPENDS to an existing value rather than replacing it. Inert on the
// serial path (each label is visited once) but carried faithfully, because the reference
// relies on the accumulated dict and the address path can pre-populate it.
func TestJoinFieldAppendsToExistingValue(t *testing.T) {
	acc := map[string]string{"Birth_place_ru": "ТОМСКАЯ"}
	if got := joinField(acc, "Birth_place_ru", "INTPASSPORT",
		[]string{"ОБЛАСТЬ"}); got != "ТОМСКАЯ ОБЛАСТЬ" {
		t.Fatalf("got %q, want the appended form", got)
	}
}

// A DATE field, by contrast, ignores any accumulated value entirely -- its branch
// assigns rather than appends.
func TestJoinFieldDateIgnoresAccumulator(t *testing.T) {
	acc := map[string]string{"Issue_date": "01.01.2000"}
	if got := joinField(acc, "Issue_date", "DL", []string{"28", "06", "2016"}); got != "28.06.2016" {
		t.Fatalf("got %q, want the date to replace, not append", got)
	}
}

// The OCR routing compares the BARE type with == "SNILS", which the '<TYPE>_<YEAR>'
// label never equals -- so the split has to happen, and a label with no underscore must
// not be an error.
func TestSplitDocType(t *testing.T) {
	cases := []struct{ in, bare, year string }{
		{"SNILS_1996", "SNILS", "1996"},
		{"INTPASSPORT_2011", "INTPASSPORT", "2011"},
		{"INTPASSPORTADDR_ALL", "INTPASSPORTADDR", "ALL"},
		{"SNILS", "SNILS", ""},
		{"", "", ""},
	}
	for _, c := range cases {
		bare, year := SplitDocType(c.in)
		if bare != c.bare || year != c.year {
			t.Errorf("SplitDocType(%q) = (%q,%q), want (%q,%q)", c.in, bare, year, c.bare, c.year)
		}
	}
}

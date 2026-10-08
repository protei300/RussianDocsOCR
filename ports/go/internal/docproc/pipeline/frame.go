package pipeline

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// PairedSides is PAIRED_SIDES: front type -> back type of a two-sided document whose sides
// carry the same number. The vehicle registration certificate prints its series and number
// on both sides (read as Licence_number on each), so a front and a back lying in one frame -
// one sheet of a two-sided scan - pair up by it (issue #26). A licence back carries no number
// the pipeline reads, so it is not here.
var PairedSides = map[string]string{"STS": "STSBACK"}

// documentNumber is _document_number: the digits of the series and number read on a side;
// "" when nothing was read.
func documentNumber(r *Results) string {
	var b strings.Builder
	for _, c := range r.Ocr["Licence_number"] {
		if unicode.IsDigit(c) {
			b.WriteRune(c)
		}
	}
	return b.String()
}

// PairSides is pair_sides: set PairedWith on the two sides of one document read from one
// frame.
//
// A front and a back pair when their types belong together (PairedSides), they read the same
// number, and that number is unique among the sides of that type in the frame - two
// certificates scanned on one sheet must not cross-pair, and an ambiguous number pairs
// nothing rather than guessing. Every other document keeps PairedWith as it was (-1).
func PairSides(documents []*Results) {
	family := func(r *Results) string {
		label := r.DocType
		if label == "" {
			label = "NONE"
		}
		if i := strings.LastIndex(label, "_"); i >= 0 {
			return label[:i]
		}
		return label
	}
	for front, back := range PairedSides {
		fronts, backs := map[string][]int{}, map[string][]int{}
		for i, r := range documents {
			number := documentNumber(r)
			if utf8.RuneCountInString(number) < 6 {
				continue
			}
			switch family(r) {
			case front:
				fronts[number] = append(fronts[number], i)
			case back:
				backs[number] = append(backs[number], i)
			}
		}
		for number, f := range fronts {
			b := backs[number]
			if len(f) == 1 && len(b) == 1 {
				documents[f[0]].PairedWith = b[0]
				documents[b[0]].PairedWith = f[0]
			}
		}
	}
}

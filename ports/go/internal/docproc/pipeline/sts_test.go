package pipeline

import (
	"reflect"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Mirrors tests/test_sts_options.py, tests/test_sts_marks.py, tests/test_sts_reading.py,
// tests/test_paired_duplicates.py and the pairing part of tests/test_document_detector.py of
// the reference. No models here.

// ---- options (test_sts_options.py)

var (
	stsFront = []string{"Reg_number", "VIN", "Vehicle_make_ru", "Vehicle_make_en", "Vehicle_type",
		"Vehicle_category", "Vehicle_year", "Chassis_number", "Body_number",
		"Vehicle_color", "Engine_power", "Eco_class", "Max_mass", "Curb_mass",
		"Expiration_date", "PTS_number", "Licence_number"}
	stsBack = []string{"Licence_number", "Last_name_ru", "Last_name_en", "First_name_ru",
		"First_name_en", "Middle_name_ru", "Living_region_ru", "House_number",
		"Apartment_number", "Special_marks", "Issue_organisation_code", "Issue_date"}
	stsNewForm = []string{"Type_approval", "Building_number"}
	stsOld2010 = []string{"Engine_model", "Engine_number", "Engine_volume",
		"Issue_organization_ru", "Issue_date", "Building_number"}
)

func TestBothStsSidesReachTheSameOptions(t *testing.T) {
	want := optionsSTS()
	for _, label := range []string{"STS", "STS_1996", "STSBACK", "STSBACK_1996", "STS_2019", "STSBACK_2019"} {
		bare, _ := SplitDocType(label)
		if got := MakeOcrOptions(bare); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: not the STS options", label)
		}
	}
}

// The dispatcher matches substrings in order; a new branch must not shadow an old type (the
// intpassportaddr/intpassport trap).
func TestNoOtherTypeIsCaughtByTheStsBranch(t *testing.T) {
	sts := optionsSTS()
	for _, label := range []string{"INTPASSPORT_2011", "INTPASSPORTADDR_ALL", "EXTPASSPORT_2003",
		"DL_2011", "SNILS_1996", "BIRTHCERT_2018", "DLBACK_ALL"} {
		bare, _ := SplitDocType(label)
		if reflect.DeepEqual(MakeOcrOptions(bare), sts) {
			t.Errorf("%s is routed to the STS options", label)
		}
	}
}

// 'dlback' contains 'dl': the licence back has no field model yet and must read nothing.
func TestDlbackReadsNothing(t *testing.T) {
	got := MakeOcrOptions("DLBACK")
	if len(got.RuFields)+len(got.EnFields)+len(got.NeededSplit) != 0 {
		t.Errorf("DLBACK has fields: %+v", got)
	}
}

func TestEveryStsFieldIsRoutedToAnEngine(t *testing.T) {
	o := optionsSTS()
	for name, fields := range map[string][]string{"front": stsFront, "back": stsBack,
		"new-form": stsNewForm, "old-2010": stsOld2010} {
		for _, f := range fields {
			if !o.IsOcrField(f) {
				t.Errorf("%s: %s is detected but never read", name, f)
			}
		}
	}
}

func TestNoStsFieldIsRoutedToBothEngines(t *testing.T) {
	o := optionsSTS()
	for _, f := range o.RuFields {
		if contains(o.EnFields, f) {
			t.Errorf("%s is in both lists", f)
		}
	}
}

// Decision of 2026-09-05: plate letters and VIN are Latin; the series keeps the passport
// precedent (digits read better on the Cyrillic engine, issue #12).
func TestRegNumberAndVinAreLatinAndTheSeriesIsCyrillic(t *testing.T) {
	o := optionsSTS()
	if !contains(o.EnFields, "Reg_number") || !contains(o.EnFields, "VIN") {
		t.Error("Reg_number and VIN must go to the Latin engine")
	}
	if !contains(o.RuFields, "Licence_number") {
		t.Error("the series must go to the Cyrillic engine")
	}
}

// The detector v10 has no model class (a7b12e81).
func TestThereIsNoModelField(t *testing.T) {
	o := optionsSTS()
	if o.IsOcrField("Vehicle_model_en") || o.NeedsSplit("Vehicle_model_en") {
		t.Error("Vehicle_model_en is still in the STS options")
	}
}

// ---- engine by year, read margin (test_sts_reading.py)

func TestTheMakeEngineFollowsTheStsForm(t *testing.T) {
	o := optionsSTS()
	if !contains(o.RuFields, "Vehicle_make_ru") {
		t.Error("the old form's route is the Cyrillic one")
	}
	if len(o.ReadMargin) != 1 || o.ReadMargin["Special_marks"] == 0 {
		t.Errorf("read margin = %v", o.ReadMargin)
	}
	if o.EngineFor("Vehicle_make_ru", "2019") != "lat" {
		t.Error("the new form prints the make in Latin")
	}
	if o.EngineFor("Vehicle_make_ru", "1996") != "" {
		t.Error("the old form keeps the Cyrillic route")
	}
	if o.EngineFor("Vehicle_color", "2019") != "" {
		t.Error("only the make follows the year")
	}
	empty := OcrOptions{}
	if empty.EngineFor("Vehicle_make_ru", "2019") != "" || len(empty.ReadMargin) != 0 {
		t.Error("other types must be untouched")
	}
}

// marginCanvas is a grey 300 x 200 image whose every pixel holds its row number.
func marginCanvas(t *testing.T) imaging.Image {
	t.Helper()
	buf := make([]byte, 200*300)
	for y := 0; y < 200; y++ {
		for x := 0; x < 300; x++ {
			buf[y*300+x] = byte(y)
		}
	}
	img, err := imaging.NewGrayFromBytes(buf, 300, 200)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func fieldAt(t *testing.T, label string, x1, y1, x2, y2 int, canvas imaging.Image) modules.Field {
	t.Helper()
	patch, err := imaging.ClampedCrop(canvas, x1, y1, x2, y2)
	if err != nil {
		t.Fatal(err)
	}
	return modules.Field{
		Box: postprocess.Box{X1: float64(x1), Y1: float64(y1), X2: float64(x2), Y2: float64(y2),
			Conf: 0.9, Label: label},
		Patch: patch,
	}
}

// rowsOf is the first row number, the last row number and the height of a patch.
func rowsOf(t *testing.T, img imaging.Image) (first, last byte, h int) {
	t.Helper()
	b, err := img.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	return b[0], b[(img.Height()-1)*img.Width()], img.Height()
}

func TestTheReadCropGrowsAndTheBoxDoesNot(t *testing.T) {
	canvas := marginCanvas(t)
	defer canvas.Close()
	fields := []modules.Field{fieldAt(t, "Special_marks", 20, 100, 280, 120, canvas)}
	defer modules.FieldsClose(fields)
	box := fields[0].Box
	frames := framesOf(fields)
	if err := readMargins(fields, OcrOptions{ReadMargin: map[string]float64{"Special_marks": 0.25}}, canvas, frames); err != nil {
		t.Fatal(err)
	}
	// the frame moved with the crop (test_the_read_crop_grows_and_the_box_does_not)
	if p, _ := frames[0].Map.ToInput([]geometry.Point{{X: 0, Y: 0}}); p[0].Y != 95 || frames[0].H != 30 {
		t.Errorf("frame puts the patch top at %v, height %d; want 95, 30", p[0].Y, frames[0].H)
	}
	first, last, h := rowsOf(t, fields[0].Patch)
	if fields[0].Box.X1 != box.X1 || fields[0].Box.Y1 != box.Y1 || fields[0].Box.Y2 != box.Y2 {
		t.Error("the box changed")
	}
	if h != 30 || first != 95 || last != 124 { // 20 px + 5 above + 5 below
		t.Errorf("height %d rows %d..%d, want 30 rows 95..124", h, first, last)
	}
}

func TestTheMarginStopsHalfwayToTheNextLine(t *testing.T) {
	canvas := marginCanvas(t)
	defer canvas.Close()
	fields := []modules.Field{
		fieldAt(t, "Special_marks", 20, 80, 280, 96, canvas),
		fieldAt(t, "Special_marks", 20, 100, 280, 116, canvas),
	}
	defer modules.FieldsClose(fields)
	if err := readMargins(fields, OcrOptions{ReadMargin: map[string]float64{"Special_marks": 0.5}}, canvas, framesOf(fields)); err != nil {
		t.Fatal(err)
	}
	_, lastUpper, _ := rowsOf(t, fields[0].Patch)
	firstLower, _, _ := rowsOf(t, fields[1].Patch)
	if lastUpper != 97 || firstLower != 98 {
		t.Errorf("upper ends at %d, lower starts at %d, want 97 and 98 (no overlap)", lastUpper, firstLower)
	}
}

func TestABoxBesideItDoesNotLimitTheMargin(t *testing.T) {
	canvas := marginCanvas(t)
	defer canvas.Close()
	fields := []modules.Field{
		fieldAt(t, "Special_marks", 20, 100, 140, 120, canvas),
		fieldAt(t, "Reg_number", 160, 80, 280, 98, canvas), // no shared width
	}
	defer modules.FieldsClose(fields)
	if err := readMargins(fields, OcrOptions{ReadMargin: map[string]float64{"Special_marks": 0.25}}, canvas, framesOf(fields)); err != nil {
		t.Fatal(err)
	}
	if h := fields[0].Patch.Height(); h != 30 {
		t.Errorf("height %d, want 30", h)
	}
}

func TestOtherFieldsAreUntouchedByTheMargin(t *testing.T) {
	canvas := marginCanvas(t)
	defer canvas.Close()
	fields := []modules.Field{fieldAt(t, "Vehicle_color", 20, 100, 280, 120, canvas)}
	defer modules.FieldsClose(fields)
	if err := readMargins(fields, OcrOptions{ReadMargin: map[string]float64{"Special_marks": 0.25}}, canvas, framesOf(fields)); err != nil {
		t.Fatal(err)
	}
	if h := fields[0].Patch.Height(); h != 20 {
		t.Errorf("height %d, want 20", h)
	}
}

// ---- paired duplicates (test_paired_duplicates.py)

func lbl(label string, conf float64, x1, y1, x2, y2 int) modules.Field {
	return modules.Field{Box: postprocess.Box{X1: float64(x1), Y1: float64(y1), X2: float64(x2),
		Y2: float64(y2), Conf: conf, Label: label}}
}

func dropped(fields ...modules.Field) []int {
	var out []int
	drop := pairedDuplicateIndices(fields)
	for i := range fields {
		if drop[i] {
			out = append(out, i)
		}
	}
	return out
}

func TestSameLineLabelledBothWaysKeepsTheConfidentLabel(t *testing.T) {
	got := dropped(lbl("Birth_place_ru", 0.922, 513, 384, 801, 423), lbl("Birth_place_en", 0.619, 513, 384, 801, 422))
	if !reflect.DeepEqual(got, []int{1}) {
		t.Errorf("dropped %v, want [1]", got)
	}
	got = dropped(lbl("Birth_place_ru", 0.55, 100, 10, 300, 40), lbl("Birth_place_en", 0.90, 100, 10, 300, 40))
	if !reflect.DeepEqual(got, []int{0}) {
		t.Errorf("dropped %v, want [0]", got)
	}
}

func TestRealPairsAreKept(t *testing.T) {
	sideBySide := dropped(lbl("Birth_place_ru", 0.95, 407, 291, 571, 317), lbl("Birth_place_en", 0.93, 582, 289, 652, 316))
	stacked := dropped(lbl("Last_name_ru", 0.9, 100, 100, 400, 140), lbl("Last_name_en", 0.9, 100, 116, 400, 156))
	other := dropped(lbl("Birth_place_ru", 0.9, 100, 10, 300, 40), lbl("Last_name_en", 0.6, 100, 10, 300, 40))
	if len(sideBySide)+len(stacked)+len(other) != 0 {
		t.Errorf("dropped %v %v %v, want nothing", sideBySide, stacked, other)
	}
}

// ---- special marks: torn words and leasing (test_sts_marks.py)

func TestAKnownWordTornByTheLineBreakIsGlued(t *testing.T) {
	cases := []struct {
		lines [][]string
		want  []string
	}{
		{[][]string{{"ПО", "ДЛ", "ЛИЗИ"}, {"НГА", "ОТ"}}, []string{"ПО", "ДЛ", "ЛИЗИНГА", "ОТ"}},
		{[][]string{{"ЛИЗИН"}, {"ГОДАТЕЛЬ", "АО"}}, []string{"ЛИЗИНГОДАТЕЛЬ", "АО"}},
		{[][]string{{"ЛИЗИНГОВАЯ", "КОМ"}, {"ПАНИЯ"}}, []string{"ЛИЗИНГОВАЯ", "КОМПАНИЯ"}},
		{[][]string{{"ГАЗПРОМБАНК", "АВТОЛ"}, {"ИЗИНГ"}}, []string{"ГАЗПРОМБАНК", "АВТОЛИЗИНГ"}},
	}
	for _, c := range cases {
		if got := GlueTornWords(c.lines); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%v -> %v, want %v", c.lines, got, c.want)
		}
	}
}

func TestWordsThatAreNotATornKnownWordStayApart(t *testing.T) {
	for _, lines := range [][][]string{
		{{"ЛИЗИНГ"}, {"ДОГОВОР"}},          // two whole words
		{{"ЛИЗИНГОДАТЕЛЬ"}, {"АО", "ВТБ"}}, // a whole word, then a name
		{{"ООО", "КАРКА"}, {"ПЕТРОВ"}},     // joined is not a known word
		{{"№АЛ2682"}, {"26/01-26"}},        // numbers are never glued
	} {
		var flat []string
		for _, l := range lines {
			flat = append(flat, l...)
		}
		if got := GlueTornWords(lines); !reflect.DeepEqual(got, flat) {
			t.Errorf("%v -> %v, want %v", lines, got, flat)
		}
	}
}

func TestEmptyLinesAreSkipped(t *testing.T) {
	got := GlueTornWords([][]string{{}, {"ЛИЗИ"}, {""}, {"НГ"}})
	if !reflect.DeepEqual(got, []string{"ЛИЗИНГ"}) {
		t.Errorf("got %v", got)
	}
}

func TestTheFullRecordWithTheLessorOnTheNextLine(t *testing.T) {
	got := ParseLeasing("ПО ДЛ №АЛ268226/01-26 ОТ 12.03.2024 ЛИЗИНГОДАТЕЛЬ АО ВТБ ЛИЗИНГ")
	want := &Leasing{Leasing: true, Role: "lessor_named", Lessor: "АО ВТБ ЛИЗИНГ",
		ContractNumber: "АЛ268226/01-26", ContractDate: "12.03.2024", ContractDateNormalized: "12.03.2024"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestRealAbbreviations(t *testing.T) {
	got := ParseLeasing("ЛИЗИНГОПОЛУЧАТЕЛЬ ИВАНОВ ИВАН ЛИЗИНГОДАТЕЛЬ АО ЛИЗИНГОВАЯ КОМПАНИЯ " +
		"КАМАЗ ЛИЗИНГ ВРЕМ. УЧЕТ ДО 01.01.2027")
	if got.Role != "lessor_named" || got.Lessor != "АО ЛИЗИНГОВАЯ КОМПАНИЯ КАМАЗ ЛИЗИНГ" {
		t.Errorf("got %+v", got)
	}
	got = ParseLeasing("ЛИЗИНГ ДО 31.12.2027. ДОГ ЛИЗ АХ ЭЛ/УЛН-123456/ДЛ ООО ЭЛЕМЕНТ")
	if got.UntilNormalized != "31.12.2027" || got.ContractNumber != "АХ ЭЛ/УЛН-123456/ДЛ" ||
		got.ContractDate != "" { // «ДО» is the term, not the contract
		t.Errorf("got %+v", got)
	}
	got = ParseLeasing("ЛИЗИНГОДАТЕЛЬ ООО АВТОЛИЗИНГ, ДЕЙСТВИТЕЛЬНО ДО 01.02.2026")
	if got.Lessor != "ООО АВТОЛИЗИНГ" {
		t.Errorf("got %+v", got)
	}
}

func TestTheLessorEndsWhereTheContractBegins(t *testing.T) {
	got := ParseLeasing(`В ЛИЗИНГЕ ЛИЗИНГОДАТЕЛЬ ООО "КАРКАДЕ" ДОГОВОР ЛИЗИНГА №45120 ОТ 05.11.2016`)
	if got.Lessor != `ООО "КАРКАДЕ"` || got.ContractNumber != "45120" || got.ContractDateNormalized != "05.11.2016" {
		t.Errorf("got %+v", got)
	}
}

func TestTheOwnerAsLesseeNamesNoLessor(t *testing.T) {
	got := ParseLeasing("ЛИЗИНГОПОЛУЧАТЕЛЬ ДОГОВОР №1234-Л ОТ 03.04.2015")
	if got.Role != "lessee" || got.Lessor != "" || got.ContractNumber != "1234-Л" {
		t.Errorf("got %+v", got)
	}
}

func TestThe2010ShortForm(t *testing.T) {
	got := ParseLeasing("77 1234 Л.Д 14.02.2013")
	if got == nil || !got.Leasing || got.ContractDateNormalized != "14.02.2013" || got.Role != "" || got.Lessor != "" {
		t.Errorf("got %+v", got)
	}
}

func TestNoLeasingRecord(t *testing.T) {
	for _, text := range []string{"ДУБЛИКАТ", "СМЕНА СОБСТВЕННИКА", "", "ЛИЗИ НГ"} { // a tear left unglued is not found
		if got := ParseLeasing(text); got != nil {
			t.Errorf("%q: got %+v", text, got)
		}
	}
}

// The word boundary of the reference is Unicode-aware; RE2's is not. «ДОГ» inside a Cyrillic
// word is not a word start, and «ДО» only ends the name when it is a word of its own.
func TestWordBoundariesAreUnicodeAware(t *testing.T) {
	got := ParseLeasing("ЛИЗИНГОДАТЕЛЬ ООО ПОДОГРЕВ ДО ВТОРНИКА")
	if got == nil || got.Lessor != "ООО ПОДОГРЕВ" {
		t.Errorf("got %+v", got)
	}
}

func TestThePipelineGluesOnlyTheFieldsTheOptionsName(t *testing.T) {
	o := optionsSTS()
	marks := FieldWords{Label: "Special_marks", Lines: []int{2, 2}}
	if got := glueTorn(marks, []string{"ПО", "ЛИЗИ", "НГА", "ОТ"}, o); !reflect.DeepEqual(got, []string{"ПО", "ЛИЗИНГА", "ОТ"}) {
		t.Errorf("got %v", got)
	}
	name := FieldWords{Label: "Last_name_ru", Lines: []int{1, 1}}
	if got := glueTorn(name, []string{"ЛИЗИ", "НГА"}, o); !reflect.DeepEqual(got, []string{"ЛИЗИ", "НГА"}) {
		t.Errorf("got %v", got)
	}
	// line lengths that do not add up to the words: left alone
	if got := glueTorn(marks, []string{"ЛИЗИ", "НГА"}, o); !reflect.DeepEqual(got, []string{"ЛИЗИ", "НГА"}) {
		t.Errorf("got %v", got)
	}
	if got := glueTorn(marks, []string{"ЛИЗИ", "НГА", "x", "y"}, OcrOptions{}); len(got) != 4 {
		t.Error("another type must not glue")
	}
}

func TestOnlyTheFlagReachesTheResults(t *testing.T) {
	rec := ParseLeasing("ПО ДЛ №АЛ1/01-26 ОТ 12.03.2024 ЛИЗИНГОДАТЕЛЬ АО ВТБ ЛИЗИНГ")
	if got := reportedLeasing(rec); !reflect.DeepEqual(got, map[string]any{"leasing": true}) {
		t.Errorf("got %v", got)
	}
}

// ---- process_frame: the two sides of a certificate (test_document_detector.py)

func sideOf(docType, number string) *Results {
	return &Results{DocType: docType, Ocr: map[string]string{"Licence_number": number}, PairedWith: -1}
}

func TestTheTwoSidesOfOneCertificatePair(t *testing.T) {
	docs := []*Results{sideOf("STS_2019", "77 УХ 123456"), sideOf("DL_2011", "12 34 567890"),
		sideOf("STSBACK_2019", "77УХ123456")}
	PairSides(docs)
	if docs[0].PairedWith != 2 || docs[2].PairedWith != 0 || docs[1].PairedWith != -1 {
		t.Errorf("paired with %d %d %d", docs[0].PairedWith, docs[1].PairedWith, docs[2].PairedWith)
	}
}

func TestAnAmbiguousNumberPairsNothing(t *testing.T) {
	docs := []*Results{sideOf("STS_2019", "77 12 345678"), sideOf("STS_2019", "77 12 345678"),
		sideOf("STSBACK_2019", "77 12 345678")}
	PairSides(docs)
	for i, d := range docs {
		if d.PairedWith != -1 {
			t.Errorf("document %d paired with %d", i, d.PairedWith)
		}
	}
}

func TestADifferentNumberOrANumberTooShortPairsNothing(t *testing.T) {
	docs := []*Results{sideOf("STS_1996", "77 12 345678"), sideOf("STSBACK_1996", "77 12 345679")}
	PairSides(docs)
	short := []*Results{sideOf("STS_1996", "1234"), sideOf("STSBACK_1996", "1234")}
	PairSides(short)
	if docs[0].PairedWith != -1 || short[0].PairedWith != -1 {
		t.Error("paired when it must not")
	}
}

// framesOf are the frames of detections cut straight from a canvas at their boxes.
func framesOf(fields []modules.Field) []fieldFrame {
	frames := make([]fieldFrame, len(fields))
	for i, f := range fields {
		frames[i] = singleCanvasFrame(f, false)
	}
	return frames
}

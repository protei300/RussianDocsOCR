package pipeline

// optionsSTS is OCROptionsSTS (pipeline.py): OCR options for the vehicle registration
// certificate (issue #17). The vehicle side (STS_<year>) and the owner side
// (STSBACK_<year>) share one options class, as the birth-certificate eras do - MakeOcrOptions
// cannot tell the sides apart and does not need to: a field the detector does not find on a
// side simply produces nothing.
//
// Engine routing follows the alphabet the field is printed in, with the two project
// precedents kept: the series/number goes to the Cyrillic engine (digits read better there,
// issue #12, and old blanks carry Cyrillic series letters), and mixed fields go where MOST of
// their values live - Chassis_number is «ОТСУТСТВУЕТ» on nearly every car (Cyrillic), while
// Body_number is the VIN on nearly every car (Latin). The PTS line mixes a Cyrillic series
// with digits («77ТС272158») - Cyrillic. House_number can carry a Cyrillic letter («38Б») -
// Cyrillic. Reg_number and VIN are Latin by decision (2026-09-05): plate letters are the GOST
// subset that shares its glyphs with Latin, and VIN never contains I, O or Q.
//
// The new form (order 267/2019, STS_2019/STSBACK_2019) adds the type-approval number
// (Latin-routed: its «ТС»/«ЕАЭС» prefix shares glyphs with Latin, the rest is Latin and
// digits) and the building as its own address line; its issue date is digits, which the
// Cyrillic engine reads as well as the worded one. There is no model class: the make is read
// by the lines of the blank, the upper one is Vehicle_make_ru, the lower one Vehicle_make_en
// (decision of 2026-10-04; Vehicle_model_en was removed with the detector v10, a7b12e81).
//
// The original edition of the old form (order 1001/2008 before its 2013 amendment) also
// prints the engine model, engine number and displacement: Latin letters and digits,
// Latin-routed. Its owner side has no unit code; it names the issuing unit in words on two
// lines («МРЭО ГИБДД ГУВД ПО / ЧЕЛЯБИНСКОЙ ОБЛАСТИ») - Issue_organization_ru, Cyrillic and
// split into words like the birth certificate's, since no OCR alphabet has a space.
//
// Each name here is ALSO in the Python reference, the .NET and Kotlin ports and in
// `service/ml/labels.py`: one list in five places, changed in all of them at once.
func optionsSTS() OcrOptions {
	return OcrOptions{
		NeededSplit: []string{"Vehicle_make_ru", "Vehicle_make_en", "Vehicle_type",
			"Special_marks", "Living_region_ru", "Licence_number",
			"Issue_date", "PTS_number", "Eco_class", "Vehicle_color",
			"Issue_organization_ru"},
		EnFields: []string{"Reg_number", "VIN", "Vehicle_make_en",
			"Type_approval", "Engine_model", "Engine_number", "Engine_volume",
			"Last_name_en",
			"First_name_en", "Vehicle_category", "Vehicle_year",
			"Body_number", "Engine_power", "Max_mass", "Curb_mass",
			"Expiration_date", "Apartment_number", "Issue_organisation_code"},
		RuFields: []string{"Last_name_ru", "First_name_ru", "Middle_name_ru",
			"Living_region_ru", "Vehicle_make_ru", "Vehicle_type",
			"Vehicle_color", "Eco_class", "Chassis_number", "Special_marks",
			"Licence_number", "Issue_date", "PTS_number", "House_number",
			"Building_number", "Issue_organization_ru"},
		// The make, by line: the upper line is Vehicle_make_ru, the lower one
		// Vehicle_make_en (no model class since 2026-10-04). The new form prints the upper
		// line in Latin («LADA GRANTA»), the old one in Cyrillic, so the engine follows the
		// year. Measured on the STS synthetic holdout with the r32 field detector candidate
		// (2026-10-06): new form Latin-routed 79 of 112 exact against 22 Cyrillic-routed, the
		// old form unchanged (23 of 32). Reading each word with BOTH engines and keeping the
		// more confident one was tried and lost: the engines are sure of look-alike letters
		// in either script («CYBAPY», «YA3» for СУБАРУ, УАЗ; «ЕХЕЕР», «ОМОРА» for EXEED,
		// OMODA) - 67 of 112 with the confidence over all CTC steps, 76 over the letters
		// only, and over every year the old form fell to 15 of 32.
		EngineByYear: map[string]map[string]string{"Vehicle_make_ru": {"2019": "lat"}},
		// The special-marks lines are labelled tight to their letters (their lines stand so
		// close that a labelling margin merged neighbours), so the crop that is read gets a
		// margin (readMargins). Same measurement: CER 0.145 -> 0.119, leasing found 23 -> 25
		// of 30 (no false ones), exact lines unchanged at 31 of 74; 0.2 and 0.3 gave nothing
		// more and cost the 2010 edition 2 of 26 exact lines.
		ReadMargin: map[string]float64{"Special_marks": 0.1},
		GlueTorn:   []string{"Special_marks"},
	}
}

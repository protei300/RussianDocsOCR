"""Which text fields each document type has and which engine reads them."""
from dataclasses import dataclass


@dataclass(init=False)
class OCROptionsClass:
    """Class for storing OCR options for different document types.

    Holds common OCR options like fields needed, split preferences, etc.
    Sub-classes implement options specific to document types.
    """

    """list: Fields that need to be split for this doc type."""
    needed_split = []

    """list: Fields read by the LATIN engine for this doc type."""
    en_fields = []

    """list: Fields read by the CYRILLIC engine for this doc type.

    The two lists select an ENGINE, not a language, and the distinction became
    load-bearing with issue #12: a passport series/number is pure digits, both
    engines carry the digit classes, and they do not agree on them. Membership
    here is decided by measurement, not by what language the field is written
    in - see OCROptionsINTPassport.
    """
    ru_fields = []

    """dict: Field -> {form year (label suffix): 'cyr' | 'lat'}, an engine that
    overrides ru_fields/en_fields on that year only. For a line whose alphabet
    follows the form edition. The field must still be listed in ru_fields or
    en_fields - that routing is what the other years get."""
    engine_by_year = {}

    """dict: Field -> vertical margin, as a share of the box height, added to the
    crop that is READ (the box itself is not changed). For fields labelled tight
    to the letters (Pipeline._read_margins)."""
    read_margin = {}

    """list: Multi-line fields whose lines are printed wrapped at the edge of the
    print area without a hyphen, so a word can be torn across two lines;
    torn known words are glued back (sts_marks.glue_torn_words)."""
    glue_torn = []

    """bool: Whether this doc type needs license number rotation."""
    needs_licence_rotation = False

    """bool: Whether this doc type has a multi-line registration address to OCR."""
    has_address = False


    @classmethod
    def make_options(cls, doc_type):
        """Factory method to make OCR options for a document type.

        Args:
            doc_type (str): Document type string

        Returns:
            OCROptionsClass instance with options for the document type.
        """
        # NOTE: 'intpassportaddr' contains 'intpassport', so it MUST be checked first.
        if 'intpassportaddr' in doc_type.lower():
            return OCROptionsINTPASSPORTADDR()
        elif 'intpassport' in doc_type.lower():
            return OCROptionsINTPassport()
        elif 'extpassport' in doc_type.lower():
            return OCROptionsEXTPassport()
        # 'dlback' contains 'dl': the licence back side (categories table) has no
        # field model yet, so it must not fall into the front-side DL options.
        elif 'dlback' in doc_type.lower():
            return OCROptionsClass()
        elif 'dl' in doc_type.lower():
            return OCROptionsDL()
        elif 'snils' in doc_type.lower():
            return OCROptionsSNILS()
        elif 'birthcert' in doc_type.lower():
            return OCROptionsBIRTHCERT()
        # 'stsback' contains 'sts': both sides of the certificate land here
        # on purpose (one options class for both, see OCROptionsSTS).
        elif 'sts' in doc_type.lower():
            return OCROptionsSTS()
        # Unknown type: return the empty base options instead of None, so the
        # caller's `.needs_licence_rotation`/`.ru_fields` access doesn't crash
        # (the pipeline then simply produces no OCR fields for it).
        return OCROptionsClass()


class OCROptionsINTPassport(OCROptionsClass):
    """OCR options for internal Russian passports."""

    # `Middle_name_ru`: see OCROptionsDL — no OCR alphabet carries a space, so a
    # double patronymic is returned glued unless the splitter runs.
    needed_split = ["Licence_number",
                    "Birth_place_ru", "Issue_organization_ru",
                    "Middle_name_ru",
                    ]

    # MRZ is deliberately NOT in needed_split: it is detected one box per line
    # and each line must reach the OCR engine whole - splitting it into "words"
    # at the filler runs would destroy the fixed 44-character layout the check
    # digits are computed over.
    en_fields = ["Issue_date", "Expiration_date", "Birth_date",
                 "Issue_organisation_code", "MRZ", ]
    # Licence_number is CYRILLIC-routed, and that is not a typo: the series and
    # number are digits only, but the Latin engine reads the passport's red '3'
    # as '8' - confidently, at p=0.94..1.00 with '3' as the runner-up at 0.004,
    # so no threshold, alphabet mask or upscaling can recover it (issue #12).
    # The Cyrillic engine reads the SAME crops correctly. Measured over
    # samples/: Latin 17/24 exact, Cyrillic 24/24; across every doc type,
    # Latin 103/112, Cyrillic 111/112, and Cyrillic is never worse except on
    # BIRTHCERT, where NEITHER engine reads the Roman-numeral series correctly.
    # That type is not touched here: it was already Cyrillic-routed, as the
    # lesser evil - see OCROptionsBIRTHCERT.
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru", "Licence_number"]
    needs_licence_rotation = True


class OCROptionsINTPASSPORTADDR(OCROptionsClass):
    """OCR options for the registration ('place of residence') page of an
    internal Russian passport. Only the multi-line registration address is
    recognized (printed text), via the Address_line detections produced by
    the TextFields detector."""

    needed_split = []
    en_fields = []
    ru_fields = []
    has_address = True
    needs_licence_rotation = False


class OCROptionsBIRTHCERT(OCROptionsClass):
    """OCR options for RF birth certificates (BIRTHCERT_1998, BIRTHCERT_2018).

    Field names follow the shared TextFields class vocabulary: the child's
    name maps onto Last_name_ru/First_name_ru, the ZAGS office onto
    Issue_organization_ru, the series/number line onto Licence_number, and
    the parents get their own *_ru classes (added with the 2026-08 detector
    retrain). Dates on this form spell the month out in Cyrillic
    («16 декабря 2001»), so they go through the Cyrillic engine.
    Licence_number mixes a Roman-numeral series with Cyrillic letters and
    «№» — routed to the Cyrillic engine as the lesser evil (the Roman-digit
    series is the expected CER cost; revisit if the eval says otherwise).

    Shared by both blank generations: BIRTHCERT_1998 (digit birth date
    DD.MM.YYYY) and BIRTHCERT_2018 (order 167/2018: worded birth date, parent
    birth dates, place of issue, 21-digit act number). The options cannot see
    the era (make_options gets the type without the year suffix), so
    Birth_date is routed to the Cyrillic engine for BOTH: it must read the
    worded 2018 form, and the Cyrillic engine reads digit-only crops fine —
    the same precedent as the passport Licence_number (issue #12).

    Act_date is the date of the civil-registry record (2026-10: a hotel
    registry system now asks for the record's number and date instead of the
    certificate's series and number; also issue #22). Both forms print it in
    words, the 1998 form in reverse order with the printed words between the
    parts - «2010 года июня месяца 15 числа» - and the field box spans them,
    so dates.py skips those words."""

    needed_split = ["First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                    "Issue_date", "Licence_number",
                    "Father_first_middle_ru", "Mother_first_middle_ru",
                    "Birth_date", "Father_birth_date", "Mother_birth_date",
                    "Issue_place_ru", "Act_date"]
    en_fields = []
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru",
                 "Issue_organization_ru", "Issue_date", "Licence_number",
                 "Father_last_name_ru", "Father_first_middle_ru",
                 "Mother_last_name_ru", "Mother_first_middle_ru",
                 "Birth_date", "Father_birth_date", "Mother_birth_date",
                 "Issue_place_ru", "Act_number", "Act_date"]


class OCROptionsEXTPassport(OCROptionsClass):
    """OCR options for external Russian passports."""

    # `Issue_organization_ru` is split for a reason that is easy to undo by
    # accident: NEITHER OCR alphabet contains a space (cyrillic 102 symbols,
    # latin 147 - both carry '.', '-' and "'", neither carries ' '). A space can
    # therefore only come from word splitting joining the words back together,
    # so a multi-word field that skips the splitter is returned glued:
    # «МИД РОССИИ» came back as «МИДРОССИИ». Measured over samples/ (2026-09-02):
    # 12 of the 99 acceptance failures were exactly this field on external
    # passports of both generations, and every one of them differed from the
    # ground truth by spaces alone.
    needed_split = ["Licence_number", "Birth_place_ru", "Birth_place_en",
                    "Issue_organization_ru", ]

    # `Middle_name_en` is listed and never arrives, ON PURPOSE — do not "clean it
    # up". A Russian foreign passport carries no Latin patronymic: the data page
    # prints surname and given name in both scripts and the patronymic in Russian
    # only. Measured rather than assumed — over `samples/`, all 42 external
    # passports have an empty `Middle_name_en` in the (hand-verified) ground truth
    # while `Middle_name_ru`, `First_name_en` and `Last_name_en` are filled in all
    # 42, and the detector returns no box for the class on any of them. The class
    # itself is fine: on driving licences, where the field does exist, it is found
    # and read on 34 of 34.
    #
    # The entry is therefore inert — these lists only filter and route boxes that
    # were already detected — and removing it would change no output at all. It
    # stays because THE SAME LIST LIVES IN FOUR PLACES: the .NET, Go and Kotlin
    # ports plus `service/ml/labels.py`. Dropping it here alone would split the
    # four implementations, and the conformance harness could not report that:
    # it compares outputs, not field lists, and a field nobody produces is
    # equally absent everywhere. An invisible price is the worst kind.
    #
    # If it is ever removed, remove it in all four at once, as its own task.
    en_fields = ["Last_name_en", "First_name_en", "Issue_date",
                 "Expiration_date", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en", "Sex_en",
                 "Issue_organisation_code", "Middle_name_en", "MRZ"]
    # Licence_number: Cyrillic-routed for the reason given on OCROptionsINTPassport.
    # Smaller effect here (Latin 41/42 exact, Cyrillic 42/42 over samples/) because
    # the number is printed larger, but it is the same digit confusion.
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru", "Licence_number"]


class OCROptionsDL(OCROptionsClass):
    """OCR options for Russian driver's licenses."""

    # `Middle_name_*` is split for the same reason as the external passport's
    # `Issue_organization_ru` (see there): no OCR alphabet contains a space, so
    # a field that skips the splitter can never return one. A double patronymic
    # («ОГЛЫ», «КЫЗЫ») then comes back glued. Four such failures over samples/
    # (2026-09-02), all differing from the ground truth by spaces alone.
    needed_split = ["Licence_number", "Driver_class", "Birth_place_ru", "Birth_place_en",
                    "Living_region_ru", "Living_region_en",
                    "Middle_name_ru", "Middle_name_en", ]
    en_fields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                 "Expiration_date", "Driver_class", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en",  "Issue_organisation_code", "Middle_name_en"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", ]


class OCROptionsSNILS(OCROptionsClass):
    """OCR options for Russian SNILS documents."""

    needed_split = ["Last_name_ru", "First_name_ru", "Licence_number", "Issue_date",
                    "Birth_date", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]
    en_fields = ["Licence_number", "Issue_date", "Birth_date"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]


class OCROptionsSTS(OCROptionsClass):
    """OCR options for the vehicle registration certificate (issue #17), old
    form: the vehicle side (STS_<year>) and the owner side (STSBACK_<year>)
    share one options class, as the birth-certificate eras do - make_options
    cannot tell the sides apart and does not need to: a field the detector
    does not find on a side simply produces nothing.

    Engine routing follows the alphabet the field is printed in, with the two
    project precedents kept: the series/number goes to the Cyrillic engine
    (digits read better there, issue #12, and old blanks carry Cyrillic series
    letters), and mixed fields go where MOST of their values live -
    Chassis_number is «ОТСУТСТВУЕТ» on nearly every car (Cyrillic), while
    Body_number is the VIN on nearly every car (Latin). The PTS line mixes a
    Cyrillic series with digits («77ТС272158») - Cyrillic. House_number can
    carry a Cyrillic letter («38Б») - Cyrillic. Reg_number and VIN are Latin
    by decision (2026-09-05): plate letters are the GOST subset that shares
    its glyphs with Latin, and VIN never contains I, O or Q.

    The new form (order 267/2019, STS_2019/STSBACK_2019) adds the model on a
    line of its own (Latin), the type-approval number (Latin-routed: its
    «ТС»/«ЕАЭС» prefix shares glyphs with Latin, the rest is Latin and
    digits) and the building as its own address line; its issue date is
    digits, which the Cyrillic engine reads as well as the worded one.

    The original edition of the old form (order 1001/2008 before its 2013
    amendment) also prints the engine model, engine number and displacement:
    Latin letters and digits, Latin-routed. Its owner side has no unit code;
    it names the issuing unit in words on two lines («МРЭО ГИБДД ГУВД ПО /
    ЧЕЛЯБИНСКОЙ ОБЛАСТИ») - Issue_organization_ru, Cyrillic and split into
    words like the birth certificate's, since no OCR alphabet has a space.

    Each name here is ALSO in the .NET, Go and Kotlin ports and in
    `service/ml/labels.py`; the ports follow after this type lands (decision
    #132), so until then only the Python reference produces these fields."""

    needed_split = ["Vehicle_make_ru", "Vehicle_make_en", "Vehicle_type",
                    "Special_marks", "Living_region_ru", "Licence_number",
                    "Issue_date", "PTS_number", "Eco_class", "Vehicle_color",
                    "Issue_organization_ru"]
    en_fields = ["Reg_number", "VIN", "Vehicle_make_en",
                 "Type_approval", "Engine_model", "Engine_number", "Engine_volume",
                 "Last_name_en",
                 "First_name_en", "Vehicle_category", "Vehicle_year",
                 "Body_number", "Engine_power", "Max_mass", "Curb_mass",
                 "Expiration_date", "Apartment_number", "Issue_organisation_code"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Middle_name_ru",
                 "Living_region_ru", "Vehicle_make_ru", "Vehicle_type",
                 "Vehicle_color", "Eco_class", "Chassis_number", "Special_marks",
                 "Licence_number", "Issue_date", "PTS_number", "House_number",
                 "Building_number", "Issue_organization_ru"]
    # The make, by line: the upper line is Vehicle_make_ru, the lower one
    # Vehicle_make_en (no model class since 2026-10-04). The new form prints the
    # upper line in Latin («LADA GRANTA»), the old one in Cyrillic, so the engine
    # follows the year. Measured on the STS synthetic holdout with the r32 field
    # detector candidate (2026-10-06): new form Latin-routed 79 of 112 exact
    # against 22 Cyrillic-routed, the old form unchanged (23 of 32). Reading each
    # word with BOTH engines and keeping the more confident one was tried and
    # lost: the engines are sure of look-alike letters in either script
    # («CYBAPY», «YA3» for СУБАРУ, УАЗ; «ЕХЕЕР», «ОМОРА» for EXEED, OMODA) -
    # 67 of 112 with the confidence over all CTC steps, 76 over the letters only,
    # and over every year the old form fell to 15 of 32.
    engine_by_year = {"Vehicle_make_ru": {"2019": "lat"}}
    # The special-marks lines are labelled tight to their letters (their lines
    # stand so close that a labelling margin merged neighbours), so the crop that
    # is read gets a margin (Pipeline._read_margins). Same measurement: CER
    # 0.145 -> 0.119, leasing found 23 -> 25 of 30 (no false ones), exact lines
    # unchanged at 31 of 74; 0.2 and 0.3 gave nothing more and cost the 2010
    # edition 2 of 26 exact lines.
    read_margin = {"Special_marks": 0.1}
    glue_torn = ["Special_marks"]

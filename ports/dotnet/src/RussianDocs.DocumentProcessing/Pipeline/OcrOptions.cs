namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// Which fields a document type has, which need splitting into words, and which script each uses.
///
/// <para>
/// Port of the <c>OCROptions*</c> class family. A single record with lists rather than a class
/// hierarchy: the subclasses in the reference differ ONLY in their data, so inheritance buys nothing
/// and costs one place per language where a base method could be called by mistake.
/// </para>
/// </summary>
public sealed record OcrOptions
{
    public string[] NeededSplit { get; init; } = [];
    public string[] EnFields { get; init; } = [];
    public string[] RuFields { get; init; } = [];
    public bool NeedsLicenceRotation { get; init; }
    public bool HasAddress { get; init; }

    /// <summary>
    /// Field to {form year (label suffix): "cyr" | "lat"}, an engine that overrides
    /// <see cref="RuFields"/>/<see cref="EnFields"/> on that year only — for a line whose alphabet
    /// follows the form edition (<c>OCROptionsClass.engine_by_year</c>). The field must still be listed
    /// in one of the two lists: that routing is what the other years get.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> EngineByYear { get; init; } =
        new Dictionary<string, IReadOnlyDictionary<string, string>>();

    /// <summary>
    /// Field to vertical margin, as a share of the box height, added to the crop that is READ (the box
    /// itself is not changed) — for fields labelled tight to the letters (<c>Pipeline._read_margins</c>).
    /// </summary>
    public IReadOnlyDictionary<string, double> ReadMargin { get; init; } =
        new Dictionary<string, double>();

    /// <summary>
    /// Multi-line fields whose lines are printed wrapped at the edge of the print area without a
    /// hyphen, so a word can be torn across two lines; torn known words are glued back
    /// (<see cref="StsMarks.GlueTornWords"/>).
    /// </summary>
    public string[] GlueTorn { get; init; } = [];

    /// <summary>
    /// The engine this field takes on this form year ("cyr" or "lat"), or null when the
    /// RuFields/EnFields routing decides. Port of <c>Pipeline._engine_by_year</c>; a label without a year
    /// suffix has no year and so no override.
    /// </summary>
    public string? EngineForYear(string field, string? year) =>
        year is not null && EngineByYear.TryGetValue(field, out var byYear)
        && byYear.TryGetValue(year, out string? engine) ? engine : null;

    public bool IsOcrField(string label) =>
        Array.IndexOf(EnFields, label) >= 0 || Array.IndexOf(RuFields, label) >= 0;

    public bool NeedsSplit(string label) => Array.IndexOf(NeededSplit, label) >= 0;

    /// <summary>
    /// Splits a document label into its bare type and issuance year.
    ///
    /// <para>
    /// The reference uses <c>rsplit('_', maxsplit=1)</c> and would raise on a label without an
    /// underscore. This returns an empty year instead — a label the model produced should not crash
    /// the pipeline, and every shipped label has the suffix anyway.
    /// </para>
    /// </summary>
    public static (string Bare, string Year) SplitDocType(string label)
    {
        int at = label.LastIndexOf('_');
        return at >= 0 ? (label[..at], label[(at + 1)..]) : (label, "");
    }

    /// <summary>
    /// Builds the options for a document type.
    ///
    /// <para>
    /// **`intpassportaddr` MUST be tested before `intpassport`.** The check is a substring match, so
    /// reversing the two sends the registration page down the ordinary text-field path and produces a
    /// document with no address and no error. The reference has the same ordering dependency and the
    /// same comment.
    /// </para>
    ///
    /// <para>
    /// An unrecognised type returns EMPTY options rather than null. The reference returns None here
    /// and the next attribute access throws <c>AttributeError</c> — a crash two lines later that says
    /// nothing about the document type. Empty options mean "no OCR fields", which is what an unknown
    /// document deserves.
    /// </para>
    /// </summary>
    public static OcrOptions For(string docType)
    {
        string t = docType.ToLowerInvariant();

        if (t.Contains("intpassportaddr", StringComparison.Ordinal))
        {
            return new OcrOptions { HasAddress = true };
        }
        if (t.Contains("intpassport", StringComparison.Ordinal))
        {
            return new OcrOptions
            {
                // `Middle_name_ru`: see the DL branch below — no OCR alphabet carries a space, so a
                // double patronymic is returned glued unless the splitter runs (pipeline.py:134-139).
                NeededSplit = ["Licence_number", "Birth_place_ru", "Issue_organization_ru",
                    "Middle_name_ru"],
                // MRZ is read by the Latin engine and is NOT in NeededSplit: the zone is detected
                // one box per LINE, and each line must reach the engine whole — splitting it at its
                // filler runs would destroy the fixed 44-character layout the check digits are
                // computed over.
                EnFields = ["Issue_date", "Expiration_date", "Birth_date",
                    "Issue_organisation_code", "MRZ"],
                // Licence_number is CYRILLIC-routed although it is digits only: the Latin engine
                // reads the passport's red '3' as '8' at p=0.94..1.00, and the Cyrillic engine
                // reads the same crops correctly (issue #12). Matches the reference,
                // OCROptionsINTPassport in pipeline.py.
                RuFields = ["Last_name_ru", "First_name_ru", "Birth_place_ru",
                    "Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru",
                    "Licence_number"],
                // The internal passport prints its series and number sideways, so the crop is rotated
                // before OCR. Only this type does.
                NeedsLicenceRotation = true,
            };
        }
        if (t.Contains("extpassport", StringComparison.Ordinal))
        {
            return new OcrOptions
            {
                // `Issue_organization_ru` is split for the same reason as the DL branch's
                // `Middle_name_*` below: no OCR alphabet contains a space, so a multi-word field that
                // skips the splitter comes back glued («МИД РОССИИ» -> «МИДРОССИИ»,
                // pipeline.py:211-219).
                NeededSplit = ["Licence_number", "Birth_place_ru", "Birth_place_en",
                    "Issue_organization_ru"],
                // MRZ: Latin engine, never split — see intpassport above.
                EnFields = ["Last_name_en", "First_name_en", "Issue_date",
                    "Expiration_date", "Birth_date", "Birth_place_en", "Issue_organization_en",
                    "Living_region_en", "Sex_en", "Issue_organisation_code", "Middle_name_en",
                    "MRZ"],
                // Licence_number: Cyrillic-routed, same reason as intpassport above.
                RuFields = ["Licence_number", "Last_name_ru", "First_name_ru", "Birth_place_ru",
                    "Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru"],
            };
        }
        // 'dlback' contains 'dl': the licence back side (categories table) has no field model yet, so
        // it must not fall into the front-side DL options (pipeline.py make_options).
        if (t.Contains("dlback", StringComparison.Ordinal))
        {
            return new OcrOptions();
        }
        if (t.Contains("dl", StringComparison.Ordinal))
        {
            return new OcrOptions
            {
                // `Middle_name_*` is split for the same reason as the external passport's
                // `Issue_organization_ru` above: no OCR alphabet contains a space, so a double
                // patronymic («ОГЛЫ», «КЫЗЫ») comes back glued unless the splitter runs
                // (pipeline.py:256-263).
                NeededSplit = ["Licence_number", "Driver_class", "Birth_place_ru", "Birth_place_en",
                    "Living_region_ru", "Living_region_en", "Middle_name_ru", "Middle_name_en"],
                EnFields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                    "Expiration_date", "Driver_class", "Birth_date", "Birth_place_en",
                    "Issue_organization_en", "Living_region_en", "Issue_organisation_code",
                    "Middle_name_en"],
                RuFields = ["Last_name_ru", "First_name_ru", "Birth_place_ru",
                    "Issue_organization_ru", "Living_region_ru", "Middle_name_ru"],
            };
        }
        if (t.Contains("snils", StringComparison.Ordinal))
        {
            return new OcrOptions
            {
                NeededSplit = ["Last_name_ru", "First_name_ru", "Licence_number", "Issue_date",
                    "Birth_date", "Birth_place_ru", "Middle_name_ru", "Sex_ru"],
                EnFields = ["Licence_number", "Issue_date", "Birth_date"],
                RuFields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Middle_name_ru",
                    "Sex_ru"],
            };
        }
        if (t.Contains("birthcert", StringComparison.Ordinal))
        {
            // Birth certificates (OCROptionsBIRTHCERT, pipeline.py:156). ONE branch for both blank
            // generations — BIRTHCERT_1998 and BIRTHCERT_2018 (order 167/2018: worded birth date,
            // parents' birth dates, place of issue, 21-digit act number) — because the dispatcher
            // never sees the year suffix.
            //
            // EnFields is EMPTY and that is the whole point: every date on these forms is spelled
            // out in Cyrillic («16 декабря 2001», «15 октября 2020 г.»), and the Cyrillic engine
            // reads the 1998 digit-only birth date just as well — the same precedent as the
            // passport Licence_number (issue #12). Licence_number mixes a Roman-numeral series with
            // Cyrillic and «№»; routed Cyrillic as the lesser evil, same as the reference.
            return new OcrOptions
            {
                NeededSplit = ["First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                    "Issue_date", "Licence_number",
                    "Father_first_middle_ru", "Mother_first_middle_ru",
                    "Birth_date", "Father_birth_date", "Mother_birth_date",
                    "Issue_place_ru", "Act_date"],
                EnFields = [],
                RuFields = ["Last_name_ru", "First_name_ru", "Birth_place_ru",
                    "Issue_organization_ru", "Issue_date", "Licence_number",
                    "Father_last_name_ru", "Father_first_middle_ru",
                    "Mother_last_name_ru", "Mother_first_middle_ru",
                    "Birth_date", "Father_birth_date", "Mother_birth_date",
                    "Issue_place_ru", "Act_number", "Act_date"],
            };
        }
        // 'stsback' contains 'sts': both sides of the certificate land here on purpose (one options
        // class for both, see OCROptionsSTS in pipeline.py).
        if (t.Contains("sts", StringComparison.Ordinal))
        {
            return new OcrOptions
            {
                NeededSplit = ["Vehicle_make_ru", "Vehicle_make_en", "Vehicle_type",
                    "Special_marks", "Living_region_ru", "Licence_number",
                    "Issue_date", "PTS_number", "Eco_class", "Vehicle_color",
                    "Issue_organization_ru"],
                // Engine routing follows the alphabet the field is printed in, with the two project
                // precedents kept: the series/number goes to the Cyrillic engine (issue #12), and mixed
                // fields go where MOST of their values live. Reg_number and VIN are Latin by decision
                // (2026-09-05): plate letters are the GOST subset that shares its glyphs with Latin, and
                // a VIN never contains I, O or Q.
                EnFields = ["Reg_number", "VIN", "Vehicle_make_en",
                    "Type_approval", "Engine_model", "Engine_number", "Engine_volume",
                    "Last_name_en",
                    "First_name_en", "Vehicle_category", "Vehicle_year",
                    "Body_number", "Engine_power", "Max_mass", "Curb_mass",
                    "Expiration_date", "Apartment_number", "Issue_organisation_code"],
                RuFields = ["Last_name_ru", "First_name_ru", "Middle_name_ru",
                    "Living_region_ru", "Vehicle_make_ru", "Vehicle_type",
                    "Vehicle_color", "Eco_class", "Chassis_number", "Special_marks",
                    "Licence_number", "Issue_date", "PTS_number", "House_number",
                    "Building_number", "Issue_organization_ru"],
                // The make, by line: the upper line is Vehicle_make_ru, the lower one Vehicle_make_en (no
                // model class since 2026-10-04). The new form prints the upper line in Latin
                // («LADA GRANTA»), the old one in Cyrillic, so the engine follows the year. Measured on the
                // STS synthetic holdout with the r32 field detector candidate (2026-10-06): new form
                // Latin-routed 79 of 112 exact against 22 Cyrillic-routed, the old form unchanged.
                EngineByYear = new Dictionary<string, IReadOnlyDictionary<string, string>>
                {
                    ["Vehicle_make_ru"] = new Dictionary<string, string> { ["2019"] = "lat" },
                },
                // The special-marks lines are labelled tight to their letters, so the crop that is read
                // gets a margin (Recognizer.ReadMargins): CER 0.145 -> 0.119, leasing found 23 -> 25 of 30.
                ReadMargin = new Dictionary<string, double> { ["Special_marks"] = 0.1 },
                GlueTorn = ["Special_marks"],
            };
        }
        return new OcrOptions();
    }
}

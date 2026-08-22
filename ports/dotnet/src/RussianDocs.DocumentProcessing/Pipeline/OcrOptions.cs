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
                NeededSplit = ["Licence_number", "Birth_place_ru", "Issue_organization_ru"],
                EnFields = ["Issue_date", "Expiration_date", "Birth_date",
                    "Issue_organisation_code"],
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
                NeededSplit = ["Licence_number", "Birth_place_ru", "Birth_place_en"],
                EnFields = ["Last_name_en", "First_name_en", "Issue_date",
                    "Expiration_date", "Birth_date", "Birth_place_en", "Issue_organization_en",
                    "Living_region_en", "Sex_en", "Issue_organisation_code", "Middle_name_en"],
                // Licence_number: Cyrillic-routed, same reason as intpassport above.
                RuFields = ["Licence_number", "Last_name_ru", "First_name_ru", "Birth_place_ru",
                    "Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru"],
            };
        }
        if (t.Contains("dl", StringComparison.Ordinal))
        {
            return new OcrOptions
            {
                NeededSplit = ["Licence_number", "Driver_class", "Birth_place_ru", "Birth_place_en",
                    "Living_region_ru", "Living_region_en"],
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
                    "Issue_place_ru"],
                EnFields = [],
                RuFields = ["Last_name_ru", "First_name_ru", "Birth_place_ru",
                    "Issue_organization_ru", "Issue_date", "Licence_number",
                    "Father_last_name_ru", "Father_first_middle_ru",
                    "Mother_last_name_ru", "Mother_first_middle_ru",
                    "Birth_date", "Father_birth_date", "Mother_birth_date",
                    "Issue_place_ru", "Act_number"],
            };
        }
        return new OcrOptions();
    }
}

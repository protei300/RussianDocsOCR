namespace RussianDocs.DocumentProcessing.ViewModel;

/// <summary>
/// Display names, field ordering and script hints.
///
/// <para>
/// D-01: the view model lives on the LIBRARY side, not in the service. The conformance CLI needs it
/// and must not depend on HTTP — which is why this is here and not in a service project, even though
/// the reference puts it in <c>service/ml/transform.py</c>.
/// </para>
/// </summary>
public static class Labels
{
    /// <summary>
    /// Human-readable names. The English UI shows these, so both the <c>_ru</c> and <c>_en</c>
    /// variants of a field map to the SAME display string — the script is carried separately.
    /// </summary>
    private static readonly Dictionary<string, string> FieldLabels = new(StringComparer.Ordinal)
    {
        ["Last_name_ru"] = "Last name",
        ["Last_name_en"] = "Last name",
        ["First_name_ru"] = "First name",
        ["First_name_en"] = "First name",
        ["Middle_name_ru"] = "Middle name",
        ["Middle_name_en"] = "Middle name",
        ["Birth_date"] = "Date of birth",
        ["Birth_place_ru"] = "Place of birth",
        ["Birth_place_en"] = "Place of birth",
        ["Sex_ru"] = "Sex",
        ["Sex_en"] = "Sex",
        ["Licence_number"] = "Document number",
        ["Issue_date"] = "Date of issue",
        ["Expiration_date"] = "Valid until",
        ["Issue_organization_ru"] = "Issuing authority",
        ["Issue_organization_en"] = "Issuing authority",
        ["Issue_organisation_code"] = "Authority code",
        ["Living_region_ru"] = "Place of residence",
        ["Living_region_en"] = "Place of residence",
        ["Driver_class"] = "Categories",
        ["Face"] = "Photo",
        ["Signature"] = "Signature",
        ["Address"] = "Registration address",
        ["Address_has_handwritten"] = "Contains handwriting",
    };

    /// <summary>
    /// Labels that are detected but never OCR'd.
    ///
    /// <para>
    /// They become <c>kind: "visual"</c> so the overlay can draw them without the UI expecting text —
    /// free value from the detector, since it finds them anyway.
    /// </para>
    /// </summary>
    private static readonly HashSet<string> NonTextLabels = new(StringComparer.Ordinal)
        { "Face", "Signature" };

    /// <summary>
    /// Fields the UI renders monospaced.
    ///
    /// <para>
    /// Numbers and dates only. Monospacing capital Cyrillic — Ш, Щ, Ж, Ы — looks wrong, which is why
    /// this is a short allowlist rather than a default.
    /// </para>
    /// </summary>
    private static readonly HashSet<string> MonospaceFields = new(StringComparer.Ordinal)
    {
        "Licence_number", "Issue_date", "Expiration_date", "Birth_date", "Issue_organisation_code",
    };

    private static readonly string[] PassportOrder =
    [
        "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
        "Birth_date", "Birth_place_ru",
        "Licence_number", "Issue_date", "Expiration_date",
        "Issue_organization_ru", "Issue_organisation_code",
        "Living_region_ru",
    ];

    private static readonly string[] ExtPassportOrder =
    [
        "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
        "Middle_name_ru", "Middle_name_en", "Sex_ru", "Sex_en",
        "Birth_date", "Birth_place_ru", "Birth_place_en",
        "Licence_number", "Issue_date", "Expiration_date",
        "Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
        "Living_region_ru", "Living_region_en",
    ];

    private static readonly string[] DlOrder =
    [
        "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
        "Middle_name_ru", "Middle_name_en",
        "Birth_date", "Birth_place_ru", "Birth_place_en",
        "Licence_number", "Issue_date", "Expiration_date",
        "Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
        "Living_region_ru", "Living_region_en",
        "Driver_class",
    ];

    private static readonly string[] SnilsOrder =
    [
        "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
        "Birth_date", "Birth_place_ru",
        "Licence_number", "Issue_date",
    ];

    private static readonly string[] AddrOrder = ["Address", "Address_has_handwritten"];

    private static readonly Dictionary<string, string[]> FieldOrder = new(StringComparer.Ordinal)
    {
        ["INTPASSPORT"] = PassportOrder,
        ["INTPASSPORTADDR"] = AddrOrder,
        ["EXTPASSPORT"] = ExtPassportOrder,
        ["EXTPASSPORTBIO"] = ExtPassportOrder,
        ["DL"] = DlOrder,
        ["SNILS"] = SnilsOrder,
    };

    /// <summary>The type without its era suffix. Returns the input unchanged when there is none.</summary>
    public static string BaseDocType(string docType)
    {
        int at = docType.LastIndexOf('_');
        return at >= 0 ? docType[..at] : docType;
    }

    /// <summary>The era suffix, or null when the label has none.</summary>
    public static string? DocTypeEra(string docType)
    {
        int at = docType.LastIndexOf('_');
        return at >= 0 ? docType[(at + 1)..] : null;
    }

    public static string FieldDisplay(string name) =>
        FieldLabels.TryGetValue(name, out string? display) ? display : name;

    public static bool IsNonText(string label) => NonTextLabels.Contains(label);

    /// <summary>
    /// Which font family the UI should use: <c>num</c>, <c>ru</c> or <c>en</c>.
    ///
    /// <para>
    /// The monospace check comes FIRST, because <c>Birth_date</c> has no script suffix and would
    /// otherwise fall through to the <c>ru</c> default.
    /// </para>
    /// </summary>
    public static string FieldScript(string name)
    {
        if (MonospaceFields.Contains(name))
        {
            return "num";
        }
        if (name.EndsWith("_ru", StringComparison.Ordinal))
        {
            return "ru";
        }
        return name.EndsWith("_en", StringComparison.Ordinal) ? "en" : "ru";
    }

    /// <summary>
    /// Orders field names for display: canonical order first, then anything unrecognised.
    ///
    /// <para>
    /// **The sort must be stable and the unknown tail sorted by name**, because the reference's
    /// dictionary iteration order is insertion order and a port cannot reproduce that — sorting is
    /// what makes the result deterministic across languages. An unknown field appearing after the
    /// known ones rather than being dropped is deliberate: a new field should show up somewhere
    /// rather than vanish.
    /// </para>
    /// </summary>
    public static List<string> OrderFields(string docType, IEnumerable<string> names)
    {
        string[] canonical = FieldOrder.TryGetValue(BaseDocType(docType), out string[]? order)
            ? order
            : [];
        var rank = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < canonical.Length; i++)
        {
            rank[canonical[i]] = i;
        }

        var known = new List<string>();
        var unknown = new List<string>();
        foreach (string name in names)
        {
            (rank.ContainsKey(name) ? known : unknown).Add(name);
        }

        // OrderBy is stable in LINQ; List.Sort is not.
        var result = known.OrderBy(n => rank[n]).ToList();
        result.AddRange(unknown.OrderBy(n => n, StringComparer.Ordinal));
        return result;
    }
}

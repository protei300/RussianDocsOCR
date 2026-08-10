using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Modules;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>One field's OCR result: the per-word strings and the joined value.</summary>
public sealed record FieldText
{
    public required string Label { get; init; }
    public List<string> Words { get; init; } = [];
    public string Value { get; set; } = "";
}

public static class Ocr
{
    /// <summary>
    /// Routes every word crop to an engine and joins the results per field.
    /// </summary>
    public static List<FieldText> Run(List<FieldWords> fields, string docType, OcrOptions options,
        OcrEngine cyrillic, OcrEngine latin)
    {
        var output = new List<FieldText>(fields.Count);

        foreach (FieldWords fw in fields)
        {
            var words = new List<string>();
            for (int i = 0; i < fw.Patches.Count; i++)
            {
                Image patch = fw.Patches[i];

                // **SNILS routes by word-index PARITY, not by field semantics.** Its dates read like
                // "26 СЕНТЯБРЯ 1997 ГОДА", so odd-indexed words go to the CYRILLIC engine even inside
                // a date field. It looks like a bug and it is load-bearing: without it the Russian
                // month name is decoded by the Latin engine and comes out as noise.
                //
                // The order of these branches is the reference's, and it matters — the parity rule is
                // checked BEFORE the date rule, or SNILS months would be routed as dates.
                if ((docType == "SNILS" && i % 2 == 1)
                    || Array.IndexOf(options.RuFields, fw.Label) >= 0)
                {
                    words.Add(cyrillic.FixErrors(fw.Label, cyrillic.Predict(patch)));
                }
                else if (fw.Label.Contains("date", StringComparison.OrdinalIgnoreCase))
                {
                    words.Add(latin.FixErrors(fw.Label, latin.Predict(patch)));
                }
                else if (Array.IndexOf(options.EnFields, fw.Label) >= 0)
                {
                    words.Add(latin.FixErrors(fw.Label, latin.Predict(patch)));
                }
                // No else: a field that is neither Russian, a date, nor English contributes no words.
                // The reference has the same gap, and a fallback here would invent text.
            }
            output.Add(new FieldText { Label = fw.Label, Words = words });
        }

        // Joining happens in a SECOND pass, because a field detected twice appends to what the first
        // detection produced — see JoinField.
        var joined = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (FieldText field in output)
        {
            field.Value = JoinField(joined, field.Label, docType, field.Words);
        }
        return output;
    }

    /// <summary>
    /// Joins one field's words.
    ///
    /// <para>
    /// Three rules, all from the reference. A date joins with DOTS — <c>17.03.1987</c> — except on
    /// SNILS, where the parts are words and join with spaces. Everything else joins with spaces, and
    /// APPENDS to whatever an earlier detection of the same label produced, which is how the internal
    /// passport's twice-printed series ends up as one value.
    /// </para>
    ///
    /// <para>
    /// The double-space squeeze and the trim are the reference's too. They matter because an empty
    /// word — a crop the OCR read as nothing — would otherwise leave a visible gap in the value.
    /// </para>
    /// </summary>
    private static string JoinField(Dictionary<string, string> joined, string label, string docType,
        List<string> words)
    {
        bool isDate = label.Contains("date", StringComparison.OrdinalIgnoreCase);

        string value;
        if (isDate && docType != "SNILS")
        {
            value = string.Join(".", words);
        }
        else if (isDate)
        {
            value = string.Join(" ", words);
        }
        else
        {
            string previous = joined.TryGetValue(label, out string? p) ? p : "";
            value = previous.Length > 0
                ? previous + " " + string.Join(" ", words)
                : string.Join(" ", words);
        }

        value = value.Replace("  ", " ").Trim();
        joined[label] = value;
        return value;
    }

    /// <summary>
    /// The FMS code beautifier, ported as the no-op it currently is.
    ///
    /// <para>
    /// <c>Pipeline._fix_fms</c> in the reference returns immediately — the dictionary lookup was
    /// disabled because a cache miss scans ~16k rows with difflib, costing 3.3-5.1 s per document, and
    /// on failure it does not correct the code but REPLACES it with the code of the most similar name.
    /// Kept as a named stub so the next port does not have to rediscover why it is absent.
    /// </para>
    /// </summary>
    public static void FixFms(List<FieldText> fields, string docType)
    {
        // Intentionally empty. See the note above.
    }

    /// <summary>
    /// Collapses the dotted ruler lines the 1998 birth-certificate form prints under every value.
    ///
    /// <para>
    /// Port of <c>Pipeline._clean_ruler_artifacts</c> (pipeline.py:1061). The rulers land inside the
    /// field crops and OCR emits runs of those marks around the real words; they carry no information
    /// on this form. Commas and quotes are in the set because that is what the engine emits on this
    /// form («28., ИЮЛЯ 2010», «"""СЕМ","" ПОННИЛОВИЧ»), not because they were expected. Only runs of
    /// two or more and marks standing alone are removed, which is what keeps real punctuation: the
    /// comma in «Г. ИРКУТСК, ИРКУТСКАЯ ОБЛАСТЬ» is attached to a word and the hyphen in «II-МЮ» sits
    /// between letters, so neither matches. Exactly as in the reference.
    /// </para>
    ///
    /// <para>
    /// The reference's second pattern uses a lookbehind, which .NET supports, so both patterns are
    /// ported verbatim. (The Go port had to replace it with token filtering: RE2 has no lookaround.)
    /// </para>
    /// </summary>
    public static string CleanRulerArtifacts(string value)
    {
        string text = RulerRuns.Replace(value, " ");
        text = LoneSeparator.Replace(text, " ");
        return Whitespace.Replace(text, " ").Trim();
    }

    private static readonly System.Text.RegularExpressions.Regex RulerRuns =
        new(@"[.,_\-""]{2,}", System.Text.RegularExpressions.RegexOptions.Compiled);

    private static readonly System.Text.RegularExpressions.Regex LoneSeparator =
        new(@"(?:^|(?<=\s))[.,_\-""](?=\s|$)", System.Text.RegularExpressions.RegexOptions.Compiled);

    private static readonly System.Text.RegularExpressions.Regex Whitespace =
        new(@"\s+", System.Text.RegularExpressions.RegexOptions.Compiled);
}

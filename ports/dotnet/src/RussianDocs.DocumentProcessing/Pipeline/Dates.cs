using System.Globalization;
using System.Text.RegularExpressions;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// Canonical <c>dd.mm.yyyy</c> view of a recognised date. Port of <c>pipeline/dates.py</c>.
///
/// <para>
/// The pipeline returns dates AS PRINTED — «15 ОКТЯБРЯ 2020 Г.» on a 2018 birth certificate, «10
/// ДЕКАБРЯ 1999 ГОДА» on a SNILS, «03.АВГУСТ.1989» on a 1997 internal passport — because that is what
/// the ground truth describes and what the accuracy measurement compares against. A consumer usually
/// wants a machine form instead, so the canonical view is built ALONGSIDE the reading, never in place
/// of it (<see cref="Results.Ocr"/> keeps the reading; <see cref="Results.OcrNormalized"/> holds this).
/// </para>
///
/// <para>
/// Two rules shape everything here (dates.py:11-17):
/// </para>
/// <list type="bullet">
/// <item>
/// <b>Never guess.</b> No year in the text -> no canonical value. A month word that does not match ->
/// no canonical value. A date outside the calendar (31.02) -> no canonical value. The caller falls
/// back to the reading instead of receiving an invention.
/// </item>
/// <item>
/// <b>Never touch the reading.</b> Trailing «Г.» / «ГОДА» stay in the reading; they are printed on
/// the document. They simply have no place in <c>dd.mm.yyyy</c>.
/// </item>
/// </list>
///
/// <para>Pure functions of a string: no image, no model, no configuration.</para>
/// </summary>
public static class Dates
{
    /// <summary>Month names as the documents print them, nominative and genitive.</summary>
    private static readonly Dictionary<string, int> Months = new(StringComparer.Ordinal)
    {
        ["ЯНВАРЬ"] = 1, ["ЯНВАРЯ"] = 1,
        ["ФЕВРАЛЬ"] = 2, ["ФЕВРАЛЯ"] = 2,
        ["МАРТ"] = 3, ["МАРТА"] = 3,
        ["АПРЕЛЬ"] = 4, ["АПРЕЛЯ"] = 4,
        ["МАЙ"] = 5, ["МАЯ"] = 5,
        ["ИЮНЬ"] = 6, ["ИЮНЯ"] = 6,
        ["ИЮЛЬ"] = 7, ["ИЮЛЯ"] = 7,
        ["АВГУСТ"] = 8, ["АВГУСТА"] = 8,
        ["СЕНТЯБРЬ"] = 9, ["СЕНТЯБРЯ"] = 9,
        ["ОКТЯБРЬ"] = 10, ["ОКТЯБРЯ"] = 10,
        ["НОЯБРЬ"] = 11, ["НОЯБРЯ"] = 11,
        ["ДЕКАБРЬ"] = 12, ["ДЕКАБРЯ"] = 12,
    };

    /// <summary>Words a document prints next to a date that carry no date information.</summary>
    private static readonly HashSet<string> Noise = new(StringComparer.Ordinal)
        { "Г", "Г.", "ГОД", "ГОДА", "ГОДУ" };

    /// <summary><c>[^\W\d_]+|\d+</c> with <c>re.UNICODE</c>: runs of letters, or runs of digits.</summary>
    private static readonly Regex Token = new(@"[^\W\d_]+|\d+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary><c>dd.mm.yyyy</c> for a real calendar date, else null (31.02 is not a date).</summary>
    private static string? AsDate(int day, int month, int year)
    {
        if (month is < 1 or > 12 || year is < 1900 or > 2100)
        {
            return null;
        }
        if (day < 1 || day > DateTime.DaysInMonth(year, month))
        {
            return null;
        }
        return $"{day:D2}.{month:D2}.{year:D4}";
    }

    /// <summary>
    /// Canonical <c>dd.mm.yyyy</c>, or null when the text does not yield one. Port of
    /// <c>dates.to_ddmmyyyy</c> (dates.py:60-104):
    /// <list type="bullet">
    /// <item><c>'22.06.2010'</c> -> <c>'22.06.2010'</c> (already canonical)</item>
    /// <item><c>'15 ОКТЯБРЯ 2020 Г.'</c> -> <c>'15.10.2020'</c></item>
    /// <item><c>'10 ДЕКАБРЯ 1999 ГОДА'</c> -> <c>'10.12.1999'</c></item>
    /// <item><c>'03.АВГУСТ.1989'</c> -> <c>'03.08.1989'</c></item>
    /// <item><c>'5 МАЯ'</c> -> null (no year: guessing one would invent data)</item>
    /// <item><c>'31.02.2020'</c> -> null (not a calendar date)</item>
    /// </list>
    /// </summary>
    public static string? ToDdmmyyyy(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return null;
        }

        var tokens = new List<string>();
        foreach (Match m in Token.Matches(text))
        {
            string t = m.Value.ToUpperInvariant();
            if (Noise.Contains(t) || t == "Г")
            {
                continue;
            }
            tokens.Add(t);
        }
        if (tokens.Count == 0)
        {
            return null;
        }

        int day = -1, month = -1, year = -1;
        foreach (string token in tokens)
        {
            if (IsDigits(token))
            {
                if (!int.TryParse(token, NumberStyles.None, CultureInfo.InvariantCulture,
                        out int value))
                {
                    return null;
                }
                if (token.Length == 4 && year < 0)
                {
                    year = value;
                }
                else if (day < 0 && value is >= 1 and <= 31)
                {
                    day = value;
                }
                else if (month < 0 && value is >= 1 and <= 12)
                {
                    month = value;
                }
                else if (year < 0 && token.Length <= 2)
                {
                    // A two-digit year is ambiguous (26 -> 1926 or 2026?) and this module does not
                    // guess, so it is left unresolved.
                    return null;
                }
                continue;
            }

            if (!Months.TryGetValue(token, out int resolved) || month >= 0)
            {
                return null;
            }
            month = resolved;
        }

        if (day < 0 || month < 0 || year < 0)
        {
            return null;
        }
        return AsDate(day, month, year);
    }

    private static bool IsDigits(string s)
    {
        if (s.Length == 0)
        {
            return false;
        }
        foreach (char c in s)
        {
            if (c is < '0' or > '9')
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Canonical view of every date field that yields one. Port of <c>dates.canonical_dates</c>.
    ///
    /// <para>
    /// Returns a NEW dictionary holding only the fields that converted — a field that did not convert
    /// is simply absent, so the consumer can tell "no canonical form" from "canonical form equals the
    /// reading". Never mutates <paramref name="ocr"/>.
    /// </para>
    /// </summary>
    public static Dictionary<string, string> CanonicalDates(Dictionary<string, string> ocr,
        IEnumerable<string> fields)
    {
        var outMap = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string name in fields)
        {
            if (!ocr.TryGetValue(name, out string? value))
            {
                continue;
            }
            string? canonical = ToDdmmyyyy(value);
            if (!string.IsNullOrEmpty(canonical))
            {
                outMap[name] = canonical;
            }
        }
        return outMap;
    }

    /// <summary>
    /// Port of <c>Pipeline._normalize_dates</c>: date fields are recognised BY NAME, the same
    /// <c>'date' in name.lower()</c> convention the field-joiner uses. Runs once, on the finished OCR
    /// dictionary, so nothing upstream sees a rewritten value. Returns an empty dictionary when no
    /// field converted, matching the absent <c>OCR_normalized</c> key of the reference.
    /// </summary>
    public static Dictionary<string, string> NormalizeDates(Dictionary<string, string> ocr,
        IEnumerable<string> order)
    {
        List<string> fields = [.. order.Where(
            name => name.Contains("date", StringComparison.OrdinalIgnoreCase))];
        return CanonicalDates(ocr, fields);
    }
}

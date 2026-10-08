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

    /// <summary>
    /// Words a document prints next to a date that carry no date information. «месяца» and «числа»
    /// belong to the 1998 birth certificate's record date, printed in reverse order around the
    /// values: «2010 года июня месяца 15 числа».
    /// </summary>
    private static readonly HashSet<string> Noise = new(StringComparer.Ordinal)
        { "Г", "Г.", "ГОД", "ГОДА", "ГОДУ", "МЕСЯЦ", "МЕСЯЦА", "ЧИСЛО", "ЧИСЛА" };

    /// <summary>
    /// Fields printed as a civil-registry record date: year, month, day in a FIXED order with
    /// printed words between them — «2010 года июня месяца 15 числа» on the 1998 birth certificate.
    /// The box spans the printed words, the word split often loses the gaps («2015ГОДАИЮНЯМЕСЯЦА16»)
    /// and the printed words come back misread («ИЕСЯЦА», «ТОДА»), so the general converter refuses
    /// most of them. Mirrors <c>RECORD_DATE_FIELDS</c> in dates.py.
    /// </summary>
    private static readonly string[] RecordDateFields = ["Act_date"];

    /// <summary>Month names in the genitive — the only case a record date prints.</summary>
    private static readonly KeyValuePair<string, int>[] Genitive =
        [.. Months.Where(kv => kv.Key.EndsWith('Я') || kv.Key.EndsWith('А'))];

    /// <summary><c>[^\W\d_]+|\d+</c> with <c>re.UNICODE</c>: runs of letters, or runs of digits.</summary>
    private static readonly Regex Token = new(@"[^\W\d_]+|\d+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// Drops a lone letter standing right next to the day number. Port of
    /// <c>dates._drop_quote_letters</c> (issue #23).
    ///
    /// <para>
    /// The 1998 birth certificate prints the issue date as «10» ЯНВАРЯ 2013 г., and the field box
    /// starts on the opening quote. «» are not in the Cyrillic engine's alphabet, so the engine reads
    /// the quote as the nearest letter it knows: «И 10 ЯНВАРЯ 2013». The letter carries no date
    /// information, but as an unknown word it made the whole date refuse.
    /// </para>
    ///
    /// <para>
    /// Only a SINGLE letter and only ADJACENT to a one- or two-digit number (the day, on either side
    /// — the closing quote sits after it) is dropped, and "adjacent" is judged on the ORIGINAL token
    /// list, as the reference does. Anything else — a longer word, a letter elsewhere, a one-letter
    /// month name — still refuses: this reads a known misreading of printed punctuation, it does not
    /// guess.
    /// </para>
    /// </summary>
    private static List<string> DropQuoteLetters(List<string> tokens)
    {
        bool IsDay(int i) => i >= 0 && i < tokens.Count && IsDigits(tokens[i]) && tokens[i].Length <= 2;

        var kept = new List<string>(tokens.Count);
        for (int i = 0; i < tokens.Count; i++)
        {
            string t = tokens[i];
            bool quoteLetter = t.Length == 1 && !IsDigits(t) && !Months.ContainsKey(t)
                && (IsDay(i - 1) || IsDay(i + 1));
            if (!quoteLetter)
            {
                kept.Add(t);
            }
        }
        return kept;
    }

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
        tokens = DropQuoteLetters(tokens);
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

    /// <summary>
    /// Canonical <c>dd.mm.yyyy</c> of a civil-registry record date, or null. Port of
    /// <c>dates.record_date_to_ddmmyyyy</c>.
    ///
    /// <para>
    /// Whatever the general converter accepts is taken as is. Otherwise the parts are found by their
    /// FORM, which is what the fixed layout allows: exactly one four-digit year, exactly one one- or
    /// two-digit day, and exactly one genitive month name found INSIDE the letters (glued or not),
    /// whatever the printed words around it were read as. Any ambiguity — two days, two months, no
    /// year — refuses, as everywhere in this module:
    /// </para>
    /// <list type="bullet">
    /// <item><c>'2015ГОДАИЮНЯИЕСЯЦА16'</c> -> <c>'16.06.2015'</c></item>
    /// <item><c>'2010 ГОДА ЦЮЛЯ МЕСЯЦА 17'</c> -> null (the month itself is misread)</item>
    /// <item><c>'2020 ГОДА ИЮЛЯ МЕСЯЦА'</c> -> null (no day)</item>
    /// </list>
    /// </summary>
    public static string? RecordDateToDdmmyyyy(string? text)
    {
        string? canonical = ToDdmmyyyy(text);
        if (!string.IsNullOrEmpty(canonical) || string.IsNullOrEmpty(text))
        {
            return canonical;
        }

        var years = new List<string>();
        var days = new List<string>();
        bool stray = false;
        var letters = new System.Text.StringBuilder();
        foreach (Match m in Token.Matches(text.ToUpperInvariant()))
        {
            string r = m.Value;
            if (!IsDigits(r))
            {
                letters.Append(r);
            }
            else if (r.Length == 4)
            {
                years.Add(r);
            }
            else if (r.Length <= 2)
            {
                days.Add(r);
            }
            else
            {
                stray = true;
            }
        }
        if (years.Count != 1 || days.Count != 1 || stray)
        {
            return null;
        }

        string joined = letters.ToString();
        var months = new HashSet<int>();
        foreach (KeyValuePair<string, int> kv in Genitive)
        {
            if (joined.Contains(kv.Key, StringComparison.Ordinal))
            {
                months.Add(kv.Value);
            }
        }
        if (months.Count != 1)
        {
            return null;
        }
        return AsDate(int.Parse(days[0], NumberStyles.None, CultureInfo.InvariantCulture),
            months.First(), int.Parse(years[0], NumberStyles.None, CultureInfo.InvariantCulture));
    }

    /// <summary>The canonical view of one field: by its printed layout. Port of <c>dates.canonical_date</c>.</summary>
    public static string? CanonicalDate(string field, string? text) =>
        Array.IndexOf(RecordDateFields, field) >= 0
            ? RecordDateToDdmmyyyy(text)
            : ToDdmmyyyy(text);

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
            string? canonical = CanonicalDate(name, value);
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

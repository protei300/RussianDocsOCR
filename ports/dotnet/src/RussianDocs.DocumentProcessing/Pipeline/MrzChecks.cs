namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// ICAO 9303 check digits and the parts of a machine-readable zone the per-field verdicts lean on.
/// Port of the used part of <c>validation/mrz.py</c>: <c>check_digit</c>, <c>validate_mrz</c>,
/// <c>parse_mrz</c> (with the internal passport's optional data), <c>printed_number</c>,
/// <c>numbers_agree</c> and <c>_mrz_date</c>. <c>cross_check</c> and <c>validate</c> are not ported:
/// nothing in the port calls them.
///
/// <para>
/// Pure functions of a string. The only outside input is the current year, which decides the century of
/// a two-digit birth year (<c>_mrz_date</c> reads <c>date.today()</c> in the reference); it is a
/// parameter here so a test can fix it, and <see cref="FieldChecks"/> passes the same "today" it uses
/// for its own calendar rules.
/// </para>
///
/// <para>
/// One deliberate difference: the reference's <c>isdigit()</c> and <c>\d</c> accept every Unicode digit
/// (Arabic-Indic, superscripts). Here only <c>0-9</c> counts. The reading never produces anything else
/// (the Latin engine's alphabet is ASCII), and the zone's own line 2 must match
/// <c>[A-Z0-9&lt;]+</c> before any digit is looked at.
/// </para>
/// </summary>
public static class MrzChecks
{
    public const char Filler = '<';
    public const int LineLen = 44;

    /// <summary>Document type of the internal passport's zone ("Passport National").</summary>
    public const string InternalType = "PN";

    /// <summary>The verdict of one check digit: <c>ok</c>, <c>failed</c> or <c>not_applicable</c>.</summary>
    public const string Ok = "ok";

    /// <summary>TD3 line-2 layout: (name, payload start, payload end, check-digit position).</summary>
    private static readonly (string Name, int Start, int End, int DigitPos)[] Line2Fields =
    [
        ("document_number", 0, 9, 9),
        ("birth_date", 13, 19, 19),
        ("expiry_date", 21, 27, 27),
        ("personal_number", 28, 42, 42),
    ];

    /// <summary>
    /// The composite digit covers the number, the birth date and everything from the expiry date to the
    /// personal-number check digit, each WITH its own digit.
    /// </summary>
    private static readonly (int Start, int End)[] CompositeParts = [(0, 10), (13, 20), (21, 43)];

    private const int CompositePos = 43;

    /// <summary>Result of <see cref="Validate"/>: the overall flag and a verdict per check digit.</summary>
    public sealed record Validation(bool Ok, IReadOnlyDictionary<string, string> Checks);

    // ---- Python str helpers --------------------------------------------------------------------

    /// <summary>
    /// Python's <c>str.isspace</c> for the characters that can occur: .NET's white space plus the
    /// information separators U+001C..U+001F, which Python treats as white space and .NET does not.
    /// </summary>
    internal static bool IsPySpace(char c) => char.IsWhiteSpace(c) || c is >= '\u001c' and <= '\u001f';

    /// <summary><c>str.strip()</c>.</summary>
    internal static string PyStrip(string s)
    {
        int start = 0;
        int end = s.Length;
        while (start < end && IsPySpace(s[start]))
        {
            start++;
        }
        while (end > start && IsPySpace(s[end - 1]))
        {
            end--;
        }
        return s[start..end];
    }

    /// <summary><c>str.splitlines()</c>: the line boundaries Python knows, without the terminators.</summary>
    internal static List<string> PySplitLines(string s)
    {
        var lines = new List<string>();
        int start = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            bool boundary = c is '\n' or '\r' or '\u000b' or '\u000c' or '\u001c' or '\u001d' or '\u001e'
                or '\x0085' or (char)0x2028 or (char)0x2029;
            if (!boundary)
            {
                continue;
            }
            lines.Add(s[start..i]);
            if (c == '\r' && i + 1 < s.Length && s[i + 1] == '\n')
            {
                i++;
            }
            start = i + 1;
        }
        if (start < s.Length)
        {
            lines.Add(s[start..]);
        }
        return lines;
    }

    /// <summary>Python slicing <c>s[a:b]</c> for non-negative bounds: out-of-range is clamped, never an error.</summary>
    private static string Slice(string s, int start, int end = int.MaxValue)
    {
        int a = Math.Min(start, s.Length);
        int b = Math.Min(end, s.Length);
        return b <= a ? "" : s[a..b];
    }

    private static bool IsAsciiDigit(char c) => c is >= '0' and <= '9';

    private static bool AllAsciiDigits(string s) => s.Length > 0 && s.All(IsAsciiDigit);

    private static string? NullIfEmpty(string s) => s.Length == 0 ? null : s;

    // ---- check digits ----------------------------------------------------------------------------

    /// <summary>
    /// ICAO 9303 check digit: weights cycle 7-3-1, <c>A</c>-<c>Z</c> are 10-35, <c>&lt;</c> is 0.
    /// Null when the payload holds a character outside the MRZ alphabet - an OCR artifact, for which no
    /// meaningful digit exists.
    /// </summary>
    public static int? CheckDigit(string payload)
    {
        ReadOnlySpan<int> weights = [7, 3, 1];
        int total = 0;
        for (int i = 0; i < payload.Length; i++)
        {
            char ch = payload[i];
            int value;
            if (ch == Filler)
            {
                value = 0;
            }
            else if (IsAsciiDigit(ch))
            {
                value = ch - '0';
            }
            else if (ch is >= 'A' and <= 'Z')
            {
                value = ch - 'A' + 10;
            }
            else
            {
                return null;
            }
            total += value * weights[i % 3];
        }
        return total % 10;
    }

    /// <summary>MRZ string to its non-empty stripped lines. The pipeline joins detected lines with '\n'.</summary>
    public static List<string> SplitLines(string? mrzText)
    {
        if (string.IsNullOrEmpty(mrzText))
        {
            return [];
        }
        var lines = new List<string>();
        foreach (string raw in PySplitLines(mrzText))
        {
            string line = PyStrip(raw);
            if (line.Length > 0)
            {
                lines.Add(line);
            }
        }
        return lines;
    }

    private static bool IsMrzChar(char c) => c is >= 'A' and <= 'Z' or >= '0' and <= '9' or Filler;

    /// <summary>
    /// A field the document does not carry: all filler, filler in place of the digit. This is what an
    /// internal passport does with the expiry date.
    /// </summary>
    private static bool IsEmptyField(string payload, string digit) =>
        payload.All(c => c == Filler) && digit == Filler.ToString();

    /// <summary>
    /// EXT-4: verify every check digit of the MRZ. <see cref="Validation.Ok"/> is false for a malformed
    /// zone (fewer than two lines, a second line of the wrong length or alphabet): a 43-character line
    /// means characters were lost, which is itself a finding. <see cref="Validation.Checks"/> then holds
    /// no digits at all.
    /// </summary>
    public static Validation Validate(string mrzText)
    {
        List<string> lines = SplitLines(mrzText);
        var checks = new Dictionary<string, string>(StringComparer.Ordinal);
        if (lines.Count < 2)
        {
            return new Validation(false, checks);
        }
        string line2 = lines[1];
        if (line2.Length != LineLen || !line2.All(IsMrzChar))
        {
            return new Validation(false, checks);
        }

        foreach ((string name, int start, int end, int digitPos) in Line2Fields)
        {
            string payload = line2[start..end];
            string digit = line2[digitPos].ToString();
            if (IsEmptyField(payload, digit))
            {
                checks[name] = "not_applicable";
                continue;
            }
            int? expected = CheckDigit(payload);
            checks[name] = expected is not null && expected.Value.ToString() == digit ? Ok : "failed";
        }

        string composite = string.Concat(CompositeParts.Select(p => line2[p.Start..p.End]));
        int? compositeExpected = CheckDigit(composite);
        checks["composite"] = compositeExpected is not null
                              && compositeExpected.Value.ToString() == line2[CompositePos].ToString()
            ? Ok
            : "failed";

        return new Validation(checks.Values.All(v => v != "failed"), checks);
    }

    // ---- parsing ---------------------------------------------------------------------------------

    /// <summary>
    /// YYMMDD to <c>dd.mm.yyyy</c>, or null if it is not a real calendar date.
    ///
    /// <para>
    /// The MRZ carries no century, so it has to come from what the field means, and the two directions
    /// are opposite. A birth date is in the past: a two-digit year later than the current one belongs to
    /// the previous century. An expiry date is in the future and is always 20xx - reading it by the
    /// birth-date rule turns a passport valid until 2029 into one that expired in 1929.
    /// </para>
    /// </summary>
    public static string? MrzDate(string raw, string kind, int currentYear)
    {
        if (raw.Length != 6 || !AllAsciiDigits(raw))
        {
            return null;
        }
        int yy = int.Parse(raw[0..2]);
        int mm = int.Parse(raw[2..4]);
        int dd = int.Parse(raw[4..6]);
        int century = kind == "future" ? 2000 : (yy <= currentYear % 100 ? 2000 : 1900);
        int year = century + yy;
        if (mm is < 1 or > 12 || dd < 1 || dd > DateTime.DaysInMonth(year, mm))
        {
            return null;
        }
        return $"{dd:D2}.{mm:D2}.{year:D4}";
    }

    /// <summary>
    /// Field values carried by the zone. Missing or malformed parts are null (and a key is present only
    /// where the reference sets it). <paramref name="internalZone"/> says whether this is an internal
    /// passport's zone, whose optional data holds three printed fields; null takes it from the zone's own
    /// type characters - a caller that knows the document type from the classifier should pass it,
    /// because those two characters can be misread like any other.
    /// </summary>
    public static Dictionary<string, string?> Parse(string mrzText, bool? internalZone, int currentYear)
    {
        List<string> lines = SplitLines(mrzText);
        var fields = new Dictionary<string, string?>(StringComparer.Ordinal);
        if (lines.Count > 0)
        {
            string line1 = lines[0];
            fields["document_type"] = NullIfEmpty(Slice(line1, 0, 2).Replace("<", ""));
            fields["issuing_state"] = NullIfEmpty(Slice(line1, 2, 5));
            string rest = Slice(line1, 5);
            int split = rest.IndexOf("<<", StringComparison.Ordinal);
            string surname = split < 0 ? rest : rest[..split];
            string given = split < 0 ? "" : rest[(split + 2)..];
            fields["surname"] = NullIfEmpty(PyStrip(surname.Replace('<', ' ')));
            fields["given_names"] = NullIfEmpty(PyStrip(given.Replace('<', ' ')));
        }
        if (lines.Count > 1 && lines[1].Length == LineLen)
        {
            string line2 = lines[1];
            fields["document_number"] = NullIfEmpty(line2[0..9].Replace("<", ""));
            fields["nationality"] = NullIfEmpty(line2[10..13]);
            fields["birth_date"] = MrzDate(line2[13..19], "past", currentYear);
            fields["sex"] = line2[20] is 'M' or 'F' ? line2[20].ToString() : null;
            fields["expiry_date"] = MrzDate(line2[21..27], "future", currentYear);
            fields["personal_number"] = NullIfEmpty(line2[28..42].Replace("<", ""));
            bool isInternal = internalZone ?? fields.GetValueOrDefault("document_type") == InternalType;
            if (isInternal)
            {
                foreach ((string key, string? value) in InternalOptionalData(line2[28..42], currentYear))
                {
                    fields[key] = value;
                }
            }
        }
        return fields;
    }

    /// <summary>
    /// The internal passport's optional data: three printed fields in one run. FMS order No. 279
    /// (2011-06-30), table 2, positions 29-42: the LAST digit of the series, the issue date as YYMMDD, the
    /// issuing subdivision's code as six digits without the hyphen, then a filler. The zone's own number
    /// (positions 1-9) holds only the first three digits of the series, so the fourth one has to be taken
    /// from here to rebuild the printed series and number.
    /// </summary>
    private static Dictionary<string, string?> InternalOptionalData(string optional, int currentYear)
    {
        string digit = Slice(optional, 0, 1);
        string code = Slice(optional, 7, 13);
        return new Dictionary<string, string?>(StringComparer.Ordinal)
        {
            ["series_last_digit"] = AllAsciiDigits(digit) ? digit : null,
            ["issue_date"] = MrzDate(Slice(optional, 1, 7), "past", currentYear),
            ["subdivision_code"] = code.Length == 6 && AllAsciiDigits(code) ? code : null,
        };
    }

    /// <summary>
    /// The zone's document number in the form the page prints it, digits only. External passport: the
    /// number field as is. Internal passport: the first three digits of the series + the fourth one from
    /// the optional data + the six digits of the number - the zone splits the ten printed digits in two
    /// places, and comparing its nine-digit field with the page's ten digits reported every correctly
    /// read internal passport as a mismatch.
    /// </summary>
    public static string? PrintedNumber(IReadOnlyDictionary<string, string?> fields, bool? internalZone)
    {
        string? number = fields.GetValueOrDefault("document_number");
        if (string.IsNullOrEmpty(number))
        {
            return null;
        }
        bool isInternal = internalZone ?? fields.GetValueOrDefault("document_type") == InternalType;
        if (!isInternal)
        {
            return number;
        }
        string? last = fields.GetValueOrDefault("series_last_digit");
        if (string.IsNullOrEmpty(last) || number.Length != 9)
        {
            return null;
        }
        return number[..3] + last + number[3..];
    }

    /// <summary>
    /// Zone number against the page's digits. Five of the 19 external passports of the 2003 form in
    /// samples/ write the number WITHOUT its two-digit series (seven digits); the rest, and every
    /// biometric one, write all nine. A seven-digit zone number is therefore compared with the end of the
    /// printed number.
    /// </summary>
    public static bool NumbersAgree(string? zoneNumber, string? pageDigits)
    {
        if (string.IsNullOrEmpty(zoneNumber) || string.IsNullOrEmpty(pageDigits))
        {
            return false;
        }
        if (zoneNumber.Length == 7 && pageDigits.Length == 9)
        {
            return pageDigits.EndsWith(zoneNumber, StringComparison.Ordinal);
        }
        return zoneNumber == pageDigits;
    }
}

using System.Text;
using System.Text.RegularExpressions;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>One check that ran over one field: <c>{id, kind, result, reference?}</c>.</summary>
/// <param name="Id">A row of <c>docs/validation-checks.md</c> (GEN-1, MRZ-2, SNILS-2, ...).</param>
/// <param name="Kind"><c>confirming</c> or <c>refuting</c>.</param>
/// <param name="Result"><c>ok</c> or <c>mismatch</c>.</param>
/// <param name="Reference">The value the field was compared with, where the check has one.</param>
public sealed record FieldCheck(string Id, string Kind, string Result, string? Reference);

/// <summary>A soft remark <c>{id, text}</c>: it never changes a verdict.</summary>
public sealed record FieldNote(string Id, string Text);

/// <summary>The verdict for one field: <c>verified</c>, <c>mismatch</c> or <c>unchecked</c>, and why.</summary>
public sealed record FieldVerdict(string Status, IReadOnlyList<FieldCheck> Checks, IReadOnlyList<FieldNote> Notes);

/// <summary>
/// A verdict for every recognised field: verified, mismatch or unchecked. Port of
/// <c>validation/fields.py</c> (decision No. 152; format and "mark only, never correct" chosen
/// 2026-10-19). Check ids are the rows of <c>docs/validation-checks.md</c>.
///
/// <para>
/// Every check declares what its agreement is worth. <b>Confirming</b>: two independent readings agree,
/// or a check digit holds - any single changed character breaks it, so agreement verifies the field.
/// <b>Refuting</b>: a plausibility rule (birth before issue, a real calendar date) - almost any misreading
/// still satisfies it, so agreement proves nothing and only disagreement counts. A <b>soft</b> rule never
/// changes a verdict and shows up as a note. A disagreement between two fields marks BOTH. The reading
/// itself is never changed: the value a field was compared with travels as <c>Reference</c>.
/// </para>
///
/// <para>
/// Pure functions of the OCR dictionary and the document type: no image, no model. The order of the
/// entries in <c>Checks</c> is the order the checks ran (dates, number forms, zone, SNILS, the soft note),
/// and the conformance stage <c>checks</c> compares it.
/// </para>
///
/// <para>
/// Differences from the reference that cannot show on a real reading: digits are ASCII only (Python's
/// <c>\d</c> and <c>isdigit()</c> take every Unicode digit; the engines never produce one), and the
/// reading is a <c>string</c> map, so the reference's non-string values (<c>Address_has_handwritten</c> is
/// a boolean there) arrive as the strings "True"/"False" - no check reads that field, it gets an
/// <c>unchecked</c> entry either way.
/// </para>
/// </summary>
public static class FieldChecks
{
    public const string Verified = "verified";
    public const string Mismatch = "mismatch";
    public const string Unchecked = "unchecked";

    public const string Confirming = "confirming";
    public const string Refuting = "refuting";

    private const string Ok = "ok";

    /// <summary>
    /// Cyrillic to zone characters of the internal passport's top MRZ line, FMS order No. 279
    /// (2011-06-30), the "modernised clear" table: letters with no Latin counterpart become digits.
    /// </summary>
    private static readonly Dictionary<char, char> ZoneLetters = new()
    {
        ['А'] = 'A', ['Б'] = 'B', ['В'] = 'V', ['Г'] = 'G', ['Д'] = 'D', ['Е'] = 'E', ['Ё'] = '2',
        ['Ж'] = 'J', ['З'] = 'Z', ['И'] = 'I', ['Й'] = 'Q', ['К'] = 'K', ['Л'] = 'L', ['М'] = 'M',
        ['Н'] = 'N', ['О'] = 'O', ['П'] = 'P', ['Р'] = 'R', ['С'] = 'S', ['Т'] = 'T', ['У'] = 'U',
        ['Ф'] = 'F', ['Х'] = 'H', ['Ц'] = 'C', ['Ч'] = '3', ['Ш'] = '4', ['Щ'] = 'W', ['Ъ'] = 'X',
        ['Ы'] = 'Y', ['Ь'] = '9', ['Э'] = '6', ['Ю'] = '7', ['Я'] = '8',
    };

    /// <summary>
    /// Sex as printed to the zone's letter. The page prints «МУЖ.»/«ЖЕН.» (the ground truth keeps the
    /// first letter), the external passport also «M»/«F».
    /// </summary>
    private static readonly Dictionary<char, char> Sex = new()
    {
        ['М'] = 'M', ['Ж'] = 'F', ['M'] = 'M', ['F'] = 'F',
    };

    /// <summary>
    /// Years between the issue date and the expiry date. Driving licences: 10. External passports: 10 for
    /// the biometric form, 5 for the 2003 one - either is accepted for both, so a form the classifier
    /// confused costs a little strength instead of a false alarm on two dates.
    /// </summary>
    private static readonly Dictionary<string, int[]> ValidityYears = new()
    {
        ["DL"] = [10],
        ["EXTPASSPORT"] = [5, 10],
    };

    /// <summary>
    /// Minimum age at issue. 14 for the internal passport, 16 for a licence. SNILS is issued to children
    /// and the external passport too, so neither has an age rule.
    /// </summary>
    private static readonly Dictionary<string, int> MinAge = new()
    {
        ["INTPASSPORT"] = 14,
        ["DL"] = 16,
    };

    /// <summary>
    /// Soft: the second pair of the internal passport's series is the year the blank was printed, usually
    /// within a few years of the issue date - a note, never a verdict, and a window of three years.
    /// </summary>
    private const int SeriesYearWindow = 3;

    private static readonly string[] DateFields = ["Birth_date", "Issue_date", "Expiration_date"];

    /// <summary>
    /// The printed form of numbers, whitespace collapsed (internal passport, its code, external passport,
    /// SNILS). The comparisons with the zone look at digits only, so without the form a reading with a
    /// stray character was 'verified' while not being what is printed.
    /// </summary>
    private static readonly Dictionary<string, (string Field, string Id, Regex Pattern)[]> NumberForms = new()
    {
        ["INTPASSPORT"] =
        [
            ("Licence_number", "PASS-1", Form(@"[0-9]{2} [0-9]{2} [0-9]{6}")),
            ("Issue_organisation_code", "PASS-4", Form(@"[0-9]{3}-[0-9]{3}")),
        ],
        ["EXTPASSPORT"] = [("Licence_number", "EXT-1", Form(@"[0-9]{2} [0-9]{7}"))],
        ["SNILS"] = [("Licence_number", "SNILS-1", Form(@"[0-9]{3}-[0-9]{3}-[0-9]{3} [0-9]{2}"))],
    };

    private static Regex Form(string pattern) =>
        new($@"\A{pattern}\z", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static readonly Dictionary<string, string> Line1Prefix = new()
    {
        ["INTPASSPORT"] = "PNRUS",
        ["EXTPASSPORT"] = "P<RUS",
    };

    private const string Latin = "ABCDEFGHIJKLMNOPQRSTUVWXYZ<";

    /// <summary>
    /// The characters the name area of the top MRZ line can hold. No check digit covers this line, but its
    /// form is strict: an external passport writes names in Latin letters only; the internal one also uses
    /// the digits of the order No. 279 table (2 3 4 6 7 8 9) - never 0, 1 or 5.
    /// </summary>
    private static readonly Dictionary<string, HashSet<char>> NameChars = new()
    {
        ["INTPASSPORT"] = [.. Latin, .. "2346789"],
        ["EXTPASSPORT"] = [.. Latin],
    };

    /// <summary>
    /// The document family the checks are defined for, or null. Order matters as in
    /// <c>OcrOptions.For</c>: the registration page and the licence back side contain the names of the
    /// families they are not. Everything outside the four families gets <c>unchecked</c> throughout.
    /// </summary>
    public static string? Family(string? docType)
    {
        string label = (docType ?? "").ToUpperInvariant();
        if (label.Contains("INTPASSPORTADDR", StringComparison.Ordinal)
            || label.Contains("DLBACK", StringComparison.Ordinal))
        {
            return null;
        }
        foreach (string name in new[] { "INTPASSPORT", "EXTPASSPORT", "SNILS", "DL" })
        {
            if (label.StartsWith(name, StringComparison.Ordinal))
            {
                return name;
            }
        }
        return null;
    }

    /// <summary>
    /// Verdict per field of <paramref name="ocr"/>. Every key gets an entry, so "no entry" never has to
    /// be told apart from "unchecked". <paramref name="today"/> defaults to the clock, as
    /// <c>date.today()</c> does in the reference: it decides "not in the future" and the century of a
    /// two-digit year in the zone.
    /// </summary>
    public static Dictionary<string, FieldVerdict> Compute(
        IReadOnlyDictionary<string, string>? ocr, string? docType, DateOnly? today = null)
    {
        var v = new Verdicts(ocr ?? new Dictionary<string, string>());
        string? fam = Family(docType);
        if (fam is null)
        {
            return v.Result();
        }
        DateOnly now = today ?? DateOnly.FromDateTime(DateTime.Now);
        CheckDates(v, fam, now);
        CheckForms(v, fam);
        if (fam is "INTPASSPORT" or "EXTPASSPORT")
        {
            CheckMrz(v, fam, now);
        }
        if (fam == "SNILS")
        {
            CheckSnils(v);
        }
        if (fam == "INTPASSPORT")
        {
            NoteSeriesYear(v);
        }
        return v.Result();
    }

    /// <summary>
    /// The verdicts as the conformance stage <c>checks</c> writes them: <c>{Field: {status, checks:
    /// [{id, kind, result, reference?}], notes: [{id, text}]}}</c>, <c>reference</c> absent where there is
    /// no value.
    /// </summary>
    public static Dictionary<string, object> ToPayload(IReadOnlyDictionary<string, FieldVerdict> verdicts)
    {
        var payload = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach ((string name, FieldVerdict verdict) in verdicts)
        {
            var checks = new List<object>();
            foreach (FieldCheck c in verdict.Checks)
            {
                var entry = new Dictionary<string, object>
                {
                    ["id"] = c.Id,
                    ["kind"] = c.Kind,
                    ["result"] = c.Result,
                };
                if (c.Reference is not null)
                {
                    entry["reference"] = c.Reference;
                }
                checks.Add(entry);
            }
            var notes = new List<object>();
            foreach (FieldNote n in verdict.Notes)
            {
                notes.Add(new Dictionary<string, object> { ["id"] = n.Id, ["text"] = n.Text });
            }
            payload[name] = new Dictionary<string, object>
            {
                ["status"] = verdict.Status,
                ["checks"] = checks,
                ["notes"] = notes,
            };
        }
        return payload;
    }

    // ---- collecting ------------------------------------------------------------------------------

    /// <summary>Collects check results per field and turns them into verdicts.</summary>
    private sealed class Verdicts
    {
        private readonly IReadOnlyDictionary<string, string> _ocr;
        private readonly Dictionary<string, (List<FieldCheck> Checks, List<FieldNote> Notes)> _entries =
            new(StringComparer.Ordinal);

        public Verdicts(IReadOnlyDictionary<string, string> ocr)
        {
            _ocr = ocr;
            foreach (string name in ocr.Keys)
            {
                _entries[name] = ([], []);
            }
        }

        /// <summary>The field's reading, stripped, or null when absent or empty.</summary>
        public string? Value(string name)
        {
            if (!_ocr.TryGetValue(name, out string? value) || value is null)
            {
                return null;
            }
            string stripped = MrzChecks.PyStrip(value);
            return stripped.Length == 0 ? null : stripped;
        }

        /// <summary>Records one check over one or several fields (only those in the reading).</summary>
        public void Add(string[] fields, string id, string kind, bool agrees, string? reference = null,
            IReadOnlyDictionary<string, string?>? perField = null)
        {
            foreach (string name in fields)
            {
                if (!_entries.TryGetValue(name, out var entry))
                {
                    continue;
                }
                string? refValue = perField is not null ? perField.GetValueOrDefault(name) : reference;
                entry.Checks.Add(new FieldCheck(id, kind, agrees ? Ok : Mismatch, refValue));
            }
        }

        public void Add(string field, string id, string kind, bool agrees, string? reference = null) =>
            Add([field], id, kind, agrees, reference);

        public void Note(string[] fields, string id, string text)
        {
            foreach (string name in fields)
            {
                if (_entries.TryGetValue(name, out var entry))
                {
                    entry.Notes.Add(new FieldNote(id, text));
                }
            }
        }

        public Dictionary<string, FieldVerdict> Result()
        {
            var result = new Dictionary<string, FieldVerdict>(StringComparer.Ordinal);
            foreach ((string name, (List<FieldCheck> checks, List<FieldNote> notes)) in _entries)
            {
                string status = checks.Any(c => c.Result == Mismatch) ? Mismatch
                    : checks.Any(c => c.Kind == Confirming) ? Verified
                    : Unchecked;
                result[name] = new FieldVerdict(status, checks, notes);
            }
            return result;
        }
    }

    // ---- helpers ---------------------------------------------------------------------------------

    private static DateOnly? AsDate(string name, string? text)
    {
        string? canonical = string.IsNullOrEmpty(text) ? null : Dates.CanonicalDate(name, text);
        if (string.IsNullOrEmpty(canonical))
        {
            return null;
        }
        string[] parts = canonical.Split('.');
        return new DateOnly(int.Parse(parts[2]), int.Parse(parts[1]), int.Parse(parts[0]));
    }

    /// <summary>The same day <paramref name="years"/> later; 29.02 maps to both 28.02 and 01.03.</summary>
    private static List<DateOnly> AddYears(DateOnly day, int years)
    {
        int year = day.Year + years;
        if (day is { Month: 2, Day: 29 } && !DateTime.IsLeapYear(year))
        {
            return [new DateOnly(year, 2, 28), new DateOnly(year, 3, 1)];
        }
        return [new DateOnly(year, day.Month, day.Day)];
    }

    private static int Age(DateOnly birth, DateOnly at) =>
        at.Year - birth.Year - ((at.Month, at.Day).CompareTo((birth.Month, birth.Day)) < 0 ? 1 : 0);

    private static string Fmt(DateOnly day) => $"{day.Day:D2}.{day.Month:D2}.{day.Year:D4}";

    /// <summary>Digits only; ASCII (see the type remarks).</summary>
    private static string Digits(string? text) =>
        string.IsNullOrEmpty(text) ? "" : new string([.. text.Where(c => c is >= '0' and <= '9')]);

    /// <summary><c>' '.join(text.split())</c>.</summary>
    private static string CollapseSpaces(string text)
    {
        var sb = new StringBuilder();
        bool pending = false;
        foreach (char c in text)
        {
            if (MrzChecks.IsPySpace(c))
            {
                pending = sb.Length > 0;
                continue;
            }
            if (pending)
            {
                sb.Append(' ');
                pending = false;
            }
            sb.Append(c);
        }
        return sb.ToString();
    }

    // ---- dates -----------------------------------------------------------------------------------

    /// <summary>GEN-1, GEN-2, PASS-7 / DL-4, DL-2 / EXT-3.</summary>
    private static void CheckDates(Verdicts v, string fam, DateOnly today)
    {
        var dates = new Dictionary<string, DateOnly>(StringComparer.Ordinal);
        foreach (string name in DateFields)
        {
            string? text = v.Value(name);
            if (text is null)
            {
                continue;
            }
            DateOnly? parsed = AsDate(name, text);
            // GEN-1: every reference date in samples/ converts (309 of 309), so a reading that does not
            // is not a date.
            v.Add(name, "GEN-1", Refuting, parsed is not null);
            if (parsed is not null)
            {
                dates[name] = parsed.Value;
            }
        }

        DateOnly? birth = dates.TryGetValue("Birth_date", out DateOnly b) ? b : null;
        DateOnly? issue = dates.TryGetValue("Issue_date", out DateOnly i) ? i : null;
        DateOnly? expiry = dates.TryGetValue("Expiration_date", out DateOnly e) ? e : null;

        if (birth is not null)
        {
            v.Add("Birth_date", "GEN-2", Refuting, birth.Value <= today);
        }
        if (issue is not null)
        {
            v.Add("Issue_date", "GEN-2", Refuting, issue.Value <= today);
        }
        if (birth is not null && issue is not null)
        {
            string[] both = ["Birth_date", "Issue_date"];
            v.Add(both, "GEN-2", Refuting, birth.Value < issue.Value);
            if (MinAge.TryGetValue(fam, out int minAge))
            {
                string checkId = fam == "INTPASSPORT" ? "PASS-7" : "DL-4";
                v.Add(both, checkId, Refuting, Age(birth.Value, issue.Value) >= minAge);
            }
        }
        if (ValidityYears.TryGetValue(fam, out int[]? years) && issue is not null && expiry is not null)
        {
            var expected = new List<DateOnly>();
            foreach (int n in years)
            {
                expected.AddRange(AddYears(issue.Value, n));
            }
            Dictionary<string, string?>? reference = null;
            if (years.Length == 1 && expected.Count == 1)
            {
                List<DateOnly> back = AddYears(expiry.Value, -years[0]);
                reference = new Dictionary<string, string?>(StringComparer.Ordinal)
                {
                    ["Expiration_date"] = Fmt(expected[0]),
                    ["Issue_date"] = back.Count == 1 ? Fmt(back[0]) : null,
                };
            }
            string checkId = fam == "DL" ? "DL-2" : "EXT-3";
            v.Add(["Issue_date", "Expiration_date"], checkId, Confirming, expected.Contains(expiry.Value),
                perField: reference);
        }
    }

    // ---- the zone --------------------------------------------------------------------------------

    /// <summary>A name for comparison with the zone: no spaces, hyphens or fillers.</summary>
    private static string ZoneNames(string text)
    {
        var sb = new StringBuilder();
        foreach (char c in text.ToUpperInvariant())
        {
            if (!MrzChecks.IsPySpace(c) && c is not '-' and not '<')
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }

    /// <summary>
    /// A Cyrillic name as the internal passport's zone writes it. Ё and Е are made equal on both sides
    /// (Ё is '2' in the zone): whether a name was typed with Ё is the clerk's choice, not something the
    /// page and the zone are guaranteed to share.
    /// </summary>
    private static string CyrillicToZone(string text)
    {
        string names = ZoneNames(text).Replace('Ё', 'Е');
        return new string([.. names.Select(c => ZoneLetters.TryGetValue(c, out char z) ? z : c)]);
    }

    /// <summary>Equal, or - when the zone's name area is full - the page starts with it.</summary>
    private static bool NamesAgree(string zone, string page, bool truncated)
    {
        if (zone.Length == 0 || page.Length == 0)
        {
            return false;
        }
        if (zone == page)
        {
            return true;
        }
        return truncated && page.StartsWith(zone, StringComparison.Ordinal);
    }

    /// <summary>MRZ-1 .. MRZ-9 (docs/validation-checks.md, section MRZ).</summary>
    private static void CheckMrz(Verdicts v, string fam, DateOnly today)
    {
        string? mrzText = v.Value("MRZ");
        if (mrzText is null)
        {
            return;
        }
        MrzChecks.Validation checkedZone = MrzChecks.Validate(mrzText);
        // The check digits cover line 2 only, so for the zone as a whole they can refute but not
        // confirm; the confirmation is the names of line 1 agreeing with the page (MRZ-8 below).
        v.Add("MRZ", "MRZ-1", Refuting, checkedZone.Ok);
        IReadOnlyDictionary<string, string> digitsOk = checkedZone.Checks;
        bool internalZone = fam == "INTPASSPORT";
        // The layout is chosen by the classifier's type, not by the zone's own first characters, which
        // may be misread.
        Dictionary<string, string?> zone = MrzChecks.Parse(mrzText, internalZone, today.Year);

        bool Trusted(params string[] parts) =>
            parts.All(p => digitsOk.TryGetValue(p, out string? r) && r == Ok);

        // MRZ-2: the number, in the form the page prints it.
        string pageNumber = Digits(v.Value("Licence_number"));
        string[] numberParts = internalZone ? ["document_number", "personal_number"] : ["document_number"];
        string? zoneNumber = MrzChecks.PrintedNumber(zone, internalZone);
        if (pageNumber.Length > 0 && !string.IsNullOrEmpty(zoneNumber) && Trusted(numberParts))
        {
            v.Add("Licence_number", "MRZ-2", Confirming, MrzChecks.NumbersAgree(zoneNumber, pageNumber), zoneNumber);
        }

        // MRZ-3..MRZ-5: dates, each behind its own check digit.
        var dateParts = new List<(string Page, string Zone, string Part, string Id)>
        {
            ("Birth_date", "birth_date", "birth_date", "MRZ-3"),
        };
        dateParts.Add(internalZone
            ? ("Issue_date", "issue_date", "personal_number", "MRZ-5")
            : ("Expiration_date", "expiry_date", "expiry_date", "MRZ-4"));
        foreach ((string pageName, string zoneName, string part, string checkId) in dateParts)
        {
            string? text = v.Value(pageName);
            string? zoneValue = zone.GetValueOrDefault(zoneName);
            if (text is null || string.IsNullOrEmpty(zoneValue) || !Trusted(part))
            {
                continue;
            }
            DateOnly? pageValue = AsDate(pageName, text);
            v.Add(pageName, checkId, Confirming, pageValue is not null && Fmt(pageValue.Value) == zoneValue,
                zoneValue);
        }

        // MRZ-6: the subdivision code (internal passport only).
        string codePage = Digits(v.Value("Issue_organisation_code"));
        string? codeZone = zone.GetValueOrDefault("subdivision_code");
        if (internalZone && codePage.Length > 0 && !string.IsNullOrEmpty(codeZone) && Trusted("personal_number"))
        {
            v.Add("Issue_organisation_code", "MRZ-6", Confirming, codePage == codeZone,
                $"{codeZone[..3]}-{codeZone[3..]}");
        }

        // MRZ-7: sex. No check digit covers it, but two independent readings agreeing is a confirmation
        // all the same.
        string? sexZone = zone.GetValueOrDefault("sex");
        if (!string.IsNullOrEmpty(sexZone))
        {
            foreach (string name in new[] { "Sex_ru", "Sex_en" })
            {
                string? text = v.Value(name);
                if (text is not null)
                {
                    bool agrees = Sex.TryGetValue(char.ToUpperInvariant(text[0]), out char letter)
                                  && letter.ToString() == sexZone;
                    v.Add(name, "MRZ-7", Confirming, agrees, sexZone);
                }
            }
        }

        // MRZ-8: names from the top line. No check digit covers that line, so a disagreement says "one of
        // the two readings is wrong", not which - and a line of the wrong length may have lost part of a
        // name: not compared.
        List<string> lines = MrzChecks.SplitLines(mrzText);
        string line1 = lines.Count > 0 ? lines[0] : "";
        string nameArea = line1.Length > 5 ? line1[5..] : "";
        bool wellFormed = line1.Length == 44
                          && line1.StartsWith(Line1Prefix[fam], StringComparison.Ordinal)
                          && nameArea.Contains("<<", StringComparison.Ordinal)
                          && nameArea.All(NameChars[fam].Contains);
        v.Add("MRZ", "MRZ-9", Refuting, wellFormed);
        if (!wellFormed)
        {
            return;
        }
        bool truncated = !lines[0].EndsWith('<');
        string surname = (zone.GetValueOrDefault("surname") ?? "").Replace(" ", "");
        string[] given = (zone.GetValueOrDefault("given_names") ?? "").Split(' ', 2);
        string zoneFirst = given[0];
        string zoneRest = given.Length > 1 ? given[1].Replace(" ", "") : "";
        Func<string, string> convert;
        string[] names;
        if (internalZone)
        {
            // Ё is '2' in the zone; made equal to Е on both sides (CyrillicToZone).
            surname = surname.Replace('2', 'E');
            zoneFirst = zoneFirst.Replace('2', 'E');
            zoneRest = zoneRest.Replace('2', 'E');
            convert = CyrillicToZone;
            names = ["Last_name_ru", "First_name_ru", "Middle_name_ru"];
        }
        else
        {
            convert = ZoneNames;
            names = ["Last_name_en", "First_name_en", "Middle_name_en"];
        }
        string[] zoneValues = [surname, zoneFirst, zoneRest];
        var agreed = new List<bool>();
        for (int k = 0; k < names.Length; k++)
        {
            string? text = v.Value(names[k]);
            if (text is null || zoneValues[k].Length == 0)
            {
                continue;
            }
            agreed.Add(NamesAgree(zoneValues[k], convert(text), truncated));
            v.Add(names[k], "MRZ-8", Confirming, agreed[^1], zoneValues[k]);
        }
        // The zone itself is confirmed only when every compared name agrees. A disagreement does not mark
        // it: the zone's own misreadings are caught for certain by MRZ-1 and MRZ-9, and which of the two
        // readings of a name is wrong is unknown - the page field already carries that mismatch.
        if (agreed.Count > 0 && agreed.All(a => a))
        {
            v.Add("MRZ", "MRZ-8", Confirming, true);
        }
    }

    // ---- SNILS and the printed forms -------------------------------------------------------------

    /// <summary>SNILS check number over nine digits (weights 9..1, mod 101).</summary>
    private static int SnilsControl(string body)
    {
        int total = 0;
        for (int i = 0; i < 9; i++)
        {
            total += (body[i] - '0') * (9 - i);
        }
        while (total > 101)
        {
            total %= 101;
        }
        return total is 100 or 101 ? 0 : total;
    }

    /// <summary>PASS-1, PASS-4, EXT-1, SNILS-1: the printed form of numbers (refuting).</summary>
    private static void CheckForms(Verdicts v, string fam)
    {
        if (!NumberForms.TryGetValue(fam, out var forms))
        {
            return;
        }
        foreach ((string name, string checkId, Regex pattern) in forms)
        {
            string? text = v.Value(name);
            if (text is not null)
            {
                v.Add(name, checkId, Refuting, pattern.IsMatch(CollapseSpaces(text)));
            }
        }
    }

    /// <summary>SNILS-2 (the check number); its form is SNILS-1 in <see cref="CheckForms"/>.</summary>
    private static void CheckSnils(Verdicts v)
    {
        string? text = v.Value("Licence_number");
        if (text is null)
        {
            return;
        }
        string digits = Digits(text);
        if (digits.Length != 11)
        {
            return;
        }
        string body = digits[..9];
        // Numbers up to 001-001-998 are issued without a valid check number.
        if (int.Parse(body) <= 1001998)
        {
            return;
        }
        int expected = SnilsControl(body);
        v.Add("Licence_number", "SNILS-2", Confirming, int.Parse(digits[9..]) == expected, $"{expected:D2}");
    }

    /// <summary>PASS-3 (soft): the blank's year in the series against the issue year.</summary>
    private static void NoteSeriesYear(Verdicts v)
    {
        string digits = Digits(v.Value("Licence_number"));
        DateOnly? issue = AsDate("Issue_date", v.Value("Issue_date"));
        if (digits.Length != 10 || issue is null)
        {
            return;
        }
        int yy = int.Parse(digits[2..4]);
        int blankYear = 2000 + yy <= issue.Value.Year + SeriesYearWindow ? 2000 + yy : 1900 + yy;
        if (Math.Abs(issue.Value.Year - blankYear) > SeriesYearWindow)
        {
            v.Note(["Licence_number", "Issue_date"], "PASS-3",
                $"series printed in {blankYear}, issued in {issue.Value.Year}");
        }
    }
}

using System.Globalization;
using System.Text;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// The text-only fixes the reference applies after decoding.
///
/// <para>
/// Every one is a pure string function, which is why they are here rather than in the engine: they are
/// the part of OCR post-processing that can be unit-tested without a model.
/// </para>
/// </summary>
public static class OcrCorrections
{
    /// <summary>
    /// Normalises a date to <c>dd.MM.yyyy</c>.
    ///
    /// <para>
    /// **This function has THREE outcomes, not two, and the third is invisible from inside it.** On
    /// eight digits forming a real date it returns the formatted date. On anything that is not eight
    /// digits it returns the SUBSTITUTED string — <c>O</c> to <c>0</c> and <c>-</c> to <c>.</c>
    /// applied. But on eight digits that are not a valid date, the reference's <c>strptime</c> raises
    /// and its <c>except</c> returns the argument THAT HANDLER received, which never saw the
    /// substitutions — so the original comes back untouched.
    /// </para>
    ///
    /// <para>
    /// Concretely: <c>"123"</c> comes back substituted, <c>"O6-13-1985"</c> comes back completely
    /// unchanged. Confirmed by calling the reference, not by reading it — a first implementation that
    /// returned the substituted string in both cases would differ only on malformed input, where
    /// nobody looks.
    /// </para>
    /// </summary>
    public static string CheckDdmmyyyy(string date)
    {
        string substituted = date.Replace("O", "0").Replace("-", ".");

        var digits = new StringBuilder();
        foreach (char c in substituted)
        {
            if (char.IsAsciiDigit(c))
            {
                digits.Append(c);
            }
        }

        if (digits.Length != 8)
        {
            return substituted;
        }

        string pure = digits.ToString();
        int day = int.Parse(pure[..2], CultureInfo.InvariantCulture);
        int month = int.Parse(pure[2..4], CultureInfo.InvariantCulture);
        int year = int.Parse(pure[4..8], CultureInfo.InvariantCulture);

        // The ORIGINAL, not the substituted string — see the note above.
        return IsValidDate(day, month, year)
            ? $"{day:D2}.{month:D2}.{year:D4}"
            : date;
    }

    private static bool IsValidDate(int day, int month, int year)
    {
        if (month is < 1 or > 12 || day < 1 || year is < 1 or > 9999)
        {
            return false;
        }
        int[] days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        int limit = days[month - 1];
        if (month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))
        {
            limit = 29;
        }
        return day <= limit;
    }

    /// <summary>
    /// Latin sex: anything containing M is male, everything else is female.
    ///
    /// <para>
    /// Note the asymmetry — it is not "F means female". An unreadable crop therefore becomes F rather
    /// than empty, which is the reference's behaviour and not obviously the right one; reproduced
    /// because the goldens encode it.
    /// </para>
    /// </summary>
    public static string CheckEnSex(string sex) =>
        sex.TrimStart('.').ToUpperInvariant().Replace(".", "").Contains('M') ? "M" : "F";

    /// <summary>Cyrillic sex, with the same asymmetry. The М here is CYRILLIC U+041C.</summary>
    public static string CheckRusSex(string sex) =>
        sex.TrimStart('.').ToUpperInvariant().Replace(".", "").Contains('М') ? "М" : "Ж";

    /// <summary>Keeps only the characters a licence class can contain.</summary>
    public static string CheckDriverClass(string driverClass)
    {
        const string allowed = "ABCDEM1";
        var result = new StringBuilder();
        foreach (char c in driverClass.Replace(" ", ""))
        {
            if (allowed.Contains(c))
            {
                result.Append(c);
            }
        }
        return result.ToString();
    }

    /// <summary>
    /// Strips LEADING dots only.
    ///
    /// <para>
    /// <c>lstrip('.')</c>, not <c>strip('.')</c>. Names pick up a spurious leading dot from the crop
    /// edge; a trailing one is rare and the reference leaves it, so trimming both ends would diverge
    /// on exactly the cases where it matters.
    /// </para>
    /// </summary>
    public static string StripEdgeDots(string name) => name.TrimStart('.');
}

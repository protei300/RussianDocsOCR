using System.Text.RegularExpressions;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// The leasing record found in the special marks (<c>sts_marks.parse_leasing</c>). A part that is not
/// printed or not found is null.
/// </summary>
/// <param name="Role">"lessor_named" when the marks name the lessor, "lessee" when they only say the
/// owner is the lessee, null otherwise.</param>
/// <param name="Until">The end of the leasing term («ЛИЗИНГ ДО 31.12.2027»).</param>
public sealed record LeasingRecord(
    string? Role, string? Lessor, string? ContractNumber, string? ContractDate,
    string? ContractDateNormalized, string? Until, string? UntilNormalized)
{
    /// <summary>Always true: a record exists only where the marks say leasing.</summary>
    public bool Leasing => true;
}

/// <summary>
/// Special marks of the vehicle registration certificate (STS): words torn by a line break, and the
/// leasing record. Port of <c>document_processing/pipeline/sts_marks.py</c>; pure functions of strings,
/// like <see cref="Dates"/>.
///
/// <para>
/// The special marks are printed by the registry's printer into a narrow area and wrapped at its edge
/// WITHOUT a hyphen, so a word can be torn in two: «ЛИЗИ» at the end of one line, «НГА» at the start of
/// the next. Measured on 140 real STS backs (external set, 2026-10-07): 4 such tears on known words.
/// The position of the line end does NOT tell a tear from a break between words, so the tear is
/// recognised by vocabulary: the two pieces are glued only when together they make a known word and
/// neither piece is a known word on its own. Unknown words (a company name outside the list, a contract
/// number) stay as read — a wrong glue would merge two real words, which is worse than leaving a torn
/// one.
/// </para>
/// </summary>
public static class StsMarks
{
    /// <summary>
    /// Words that stand as they are (no ending needed). Special-marks terms, the words seen on real backs
    /// (2026-10-07) and lessor names (public leasing companies).
    /// </summary>
    private static readonly string[] Words =
    [
        "ЛИЗИНГОДАТЕЛЬ", "ЛИЗИНГОПОЛУЧАТЕЛЬ", "ЛИЗИНГ", "ДОГОВОР", "СОБСТВЕННИК",
        "ВЛАДЕЛЕЦ", "ДУБЛИКАТ", "ВЗАМЕН", "УВЭОС", "ВЫДАН", "УЧЕТ", "ТЕНТ",
        "ФУРГОН", "РЕФРИЖЕРАТОР", "ОБТЕКАТЕЛЬ", "МОЩНОСТЬ", "ДЕЙСТВИТЕЛЬНО",
        "ОБОРУДОВАНО", "УСТАНОВЛЕНО", "АДРЕС",
        "АВТОЛИЗИНГ", "ИНТЕРЛИЗИНГ", "РОСЛИЗИНГ", "ЕВРОПЛАН", "ГАЗПРОМБАНК",
        "СБЕРБАНК", "СОВКОМБАНК", "КАРКАДЕ", "АЛЬФАМОБИЛЬ", "МЭЙДЖОР", "ЭЛЕМЕНТ",
    ];

    /// <summary>Stems that only occur with an ending: a bare stem is not a word («РЕГИСТРАЦИ» at a line end is a torn piece).</summary>
    private static readonly string[] Stems =
    [
        "ЛИЗИНГОВ", "СОБСТВЕННОСТ", "ВЛАДЕЛЬЦ", "ОБЩЕСТВ", "ОГРАНИЧЕНН",
        "ОТВЕТСТВЕННОСТ", "АКЦИОНЕРН", "ПУБЛИЧН", "КОМПАНИ", "УТРАЧЕНН",
        "РЕГИСТРАЦИ", "ВРЕМЕНН", "ЗАМЕН", "СМЕН", "ИЗМЕНЕНИ", "ПЛАТФОРМ", "ВОРОТ",
        "ПОДРАЗДЕЛЕНИ", "КОНСТРУКЦИ", "БАЛТИЙСК",
    ];

    /// <summary>Noun and adjective endings of the case forms the marks use.</summary>
    private static readonly string[] Endings =
    [
        "А", "Я", "У", "Ю", "Е", "И", "Ы", "О", "Ь", "ОМ", "ЕМ", "ЁМ", "ОВ", "ЕВ", "АМ",
        "ЯМ", "АХ", "ЯХ", "ОЙ", "ЕЙ", "ИЙ", "ЫЙ", "АЯ", "ЯЯ", "ОЕ", "ЕЕ", "ЫЕ", "ИЕ",
        "ИЯ", "ИИ", "ИЮ", "ЬЮ", "ОЮ", "ЕЮ", "ОГО", "ЕГО", "ОМУ", "ЕМУ", "ЫМ", "ИМ",
        "ЫХ", "ИХ", "АМИ", "ЯМИ", "ЫМИ", "ИМИ",
    ];

    private static readonly Regex NotLetters = new("[^А-ЯЁA-Z0-9]", RegexOptions.Compiled);

    private static string Clean(string? word) =>
        NotLetters.Replace((word ?? "").ToUpperInvariant(), "");

    /// <summary>A vocabulary word as is, or a word or stem with a case ending.</summary>
    private static bool Known(string word)
    {
        if (Array.IndexOf(Words, word) >= 0)
        {
            return true;
        }
        foreach (string baseWord in Words.Concat(Stems))
        {
            if (word.StartsWith(baseWord, StringComparison.Ordinal)
                && Array.IndexOf(Endings, word[baseWord.Length..]) >= 0)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Words of a multi-line field, line by line, to words with tears glued. Port of
    /// <c>glue_torn_words</c>.
    ///
    /// <para>
    /// The last word of a line and the first of the next are glued when together they are a known word
    /// and neither alone is: «ЛИЗИ» + «НГА» to «ЛИЗИНГА», while «ЛИЗИНГОДАТЕЛЬ» + «АО» stay two words.
    /// Deliberately NOT glued by the length of the line, although the registry does wrap by character
    /// count: on 140 real backs the rule «a full line goes on into the next one» glued about half of its
    /// cases wrongly («КРОНШ.» + «БАЗЫ», «КВТ» + «Л.С», a date line onto the line above) — the width
    /// differs between documents and the reading of real marks is often noisy.
    /// </para>
    /// </summary>
    public static List<string> GlueTornWords(IReadOnlyList<IReadOnlyList<string>> lines)
    {
        var output = new List<string>();
        foreach (IReadOnlyList<string> line in lines)
        {
            List<string> words = [.. line.Where(w => w.Length > 0)];
            if (words.Count == 0)
            {
                continue;
            }
            if (output.Count > 0)
            {
                string tail = Clean(output[^1]), head = Clean(words[0]);
                if (tail.Length > 0 && head.Length > 0 && Known(tail + head)
                    && !Known(tail) && !Known(head))
                {
                    output[^1] += words[0];
                    words.RemoveAt(0);
                }
            }
            output.AddRange(words);
        }
        return output;
    }

    private const string DatePattern = @"(\d{1,2}[.,]\d{1,2}[.,]\d{4})";

    private static readonly Regex Number = new(@"№\s*([0-9A-ZА-ЯЁ][0-9A-ZА-ЯЁ/\-]*)", RegexOptions.Compiled);

    // «ДОГ ЛИЗ АХ ЭЛ/УЛН-123/ДЛ», «ПО ДОГ ЛИЗИНГА 12/34-СКТ»: the number is the first token with a digit
    // after the abbreviated «договор лизинга», within three tokens.
    private static readonly Regex AbbrNumber =
        new(@"\bДОГ\w*\.?\s+(?:ЛИЗ\w*\.?\s+)?((?:\S+\s+){0,2}?\S*\d\S*)", RegexOptions.Compiled);

    private static readonly Regex Date = new(DatePattern, RegexOptions.Compiled);
    private static readonly Regex ShortLease = new(@"\bЛ\s*\.\s*Д\b", RegexOptions.Compiled);   // the 2010 edition: «<nn> <nnnn> Л.Д <дата>»
    private static readonly Regex LeaseContract = new(@"\bПО\s+ДЛ\b", RegexOptions.Compiled);   // «по договору лизинга»
    private static readonly Regex Until =
        new(@"\bЛИЗИНГ\w*\s+(?:ДЕЙСТВ\w*\s+)?ДО\W{0,2}" + DatePattern, RegexOptions.Compiled);
    private static readonly Regex DateAfterFrom = new(@"\bОТ\s+" + DatePattern, RegexOptions.Compiled);

    // What follows the lessor's name on real marks: the contract, the lessee, the registration term, the
    // validity, the next mark (engine power ...).
    private static readonly Regex LessorEnd = new(
        @"[.,;(]|№|\bДОГ\w*|\bПО\s+ДЛ\b|ЛИЗИНГОПОЛУЧАТЕЛЬ|\bВРЕМ\w*|\bДЕЙСТВ\w*|\bМОЩНОСТ\w*|\bСРОК\w*|\bДО\b",
        RegexOptions.Compiled);

    private static string? Normal(string? dateText) =>
        dateText is null ? null : Dates.ToDdmmyyyy(dateText.Replace(',', '.'));

    /// <summary>
    /// The leasing record in the special marks, or null when there is none. Port of
    /// <c>parse_leasing</c>.
    ///
    /// <para>
    /// The lessee's own name is never taken out: it is the owner, already read. The text is the reading
    /// AFTER <see cref="GlueTornWords"/>: a torn «ЛИЗИ НГ» is not found. Never guesses: no leasing word to
    /// null, and every part is taken only where the marks print it. Real marks are abbreviated freely and
    /// read noisily, so expect the flag far more often than the parts.
    /// </para>
    /// </summary>
    public static LeasingRecord? ParseLeasing(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return null;
        }
        string t = string.Join(" ", text.ToUpperInvariant().Replace('Ё', 'Е')
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        if (!(t.Contains("ЛИЗИНГ", StringComparison.Ordinal) || ShortLease.IsMatch(t)
              || LeaseContract.IsMatch(t)))
        {
            return null;
        }

        string? lessor = null;
        string? role = null;
        if (t.Contains("ЛИЗИНГОДАТЕЛЬ", StringComparison.Ordinal))
        {
            role = "lessor_named";
            string after = t[(t.IndexOf("ЛИЗИНГОДАТЕЛЬ", StringComparison.Ordinal) + "ЛИЗИНГОДАТЕЛЬ".Length)..];
            Match end = LessorEnd.Match(after);
            string piece = (end.Success ? after[..end.Index] : after).Trim(' ', ':', '-');   // quotes stay
            lessor = piece.Length > 0 ? piece : null;
        }
        else if (t.Contains("ЛИЗИНГОПОЛУЧАТЕЛЬ", StringComparison.Ordinal))
        {
            role = "lessee";
        }

        Match number = Number.Match(t);
        if (!number.Success)
        {
            number = AbbrNumber.Match(t);
        }

        string? dateText = null;
        Match from = DateAfterFrom.Match(t);
        if (from.Success)
        {
            dateText = from.Groups[1].Value;
        }
        else
        {
            Match shortLease = ShortLease.Match(t);
            if (shortLease.Success)
            {
                Match m = Date.Match(t, shortLease.Index + shortLease.Length);
                dateText = m.Success ? m.Groups[1].Value : null;
            }
        }
        Match until = Until.Match(t);
        string? untilText = until.Success ? until.Groups[1].Value : null;

        return new LeasingRecord(role, lessor, number.Success ? number.Groups[1].Value.Trim() : null,
            dateText, Normal(dateText), untilText, Normal(untilText));
    }
}

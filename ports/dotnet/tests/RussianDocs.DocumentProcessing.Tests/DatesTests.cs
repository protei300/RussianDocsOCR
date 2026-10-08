using RussianDocs.DocumentProcessing.Pipeline;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// <see cref="Dates"/> — the canonical <c>dd.mm.yyyy</c> view of a date reading. Pins the reference's
/// own worked examples (dates.py:66-71) plus the two "never guess" rules, so a regression here shows
/// up as a failing unit test rather than an unexplained <c>viewmodel.fields[*].normalized</c>
/// divergence three stages later.
/// </summary>
[TestFixture]
public class DatesTests
{
    [TestCase("22.06.2010", "22.06.2010", TestName = "AlreadyCanonical")]
    [TestCase("15 ОКТЯБРЯ 2020 Г.", "15.10.2020", TestName = "WordedMonth_WithTrailingNoiseWord")]
    [TestCase("10 ДЕКАБРЯ 1999 ГОДА", "10.12.1999", TestName = "WordedMonth_GenitiveNoise")]
    [TestCase("03.АВГУСТ.1989", "03.08.1989", TestName = "DotSeparated_NominativeMonth")]
    public void ToDdmmyyyy_MatchesTheReferencesWorkedExamples(string text, string expected)
    {
        Assert.That(Dates.ToDdmmyyyy(text), Is.EqualTo(expected));
    }

    // Record date of the civil-registry entry (Act_date): the 1998 form prints the parts in reverse
    // order with printed words between them and the box spans them; the 2018 form uses the usual
    // order. Mirrors PRINTED in tests/test_date_canon.py.
    [TestCase("2010 ГОДА ИЮНЯ МЕСЯЦА 15 ЧИСЛА", "15.06.2010", TestName = "RecordDate_1998Layout")]
    [TestCase("2010 года июня месяца 15", "15.06.2010", TestName = "RecordDate_NumberWordOutsideBox")]
    [TestCase("2 МАРТА 2025 Г.", "02.03.2025", TestName = "RecordDate_2018Layout")]
    public void ToDdmmyyyy_ReadsTheRecordDate(string text, string expected)
    {
        Assert.That(Dates.ToDdmmyyyy(text), Is.EqualTo(expected));
    }

    // BIRTHCERT_1998 prints «10» ЯНВАРЯ 2013 г.; the box starts on the quote, which the engine reads
    // as the nearest letter (issue #23). A lone letter right next to the day is that quote. Mirrors
    // PRINTED in tests/test_date_canon.py.
    [TestCase("И 10 ЯНВАРЯ 2013", "10.01.2013", TestName = "QuoteLetter_Opening")]
    [TestCase("И10 ЯНВАРЯ 2013", "10.01.2013", TestName = "QuoteLetter_NoSpace")]
    [TestCase("И 10 Н ЯНВАРЯ 2013", "10.01.2013", TestName = "QuoteLetter_BothQuotes")]
    [TestCase("10 П ЯНВАРЯ 2013 Г.", "10.01.2013", TestName = "QuoteLetter_Closing")]
    public void ToDdmmyyyy_DropsAQuoteReadAsALetter(string text, string expected)
    {
        Assert.That(Dates.ToDdmmyyyy(text), Is.EqualTo(expected));
    }

    // A letter NOT next to the day, and a word longer than a letter, are garbage and still refuse.
    // Mirrors REFUSED in tests/test_date_canon.py.
    [TestCase("ИЗ 10 ЯНВАРЯ 2013", TestName = "Refuses_AWordNotALetter")]
    [TestCase("10 ЯНВАРЯ И 2013", TestName = "Refuses_LetterNextToTheYear")]
    [TestCase("И 10 ЯНВАРЯ", TestName = "Refuses_QuoteDroppedButNoYear")]
    public void ToDdmmyyyy_StillRefusesGarbageThatIsNotAQuote(string text)
    {
        Assert.That(Dates.ToDdmmyyyy(text), Is.Null);
    }

    [TestCase("2010 ГОДА ИЮНЯ МЕСЯЦА 31 ЧИСЛА", TestName = "NoSuchDay")]
    [TestCase("2010 ГОДА ИЮНЯ МЕСЯЦА ЧИСЛА", TestName = "NoDay")]
    public void ToDdmmyyyy_RefusesAnImpossibleRecordDate(string text)
    {
        Assert.That(Dates.ToDdmmyyyy(text), Is.Null);
    }

    // Mirrors RECORD_READ in tests/test_date_canon.py.
    [TestCase("2015ГОДАИЮНЯИЕСЯЦА16", "16.06.2015", TestName = "Glued_MisreadMonthsWord")]
    [TestCase("2026ТОДАМАЯНСЯЦА3", "03.05.2026", TestName = "Glued_MisreadYearAndMonthsWords")]
    [TestCase("2002ТОДАИЮНЯ,МЕСЯЦА18", "18.06.2002", TestName = "Glued_WithComma")]
    [TestCase("2003 ДЕКАБРЯ МЕСЯЦА 27", "27.12.2003", TestName = "Spaced")]
    [TestCase("2 МАРТА 2025 Г.", "02.03.2025", TestName = "UsualOrderUsesTheGeneralReading")]
    public void RecordDateToDdmmyyyy_ReadsThroughGlueAndMisreadWords(string text, string expected)
    {
        Assert.That(Dates.RecordDateToDdmmyyyy(text), Is.EqualTo(expected));
    }

    // Mirrors RECORD_REFUSED in tests/test_date_canon.py.
    [TestCase("2010 ГОДА ЦЮЛЯ МЕСЯЦА 17", TestName = "MonthItselfMisread")]
    [TestCase("2020 ГОДА ИЮЛЯ МЕСЯЦА", TestName = "NoDay")]
    [TestCase("2020 ЯНВАРЯ 110201114335", TestName = "RecordNumberInTheBox")]
    [TestCase(".110266032", TestName = "OnlyANumber")]
    [TestCase("2010 ИЮНЯ МАЯ 15", TestName = "TwoMonths")]
    [TestCase("2010ГОДАИЮНЯ 15 16", TestName = "TwoDays")]
    public void RecordDateToDdmmyyyy_StillRefusesRatherThanGuesses(string text)
    {
        Assert.That(Dates.RecordDateToDdmmyyyy(text), Is.Null);
    }

    [Test]
    public void CanonicalDates_OnlyTheRecordDateGetsTheLenientReading()
    {
        var ocr = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Act_date"] = "2015ГОДАИЮНЯИЕСЯЦА16",
            ["Issue_date"] = "2015ГОДАИЮНЯИЕСЯЦА16",
        };

        Dictionary<string, string> canonical = Dates.CanonicalDates(ocr, ["Act_date", "Issue_date"]);

        Assert.Multiple(() =>
        {
            Assert.That(canonical, Has.Count.EqualTo(1));
            Assert.That(canonical["Act_date"], Is.EqualTo("16.06.2015"));
        });
    }

    [Test]
    public void ToDdmmyyyy_NeverGuessesAMissingYear()
    {
        Assert.That(Dates.ToDdmmyyyy("5 МАЯ"), Is.Null);
    }

    [Test]
    public void ToDdmmyyyy_RejectsACalendarImpossibleDate()
    {
        Assert.That(Dates.ToDdmmyyyy("31.02.2020"), Is.Null);
    }

    [Test]
    public void ToDdmmyyyy_RejectsAnAmbiguousTwoDigitYear()
    {
        // '26' could be 1926 or 2026 — the module does not guess.
        Assert.That(Dates.ToDdmmyyyy("5 МАЯ 26"), Is.Null);
    }

    [Test]
    public void ToDdmmyyyy_NullAndEmptyAreNotDates()
    {
        Assert.Multiple(() =>
        {
            Assert.That(Dates.ToDdmmyyyy(null), Is.Null);
            Assert.That(Dates.ToDdmmyyyy(""), Is.Null);
        });
    }

    /// <summary>
    /// Fields are recognised BY NAME — case-insensitive substring "date" — never by content, matching
    /// <c>Pipeline._normalize_dates</c> (pipeline.py:1977-1992: <c>'date' in name.lower()</c>).
    /// </summary>
    [Test]
    public void NormalizeDates_OnlyConvertsFieldsWhoseNameContainsDate()
    {
        var ocr = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Birth_date"] = "15 ОКТЯБРЯ 2020 Г.",
            ["Issue_date"] = "5 МАЯ", // no year: must not appear in the result at all
            ["Last_name_ru"] = "22.06.2010", // looks like a date, but the FIELD is not one
        };

        Dictionary<string, string> normalized =
            Dates.NormalizeDates(ocr, ["Birth_date", "Issue_date", "Last_name_ru"]);

        Assert.Multiple(() =>
        {
            Assert.That(normalized, Has.Count.EqualTo(1));
            Assert.That(normalized["Birth_date"], Is.EqualTo("15.10.2020"));
            Assert.That(normalized, Does.Not.ContainKey("Issue_date"));
            Assert.That(normalized, Does.Not.ContainKey("Last_name_ru"));
        });
    }
}

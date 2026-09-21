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

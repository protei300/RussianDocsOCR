using RussianDocs.DocumentProcessing.Pipeline;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// Per-field verdicts (<see cref="FieldChecks"/>), after <c>tests/test_field_checks.py</c> and the check-digit
/// tests of <c>tests/test_mrz_validation.py</c>.
///
/// <para>
/// Every check has a negative control: a deliberately wrong value must come out as <c>mismatch</c>. A check
/// that cannot show red guards nothing. The zones are the fictitious documents of the reference's tests; the page
/// values next to them are what those zones say, so a consistent document is the starting point and each
/// test changes exactly one thing. Today's date is fixed, the reference's tests do the same.
/// </para>
/// </summary>
[TestFixture]
public class FieldChecksTests
{
    private const string ExtMrz = "P<RUSROGOVA<<IRINA<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
                                  + "7648201572RUS8803148F3105205<<<<<<<<<<<<<<04";

    private const string IntMrz = "PNRUSLE4KIN<<PAVEL<PETROVI3<<<<<<<<<<<<<<<<<\n"
                                  + "4503182940RUS8704234M<<<<<<<7070515770021<46";

    private static readonly Dictionary<string, string> ExtOcr = new()
    {
        ["MRZ"] = ExtMrz,
        ["Licence_number"] = "76 4820154",     // differs from the valid zone in its last digit
        ["Birth_date"] = "14.03.1988",
        ["Expiration_date"] = "20.05.2031",
        ["Last_name_en"] = "ROGOVA",
        ["First_name_en"] = "IRINA",
        ["Sex_en"] = "F",
    };

    private static readonly Dictionary<string, string> ExtOcrConsistent = With(ExtOcr, ("Licence_number", "76 4820157"));

    // What IntMrz says, in the form the page prints it. The number: the zone holds the first three digits
    // of the series and the six of the number; the fourth digit of the series opens the optional data.
    private static readonly Dictionary<string, string> IntOcr = new()
    {
        ["MRZ"] = IntMrz,
        ["Licence_number"] = "45 07 318294",
        ["Birth_date"] = "23.04.1987",
        ["Issue_date"] = "15.05.2007",
        ["Issue_organisation_code"] = "770-021",
        ["Sex_ru"] = "МУЖ.",
        ["Last_name_ru"] = "ЛЕШКИН",
        ["First_name_ru"] = "ПАВЕЛ",
        ["Middle_name_ru"] = "ПЕТРОВИЧ",
        ["Birth_place_ru"] = "ГОР. МОСКВА",
    };

    private static readonly DateOnly Today = new(2026, 10, 19);

    private static Dictionary<string, string> With(Dictionary<string, string> ocr,
        params (string Key, string Value)[] changes)
    {
        var copy = new Dictionary<string, string>(ocr, StringComparer.Ordinal);
        foreach ((string key, string value) in changes)
        {
            copy[key] = value;
        }
        return copy;
    }

    private static Dictionary<string, string> Ocr(params (string Key, string Value)[] items) =>
        With([], items);

    private static Dictionary<string, FieldVerdict> Run(Dictionary<string, string> ocr, string? docType) =>
        FieldChecks.Compute(ocr, docType, Today);

    private static Dictionary<string, string> Statuses(Dictionary<string, string> ocr, string? docType) =>
        Run(ocr, docType).ToDictionary(kv => kv.Key, kv => kv.Value.Status);

    private static FieldCheck CheckOf(FieldVerdict verdict, string id) => verdict.Checks.First(c => c.Id == id);

    private static bool Has(FieldVerdict verdict, string id, string result) =>
        verdict.Checks.Any(c => c.Id == id && c.Result == result);

    // ---- shape ------------------------------------------------------------------------------------

    [Test]
    public void EveryFieldGetsAnEntryEvenEmptyOnes()
    {
        Dictionary<string, string> ocr = With(IntOcr, ("Living_region_ru", ""), ("Address_has_handwritten", "False"));
        Dictionary<string, FieldVerdict> result = Run(ocr, "INTPASSPORT");
        Assert.That(result.Keys, Is.EquivalentTo(ocr.Keys));
        Assert.That(result["Living_region_ru"].Status, Is.EqualTo("unchecked"));
        Assert.That(result["Living_region_ru"].Checks, Is.Empty);
        Assert.That(result["Living_region_ru"].Notes, Is.Empty);
        Assert.That(result["Address_has_handwritten"].Status, Is.EqualTo("unchecked"));
    }

    [Test]
    public void TheReadingIsNeverChanged()
    {
        Dictionary<string, string> ocr = With(IntOcr, ("Licence_number", "45 07 318295"));
        var before = new Dictionary<string, string>(ocr);
        Run(ocr, "INTPASSPORT");
        Assert.That(ocr, Is.EquivalentTo(before));
    }

    [TestCase("INTPASSPORTADDR")]
    [TestCase("DLBACK")]
    [TestCase("BIRTHCERT")]
    [TestCase("STS")]
    [TestCase("NONE")]
    [TestCase(null)]
    public void UnmeasuredFamiliesAreUncheckedThroughout(string? docType)
    {
        Dictionary<string, string> result = Statuses(
            Ocr(("Birth_date", "31.02.2020"), ("MRZ", IntMrz)), docType);
        Assert.That(result.Values, Is.All.EqualTo("unchecked"));
    }

    [Test]
    public void FamilyDoesNotConfuseTheRegistrationPageOrTheLicenceBack()
    {
        Assert.That(FieldChecks.Family("INTPASSPORT"), Is.EqualTo("INTPASSPORT"));
        Assert.That(FieldChecks.Family("INTPASSPORTADDR"), Is.Null);
        Assert.That(FieldChecks.Family("DL"), Is.EqualTo("DL"));
        Assert.That(FieldChecks.Family("DLBACK"), Is.Null);
        Assert.That(FieldChecks.Family("EXTPASSPORTBIO"), Is.EqualTo("EXTPASSPORT"));
        Assert.That(FieldChecks.Family("SNILS"), Is.EqualTo("SNILS"));
    }

    [Test]
    public void TheDefaultTodayComesFromTheClock()
    {
        // A date far in the future is "in the future" whatever the clock says; one in the past is not.
        Dictionary<string, FieldVerdict> result = FieldChecks.Compute(
            Ocr(("Issue_date", "01.01.2099"), ("Birth_date", "01.01.1990")), "SNILS");
        Assert.That(result["Issue_date"].Status, Is.EqualTo("mismatch"));
        Assert.That(result["Birth_date"].Status, Is.EqualTo("unchecked"));
    }

    [Test]
    public void PayloadHasNoReferenceKeyWhenThereIsNoValue()
    {
        Dictionary<string, object> payload = FieldChecks.ToPayload(Run(
            With(IntOcr, ("Licence_number", "45 07 318295")), "INTPASSPORT"));
        var number = (Dictionary<string, object>)payload["Licence_number"];
        var checks = ((List<object>)number["checks"]).Cast<Dictionary<string, object>>().ToList();
        Assert.That(checks.First(c => (string)c["id"] == "PASS-1").ContainsKey("reference"), Is.False);
        Assert.That(checks.First(c => (string)c["id"] == "MRZ-2")["reference"], Is.EqualTo("4507318294"));
        Assert.That(number["status"], Is.EqualTo("mismatch"));
    }

    // ---- internal passport zone -------------------------------------------------------------------

    [Test]
    public void ConsistentInternalPassportIsVerifiedFieldByField()
    {
        Dictionary<string, string> result = Statuses(IntOcr, "INTPASSPORT");
        Assert.That(result["Birth_place_ru"], Is.EqualTo("unchecked"));   // nothing to compare it with
        result.Remove("Birth_place_ru");
        Assert.That(result.Values, Is.All.EqualTo("verified"));
    }

    [Test]
    public void InternalNumberIsRebuiltFromTwoPlacesOfTheZone()
    {
        // The defect this work found: 10 printed digits against the zone's 9.
        FieldVerdict verdict = Run(IntOcr, "INTPASSPORT")["Licence_number"];
        Assert.That(Has(verdict, "MRZ-2", "ok"), Is.True);
        var zone = MrzChecks.Parse(IntMrz, true, Today.Year);
        Assert.That(MrzChecks.PrintedNumber(zone, true), Is.EqualTo("4507318294"));
    }

    // a digit of the number, and the series' fourth digit (lives in the optional data)
    [TestCase("45 07 318295")]
    [TestCase("45 08 318294")]
    public void InternalNumberNegativeControls(string wrong)
    {
        FieldVerdict verdict = Run(With(IntOcr, ("Licence_number", wrong)), "INTPASSPORT")["Licence_number"];
        Assert.That(verdict.Status, Is.EqualTo("mismatch"));
        Assert.That(CheckOf(verdict, "MRZ-2").Reference, Is.EqualTo("4507318294"));
    }

    [Test]
    public void InternalIssueDateAndCodeNegativeControls()
    {
        Dictionary<string, string> result = Statuses(
            With(IntOcr, ("Issue_date", "16.05.2007"), ("Issue_organisation_code", "770-022")), "INTPASSPORT");
        Assert.That(result["Issue_date"], Is.EqualTo("mismatch"));
        Assert.That(result["Issue_organisation_code"], Is.EqualTo("mismatch"));
        FieldVerdict code = Run(With(IntOcr, ("Issue_organisation_code", "770-022")), "INTPASSPORT")
            ["Issue_organisation_code"];
        Assert.That(CheckOf(code, "MRZ-6").Reference, Is.EqualTo("770-021"));
    }

    // Ч -> 3, Ш -> 4 follow the order's table; one changed letter must break it
    [TestCase("Last_name_ru", "ЛЕЩКИН")]
    [TestCase("First_name_ru", "ПАВЕК")]
    [TestCase("Middle_name_ru", "ПЕТРОВИ")]
    public void InternalNamesThroughTheOrder279Table(string name, string wrong)
    {
        Assert.That(Statuses(With(IntOcr, (name, wrong)), "INTPASSPORT")[name], Is.EqualTo("mismatch"));
    }

    [Test]
    public void YoAndYeAreTheSameLetterForTheZone()
    {
        // Whether a clerk typed Ё is not shared between page and zone.
        Dictionary<string, string> ocr = With(IntOcr, ("MRZ", IntMrz.Replace("LE4KIN", "L24KIN")));
        Assert.That(Statuses(ocr, "INTPASSPORT")["Last_name_ru"], Is.EqualTo("verified"));
        Assert.That(Statuses(With(ocr, ("Last_name_ru", "ЛЁШКИН")), "INTPASSPORT")["Last_name_ru"],
            Is.EqualTo("verified"));
    }

    [Test]
    public void SexNegativeControl()
    {
        Assert.That(Statuses(With(IntOcr, ("Sex_ru", "ЖЕН.")), "INTPASSPORT")["Sex_ru"], Is.EqualTo("mismatch"));
    }

    [Test]
    public void AFailedCheckDigitTakesTheZonePartOutOfPlay()
    {
        // A zone part whose own digit fails is not a reference: the page field loses that comparison
        // (unchecked), it does not become a mismatch.
        string zone = IntMrz.Replace("RUS8704234", "RUS8704244");     // birth date, digit kept
        Dictionary<string, FieldVerdict> result = Run(With(IntOcr, ("MRZ", zone)), "INTPASSPORT");
        Assert.That(result["MRZ"].Status, Is.EqualTo("mismatch"));
        Assert.That(result["Birth_date"].Checks.Select(c => c.Id), Does.Not.Contain("MRZ-3"));
        Assert.That(result["Birth_date"].Status, Is.EqualTo("unchecked"));
        Assert.That(result["Licence_number"].Status, Is.EqualTo("verified"));   // other parts still hold
    }

    [Test]
    public void AMalformedZoneIsAMismatchAndNoReference()
    {
        Dictionary<string, FieldVerdict> result = Run(With(IntOcr, ("MRZ", IntMrz[..^1])), "INTPASSPORT");
        Assert.That(result["MRZ"].Status, Is.EqualTo("mismatch"));
        Assert.That(result["Licence_number"].Status, Is.EqualTo("unchecked"));
    }

    [Test]
    public void Line1WithAnImpossibleCharacterIsNoReference()
    {
        // Seen on samples/: the letter O of a surname read as the digit 0 in the zone. The page reading is right;
        // it must not be blamed for the zone's misreading.
        string zone = ExtMrz.Replace("ROGOVA", "R0GOVA");
        Dictionary<string, FieldVerdict> result = Run(With(ExtOcrConsistent, ("MRZ", zone)), "EXTPASSPORTBIO");
        Assert.That(result["MRZ"].Status, Is.EqualTo("mismatch"));
        Assert.That(Has(result["MRZ"], "MRZ-9", "mismatch"), Is.True);
        Assert.That(result["Last_name_en"].Status, Is.EqualTo("unchecked"));
        Assert.That(result["Licence_number"].Status, Is.EqualTo("verified"));   // line 2 still holds
    }

    [TestCase("LE0KIN")]
    [TestCase("LE1KIN")]
    [TestCase("LE5KIN")]
    public void InternalZoneDigitsOutsideTheTableAreImpossible(string wrong)
    {
        Dictionary<string, FieldVerdict> result = Run(
            With(IntOcr, ("MRZ", IntMrz.Replace("LE4KIN", wrong))), "INTPASSPORT");
        Assert.That(result["MRZ"].Status, Is.EqualTo("mismatch"));
        Assert.That(result["Last_name_ru"].Status, Is.EqualTo("unchecked"));
    }

    [Test]
    public void TheZoneIsVerifiedOnlyThroughItsNames()
    {
        // Line-2 check digits say nothing about line 1.
        string[] names = ["Last_name_ru", "First_name_ru", "Middle_name_ru"];
        Dictionary<string, string> noNames = IntOcr.Where(kv => !names.Contains(kv.Key))
            .ToDictionary(kv => kv.Key, kv => kv.Value);
        Assert.That(Statuses(noNames, "INTPASSPORT")["MRZ"], Is.EqualTo("unchecked"));
        Assert.That(Statuses(IntOcr, "INTPASSPORT")["MRZ"], Is.EqualTo("verified"));
        // a page name that disagrees withholds the confirmation, it does not blame the zone
        Dictionary<string, string> result = Statuses(With(IntOcr, ("First_name_ru", "ПАВЕК")), "INTPASSPORT");
        Assert.That(result["MRZ"], Is.EqualTo("unchecked"));
        Assert.That(result["First_name_ru"], Is.EqualTo("mismatch"));
    }

    [Test]
    public void ZoneLayoutFollowsTheClassifierNotTheZoneTypeCharacters()
    {
        // 'PH' also fails the line-1 form (MRZ-9), so names are not compared; the optional data of line 2
        // is still read with the internal layout.
        string zone = "PH" + IntMrz[2..];
        Assert.That(Statuses(With(IntOcr, ("MRZ", zone)), "INTPASSPORT")["Issue_date"], Is.EqualTo("verified"));
    }

    [Test]
    public void DigitsAgreeingDoNotHideStrayCharacters()
    {
        // Damir set: a glare read as quotes inside the number, a doubled hyphen in the code. The zone
        // comparisons look at digits only; the form catches it.
        Dictionary<string, string> result = Statuses(
            With(IntOcr, ("Licence_number", "88 12 106\"436\"\""), ("Issue_organisation_code", "770--021")),
            "INTPASSPORT");
        Assert.That(result["Licence_number"], Is.EqualTo("mismatch"));
        Assert.That(result["Issue_organisation_code"], Is.EqualTo("mismatch"));
        // whitespace is not part of the form
        Assert.That(Statuses(With(IntOcr, ("Licence_number", "45  07 318294")), "INTPASSPORT")["Licence_number"],
            Is.EqualTo("verified"));
    }

    [Test]
    public void ExternalNumberForm()
    {
        Assert.That(Statuses(With(ExtOcrConsistent, ("Licence_number", "764820157")), "EXTPASSPORTBIO")["Licence_number"],
            Is.EqualTo("mismatch"));
    }

    // ---- external passport zone -------------------------------------------------------------------

    [Test]
    public void ConsistentExternalPassport()
    {
        Dictionary<string, string> result = Statuses(With(ExtOcrConsistent, ("Issue_date", "20.05.2021")),
            "EXTPASSPORTBIO");
        Assert.That(result.Values, Is.All.EqualTo("verified"));
    }

    [Test]
    public void ExternalNegativeControls()
    {
        Dictionary<string, string> result = Statuses(With(ExtOcrConsistent,
            ("Expiration_date", "20.05.2032"), ("Last_name_en", "ROGOVA"), ("First_name_en", "IRENA")),
            "EXTPASSPORTBIO");
        Assert.That(result["Expiration_date"], Is.EqualTo("mismatch"));
        Assert.That(result["First_name_en"], Is.EqualTo("mismatch"));
    }

    [Test]
    public void TheSpecimensAlteredNumberIsAMismatch()
    {
        // As on the anonymised samples/: the printed number differs, the zone is valid.
        Assert.That(Statuses(ExtOcr, "EXTPASSPORTBIO")["Licence_number"], Is.EqualTo("mismatch"));
    }

    /// <summary>A TD3 second line with every check digit computed, for building cases.</summary>
    private static string Td3Line2(string number, string birth, string sex, string expiry, string personal = "")
    {
        static string Digit(string s) => MrzChecks.CheckDigit(s)!.Value.ToString();
        number = number.PadRight(9, '<');
        personal = personal.PadRight(14, '<');
        string line = number + Digit(number) + "RUS" + birth + Digit(birth) + sex
                      + expiry + Digit(expiry) + personal + Digit(personal);
        string composite = line[0..10] + line[13..20] + line[21..43];
        return line + Digit(composite);
    }

    [Test]
    public void Td3BuilderReproducesARealZone()
    {
        Assert.That(Td3Line2("764820157", "880314", "F", "310520"), Is.EqualTo(ExtMrz.Split('\n')[1]));
    }

    [Test]
    public void SevenDigitZoneNumberOfThe2003Form()
    {
        // Five of 19 references of the 2003 form write the number without its series.
        string zone = ExtMrz.Split('\n')[0] + "\n" + Td3Line2("4820157", "880314", "F", "310520");
        Dictionary<string, string> ocr = With(ExtOcrConsistent, ("MRZ", zone));
        Dictionary<string, FieldVerdict> result = Run(ocr, "EXTPASSPORT");
        Assert.That(result["MRZ"].Status, Is.EqualTo("verified"));
        Assert.That(Has(result["Licence_number"], "MRZ-2", "ok"), Is.True);
        FieldVerdict bad = Run(With(ocr, ("Licence_number", "76 4820158")), "EXTPASSPORT")["Licence_number"];
        Assert.That(bad.Status, Is.EqualTo("mismatch"));
    }

    [Test]
    public void TruncatedNameComparesByPrefix()
    {
        string longName = string.Concat(Enumerable.Repeat("KONSTANTINOPOLSKAYA", 3));
        string line1 = ("P<RUS" + "ROGOVA<<" + longName)[..44];
        string zone = line1 + "\n" + ExtMrz.Split('\n')[1];
        Dictionary<string, string> ocr = With(ExtOcrConsistent, ("MRZ", zone), ("First_name_en", longName));
        Assert.That(Statuses(ocr, "EXTPASSPORTBIO")["First_name_en"], Is.EqualTo("verified"));
    }

    // ---- dates ------------------------------------------------------------------------------------

    private static readonly Dictionary<string, string> DlOcr = new()
    {
        ["Birth_date"] = "01.02.1990",
        ["Issue_date"] = "15.03.2015",
        ["Expiration_date"] = "15.03.2025",
        ["Licence_number"] = "99 12 345678",
    };

    [Test]
    public void LicenceValidityConfirmsBothDates()
    {
        Dictionary<string, string> result = Statuses(DlOcr, "DL");
        Assert.That(result["Issue_date"], Is.EqualTo("verified"));
        Assert.That(result["Expiration_date"], Is.EqualTo("verified"));
        Assert.That(result["Birth_date"], Is.EqualTo("unchecked"));      // only refuting checks reach it
        Assert.That(result["Licence_number"], Is.EqualTo("unchecked"));
    }

    [Test]
    public void LicenceValidityNegativeControlMarksBothWithReferences()
    {
        Dictionary<string, FieldVerdict> result = Run(With(DlOcr, ("Expiration_date", "16.03.2025")), "DL");
        Assert.That(result["Issue_date"].Status, Is.EqualTo("mismatch"));
        Assert.That(result["Expiration_date"].Status, Is.EqualTo("mismatch"));
        Assert.That(CheckOf(result["Expiration_date"], "DL-2").Reference, Is.EqualTo("15.03.2025"));
        Assert.That(CheckOf(result["Issue_date"], "DL-2").Reference, Is.EqualTo("16.03.2015"));
    }

    [Test]
    public void ExternalValidityAcceptsFiveOrTenYears()
    {
        foreach (int years in new[] { 5, 10 })
        {
            Dictionary<string, string> ocr = Ocr(("Issue_date", "01.06.2010"), ("Expiration_date", $"01.06.{2010 + years}"));
            Assert.That(Statuses(ocr, "EXTPASSPORT")["Issue_date"], Is.EqualTo("verified"));
        }
        Dictionary<string, string> wrong = Ocr(("Issue_date", "01.06.2010"), ("Expiration_date", "01.06.2017"));
        Assert.That(Statuses(wrong, "EXTPASSPORTBIO")["Issue_date"], Is.EqualTo("mismatch"));
        // two candidate dates: no single reference to show
        FieldVerdict verdict = Run(wrong, "EXTPASSPORTBIO")["Issue_date"];
        Assert.That(CheckOf(verdict, "EXT-3").Reference, Is.Null);
    }

    [Test]
    public void LeapDayValidity()
    {
        Dictionary<string, string> ocr = Ocr(("Issue_date", "29.02.2016"), ("Expiration_date", "28.02.2026"));
        Assert.That(Statuses(ocr, "DL")["Expiration_date"], Is.EqualTo("verified"));
        // 01.03 is the other accepted day, and the issue date has no single back-reference
        Dictionary<string, string> march = Ocr(("Issue_date", "29.02.2016"), ("Expiration_date", "01.03.2026"));
        Assert.That(Statuses(march, "DL")["Expiration_date"], Is.EqualTo("verified"));
        FieldVerdict verdict = Run(Ocr(("Issue_date", "29.02.2016"), ("Expiration_date", "27.02.2026")), "DL")
            ["Issue_date"];
        Assert.That(verdict.Status, Is.EqualTo("mismatch"));
    }

    [Test]
    public void NotACalendarDate()
    {
        Assert.That(Statuses(Ocr(("Birth_date", "31.02.1990")), "SNILS")["Birth_date"], Is.EqualTo("mismatch"));
        Assert.That(Statuses(Ocr(("Birth_date", "10 ДЕКАБРЯ 1999 ГОДА")), "SNILS")["Birth_date"],
            Is.EqualTo("unchecked"));
    }

    [Test]
    public void IssueBeforeBirthMarksBoth()
    {
        Dictionary<string, string> result = Statuses(
            Ocr(("Birth_date", "01.01.2000"), ("Issue_date", "01.01.1999")), "SNILS");
        Assert.That(result["Birth_date"], Is.EqualTo("mismatch"));
        Assert.That(result["Issue_date"], Is.EqualTo("mismatch"));
    }

    [Test]
    public void FutureDate()
    {
        Assert.That(Statuses(Ocr(("Issue_date", "01.01.2030")), "SNILS")["Issue_date"], Is.EqualTo("mismatch"));
        // today itself is not the future
        Assert.That(Statuses(Ocr(("Issue_date", "19.10.2026")), "SNILS")["Issue_date"], Is.EqualTo("unchecked"));
    }

    [Test]
    public void MinimumAgeAtIssue()
    {
        Dictionary<string, string> young = Ocr(("Birth_date", "01.01.2000"), ("Issue_date", "31.12.2013"));
        Assert.That(Statuses(young, "INTPASSPORT")["Issue_date"], Is.EqualTo("mismatch"));
        Assert.That(Statuses(With(young, ("Issue_date", "01.01.2014")), "INTPASSPORT")["Issue_date"],
            Is.EqualTo("unchecked"));
        Assert.That(Statuses(Ocr(("Birth_date", "01.01.2000"), ("Issue_date", "01.06.2015")), "DL")["Birth_date"],
            Is.EqualTo("mismatch"));
        // SNILS is issued to children: no age rule
        Assert.That(Statuses(young, "SNILS")["Issue_date"], Is.EqualTo("unchecked"));
    }

    // ---- SNILS ------------------------------------------------------------------------------------

    [Test]
    public void SnilsCheckNumber()
    {
        // the worked example of docs/validation-checks.md
        Assert.That(Statuses(Ocr(("Licence_number", "112-233-445 95")), "SNILS")["Licence_number"],
            Is.EqualTo("verified"));
        FieldVerdict verdict = Run(Ocr(("Licence_number", "112-233-445 96")), "SNILS")["Licence_number"];
        Assert.That(verdict.Status, Is.EqualTo("mismatch"));
        Assert.That(verdict.Checks[^1].Reference, Is.EqualTo("95"));
        // a changed digit in the body
        Assert.That(Statuses(Ocr(("Licence_number", "112-233-446 95")), "SNILS")["Licence_number"],
            Is.EqualTo("mismatch"));
    }

    [Test]
    public void SnilsCheckNumberWrapsMod101()
    {
        // all nines: 9 * (9+8+...+1) = 405 -> 405 mod 101 = 1
        Assert.That(Statuses(Ocr(("Licence_number", "999-999-999 01")), "SNILS")["Licence_number"],
            Is.EqualTo("verified"));
        // 9*9 + 2*8 + 3*1 = 100 -> 00, not 100
        Assert.That(Statuses(Ocr(("Licence_number", "920-000-003 00")), "SNILS")["Licence_number"],
            Is.EqualTo("verified"));
        Assert.That(Statuses(Ocr(("Licence_number", "920-000-003 01")), "SNILS")["Licence_number"],
            Is.EqualTo("mismatch"));
    }

    [Test]
    public void SnilsLengthAndSmallNumbers()
    {
        Assert.That(Statuses(Ocr(("Licence_number", "112-233-445 9")), "SNILS")["Licence_number"],
            Is.EqualTo("mismatch"));
        Assert.That(Statuses(Ocr(("Licence_number", "112-233-445-95")), "SNILS")["Licence_number"],
            Is.EqualTo("mismatch"));
        // up to 001-001-998 there is no valid check number to verify
        Assert.That(Statuses(Ocr(("Licence_number", "001-001-998 00")), "SNILS")["Licence_number"],
            Is.EqualTo("unchecked"));
    }

    // ---- soft -------------------------------------------------------------------------------------

    [Test]
    public void SeriesYearIsANoteNotAVerdict()
    {
        Dictionary<string, string> ocr = With(IntOcr, ("Licence_number", "45 02 318294"));
        Dictionary<string, FieldVerdict> result = Run(ocr, "INTPASSPORT");
        FieldNote note = result["Licence_number"].Notes.Single();
        Assert.That(note.Id, Is.EqualTo("PASS-3"));
        Assert.That(note.Text, Is.EqualTo("series printed in 2002, issued in 2007"));
        Assert.That(result["Issue_date"].Notes.Single().Id, Is.EqualTo("PASS-3"));
        // the note does not change any verdict by itself
        FieldVerdict plain = Run(With(ocr, ("MRZ", "")), "INTPASSPORT")["Licence_number"];
        Assert.That(plain.Status, Is.EqualTo("unchecked"));
        Assert.That(plain.Notes.Single().Id, Is.EqualTo("PASS-3"));
    }

    [Test]
    public void ChecksKeepTheOrderTheyRan()
    {
        // dates, forms, zone, SNILS, note - the order the conformance stage compares
        FieldVerdict verdict = Run(IntOcr, "INTPASSPORT")["Issue_date"];
        Assert.That(verdict.Checks.Select(c => c.Id),
            Is.EqualTo(new[] { "GEN-1", "GEN-2", "GEN-2", "PASS-7", "MRZ-5" }));
    }

    // ---- the results ------------------------------------------------------------------------------

    [Test]
    public void ResultsWithoutReadingHaveNoVerdicts()
    {
        using var results = new Results();
        Assert.That(results.FieldChecks, Is.Empty);
    }

    // ---- the zone arithmetic ----------------------------------------------------------------------

    [Test]
    public void CheckDigitMatchesIcaoWeights()
    {
        // 7-3-1 over "764820157": digits only, so the weights alone decide.
        Assert.That(MrzChecks.CheckDigit("764820157"), Is.EqualTo(2));
        // Letters count as 10..35; fillers as 0.
        Assert.That(MrzChecks.CheckDigit("<<<<<<<<<<<<<<"), Is.EqualTo(0));
        Assert.That(MrzChecks.CheckDigit("AB"), Is.EqualTo(3));          // 10*7 + 11*3 = 103
    }

    [Test]
    public void CheckDigitRejectsCharactersOutsideTheAlphabet()
    {
        Assert.That(MrzChecks.CheckDigit("76482015!"), Is.Null);
    }

    [Test]
    public void ValidateAcceptsTheRealZonesAndRejectsAChangedDigit()
    {
        Assert.That(MrzChecks.Validate(ExtMrz).Ok, Is.True);
        Assert.That(MrzChecks.Validate(IntMrz).Ok, Is.True);
        Assert.That(MrzChecks.Validate(IntMrz).Checks["expiry_date"], Is.EqualTo("not_applicable"));
        Assert.That(MrzChecks.Validate(ExtMrz.Replace("7648201572", "7648201582")).Ok, Is.False);
        Assert.That(MrzChecks.Validate(ExtMrz[..^1]).Ok, Is.False);
        Assert.That(MrzChecks.Validate("P<RUSROGOVA<<IRINA").Ok, Is.False);
    }

    [Test]
    public void BirthCenturyFollowsTheCurrentYearAndExpiryIsAlways20xx()
    {
        Assert.That(MrzChecks.MrzDate("870423", "past", 2026), Is.EqualTo("23.04.1987"));
        Assert.That(MrzChecks.MrzDate("251007", "past", 2026), Is.EqualTo("07.10.2025"));
        Assert.That(MrzChecks.MrzDate("261007", "past", 2026), Is.EqualTo("07.10.2026"));
        Assert.That(MrzChecks.MrzDate("271007", "past", 2026), Is.EqualTo("07.10.1927"));
        Assert.That(MrzChecks.MrzDate("870423", "future", 2026), Is.EqualTo("23.04.2087"));
        Assert.That(MrzChecks.MrzDate("920231", "past", 2026), Is.Null);
        Assert.That(MrzChecks.MrzDate("87042x", "past", 2026), Is.Null);
    }

    [Test]
    public void ParseReadsTheOptionalDataOfTheInternalZone()
    {
        var zone = MrzChecks.Parse(IntMrz, null, 2026);   // the layout from the zone's own type characters
        Assert.That(zone["document_type"], Is.EqualTo("PN"));
        Assert.That(zone["series_last_digit"], Is.EqualTo("7"));
        Assert.That(zone["issue_date"], Is.EqualTo("15.05.2007"));
        Assert.That(zone["subdivision_code"], Is.EqualTo("770021"));
        Assert.That(zone["surname"], Is.EqualTo("LE4KIN"));
        Assert.That(zone["given_names"], Is.EqualTo("PAVEL PETROVI3"));
        Assert.That(MrzChecks.Parse(ExtMrz, null, 2026).ContainsKey("issue_date"), Is.False);
    }

    [Test]
    public void NumbersAgreeComparesTheEndOfASevenDigitZoneNumber()
    {
        Assert.That(MrzChecks.NumbersAgree("4820157", "764820157"), Is.True);
        Assert.That(MrzChecks.NumbersAgree("4820157", "764820158"), Is.False);
        Assert.That(MrzChecks.NumbersAgree("764820157", "764820157"), Is.True);
        Assert.That(MrzChecks.NumbersAgree(null, "764820157"), Is.False);
    }

    [Test]
    public void PythonStyleLineSplittingHandlesEveryTerminator()
    {
        Assert.That(MrzChecks.SplitLines("AB\r\nCD\rEF\u0085GH\n\n  \nIJ"),
            Is.EqualTo(new[] { "AB", "CD", "EF", "GH", "IJ" }));
        Assert.That(MrzChecks.SplitLines(null), Is.Empty);
    }
}

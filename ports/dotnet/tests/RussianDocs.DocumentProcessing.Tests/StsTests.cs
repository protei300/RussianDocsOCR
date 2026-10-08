using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// The vehicle registration certificate (STS, issue #17) that needs no weights: the special marks (torn
/// words, leasing), the options, the VIN fix, the reading rules and the pairing of the two sides. Mirrors
/// tests/test_sts_marks.py, test_sts_options.py, test_sts_reading.py and the pairing tests of
/// tests/test_document_detector.py.
/// </summary>
[TestFixture]
public class StsMarksTests
{
    private static IReadOnlyList<IReadOnlyList<string>> Lines(params string[][] lines) => lines;

    [TestCase(new[] { "ПО", "ДЛ", "ЛИЗИ" }, new[] { "НГА", "ОТ" }, new[] { "ПО", "ДЛ", "ЛИЗИНГА", "ОТ" })]
    [TestCase(new[] { "ЛИЗИН" }, new[] { "ГОДАТЕЛЬ", "АО" }, new[] { "ЛИЗИНГОДАТЕЛЬ", "АО" })]
    [TestCase(new[] { "ЛИЗИНГОВАЯ", "КОМ" }, new[] { "ПАНИЯ" }, new[] { "ЛИЗИНГОВАЯ", "КОМПАНИЯ" })]
    [TestCase(new[] { "ГАЗПРОМБАНК", "АВТОЛ" }, new[] { "ИЗИНГ" }, new[] { "ГАЗПРОМБАНК", "АВТОЛИЗИНГ" })]
    public void AKnownWordTornByTheLineBreak_IsGlued(string[] first, string[] second, string[] glued) =>
        Assert.That(StsMarks.GlueTornWords(Lines(first, second)), Is.EqualTo(glued));

    [TestCase(new[] { "ЛИЗИНГ" }, new[] { "ДОГОВОР" })]                 // two whole words
    [TestCase(new[] { "ЛИЗИНГОДАТЕЛЬ" }, new[] { "АО", "ВТБ" })]        // a whole word, then a name
    [TestCase(new[] { "ООО", "КАРКА" }, new[] { "ПЕТРОВ" })]            // joined is not a known word
    [TestCase(new[] { "№АЛ2682" }, new[] { "26/01-26" })]               // numbers are never glued
    public void WordsThatAreNotATornKnownWord_StayApart(string[] first, string[] second) =>
        Assert.That(StsMarks.GlueTornWords(Lines(first, second)), Is.EqualTo(first.Concat(second).ToArray()));

    [Test]
    public void EmptyLinesAreSkipped() =>
        Assert.That(StsMarks.GlueTornWords(Lines([], ["ЛИЗИ"], [""], ["НГ"])), Is.EqualTo(new[] { "ЛИЗИНГ" }));

    [Test]
    public void TheFullRecord_WithTheLessorOnTheNextLine()
    {
        LeasingRecord? got = StsMarks.ParseLeasing("ПО ДЛ №АЛ268226/01-26 ОТ 12.03.2024 ЛИЗИНГОДАТЕЛЬ АО ВТБ ЛИЗИНГ");
        Assert.That(got, Is.EqualTo(new LeasingRecord("lessor_named", "АО ВТБ ЛИЗИНГ", "АЛ268226/01-26",
            "12.03.2024", "12.03.2024", null, null)));
    }

    [Test]
    public void RealAbbreviations()
    {
        // Forms seen on real backs (2026-10-07), values invented.
        LeasingRecord got = StsMarks.ParseLeasing(
            "ЛИЗИНГОПОЛУЧАТЕЛЬ ИВАНОВ ИВАН ЛИЗИНГОДАТЕЛЬ АО ЛИЗИНГОВАЯ КОМПАНИЯ КАМАЗ ЛИЗИНГ ВРЕМ. УЧЕТ ДО 01.01.2027")!;
        Assert.That(got.Role, Is.EqualTo("lessor_named"));
        Assert.That(got.Lessor, Is.EqualTo("АО ЛИЗИНГОВАЯ КОМПАНИЯ КАМАЗ ЛИЗИНГ"));

        got = StsMarks.ParseLeasing("ЛИЗИНГ ДО 31.12.2027. ДОГ ЛИЗ АХ ЭЛ/УЛН-123456/ДЛ ООО ЭЛЕМЕНТ")!;
        Assert.That(got.UntilNormalized, Is.EqualTo("31.12.2027"));
        Assert.That(got.ContractNumber, Is.EqualTo("АХ ЭЛ/УЛН-123456/ДЛ"));
        Assert.That(got.ContractDate, Is.Null, "«ДО» is the term, not the contract");

        got = StsMarks.ParseLeasing("ЛИЗИНГОДАТЕЛЬ ООО АВТОЛИЗИНГ, ДЕЙСТВИТЕЛЬНО ДО 01.02.2026")!;
        Assert.That(got.Lessor, Is.EqualTo("ООО АВТОЛИЗИНГ"));
    }

    [Test]
    public void TheLessorEndsWhereTheContractBegins()
    {
        LeasingRecord got = StsMarks.ParseLeasing(
            "В ЛИЗИНГЕ ЛИЗИНГОДАТЕЛЬ ООО \"КАРКАДЕ\" ДОГОВОР ЛИЗИНГА №45120 ОТ 05.11.2016")!;
        Assert.That(got.Lessor, Is.EqualTo("ООО \"КАРКАДЕ\""));
        Assert.That(got.ContractNumber, Is.EqualTo("45120"));
        Assert.That(got.ContractDateNormalized, Is.EqualTo("05.11.2016"));
    }

    [Test]
    public void TheOwnerAsLessee_NamesNoLessor()
    {
        LeasingRecord got = StsMarks.ParseLeasing("ЛИЗИНГОПОЛУЧАТЕЛЬ ДОГОВОР №1234-Л ОТ 03.04.2015")!;
        Assert.That(got.Role, Is.EqualTo("lessee"));
        Assert.That(got.Lessor, Is.Null);
        Assert.That(got.ContractNumber, Is.EqualTo("1234-Л"));
    }

    [Test]
    public void The2010ShortForm()
    {
        LeasingRecord got = StsMarks.ParseLeasing("77 1234 Л.Д 14.02.2013")!;
        Assert.That(got.ContractDateNormalized, Is.EqualTo("14.02.2013"));
        Assert.That(got.Role, Is.Null);
        Assert.That(got.Lessor, Is.Null);
    }

    [TestCase("ДУБЛИКАТ")]
    [TestCase("СМЕНА СОБСТВЕННИКА")]
    [TestCase("")]
    [TestCase(null)]
    [TestCase("ЛИЗИ НГ")]   // a tear left unglued is not found
    public void NoLeasingRecord(string? text) => Assert.That(StsMarks.ParseLeasing(text), Is.Null);

    [Test]
    public void TheGlueIsAppliedOnlyToTheFieldsTheOptionsName_AndOnlyWhenTheLineLengthsAddUp()
    {
        OcrOptions sts = OcrOptions.For("STSBACK");
        Assert.Multiple(() =>
        {
            Assert.That(Ocr.GlueTorn("Special_marks", [2, 2], sts, ["ПО", "ЛИЗИ", "НГА", "ОТ"]),
                Is.EqualTo(new[] { "ПО", "ЛИЗИНГА", "ОТ" }));
            Assert.That(Ocr.GlueTorn("Last_name_ru", [1, 1], sts, ["ЛИЗИ", "НГА"]),
                Is.EqualTo(new[] { "ЛИЗИ", "НГА" }));
            // line lengths that do not add up to the words: left alone
            Assert.That(Ocr.GlueTorn("Special_marks", [2, 2], sts, ["ЛИЗИ", "НГА"]),
                Is.EqualTo(new[] { "ЛИЗИ", "НГА" }));
            Assert.That(OcrOptions.For("DL_2011").GlueTorn, Is.Empty);
        });
    }
}

[TestFixture]
public class StsOptionsTests
{
    private static readonly string[] FieldsFront =
    [
        "Reg_number", "VIN", "Vehicle_make_ru", "Vehicle_make_en", "Vehicle_type", "Vehicle_category",
        "Vehicle_year", "Chassis_number", "Body_number", "Vehicle_color", "Engine_power", "Eco_class",
        "Max_mass", "Curb_mass", "Expiration_date", "PTS_number", "Licence_number",
    ];
    private static readonly string[] FieldsBack =
    [
        "Licence_number", "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en", "Middle_name_ru",
        "Living_region_ru", "House_number", "Apartment_number", "Special_marks", "Issue_organisation_code",
        "Issue_date",
    ];
    private static readonly string[] FieldsNewForm = ["Type_approval", "Building_number"];
    private static readonly string[] FieldsOld2010 =
        ["Engine_model", "Engine_number", "Engine_volume", "Issue_organization_ru", "Issue_date", "Building_number"];

    [TestCase("STS_1996")]
    [TestCase("STSBACK_1996")]
    [TestCase("STS_2019")]
    [TestCase("STSBACK_2019")]
    public void BothSidesReachTheSameOptions(string docType) =>
        Assert.That(OcrOptions.For(docType).RuFields, Is.EqualTo(OcrOptions.For("STS").RuFields));

    [TestCase("INTPASSPORT_2011")]
    [TestCase("INTPASSPORTADDR_ALL")]
    [TestCase("EXTPASSPORT_2003")]
    [TestCase("DL_2011")]
    [TestCase("SNILS_1996")]
    [TestCase("BIRTHCERT_2018")]
    public void NoOtherTypeIsCaughtByTheStsBranch(string docType) =>
        Assert.That(OcrOptions.For(docType).ReadMargin, Is.Empty, "only the STS options carry the read margin");

    [Test]
    public void TheLicenceBackHasNoFieldModelAndIsNotTakenForTheFrontSide() =>
        Assert.That(OcrOptions.For("DLBACK_ALL").IsOcrField("Last_name_en"), Is.False);

    [Test]
    public void EveryFieldOfEachSideIsRoutedToAnEngine()
    {
        OcrOptions options = OcrOptions.For("STS");
        foreach (string[] fields in new[] { FieldsFront, FieldsBack, FieldsNewForm, FieldsOld2010 })
        {
            Assert.That(fields.Where(f => !options.IsOcrField(f)), Is.Empty, "detected but never read");
        }
    }

    [Test]
    public void NoFieldIsRoutedToBothEngines() =>
        Assert.That(OcrOptions.For("STS").RuFields.Intersect(OcrOptions.For("STS").EnFields), Is.Empty);

    [Test]
    public void RegNumberAndVin_AreLatin_AndTheSeriesKeepsThePassportPrecedent()
    {
        OcrOptions options = OcrOptions.For("STS");
        Assert.Multiple(() =>
        {
            Assert.That(options.EnFields, Does.Contain("Reg_number").And.Contain("VIN"));
            Assert.That(options.RuFields, Does.Contain("Licence_number"));
        });
    }

    [Test]
    public void ThereIsNoModelField()
    {
        OcrOptions options = OcrOptions.For("STS");
        Assert.That(options.RuFields.Concat(options.EnFields).Concat(options.NeededSplit),
            Does.Not.Contain("Vehicle_model_en"));
    }

    [TestCase("WFODXXGAJD1A00001", "WF0DXXGAJD1A00001")]   // the issue #17 case (a made-up VIN)
    [TestCase("XTAOOOOOOOO123456", "XTA00000000123456")]
    [TestCase("WF0DXXGAJD1A00001", "WF0DXXGAJD1A00001")]   // already right: untouched
    public void VinLetterO_BecomesZero(string read, string expected) =>
        Assert.That(OcrCorrections.CheckVin(read), Is.EqualTo(expected));

    [Test]
    public void TheMakeEngineFollowsTheStsForm()
    {
        OcrOptions sts = OcrOptions.For("STS");
        Assert.Multiple(() =>
        {
            Assert.That(sts.RuFields, Does.Contain("Vehicle_make_ru"), "the old form's route");
            Assert.That(sts.EngineForYear("Vehicle_make_ru", "2019"), Is.EqualTo("lat"), "the new form prints the make in Latin");
            Assert.That(sts.EngineForYear("Vehicle_make_ru", "1996"), Is.Null, "the old form keeps the Cyrillic route");
            Assert.That(sts.EngineForYear("Vehicle_make_ru", null), Is.Null, "no year, no override");
            Assert.That(sts.EngineForYear("Vehicle_color", "2019"), Is.Null);
            Assert.That(OcrOptions.For("DL_2011").EngineByYear, Is.Empty);
        });
    }

    // ---- the read margin (Pipeline._read_margins) ------------------------------------------------

    private static Box B(int x1, int y1, int x2, int y2, string label) =>
        new() { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Conf = 0.9, Cls = 0, Label = label };

    private static OcrOptions Margin(double share) =>
        new() { ReadMargin = new Dictionary<string, double> { ["Special_marks"] = share } };

    [Test]
    public void TheReadCropGrows_AndTheBoxDoesNot()
    {
        Box box = B(20, 100, 280, 120, "Special_marks");
        var rows = Recognizer.ReadMarginRows([box], Margin(0.25), 200);
        Assert.That(rows[0], Is.EqualTo((95, 125)), "20 px + 5 above + 5 below");
        Assert.That((box.Y1, box.Y2), Is.EqualTo((100.0, 120.0)));
    }

    [Test]
    public void TheMarginStopsHalfwayToTheNextLine()
    {
        var rows = Recognizer.ReadMarginRows(
            [B(20, 80, 280, 96, "Special_marks"), B(20, 100, 280, 116, "Special_marks")], Margin(0.5), 200);
        // the upper crop ends where the lower begins: rows 98.. belong to the lower line
        Assert.That(rows[0]!.Value.Bottom, Is.EqualTo(98));
        Assert.That(rows[1]!.Value.Top, Is.EqualTo(98));
    }

    [Test]
    public void ABoxBesideItDoesNotLimitTheMargin()
    {
        var rows = Recognizer.ReadMarginRows(
            [B(20, 100, 140, 120, "Special_marks"), B(160, 80, 280, 98, "Reg_number")], Margin(0.25), 200);
        Assert.That(rows[0], Is.EqualTo((95, 125)));
    }

    [Test]
    public void OtherFieldsAndOtherTypesAreUntouched()
    {
        Assert.That(Recognizer.ReadMarginRows([B(20, 100, 280, 120, "Vehicle_color")], Margin(0.25), 200)[0],
            Is.Null);
        Assert.That(Recognizer.ReadMarginRows([B(20, 100, 280, 120, "Special_marks")], new OcrOptions(), 200)[0],
            Is.Null);
    }

    [Test]
    public void TheMarginStaysInsideTheCanvas() =>
        Assert.That(Recognizer.ReadMarginRows([B(20, 4, 280, 24, "Special_marks")], Margin(0.5), 30)[0],
            Is.EqualTo((0, 30)));
}

[TestFixture]
public class PairSidesTests
{
    [Test]
    public void FrontAndBack_WithOneNumber_PairUp()
    {
        int?[] got = Recognizer.PairSides([("STS_2019", "99 87 786940"), ("DL_2020", "99 12 345678"),
            ("STSBACK_2019", "9987 786940")]);
        Assert.That(got, Is.EqualTo(new int?[] { 2, null, 0 }));
    }

    [Test]
    public void TwoCertificatesOnOneSheet_DoNotCrossPair()
    {
        int?[] got = Recognizer.PairSides([("STS_2019", "99 87 786940"), ("STSBACK_2019", "99 80 895276"),
            ("STS_2019", "99 80 895276"), ("STSBACK_2019", "99 87 786940")]);
        Assert.That(got, Is.EqualTo(new int?[] { 3, 2, 1, 0 }));
    }

    [Test]
    public void AnAmbiguousOrUnreadNumber_PairsNothing()
    {
        int?[] got = Recognizer.PairSides([("STS_2019", "99 87 786940"), ("STS_2019", "99 87 786940"),
            ("STSBACK_2019", "99 87 786940"), ("STSBACK_1996", null)]);
        Assert.That(got, Is.EqualTo(new int?[] { null, null, null, null }));
    }

    [Test]
    public void ANumberShorterThanSixDigits_PairsNothing()
    {
        int?[] got = Recognizer.PairSides([("STS_2019", "12345"), ("STSBACK_2019", "12345")]);
        Assert.That(got, Is.EqualTo(new int?[] { null, null }));
    }
}

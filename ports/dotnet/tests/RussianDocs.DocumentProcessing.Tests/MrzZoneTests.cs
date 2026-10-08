using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// The machine-readable zone's length self-check, the part that needs no model: trimming the edge
/// characters outside the zone's closed alphabet, and building the zone from the detections.
/// </summary>
[TestFixture]
public class MrzZoneTests
{
    [TestCase("P<RUSIVANOV<<IVAN<<<", "P<RUSIVANOV<<IVAN<<<", TestName = "Trim_CleanLineUntouched")]
    [TestCase(".P<RUSIVANOV<<IVAN<<<_", "P<RUSIVANOV<<IVAN<<<", TestName = "Trim_PageBorderAtBothEdges")]
    [TestCase("..._P<RUS", "P<RUS", TestName = "Trim_SeveralEdgeCharacters")]
    [TestCase("P<RUS.IVANOV", "P<RUS.IVANOV", TestName = "Trim_InsideTheLineIsLeftAlone")]
    [TestCase("", "", TestName = "Trim_Empty")]
    [TestCase("...", "", TestName = "Trim_OnlyJunk")]
    public void TrimToAlphabet_StripsOnlyTheEdges(string text, string expected)
    {
        Assert.That(MrzZone.TrimToAlphabet(text), Is.EqualTo(expected));
    }

    private static Box Mrz(double x1, double y1, double x2, double y2)
        => new() { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Conf = 0.9, Label = "MRZ" };

    [Test]
    public void Note_NoMrzBoxes_NoZone()
    {
        using Image canvas = Io.NewFilled(100, 100, 255, 255, 255);
        Box other = new() { X1 = 0, Y1 = 0, X2 = 10, Y2 = 10, Label = "Last_name_en" };
        Assert.That(MrzZone.Note([other], canvas), Is.Null);
    }

    [Test]
    public void Note_WithMrzBoxes_BuildsAZone()
    {
        using Image canvas = Io.NewFilled(100, 1000, 255, 255, 255);
        Assert.That(MrzZone.Note([Mrz(10, 70, 900, 90), Mrz(20, 40, 910, 60)], canvas), Is.Not.Null);
    }

    [Test]
    public void Read_ALineOfTheRightLengthIsKeptWithoutAnyRetry()
    {
        // No engine is needed (null is never touched): a 44-character line returns before the ladder.
        string line = "P<RUSIVANOV<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<<<<";
        Assert.That(line.Length, Is.EqualTo(MrzZone.LineLength));
        Assert.That(MrzZone.Read(null, null!, 0, line), Is.EqualTo(line));
        Assert.That(MrzZone.Read(null, null!, 0, "." + line + "_"), Is.EqualTo(line));
    }

    [Test]
    public void Read_WithNoZone_ReturnsTheTrimmedShortLine()
    {
        Assert.That(MrzZone.Read(null, null!, 0, ".P<RUS<<"), Is.EqualTo("P<RUS<<"));
    }
}

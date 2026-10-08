using OpenCvSharp;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// The signal behind the gap guard on the word split, and the no-ink measure that stops it from
/// re-reading an empty strip. Both are pure functions, tested without a model. Mirrors
/// tests/test_words_gap.py and tests/test_words_ink.py.
///
/// <para>
/// The signal is a ratio to the line's own median word width, not a share of the line: an internal
/// passport's Licence_number is three digit groups with wide gaps and is read CORRECTLY, and any
/// absolute measure lumps it together with real damage.
/// </para>
/// </summary>
[TestFixture]
public class WordsGapTests
{
    private static Box B(double x1, double x2)
        => new() { X1 = x1, Y1 = 0, X2 = x2, Y2 = 10, Conf = 0.9, Label = "word" };

    [TestCaseSource(nameof(Cases))]
    public void Gap_IsMeasuredInTypicalWordWidths(Box[] boxes, double width, double expected)
    {
        Assert.That(SplitWords.WidestGap([.. boxes], width), Is.EqualTo(expected).Within(1e-9));
    }

    private static IEnumerable<TestCaseData> Cases()
    {
        yield return new TestCaseData(new[] { B(0, 100) }, 100.0, 0.0).SetName("OneWordFillsTheLine");
        yield return new TestCaseData(new[] { B(0, 45), B(55, 100) }, 100.0, 10.0 / 45).SetName("NormalSpace");
        yield return new TestCaseData(new[] { B(0, 20), B(80, 100) }, 100.0, 3.0).SetName("HoleThreeWordsWide");
        yield return new TestCaseData(new[] { B(0, 25), B(40, 65), B(80, 100) }, 100.0, 0.6).SetName("EvenlySpacedNumber");
        yield return new TestCaseData(new[] { B(40, 60) }, 100.0, 2.0).SetName("MissingFromBothEdges");
        yield return new TestCaseData(new[] { B(0, 20) }, 100.0, 4.0).SetName("EverythingAfterTheFirstLost");
        yield return new TestCaseData(new[] { B(80, 100) }, 100.0, 4.0).SetName("EverythingBeforeTheLastLost");
    }

    [Test]
    public void NothingFound_IsOneWholeHole()
    {
        // No boxes means no denominator, and the line is entirely missing: the measured case where a
        // field used to vanish silently must read as "worst possible", not as zero.
        Assert.Multiple(() =>
        {
            Assert.That(SplitWords.WidestGap([], 100), Is.EqualTo(double.PositiveInfinity));
            Assert.That(SplitWords.WidestGap(null, 100), Is.EqualTo(double.PositiveInfinity));
        });
    }

    [Test]
    public void WideButEvenSpacing_IsNotAHole()
    {
        // Both lines leave the same total emptiness; only the second one is missing something.
        double even = SplitWords.WidestGap([B(0, 10), B(30, 40), B(60, 70), B(90, 100)], 100);
        double holed = SplitWords.WidestGap([B(0, 10), B(12, 22), B(24, 34), B(90, 100)], 100);
        Assert.That(even, Is.LessThan(holed));
    }

    [Test]
    public void TheThresholdsAreTheReferences()
    {
        Assert.Multiple(() =>
        {
            Assert.That(SplitWords.WordsMaxGap, Is.EqualTo(3.0));
            Assert.That(SplitWords.LineMinInk, Is.EqualTo(100.0));
        });
    }

    [Test]
    public void Ink_ABlankStripHasNone_AStrokedStripHasPlenty()
    {
        using Image blank = Io.NewFilled(32, 200, 235, 235, 235);
        using Image stroked = Io.NewFilled(32, 200, 235, 235, 235);
        // Black vertical strokes, 2 px wide, every 6 px: what printed text looks like to a Laplacian.
        for (int x = 4; x < 196; x += 6)
        {
            Cv2.Rectangle(stroked.Mat, new Rect(x, 6, 2, 20), new Scalar(0, 0, 0), -1);
        }

        double blankInk = Ink.LaplacianVariance(blank);
        double strokedInk = Ink.LaplacianVariance(stroked);

        Assert.Multiple(() =>
        {
            Assert.That(blankInk, Is.LessThan(SplitWords.LineMinInk));
            Assert.That(strokedInk, Is.GreaterThan(SplitWords.LineMinInk));
        });
    }

    [Test]
    public void Ink_AnEmptyCropHasNone()
    {
        using Image empty = Image.Wrap(new Mat(0, 0, MatType.CV_8UC3));
        Assert.That(Ink.LaplacianVariance(empty), Is.EqualTo(0.0));
    }
}

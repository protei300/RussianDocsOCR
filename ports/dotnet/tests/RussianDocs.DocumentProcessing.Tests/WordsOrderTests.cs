using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// Word ordering and the word-crop margin — the two rules that decide which pixels reach the OCR
/// engine and in which order. Both are pure functions of the boxes, so they are pinned here rather
/// than only through a conformance run that needs models.
/// </summary>
[TestFixture]
public class WordsOrderTests
{
    private static Box B(double x1, double y1, double x2, double y2)
        => new() { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Label = "Word" };

    private static (double, double, double, double)[] Quads(List<Box> boxes)
        => [.. boxes.Select(b => (b.X1, b.Y1, b.X2, b.Y2))];

    /// <summary>
    /// A two-line field must be read line by line. A plain x-sort interleaves the lines, which the
    /// reference measured as word salad on the birth certificates' Birth_place and ZAGS fields.
    /// The expected order is what <c>WordsDetector._reading_order</c> returns for these boxes.
    /// </summary>
    [Test]
    public void ReadingOrder_KeepsLinesTogether()
    {
        List<Box> input =
        [
            B(10, 0, 60, 18),   // line 1, word 1
            B(70, 1, 130, 19),  // line 1, word 2
            B(140, 0, 200, 18), // line 1, word 3
            B(5, 22, 55, 40),   // line 2, word 1
            B(65, 23, 190, 41), // line 2, word 2
        ];

        Assert.That(Quads(WordsDetector.ReadingOrder(input)), Is.EqualTo(new[]
        {
            (10.0, 0.0, 60.0, 18.0), (70.0, 1.0, 130.0, 19.0), (140.0, 0.0, 200.0, 18.0),
            (5.0, 22.0, 55.0, 40.0), (65.0, 23.0, 190.0, 41.0),
        }));

        // And the naive sort really does disagree — a test that cannot fail proves nothing.
        Assert.That(Quads([.. input.OrderBy(b => b.X1)])[1], Is.EqualTo((10.0, 0.0, 60.0, 18.0)),
            "x1-sorted, the second line's first word lands between two words of the first line");
    }

    /// <summary>A single-line field comes out exactly as the old x1 sort produced it.</summary>
    [Test]
    public void ReadingOrder_IsAnX1SortOnOneLine()
    {
        List<Box> input = [B(140, 0, 200, 18), B(10, 2, 60, 20), B(70, 1, 130, 19)];

        Assert.That(Quads(WordsDetector.ReadingOrder(input)), Is.EqualTo(new[]
        {
            (10.0, 2.0, 60.0, 20.0), (70.0, 1.0, 130.0, 19.0), (140.0, 0.0, 200.0, 18.0),
        }));
    }
}

using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// Documents first (decision #142): grouping the detector's boxes and cutting the crop. Pure
/// functions of boxes, so they are pinned here instead of only through a conformance run that needs
/// the weights. Mirrors tests/test_document_detector.py.
/// </summary>
[TestFixture]
public class DocumentsFirstTests
{
    private static Box B(double x1, double y1, double x2, double y2, double conf, string label)
        => new() { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Conf = conf, Cls = label == "page" ? 1 : 0, Label = label };

    [Test]
    public void PagesBelongToTheDocumentTheyLieIn_AndDocumentsGoLargestFirst()
    {
        List<Box> boxes =
        [
            B(500, 50, 700, 200, 0.9, "document"),     // a small card
            B(0, 0, 400, 600, 0.95, "document"),       // a passport spread
            B(10, 310, 390, 590, 0.9, "page"),         // its lower page
            B(10, 10, 390, 290, 0.9, "page"),          // its upper page
            B(900, 900, 950, 950, 0.5, "page"),        // a page in no document
        ];

        List<DetectedDocument> docs = DocumentDetector.GroupDocuments(boxes);

        Assert.Multiple(() =>
        {
            Assert.That(docs.Select(d => d.Box), Is.EqualTo(new[]
            {
                new double[] { 0, 0, 400, 600 }, new double[] { 500, 50, 700, 200 },
            }));
            Assert.That(docs[0].Pages.Select(p => p.Box), Is.EqualTo(new[]
            {
                new double[] { 10, 10, 390, 290 }, new double[] { 10, 310, 390, 590 },
            }), "top to bottom");
            Assert.That(docs[1].Pages, Is.Empty);
        });
    }

    [Test]
    public void NoDocumentFound_NoDocuments()
    {
        Assert.That(DocumentDetector.GroupDocuments([]), Is.Empty);
        Assert.That(DocumentDetector.GroupDocuments([B(0, 0, 10, 10, 0.9, "page")]), Is.Empty);
    }

    [Test]
    public void TwoDocumentsOfEqualArea_KeepTheDetectorsOrder()
    {
        // Python's sort(reverse=True) is stable; so is OrderByDescending.
        List<DetectedDocument> docs = DocumentDetector.GroupDocuments(
        [
            B(0, 0, 100, 100, 0.9, "document"),
            B(200, 0, 300, 100, 0.8, "document"),
        ]);
        Assert.That(docs.Select(d => d.Conf), Is.EqualTo(new[] { 0.9, 0.8 }));
    }

    [Test]
    public void TheCrop_IsTheBoxPlusThreePercentOfItsLongerSide()
    {
        // 600 x 400 box: the margin is 18 px on every side.
        Assert.That(Recognizer.DocumentCrop(2000, 1500, [100, 100, 700, 500]),
            Is.EqualTo((82, 82, 718, 518)));
    }

    [Test]
    public void TheCrop_FloorsTheStartCeilsTheEnd_AndStaysInTheFrame()
    {
        // 10.4 - 3 = 7.4 -> 7 ; 110.6 + 3 = 113.6 -> 114 ; and never outside the frame.
        Assert.That(Recognizer.DocumentCrop(1000, 1000, [10.4, 10.4, 110.6, 110.6]),
            Is.EqualTo((7, 7, 114, 114)));
        Assert.That(Recognizer.DocumentCrop(110, 105, [0, 0, 110, 105]),
            Is.EqualTo((0, 0, 110, 105)));
    }
}

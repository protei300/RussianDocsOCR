using OpenCvSharp;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Maps;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.Postprocess;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// The way back from the canvas to the input photo (PR #19, <c>geometry.py</c>): the maps themselves and the
/// quadrilaterals built from them. No models: the pieces are checked against what OpenCV really does to a
/// marked pixel. Mirrors the part of tests/test_pipeline_geometry.py that does not need the detectors.
/// </summary>
[TestFixture]
public class MapsTests
{
    private static void AssertPoint(Point? actual, double x, double y, double tolerance = 1e-9)
    {
        Assert.That(actual, Is.Not.Null);
        Assert.That(actual!.Value.X, Is.EqualTo(x).Within(tolerance));
        Assert.That(actual.Value.Y, Is.EqualTo(y).Within(tolerance));
    }

    [Test]
    public void Scale_AndOffset_UndoTheirOwnStage()
    {
        AssertPoint(new ScaleMap(0.5, 0.25).ToInput([new(20, 30)])![0], 40, 120);
        AssertPoint(new OffsetMap(-100, -200).ToInput([new(5, 6)])![0], 105, 206);
    }

    [TestCase(0)]
    [TestCase(1)]
    [TestCase(2)]
    [TestCase(3)]
    [TestCase(5)]
    public void QuarterTurns_SendEveryPixelCentreBackWhereOpenCvTookItFrom(int turns)
    {
        const int w = 5, h = 3;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                using var src = new Mat(h, w, MatType.CV_8UC1, Scalar.All(0));
                src.Set(y, x, (byte)255);
                Mat current = src.Clone();
                for (int t = 0; t < turns % 4; t++)
                {
                    var next = new Mat();
                    Cv2.Rotate(current, next, RotateFlags.Rotate90Counterclockwise);
                    current.Dispose();
                    current = next;
                }
                Cv2.MinMaxLoc(current, out _, out _, out _, out OpenCvSharp.Point at);
                (int rw, int rh) = (current.Cols, current.Rows);
                current.Dispose();
                Assert.That(at.X < rw && at.Y < rh);

                Point[]? back = new QuarterTurnsMap(w, h, turns).ToInput([new(at.X + 0.5, at.Y + 0.5)]);
                AssertPoint(back![0], x + 0.5, y + 0.5);
            }
        }
    }

    [Test]
    public void Homography_UsesTheHalfPixelConventionOfTheWarps()
    {
        // warpAffine with a 2x enlargement samples the input at (i / 2, j / 2) in OpenCV's integer
        // pixel coordinates; in continuous coordinates the centre (10.5, 6.5) of output pixel (10, 6) is the
        // centre (5.5, 3.5) of input pixel (5, 3), not (5.25, 3.25).
        var map = new HomographyMap(new double[,] { { 2, 0, 0 }, { 0, 2, 0 } });
        AssertPoint(map.ToInput([new(10.5, 6.5)])![0], 5.5, 3.5);
    }

    [Test]
    public void Homography_InvertsARealPerspectiveWarp()
    {
        var quad = new Point2f[] { new(40, 30), new(380, 50), new(360, 260), new(20, 240) };
        var dst = new Point2f[] { new(0, 0), new(299, 0), new(299, 199), new(0, 199) };
        using Mat m = Cv2.GetPerspectiveTransform(quad, dst);
        var matrix = new double[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                matrix[i, j] = m.At<double>(i, j);
            }
        }
        // The pixel centre (0.5, 0.5) of the warped page is the pixel at integer position (0, 0), which is
        // the first corner of the quad; the far corner is the centre of pixel (299, 199).
        Point[]? back = new HomographyMap(matrix).ToInput([new(0.5, 0.5), new(299.5, 199.5)]);
        AssertPoint(back![0], 40.5, 30.5, 1e-6);
        AssertPoint(back[1], 360.5, 260.5, 1e-6);
    }

    [Test]
    public void ASingularHomography_HasNoWayBack() =>
        Assert.That(new HomographyMap(new double[,] { { 1, 0, 0 }, { 0, 0, 0 }, { 0, 0, 1 } }).ToInput([new(1, 1)]),
            Is.Null);

    [Test]
    public void AChain_IsReadFromTheOutputToTheInput()
    {
        // stages in the order they ran: shrink to half, then cut 10 px off the left and 20 off the top
        var chain = new ChainMap([new ScaleMap(0.5, 0.5), new OffsetMap(-10, -20)]);
        AssertPoint(chain.ToInput([new(100, 100)])![0], (100 + 10) / 0.5, (100 + 20) / 0.5);

        // the same maps in the order a map built "forward" would list them send the point elsewhere
        var wrong = new ChainMap([new OffsetMap(-10, -20), new ScaleMap(0.5, 0.5)]);
        Assert.That(wrong.ToInput([new(100, 100)])![0].X, Is.Not.EqualTo(220.0));
    }

    [Test]
    public void AStageThatPassesItsImageOn_AddsNothing_AndAStageWithNoWayBack_EndsTheWalk()
    {
        var chain = new ChainMap();
        Assert.That(chain.Then(null), Is.SameAs(chain));

        var broken = chain.Then(new ScaleMap(1, 1)).Then(new UnknownMap()).Then(new OffsetMap(-1, -1));
        Assert.That(broken.ToInput([new(3, 3)]), Is.Null);
    }

    [Test]
    public void APiecesMap_PicksThePieceByTheCentroidOfTheShape()
    {
        var pieces = new PiecesMap(
        [
            ((0.0, 0.0, 100.0, 50.0), new OffsetMap(-1000, 0)),
            ((0.0, 50.0, 100.0, 100.0), new OffsetMap(-2000, 0)),
        ]);
        // a box whose corners straddle the seam belongs to the page its centre is on, all four corners
        Point[]? onFirst = pieces.ToInput([new(10, 40), new(30, 40), new(30, 58), new(10, 58)]);   // centre y = 49
        Assert.That(onFirst!.Select(p => p.X), Is.EqualTo(new[] { 1010.0, 1030.0, 1030.0, 1010.0 }));
        Point[]? onSecond = pieces.ToInput([new(10, 44), new(30, 44), new(30, 58), new(10, 58)]);  // centre y = 51
        Assert.That(onSecond!.Select(p => p.X), Is.EqualTo(new[] { 2010.0, 2030.0, 2030.0, 2010.0 }));
    }

    [Test]
    public void APiecesMap_WithAnUnknownPiece_HasNoWayBackForThatPiece()
    {
        var pieces = new PiecesMap([((0.0, 0.0, 10.0, 10.0), new UnknownMap()), ((10.0, 0.0, 20.0, 10.0), new OffsetMap(0, 0))]);
        Assert.That(pieces.ToInput([new(5, 5)]), Is.Null);
        Assert.That(pieces.ToInput([new(15, 5)]), Is.Not.Null);
    }

    [Test]
    public void TheBendMap_IsTheWayBackItself_ReadBilinearly_AndHeldAtTheEdge()
    {
        // 3 x 2 map, v(x, y) = 1 + x + 10 * y at the pixel centres
        float[] v = [1, 2, 3, 11, 12, 13];
        var map = new BendMap(v, 3, 2);
        // at pixel centre (1, 0): v = 2
        AssertPoint(map.ToInput([new(1.5, 0.5)])![0], 1.5, 0.5 + 2);
        // halfway between the centres of pixels (0, 0) and (1, 0): v = 1.5; x is unchanged
        AssertPoint(map.ToInput([new(1.0, 0.5)])![0], 1.0, 0.5 + 1.5);
        // outside the map the edge value is held
        AssertPoint(map.ToInput([new(-4, 9)])![0], -4, 9 + 11);
    }

    [Test]
    public void Placements_ReproduceTheStitch_AndTheStitchedMapSendsEachPageBackThroughItsOwnMap()
    {
        // two pages stacked vertically: 100x50 and 80x40; the common width is the smaller, 80
        (int W, int H)[] sizes = [(100, 50), (80, 40)];
        (Placement[] placements, int[] newW, int[] newH) = Stitched.Place(sizes, [0, 1], horizontal: false);
        Assert.Multiple(() =>
        {
            Assert.That(newW, Is.EqualTo(new[] { 80, 80 }));
            Assert.That(newH, Is.EqualTo(new[] { 40, 40 }));
            Assert.That(placements[0], Is.EqualTo(new Placement(0.8, 0, 0)));
            Assert.That(placements[1], Is.EqualTo(new Placement(1.0, 0, 40)));
        });

        var first = new OffsetMap(-1000, -2000);    // page 0 was cut from the photo at (1000, 2000)
        var second = new OffsetMap(-5000, -6000);
        PiecesMap canvas = Stitched.Geometry(sizes, placements, [first, second]);
        // (40, 20) on the canvas is the middle of the resized first page: (50, 25) on the page itself
        AssertPoint(canvas.ToInput([new(40, 20)])![0], 1050, 2025);
        // (40, 60) is 20 below the top of the second page
        AssertPoint(canvas.ToInput([new(40, 60)])![0], 5040, 6020);
    }

    // ---- the quadrilaterals ---------------------------------------------------------------------

    private static Field FieldAt(string label, int x, int y, int cutW, int cutH, int patchW, int patchH)
    {
        var box = new Box { X1 = x, Y1 = y, X2 = x + cutW, Y2 = y + cutH, Conf = 0.9, Label = label };
        return new Field(box, Image.Wrap(new Mat(patchH, patchW, MatType.CV_8UC3, Scalar.All(0)))).WithCut(cutW, cutH);
    }

    private static SplitOutcome Split(params SplitDetection[] detections) => new()
    {
        Fields = [], Flags = new SplitFlags(), DateLines = [], Detections = [.. detections],
    };

    [Test]
    public void AFieldQuad_IsItsCutOnTheCanvas_ThenTheWayBackOfTheCanvas()
    {
        using Field field = FieldAt("Last_name_ru", 10, 20, 50, 20, 50, 20);
        var canvasMap = new ChainMap([new ScaleMap(0.5, 0.5), new OffsetMap(-100, -200)]);

        FieldQuadSet set = FieldQuads.Build(canvasMap, [field], Split(new SplitDetection(0, "Last_name_ru", null)),
            new OcrOptions());

        // box (10,20)-(60,40) on the canvas -> (+100, +200) -> / 0.5; an unsplit field's word is the field
        Point[] quad = set.Fields!["Last_name_ru"].Single();
        Assert.That(quad.Select(p => (p.X, p.Y)), Is.EqualTo(new[]
        {
            (220.0, 440.0), (320.0, 440.0), (320.0, 480.0), (220.0, 480.0),
        }));
        Assert.That(set.Words!["Last_name_ru"].Single().Select(p => (p.X, p.Y)), Is.EqualTo(quad.Select(p => (p.X, p.Y))));
    }

    [Test]
    public void AWordQuad_IsItsBoxInsideTheFieldPatch_ClampedToIt()
    {
        using Field field = FieldAt("Birth_place_ru", 100, 200, 80, 30, 80, 30);
        var words = new List<Box>
        {
            new() { X1 = 5.9, Y1 = 2.2, X2 = 40.7, Y2 = 28.1 },
            new() { X1 = -3, Y1 = -2, X2 = 200, Y2 = 90 },      // outside the patch: clamped to it
        };

        FieldQuadSet set = FieldQuads.Build(new ChainMap(), [field],
            Split(new SplitDetection(0, "Birth_place_ru", words)), new OcrOptions());

        var quads = set.Words!["Birth_place_ru"];
        Assert.That(quads[0][0], Is.EqualTo(new Point(105, 202)));      // (int) 5.9 = 5, (int) 2.2 = 2
        Assert.That(quads[0][2], Is.EqualTo(new Point(140, 228)));
        Assert.That(quads[1][0], Is.EqualTo(new Point(100, 200)));
        Assert.That(quads[1][2], Is.EqualTo(new Point(180, 230)));
    }

    [Test]
    public void TheTurnedSeriesAndNumber_WordsLandOnTheirOwnEndOfTheField()
    {
        // a 100 x 20 field with a mark at its far end, turned once counter-clockwise like the pipeline does
        using var cut = new Mat(20, 100, MatType.CV_8UC1, Scalar.All(0));
        cut[new Rect(80, 5, 10, 10)].SetTo(Scalar.All(255));
        using var turned = new Mat();
        Cv2.Rotate(cut, turned, RotateFlags.Rotate90Counterclockwise);
        using var nonZero = new Mat();
        Cv2.FindNonZero(turned, nonZero);
        nonZero.GetArray(out OpenCvSharp.Point[] marked);
        double x0 = marked.Min(p => p.X), y0 = marked.Min(p => p.Y);
        double x1 = marked.Max(p => p.X) + 1, y1 = marked.Max(p => p.Y) + 1;

        var box = new Box { X1 = 300, Y1 = 400, X2 = 400, Y2 = 420, Conf = 0.9, Label = "Licence_number" };
        using var field = new Field(box, Image.Wrap(turned.Clone())).WithCut(100, 20);
        var word = new Box { X1 = x0, Y1 = y0, X2 = x1, Y2 = y1 };

        FieldQuadSet set = FieldQuads.Build(new ChainMap(), [field],
            Split(new SplitDetection(0, "Licence_number", [word])), new OcrOptions { NeedsLicenceRotation = true });

        Point[] quad = set.Words!["Licence_number"].Single();
        Assert.Multiple(() =>
        {
            Assert.That(quad.Min(p => p.X), Is.EqualTo(300 + 80));
            Assert.That(quad.Max(p => p.X), Is.EqualTo(300 + 90));
            Assert.That(quad.Min(p => p.Y), Is.EqualTo(400 + 5));
            Assert.That(quad.Max(p => p.Y), Is.EqualTo(400 + 15));
        });
    }

    [Test]
    public void WhenTheWayBackIsNotKnown_BothAreNull_OnceForTheWholeRun()
    {
        using Field a = FieldAt("Last_name_ru", 10, 20, 50, 20, 50, 20);
        using Field b = FieldAt("First_name_ru", 10, 60, 50, 20, 50, 20);
        FieldQuadSet set = FieldQuads.Build(new ChainMap().Then(new UnknownMap()), [a, b],
            Split(new SplitDetection(0, "Last_name_ru", null), new SplitDetection(1, "First_name_ru", null)),
            new OcrOptions());
        Assert.That(set.Fields, Is.Null);
        Assert.That(set.Words, Is.Null);
        Assert.That(set.Payload()["fields"], Is.Null);
        Assert.That(set.Payload()["words"], Is.Null);
    }

    [Test]
    public void TheStagePayload_HasTheShapeOfTheReference()
    {
        using Field field = FieldAt("Last_name_ru", 10, 20, 50, 20, 50, 20);
        FieldQuadSet set = FieldQuads.Build(new ChainMap(), [field],
            Split(new SplitDetection(0, "Last_name_ru", null)), new OcrOptions());
        var fields = (Dictionary<string, double[][][]>)set.Payload()["fields"]!;
        Assert.That(fields["Last_name_ru"][0], Is.EqualTo(new[]
        {
            new[] { 10.0, 20.0 }, new[] { 60.0, 20.0 }, new[] { 60.0, 40.0 }, new[] { 10.0, 40.0 },
        }));
    }
}

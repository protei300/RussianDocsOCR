using OpenCvSharp;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Maps;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Preprocess;
using RussianDocs.DocumentProcessing.Tensors;
using RussianDocs.DocumentProcessing.ViewModel;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// The registration page (INTPASSPORTADDR) and the registration stamps (issue #27): the oriented-box detector's
/// decode and suppression, the rotated crop and its way back, the stamp order and tie-up, the view model's
/// <c>address</c> block. No models: the pieces are checked on literals and on what OpenCV really does to a pixel.
/// Mirrors the part of the reference's tests that does not need the detectors; the models themselves are graded by
/// the conformance case INTPASSPORTADDR_ALL.
/// </summary>
[TestFixture]
public class AddressTests
{
    private static ObbBox Obb(double cx, double cy, double w, double h, double angle = 0, string label = "L") =>
        new() { Cx = cx, Cy = cy, W = w, H = h, Angle = angle, Conf = 0.9, Label = label };

    private static Point[] Quad(double x0, double y0, double x1, double y1) => Corners.Of(x0, y0, x1, y1);

    // ---- the oriented detector ------------------------------------------------------------------------

    /// <summary>
    /// A <c>[features, anchors]</c> export, the way ultralytics writes it: cx cy w h score... angle. Padded with empty
    /// anchors up to 32: the reference tells the layout by "fewer rows than columns", which a real export (thousands
    /// of anchors) always satisfies and a handful of literals would not.
    /// </summary>
    private static NdArray Export(params float[][] rows)
    {
        int features = rows[0].Length, n = Math.Max(32, rows.Length);
        float[][] anchors = [.. rows, .. Enumerable.Range(0, n - rows.Length).Select(_ => new float[features])];
        var data = new float[features * n];
        for (int a = 0; a < n; a++)
        {
            for (int f = 0; f < features; f++)
            {
                data[f * n + a] = anchors[a][f];
            }
        }
        return NdArray.FromFloat32(data, 1, features, n);
    }

    private static Context Letterboxed(double ratio, double padX, double padY) => new()
    {
        Ratio = ratio, PadLetter = [padX, padY], PadExtra = [0, 0], Resize = true,
    };

    [Test]
    public void ObbDecode_ThresholdsSuppressesAndPutsLinesInReadingOrder()
    {
        var detector = new YoloObbDetector(["L"], iou: 0.45, cls: 0.25);
        NdArray output = Export(
            [200f, 120f, 80f, 20f, 0.90f, 0f],      // lower line, left
            [210f, 122f, 80f, 20f, 0.60f, 0f],      // the same line, weaker: suppressed
            [200f, 60f, 80f, 20f, 0.80f, 0f],       // upper line
            [300f, 300f, 40f, 10f, 0.10f, 0f]);     // under the threshold

        List<ObbBox> kept = detector.Decode(output, Letterboxed(1, 0, 0));

        Assert.That(kept, Has.Count.EqualTo(2));
        Assert.That(kept[0].Cy, Is.EqualTo(60).Within(1e-6), "reading order: the upper line first");
        Assert.That(kept[1].Cy, Is.EqualTo(120).Within(1e-6));
        Assert.That(kept[1].Conf, Is.EqualTo(0.90).Within(1e-6), "the stronger of two overlapping boxes survives");
    }

    [Test]
    public void ObbDecode_UndoesTheLetterbox_AndSortsRowsBeforeColumns()
    {
        var detector = new YoloObbDetector(["L"], iou: 0.45, cls: 0.25);
        // ratio 0.5, 10 px of padding above: the image coordinates are ((x - 0) / 0.5, (y - 10) / 0.5)
        NdArray output = Export(
            [100f, 110f, 20f, 10f, 0.9f, 0.1f],
            [60f, 112f, 20f, 10f, 0.9f, 0.2f]);      // the same row (cy / 10 rounds alike), further left

        List<ObbBox> kept = detector.Decode(output, Letterboxed(0.5, 0, 10));

        Assert.That(kept, Has.Count.EqualTo(2));
        Assert.That(kept[0].Cx, Is.EqualTo(120).Within(1e-4));   // x = 60 / 0.5, first: the rows tie, then cx
        Assert.That(kept[0].Cy, Is.EqualTo(204).Within(1e-4));
        Assert.That(kept[1].Cx, Is.EqualTo(200).Within(1e-4));
        Assert.That(kept[1].Cy, Is.EqualTo(200).Within(1e-4));
        Assert.That(kept[0].W, Is.EqualTo(40).Within(1e-4));
        Assert.That(kept[0].Angle, Is.EqualTo(0.2).Within(1e-6));
    }

    [Test]
    public void ObbDecode_RefusesAShapeItCannotRead_AndTakesAnEmptyOne()
    {
        var detector = new YoloObbDetector(["L"], iou: 0.45, cls: 0.25);
        // A single anchor squeezes to a flat vector of 6 values. The reference raises IndexError on it
        // (`vector.shape[1]`); the port refuses it loudly rather than guess.
        NdArray flat = NdArray.FromFloat32([50f, 50f, 20f, 10f, 0.9f, 0f], 1, 6);
        Assert.Throws<InvalidDataException>(() => detector.Decode(flat, Letterboxed(1, 0, 0)));

        var nothing = NdArray.FromFloat32([], 1, 6, 0);
        Assert.That(detector.Decode(nothing, Letterboxed(1, 0, 0)), Is.Empty);
    }

    [Test]
    public void ObbIou_OfTwoEqualBoxesIsOne_OfDisjointOnesIsZero_AndFollowsTheAngle()
    {
        Assert.That(YoloObbDetector.RotatedIou(50, 50, 40, 10, 0, 50, 50, 40, 10, 0), Is.EqualTo(1f).Within(1e-5));
        Assert.That(YoloObbDetector.RotatedIou(50, 50, 40, 10, 0, 500, 500, 40, 10, 0), Is.EqualTo(0f));
        // a 40x10 box against the same box turned a quarter: a 10x10 square of 400 + 400 - 100
        Assert.That(YoloObbDetector.RotatedIou(50, 50, 40, 10, 0, 50, 50, 40, 10, (float)(Math.PI / 2)),
            Is.EqualTo(100f / 700f).Within(1e-3));
    }

    [Test]
    public void TheLoaderBuildsTheObbPair()
    {
        var input = new ModelInput { Type = "YOLOOBB", Shape = [64, 64, 3] };
        Assert.That(Loader.NewPreprocessor(input), Is.InstanceOf<YoloObb>());

        var output = new ModelOutput
        {
            Type = "YOLOOBBDetector", Iou = 0.45, Cls = 0.25,
            Labels = System.Text.Json.JsonDocument.Parse("[\"L\"]").RootElement,
        };
        Assert.That(Loader.NewPostprocessor(output, "."), Is.InstanceOf<YoloObbDetector>());
    }

    [Test]
    public void ObbPreprocess_IsAFloatNchwTensorInUnitRange_WithTheGreyLetterbox()
    {
        using var mat = new Mat(20, 40, MatType.CV_8UC3, new Scalar(255, 0, 51));   // wide: padded top and bottom
        using Image image = Image.Wrap(mat.Clone());

        (NdArray tensor, Meta meta) = new YoloObb(new ModelInput { Type = "YOLOOBB", Shape = [64, 64, 3] })
            .Apply(image);

        Assert.That(tensor.Shape, Is.EqualTo(new[] { 1, 3, 64, 64 }));
        Assert.That(tensor.Dtype, Is.EqualTo(Dtype.Float32));
        ReadOnlySpan<float> values = tensor.AsFloat32();
        Assert.That(values[(0 * 64 + 32) * 64 + 32], Is.EqualTo(1f), "channel 0 of a pixel in the middle");
        Assert.That(values[(2 * 64 + 32) * 64 + 32], Is.EqualTo(51f / 255f));
        Assert.That(values[0], Is.EqualTo(114f / 255f), "the corner is the letterbox grey");
        Assert.That(meta.Ratio, Is.EqualTo(1.6).Within(1e-9));
    }

    // ---- the rotated crop and the way back -----------------------------------------------------------

    [Test]
    public void CropRotated_OfAnUprightBoxIsTheSameRectangle()
    {
        using var mat = new Mat(100, 100, MatType.CV_8UC3);
        for (int y = 0; y < 100; y++)
        {
            for (int x = 0; x < 100; x++)
            {
                mat.Set(y, x, new Vec3b((byte)x, (byte)y, (byte)(x ^ y)));
            }
        }
        using Image image = Image.Wrap(mat.Clone());

        using Image patch = AddressLinesDetector.CropRotated(image, Obb(50, 40, 11, 5));
        using Image expected = Crop.ClampedCrop(image, 45, 38, 56, 43);

        Assert.That(patch.Width, Is.EqualTo(11));
        Assert.That(patch.Height, Is.EqualTo(5));
        using var difference = new Mat();
        Cv2.Absdiff(patch.Mat, expected.Mat, difference);
        Assert.That(Cv2.CountNonZero(difference.Reshape(1)), Is.Zero);
    }

    [Test]
    public void CropRotated_NeverMakesAnEmptyPatch()
    {
        using Image image = Image.Wrap(new Mat(50, 50, MatType.CV_8UC3, Scalar.All(7)));
        using Image patch = AddressLinesDetector.CropRotated(image, Obb(25, 25, 0.2, 0.3));
        Assert.That((patch.Width, patch.Height), Is.EqualTo((1, 1)));
    }

    [TestCase(0.0)]
    [TestCase(0.3)]
    [TestCase(-0.5)]
    [TestCase(1.2)]
    public void CropGeometry_SendsThePatchCentreToTheBoxCentre(double angle)
    {
        ObbBox box = Obb(120.5, 80.25, 61, 17, angle);
        (int w, int h) = AddressLinesDetector.PatchSize(box);

        Point[]? onImage = AddressLinesDetector.CropGeometry(box).ToInput([new Point(w / 2.0, h / 2.0)]);

        Assert.That(onImage, Is.Not.Null);
        // the centre of a patch of an odd size is a pixel centre; the continuous coordinate is the index + 0.5
        Assert.That(onImage![0].X, Is.EqualTo(box.Cx + 0.5).Within(1e-9));
        Assert.That(onImage[0].Y, Is.EqualTo(box.Cy + 0.5).Within(1e-9));
    }

    [Test]
    public void CropGeometry_FindsAMarkedPixelWhereOpenCvTookItFrom()
    {
        // a bright pixel on a dark image, cut with a tilted box, found on the patch, sent back
        using var mat = new Mat(200, 300, MatType.CV_8UC3, Scalar.All(0));
        mat.Set(100, 140, new Vec3b(255, 255, 255));
        using Image image = Image.Wrap(mat.Clone());
        ObbBox box = Obb(150, 100, 80, 30, 0.25);

        using Image patch = AddressLinesDetector.CropRotated(image, box);
        using var gray = new Mat();
        Cv2.CvtColor(patch.Mat, gray, ColorConversionCodes.RGB2GRAY);
        Cv2.MinMaxLoc(gray, out _, out _, out _, out OpenCvSharp.Point peak);

        Point[]? back = AddressLinesDetector.CropGeometry(box)
            .ToInput([new Point(peak.X + 0.5, peak.Y + 0.5)]);

        Assert.That(back, Is.Not.Null);
        Assert.That(back![0].X, Is.EqualTo(140.5).Within(1.0));
        Assert.That(back[0].Y, Is.EqualTo(100.5).Within(1.0));
    }

    // ---- stamps ---------------------------------------------------------------------------------------

    private static RegistrationStamp Stamp(string kind) => new() { Kind = kind };

    [Test]
    public void StampOrder_OnASpread_IsTheLeftPageBeforeTheRight_ThenTopToBottom_FreshestFirst()
    {
        // 2000 x 1000 is a spread (wider than 1.2 heights): the left page is filled first
        List<Point[]> quads =
        [
            Quad(1200, 100, 1400, 200),    // 0 right page, top
            Quad(100, 600, 300, 700),      // 1 left page, bottom
            Quad(100, 100, 300, 200),      // 2 left page, top
        ];

        List<int> order = Stamps.Order(quads, 2000, 1000);

        // filled: 2, 1, 0 -> freshest first: 0, 1, 2
        Assert.That(order, Is.EqualTo(new[] { 0, 1, 2 }));
    }

    [Test]
    public void StampOrder_OnOnePage_IsTopToBottom()
    {
        List<Point[]> quads = [Quad(100, 600, 300, 700), Quad(100, 100, 300, 200)];
        Assert.That(Stamps.Order(quads, 1000, 1000), Is.EqualTo(new[] { 0, 1 }), "the lower stamp is the fresher");
    }

    [Test]
    public void TieCancellations_ATerminationCancelsTheRegistrationRightBeforeIt_AndTheCurrentOneIsFree()
    {
        // freshest first: registered(0), deregistered(1), registered(2)
        List<RegistrationStamp> stamps = [Stamp("registered"), Stamp("deregistered"), Stamp("registered")];

        Stamps.TieCancellations(stamps);

        Assert.That(stamps[1].Cancels, Is.EqualTo(2));
        Assert.That(stamps[0].Cancels, Is.Null);
        Assert.That(stamps.Select(s => s.Current), Is.EqualTo(new[] { true, false, false }));
    }

    [Test]
    public void TieCancellations_ATerminationOpeningTheCanvasCancelsAnInvisibleRegistration()
    {
        List<RegistrationStamp> stamps = [Stamp("deregistered"), Stamp("registered")];
        Stamps.TieCancellations(stamps);

        Assert.That(stamps[0].Cancels, Is.EqualTo(1));
        Assert.That(stamps.Any(s => s.Current), Is.False, "the only registration is cancelled: none is current");

        List<RegistrationStamp> alone = [Stamp("deregistered")];
        Stamps.TieCancellations(alone);
        Assert.That(alone[0].Cancels, Is.Null);
    }

    [Test]
    public void ContainingStamp_FindsTheQuadThatHoldsThePoint_AndTheNearestCentreAmongOverlapping()
    {
        List<Point[]> quads = [Quad(0, 0, 100, 100), Quad(80, 0, 180, 100)];

        Assert.That(Stamps.ContainingStamp(new Point(20, 50), quads), Is.EqualTo(0));
        Assert.That(Stamps.ContainingStamp(new Point(150, 50), quads), Is.EqualTo(1));
        Assert.That(Stamps.ContainingStamp(new Point(95, 50), quads), Is.EqualTo(1), "inside both: the nearer centre");
        Assert.That(Stamps.ContainingStamp(new Point(85, 50), quads), Is.EqualTo(0));
        Assert.That(Stamps.ContainingStamp(new Point(500, 50), quads), Is.Null);
    }

    [Test]
    public void ContainingStamp_WorksOnATiltedQuad_WhateverTheWinding()
    {
        Point[] diamond = [new(50, 0), new(100, 50), new(50, 100), new(0, 50)];
        Assert.That(Stamps.ContainingStamp(new Point(50, 50), [diamond]), Is.EqualTo(0));
        Assert.That(Stamps.ContainingStamp(new Point(95, 95), [diamond]), Is.Null);
        Assert.That(Stamps.ContainingStamp(new Point(50, 50), [[.. Enumerable.Reverse(diamond)]]), Is.EqualTo(0));
    }

    [Test]
    public void ObbQuad_OfAnUprightBox_IsItsRectangleFromTheTopLeft()
    {
        Point[] quad = Stamps.ObbQuad(Obb(100, 50, 40, 20));
        // the patch's pixel u samples the image at index u + cx - (w - 1) / 2; the corner of the patch is half a
        // pixel before its first sample, and a continuous coordinate is the index + 0.5: left = cx - w / 2 + 0.5
        Assert.That(quad[0].X, Is.EqualTo(100 - 20 + 0.5).Within(1e-9));
        Assert.That(quad[0].Y, Is.EqualTo(50 - 10 + 0.5).Within(1e-9));
        Assert.That(quad[2].X, Is.EqualTo(100 + 20 + 0.5).Within(1e-9));
        Assert.That(quad[2].Y, Is.EqualTo(50 + 10 + 0.5).Within(1e-9));
    }

    // ---- the reading ---------------------------------------------------------------------------------

    [TestCase("12", true)]
    [TestCase("16К1", true)]             // three digits, one letter (Cyrillic К)
    [TestCase("К1К", false)]             // one digit, two letters: the digits are outnumbered
    [TestCase("А5Б", false)]
    [TestCase("16K", true)]              // two digits, one letter
    [TestCase("КВ", false)]
    [TestCase("", false)]
    [TestCase(null, false)]
    [TestCase("-.", false)]
    [TestCase("5А", true)]               // a tie goes to the number
    public void IsNumberToken_IsDigitDominated(string? text, bool expected) =>
        Assert.That(AddressReader.IsNumberToken(text), Is.EqualTo(expected));

    [Test]
    public void LetterboxSquare_PutsALineInTheMiddleOfAGreySquare()
    {
        using Image line = Image.Wrap(new Mat(10, 40, MatType.CV_8UC3, new Scalar(0, 0, 0)));
        using Image square = AddressTextKindClassifier.LetterboxSquare(line);

        Assert.That((square.Width, square.Height), Is.EqualTo((40, 40)));
        Assert.That(square.Mat.At<Vec3b>(0, 0), Is.EqualTo(new Vec3b(114, 114, 114)));
        Assert.That(square.Mat.At<Vec3b>(20, 20), Is.EqualTo(new Vec3b(0, 0, 0)));
        Assert.That(square.Mat.At<Vec3b>(14, 5), Is.EqualTo(new Vec3b(114, 114, 114)), "row 14 is above the line (15..24)");
        Assert.That(square.Mat.At<Vec3b>(15, 5), Is.EqualTo(new Vec3b(0, 0, 0)));
    }

    // ---- the view model --------------------------------------------------------------------------------

    private static Payload Build(List<ObbBox>? detections, List<AddressLineRow>? lines) =>
        Builder.Build(new Input { DocType = "INTPASSPORTADDR_ALL", AddressDetections = detections, AddressLines = lines },
            includeDebug: false);

    [Test]
    public void ViewModelAddress_IsNullOffTheRegistrationPage()
    {
        Assert.That(Build(null, null).Address, Is.Null);
        Assert.That(Build([], []).Address, Is.Null);
    }

    [Test]
    public void ViewModelAddress_CarriesTheBoxesWhenTheyLineUp_AndOnlyTheTextWhenTheyDoNot()
    {
        ObbBox box = Obb(688.690812, 143.27181, 146.95163, 20.63384, 0.0156034, "Living_region_ru");
        box.Conf = 0.8579834;
        var line = new AddressLineRow("printed", 0.0, "TEXT");

        Address aligned = Build([box], [line]).Address!;

        Assert.That(aligned.Aligned, Is.True);
        Assert.That(aligned.Lines[0].Id, Is.EqualTo("o0"));
        Assert.That(aligned.Lines[0].Obbox!.Cx, Is.EqualTo(688.6908));
        Assert.That(aligned.Lines[0].Obbox!.AngleRad, Is.EqualTo(0.015603), "radians keep six places");
        Assert.That(aligned.Lines[0].Obbox!.Conf, Is.EqualTo(0.858));
        Assert.That(aligned.Lines[0].Obbox!.Label, Is.EqualTo("Living_region_ru"));

        // a detection whose patch came out empty has no line: the lists are out of step
        Address skewed = Build([box, box], [line]).Address!;
        Assert.That(skewed.Aligned, Is.False);
        Assert.That(skewed.Lines, Has.Count.EqualTo(1));
        Assert.That(skewed.Lines[0].Obbox, Is.Null);
        Assert.That(skewed.Lines[0].Text, Is.EqualTo("TEXT"));
    }
}

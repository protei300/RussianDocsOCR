using OpenCvSharp;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.PageRegistration;
using RussianDocs.DocumentProcessing.Pipeline;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// The vehicle registration certificate (STS) straightened by its printed blank
/// (<c>Pipeline._register_card</c>). No models here: the blank itself, put into a photo in perspective,
/// is the card. Mirrors tests/test_card_registration.py, plus the pieces the port has to get bit-for-bit
/// (the float32 native scale, the paper colour, the skew measure).
/// </summary>
[TestFixture]
public class CardRegistrationTests
{
    private static readonly string[] Types = ["STS_1996", "STSBACK_1996", "STS_2019", "STSBACK_2019"];

    /// <summary>The first reference of each type's template, to put into a photo.</summary>
    private static readonly Dictionary<string, string> FirstReference = new()
    {
        ["STS_1996"] = "sts_1996_old.jpg",
        ["STSBACK_1996"] = "stsback_1996_old.jpg",
        ["STS_2019"] = "sts_2019_new.jpg",
        ["STSBACK_2019"] = "stsback_2019_new.jpg",
    };

    private static string Root => ModelPaths.Root();

    private static string TemplatesDir => Path.Combine(Root, "document_processing", "pipeline_modules",
        "page_registration", "templates");

    /// <summary>The card warped onto a plain background at <paramref name="corners"/> (TL, TR, BR, BL).</summary>
    private static Image PhotoOf(string docType, Point2f[] corners, Size size)
    {
        using Mat gray = Cv2.ImRead(Path.Combine(TemplatesDir, FirstReference[docType]), ImreadModes.Grayscale);
        using Mat card = new();
        Cv2.CvtColor(gray, card, ColorConversionCodes.GRAY2RGB);
        Point2f[] src = [new(0, 0), new(gray.Cols, 0), new(gray.Cols, gray.Rows), new(0, gray.Rows)];
        using Mat m = Cv2.GetPerspectiveTransform(src, corners);
        using var warped = new Mat();
        Cv2.WarpPerspective(card, warped, m, size, InterpolationFlags.Linear, BorderTypes.Constant, Scalar.All(0));
        using var ones = new Mat(gray.Rows, gray.Cols, MatType.CV_8UC1, Scalar.All(255));
        using var inside = new Mat();
        Cv2.WarpPerspective(ones, inside, m, size);
        var bg = new Mat(size.Height, size.Width, MatType.CV_8UC3, new Scalar(90, 110, 130));
        warped.CopyTo(bg, inside);
        return Image.Wrap(bg);
    }

    [TestCaseSource(nameof(Types))]
    public void EverySts_TypeHasTemplates(string docType)
    {
        using var reg = new PageRegistrar(Root, docType);
        Assert.Multiple(() =>
        {
            Assert.That(reg.PageNames, Is.EqualTo(new[] { "card" }));
            Assert.That((reg.PageW, reg.PageH), Is.EqualTo((1000, 1419)));
        });
    }

    [Test]
    public void AType_WithoutTemplates_IsReportedAsMissing() =>
        Assert.Throws<FileNotFoundException>(() => new PageRegistrar(Root, "STS_1850"));

    [TestCaseSource(nameof(Types))]
    public void ACardInPerspective_IsFoundToAFewPixels(string docType)
    {
        using var reg = new PageRegistrar(Root, docType);
        Point2f[] corners = [new(260, 180), new(1320, 240), new(1380, 1700), new(210, 1640)];
        using Image photo = PhotoOf(docType, corners, new Size(1600, 1900));

        PageRegistrationResult r = reg.Register(photo)[0];

        Assert.That(r.Ok, Is.True);
        Assert.That(r.Inliers, Is.GreaterThanOrEqualTo(40));
        double worst = r.Quad!.Zip(corners, (a, b) => Math.Sqrt(Math.Pow(a.X - b.X, 2) + Math.Pow(a.Y - b.Y, 2))).Max();
        Assert.That(worst, Is.LessThan(4.0), "card corners off by that many px");
    }

    [Test]
    public void TheFill_PaintsPastThePhoto_InsteadOfSmearingItsEdge()
    {
        using var reg = new PageRegistrar(Root, "STS_2019");
        using var mat = new Mat(200, 100, MatType.CV_8UC3, Scalar.All(255));
        using Image photo = Image.Wrap(mat.Clone());
        double[,] identity = Homography.Identity();

        using Image smeared = reg.WarpMatrix(photo, identity, 0.2);
        using Image painted = reg.WarpMatrix(photo, identity, 0.2, [10, 20, 30]);

        (int w, int h) = reg.OutSize(0.2);
        Assert.Multiple(() =>
        {
            // the page is 0.2 * 1060 = 212 wide, the photo 100: columns past 100 are outside it
            Assert.That(smeared.Mat.At<Vec3b>(10, w - 5), Is.EqualTo(new Vec3b(255, 255, 255)),
                "the default repeats the edge (the passport path)");
            Assert.That(painted.Mat.At<Vec3b>(10, w - 5), Is.EqualTo(new Vec3b(10, 20, 30)));
            Assert.That(h, Is.GreaterThan(0));
        });
    }

    [Test]
    public void ThePaperColour_IsThePerChannelMedianInsideTheCard_Truncated()
    {
        using var reg = new PageRegistrar(Root, "STS_2019");
        using var mat = new Mat(60, 60, MatType.CV_8UC3, new Scalar(0, 0, 0));
        // inside the quad (10..19 x 10..19): left half R=10, right half R=11 -> median 10.5 -> 10
        mat[new Rect(10, 10, 5, 10)].SetTo(new Scalar(10, 20, 31));
        mat[new Rect(15, 10, 5, 10)].SetTo(new Scalar(11, 20, 31));
        using Image photo = Image.Wrap(mat.Clone());

        int[]? fill = reg.CardFill(photo, [new(10, 10), new(19, 10), new(19, 19), new(10, 19)]);

        Assert.That(fill, Is.EqualTo(new[] { 10, 20, 31 }));
    }

    [Test]
    public void ThePaperColour_OfAQuadOutsideThePhoto_IsNone()
    {
        using var reg = new PageRegistrar(Root, "STS_2019");
        using Image photo = Image.Wrap(new Mat(40, 40, MatType.CV_8UC3, Scalar.All(7)));
        Assert.That(reg.CardFill(photo, [new(100, 100), new(120, 100), new(120, 120), new(100, 120)]), Is.Null);
    }

    [Test]
    public void TheSkew_IsZeroForACardTheBordersQuadSquaresUp_AndGrowsWhenOneCornerMoves()
    {
        using var reg = new PageRegistrar(Root, "STS_2019");
        Point[] borders = [new(300, 200), new(1300, 260), new(1360, 1680), new(260, 1620)];

        double straight = reg.CardSkew(borders, borders);
        Point[] bent = [borders[0], borders[1], new(1360, 1780), borders[3]];
        double skewed = reg.CardSkew(borders, bent);

        Assert.Multiple(() =>
        {
            Assert.That(straight, Is.LessThan(1e-4));
            Assert.That(skewed, Is.GreaterThan(Recognizer.CardSkewKeep),
                "a corner 100 px off is well past the 1 % that keeps the Borders canvas");
        });
    }

    [Test]
    public void TheNativeScale_IsAFloat32Value_AsInTheReference()
    {
        using var reg = new PageRegistrar(Root, "STS_2019");
        // a card 861.3 px wide on the top edge and 862.9 on the bottom: scale = 0.5 * (n1 + n2) / 1000
        var quad = new Point[] { new(10.25f, 20.5f), new(871.5f, 20.5f), new(873.75f, 1440f), new(10.25f, 1440f) };
        var regs = new[] { new PageRegistrationResult("card", Homography.Identity(), 100, "x", quad) };

        double scale = reg.NativeScale(regs);

        Assert.That(scale, Is.EqualTo((double)(float)scale), "rounded to float32, like the NumPy chain");
        Assert.That(scale, Is.EqualTo(0.5f * (861.25f + 863.5f) / 1000f).Within(0));
    }

    [Test]
    public void TheNativeScale_NeverUpsamples_AndIsFlooredAtAQuarter()
    {
        using var reg = new PageRegistrar(Root, "STS_2019");
        Point[] large = [new(0, 0), new(1500, 0), new(1500, 2000), new(0, 2000)];
        Point[] tiny = [new(0, 0), new(100, 0), new(100, 140), new(0, 140)];
        Assert.Multiple(() =>
        {
            Assert.That(reg.NativeScale([new PageRegistrationResult("card", Homography.Identity(), 100, "x", large)]),
                Is.EqualTo(1.0));
            Assert.That(reg.NativeScale([new PageRegistrationResult("card", Homography.Identity(), 100, "x", tiny)]),
                Is.EqualTo(0.25));
        });
    }

    [Test]
    public void TheLineSegmentDetector_FindsALongHorizontalLine()
    {
        using var mat = new Mat(200, 400, MatType.CV_8UC1, Scalar.All(0));
        Cv2.Line(mat, new OpenCvSharp.Point(20, 100), new OpenCvSharp.Point(380, 100), Scalar.All(255), 3);

        var segments = Lsd.Detect(mat);

        Assert.That(segments.Any(s => Math.Abs(s.X2 - s.X1) > 300 && Math.Abs(s.Y2 - s.Y1) < 3), Is.True);
    }
}

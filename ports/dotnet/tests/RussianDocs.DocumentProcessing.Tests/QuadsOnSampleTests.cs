using OpenCvSharp;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Pipeline;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// The check a synthetic mark cannot make: the same pixels, to a fraction of a pixel. Port of
/// <c>test_a_field_of_a_real_sample_is_cut_from_the_photo_by_its_quadrilateral</c>.
///
/// <para>
/// The pipeline cut the field out of its canvas; cutting the same field out of the PHOTO through the
/// quadrilateral must give the same picture. The residual shift between the two is found by correlation and
/// refined on the parabola through the peak, so the half-pixel shift OpenCV's warps need - the usual mistake
/// here - shows up as a shift of about 0.5 and not as a blur of the numbers. Needs the weights and the
/// samples; ignored on a tree that has neither.
/// </para>
/// </summary>
[TestFixture]
public class QuadsOnSampleTests
{
    private Recognizer? _recognizer;
    private string _root = "";

    [OneTimeSetUp]
    public void Load()
    {
        try
        {
            _root = ModelPaths.Root();
            _recognizer = new Recognizer(Device.Cpu, 1, OcrTier.Accurate);
        }
        catch (Exception ex) when (ex is DirectoryNotFoundException or FileNotFoundException)
        {
            Assert.Ignore("no weights in this tree: " + ex.Message);
        }
    }

    [OneTimeTearDown]
    public void Unload() => _recognizer?.Dispose();

    private static readonly string[] Samples =
    [
        "samples/DL_2011/10_BG_DL_2010.jpg",
        "samples/INTPASSPORT_2011/12_CR_INTPASSPORT_2011.jpg",
        "conformance/material/STS_1996/synth_01_STS_1996.jpg",
        "conformance/material/STSBACK_2019/synth_01_STSBACK_2019.jpg",
    ];

    [TestCaseSource(nameof(Samples))]
    public void AFieldOfARealSample_IsCutFromThePhotoByItsQuadrilateral(string relative)
    {
        string path = Path.Combine(_root, relative);
        if (!File.Exists(path))
        {
            Assert.Ignore("sample not in the tree");
        }
        using Image photo = Io.LoadRgb(path);
        using Results results = _recognizer!.Run(path, new RunOptions());
        Assert.That(results.FieldQuads, Is.Not.Null, "the way back is known for this sample");
        Assert.That(results.Canvas, Is.Not.Null);

        int checkedFields = 0;
        foreach ((string label, List<Point[]> quads) in results.FieldQuads!)
        {
            var boxes = results.Boxes.Where(b => b.Label == label).ToList();
            if (boxes.Count != 1 || quads.Count != 1 || label == "Special_marks" || label == "Licence_number")
            {
                continue;   // one detection per label; the marks are re-cut with a margin, the number is turned
            }
            ViewModel.Box2 box = boxes[0];
            using Image patch = Crop.ClampedCrop(results.Canvas!, (int)box.X1, (int)box.Y1, (int)box.X2, (int)box.Y2);
            if (patch.Width < 24 || patch.Height < 12)
            {
                continue;
            }
            const int margin = 4;
            Point2f[] source = [.. quads[0].Select(p => new Point2f((float)p.X, (float)p.Y))];
            Point2f[] target =
            [
                new(margin, margin), new(patch.Width + margin, margin),
                new(patch.Width + margin, patch.Height + margin), new(margin, patch.Height + margin),
            ];
            using Mat matrix = Cv2.GetPerspectiveTransform(source, target);
            using var cut = new Mat();
            Cv2.WarpPerspective(photo.Mat, cut, matrix, new Size(patch.Width + 2 * margin, patch.Height + 2 * margin));

            using var response = new Mat();
            Cv2.MatchTemplate(cut, patch.Mat, response, TemplateMatchModes.CCoeffNormed);
            Cv2.MinMaxLoc(response, out _, out double best, out _, out OpenCvSharp.Point at);
            double dx = Vertex(response, at, horizontal: true), dy = Vertex(response, at, horizontal: false);

            Assert.That(best, Is.GreaterThan(0.9), $"{label}: the quadrilateral does not cut out the same field");
            Assert.That(Math.Abs(at.X + dx - margin), Is.LessThan(0.5), $"{label}: shift in x");
            Assert.That(Math.Abs(at.Y + dy - margin), Is.LessThan(0.5), $"{label}: shift in y");
            checkedFields++;
        }
        Assert.That(checkedFields, Is.GreaterThan(0), "none of the expected fields was read on the sample");
    }

    /// <summary>The peak of the response refined to a fraction of a pixel (parabola through three values).</summary>
    private static double Vertex(Mat response, OpenCvSharp.Point at, bool horizontal)
    {
        int x = at.X, y = at.Y;
        if (horizontal ? x <= 0 || x >= response.Cols - 1 : y <= 0 || y >= response.Rows - 1)
        {
            return 0.0;
        }
        float left = horizontal ? response.At<float>(y, x - 1) : response.At<float>(y - 1, x);
        float middle = response.At<float>(y, x);
        float right = horizontal ? response.At<float>(y, x + 1) : response.At<float>(y + 1, x);
        float denominator = left - 2 * middle + right;
        return denominator == 0 ? 0.0 : 0.5 * (left - right) / denominator;
    }
}

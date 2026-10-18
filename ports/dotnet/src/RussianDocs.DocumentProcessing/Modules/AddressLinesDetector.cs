using OpenCvSharp;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Maps;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// Finds the oriented address lines on the registration page of an internal passport (INTPASSPORTADDR) and cuts
/// each one out upright. Port of <c>AddressLinesDetector</c>.
///
/// <para>
/// A YOLO-OBB detector, so a tilted or perspective-skewed line is captured directly, without a deskew first.
/// The lines come back in reading order (rows top to bottom, left to right within a row), ready for word
/// splitting and reading.
/// </para>
/// </summary>
public sealed class AddressLinesDetector : IDisposable
{
    /// <summary>The key in <c>models_path.yaml</c>, and the folder under <c>models/</c>.</summary>
    public const string ModelKey = "AddressLinesDetector";

    private readonly ObbDetectionModel _model;

    public AddressLinesDetector(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
        => _model = new ObbDetectionModel(
            Path.Combine(ModelPaths.Resolve(root, paths, ModelKey), "ONNX"), device, threads);

    /// <summary>The oriented boxes only (<c>predict</c>).</summary>
    public List<ObbBox> Predict(Image canvas) => _model.Predict(canvas);

    /// <summary>
    /// The boxes and an upright patch of each, in the same order (<c>predict_transform</c>). **The patches are
    /// OWNED BY THE CALLER.**
    /// </summary>
    public (List<ObbBox> Boxes, List<Image> Patches) PredictTransform(Image canvas)
    {
        List<ObbBox> boxes = _model.Predict(canvas);
        var patches = new List<Image>(boxes.Count);
        try
        {
            foreach (ObbBox box in boxes)
            {
                patches.Add(CropRotated(canvas, box));
            }
            return (boxes, patches);
        }
        catch
        {
            foreach (Image patch in patches)
            {
                patch.Dispose();
            }
            throw;
        }
    }

    /// <summary>
    /// The patch size of a box: its width and height rounded half to even (Python's <c>round</c>), at least 1.
    /// </summary>
    public static (int W, int H) PatchSize(ObbBox box) =>
        (Math.Max(1, PyNum.RoundHalfEvenToInt(box.W)), Math.Max(1, PyNum.RoundHalfEvenToInt(box.H)));

    /// <summary>
    /// The rotation that stands a box upright, about its centre, in degrees: <c>np.degrees</c> on a Python float,
    /// that is <c>x * (180 / pi)</c>.
    /// </summary>
    private static Mat RotationOf(ObbBox box) =>
        Cv2.GetRotationMatrix2D(new Point2f((float)box.Cx, (float)box.Cy), box.Angle * (180.0 / Math.PI), 1.0);

    /// <summary>
    /// Cuts a rotated rectangle out and returns it upright (<c>crop_rotated</c>): the whole image is rotated about
    /// the box's centre (bilinear, zero border), then a sub-pixel window of the box's size is read around the
    /// centre. The same two calls in the same order as the reference - a different route to the same rectangle
    /// differs by interpolation, and the reading of a thin line is sensitive to that.
    /// </summary>
    public static Image CropRotated(Image img, ObbBox box)
    {
        (int w, int h) = PatchSize(box);
        using Mat m = RotationOf(box);
        using var rotated = new Mat();
        Cv2.WarpAffine(img.Mat, rotated, m, new Size(img.Width, img.Height), InterpolationFlags.Linear);
        var patch = new Mat();
        Cv2.GetRectSubPix(rotated, new Size(w, h), new Point2f((float)box.Cx, (float)box.Cy), patch);
        return Image.Wrap(patch);
    }

    /// <summary>
    /// The map from a <see cref="CropRotated"/> patch back to the image it was cut from (<c>crop_geometry</c>): the
    /// same rotation and the same sub-pixel window, so a box found on the patch lands on the image.
    /// </summary>
    public static ChainMap CropGeometry(ObbBox box)
    {
        (int w, int h) = PatchSize(box);
        using Mat m = RotationOf(box);
        var matrix = new double[2, 3];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                matrix[i, j] = m.At<double>(i, j);
            }
        }
        // getRectSubPix: patch pixel (u, v) samples the rotated image at
        // (u + cx - (w - 1) / 2, v + cy - (h - 1) / 2), pixel centres on integers.
        var window = new OffsetMap(-(box.Cx - (w - 1) / 2.0), -(box.Cy - (h - 1) / 2.0));
        return new ChainMap([new HomographyMap(matrix), window]);
    }

    public void Dispose() => _model.Dispose();
}

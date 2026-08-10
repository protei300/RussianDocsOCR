using OpenCvSharp;

namespace RussianDocs.DocumentProcessing.Imaging;

/// <summary>
/// Cropping. **The only sanctioned crop path in the port.**
///
/// <para>
/// This is the single most dangerous divergence in the whole exercise, and it is worth spelling out
/// because the failure mode is a crash on a rare input rather than a wrong number on a common one.
/// </para>
///
/// <para>
/// <c>img[y1:y2, x1:x2]</c> in Python does not validate: an upper bound past the edge is silently
/// CLAMPED, and a negative start counts from the end. <c>new Mat(mat, rect)</c> in OpenCvSharp —
/// like <c>Region</c> in gocv and <c>submat</c> on the JVM — THROWS. So the detectors, which
/// routinely return a box a pixel or two outside the image, produce a working crop in the reference
/// and an exception in a port that translates the slice literally.
/// </para>
///
/// <para>
/// A port that "works" is therefore a port that clamps, and it must clamp the way the slice
/// effectively does. Hence one function, used everywhere, with unit tests per language.
/// </para>
/// </summary>
public static class Crop
{
    /// <summary>
    /// The clamped equivalent of <c>img[y1:y2, x1:x2]</c>.
    ///
    /// <para>
    /// Negative starts clamp to 0 rather than counting from the end. The reference's own coordinates
    /// come from detector output that has already been clipped to non-negative, so the
    /// count-from-the-end branch is unreachable there; implementing it would add behaviour the
    /// reference does not exercise, which is a worse kind of wrong than not implementing it.
    /// </para>
    ///
    /// <para>
    /// An empty intersection yields a zero-sized image rather than an error, because the reference's
    /// slice does too — and the OCR path has a documented degenerate route for exactly that.
    /// </para>
    /// </summary>
    public static Image ClampedCrop(Image src, int x1, int y1, int x2, int y2)
    {
        int w = src.Width, h = src.Height;

        int left = Math.Clamp(x1, 0, w);
        int top = Math.Clamp(y1, 0, h);
        int right = Math.Clamp(x2, 0, w);
        int bottom = Math.Clamp(y2, 0, h);

        // A reversed range is empty in Python, not an error and not a flipped crop.
        int cropW = Math.Max(0, right - left);
        int cropH = Math.Max(0, bottom - top);

        if (cropW == 0 || cropH == 0)
        {
            // Zero-sized, with the source's type so downstream code sees the expected channel count.
            return Image.Wrap(new Mat(cropH, cropW, src.Mat.Type()));
        }

        // Clone rather than return a view: a submat shares the parent's buffer, so the crop would
        // dangle the moment the parent is disposed — and the parent here is a pipeline intermediate
        // that is released as soon as the stage ends.
        using var view = new Mat(src.Mat, new Rect(left, top, cropW, cropH));
        return Image.Wrap(view.Clone());
    }
}

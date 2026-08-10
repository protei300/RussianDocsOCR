using OpenCvSharp;

namespace RussianDocs.DocumentProcessing.Imaging;

/// <summary>
/// A single-channel float32 mask, at whatever resolution the proto masks came in.
///
/// <para>
/// Kept as float until the very last step. The threshold that turns it binary is the ONLY place a
/// decision is made, so any rounding earlier would move the contour — which is the thing being
/// compared.
/// </para>
///
/// <para>
/// **float32 throughout, never widened.** The reference accumulates the mask in float32, and
/// "improving" that to double changes the mask boundary and therefore the extracted quadrilateral.
/// </para>
/// </summary>
public sealed class FloatMask : IDisposable
{
    private Mat _mat;

    private FloatMask(Mat mat) => _mat = mat;

    public int Height => _mat.Rows;
    public int Width => _mat.Cols;

    public static FloatMask FromValues(float[] values, int height, int width)
    {
        var mat = new Mat(height, width, MatType.CV_32FC1);
        mat.SetArray(values);
        return new FloatMask(mat);
    }

    /// <summary>Crops by row/column bounds, exclusive at the far edge — a numpy slice.</summary>
    public FloatMask Crop(int top, int bottom, int left, int right)
    {
        using var view = new Mat(_mat, new Rect(left, top, right - left, bottom - top));
        return new FloatMask(view.Clone());
    }

    /// <summary>Resizes with bilinear interpolation, matching the reference's mask upscale.</summary>
    public FloatMask Resize(int width, int height)
    {
        var dst = new Mat();
        Cv2.Resize(_mat, dst, new Size(width, height), 0, 0, InterpolationFlags.Linear);
        return new FloatMask(dst);
    }

    /// <summary>
    /// Zeroes everything outside a box, so two adjacent documents cannot bleed into each other's
    /// contour.
    ///
    /// <para>
    /// **The comparisons are STRICT**, so the boundary row and column are zeroed too — matching the
    /// reference's <c>clip_boxes</c> exactly. Using inclusive bounds adds a one-pixel rim to every
    /// mask, which survives thresholding and shifts the contour.
    /// </para>
    /// </summary>
    public void ZeroOutsideBox(double x1, double y1, double x2, double y2)
    {
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                bool inside = x > x1 && x < x2 && y > y1 && y < y2;
                if (!inside)
                {
                    _mat.Set(y, x, 0f);
                }
            }
        }
    }

    /// <summary>Thresholds to an 8-bit binary mask, which is what <c>findContours</c> wants.</summary>
    public Image Threshold(double value)
    {
        using var binary = new Mat();
        Cv2.Threshold(_mat, binary, value, 255, ThresholdTypes.Binary);
        var eightBit = new Mat();
        binary.ConvertTo(eightBit, MatType.CV_8UC1);
        return Image.Wrap(eightBit);
    }

    public void Dispose()
    {
        _mat?.Dispose();
        _mat = null!;
    }
}

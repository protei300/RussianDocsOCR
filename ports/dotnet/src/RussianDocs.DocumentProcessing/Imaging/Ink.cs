using OpenCvSharp;

namespace RussianDocs.DocumentProcessing.Imaging;

/// <summary>Measures of how much a crop carries on it.</summary>
public static class Ink
{
    /// <summary>
    /// Variance of the Laplacian of a crop: how much fine detail (strokes, not darkness) it carries.
    /// Port of <c>Pipeline._line_ink</c>: RGB to GRAY, float32, <c>cv2.Laplacian(..., CV_32F)</c> with
    /// the default 3x3 aperture, then <c>.var()</c> — the population variance, two-pass.
    ///
    /// <para>
    /// A printed stroke is a sharp local change; a strip that has none (blank paper, or a strip blurred
    /// past the width of a stroke) has almost no such change left, whatever its brightness. An empty
    /// crop has no ink: 0.
    /// </para>
    /// </summary>
    public static double LaplacianVariance(Image rgb)
    {
        if (rgb.IsEmpty || rgb.Width == 0 || rgb.Height == 0)
        {
            return 0.0;
        }

        using Image gray = Io.ToGray(rgb);
        using var floats = new Mat();
        gray.Mat.ConvertTo(floats, MatType.CV_32FC1);
        using var laplacian = new Mat();
        Cv2.Laplacian(floats, laplacian, MatType.CV_32F, 1, 1, 0, BorderTypes.Default);

        int count = laplacian.Rows * laplacian.Cols;
        if (count == 0)
        {
            return 0.0;
        }
        float[] values = laplacian.AsSpan<float>().ToArray();

        double mean = 0.0;
        foreach (float v in values)
        {
            mean += v;
        }
        mean /= count;

        double sum = 0.0;
        foreach (float v in values)
        {
            double d = v - mean;
            sum += d * d;
        }
        return sum / count;
    }
}

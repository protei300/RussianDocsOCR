using OpenCvSharp;

namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// OpenCV's LSD line segment detector (<c>cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)</c>), the
/// line evidence of <c>line_refine</c> and <c>line_dewarp</c> in the reference.
///
/// <para>
/// OpenCvSharp4 DOES bind it: <c>LineSegmentDetector.Create</c> / <c>Detect</c>. The first port of the
/// page registration (and the notes that followed it, deviation D-03) took it for unbound and put
/// Canny+HoughLinesP in its place; the line evidence then differed from the reference's, and so did the
/// refinement homography and the dewarp decision. The defaults below are those of
/// <c>cv2.createLineSegmentDetector</c> (scale 0.8, sigma_scale 0.6, quant 2, ang_th 22.5, log_eps 0,
/// density_th 0.7, n_bins 1024).
/// </para>
///
/// <para>One shared detector, as the reference keeps one (<c>_LSD</c>), behind a lock: the detector keeps
/// state between calls, and several recognitions may run in one process.</para>
/// </summary>
public static class Lsd
{
    private static readonly object Gate = new();
    private static LineSegmentDetector? _detector;

    /// <summary>
    /// Segments of an 8-bit grey image, as <c>(x1, y1, x2, y2)</c> in float32 precision widened to double
    /// (the reference's <c>lines.reshape(-1, 4).astype(np.float64)</c>); empty when none were found.
    /// </summary>
    public static (double X1, double Y1, double X2, double Y2)[] Detect(Mat gray)
    {
        lock (Gate)
        {
            _detector ??= LineSegmentDetector.Create(LineSegmentDetectorModes.RefineStd, 0.8, 0.6, 2.0, 22.5, 0, 0.7, 1024);
            using InputArray image = InputArray.Create(gray);
            _detector.Detect(image, out Vec4f[] lines, out _, out _, out _);
            var result = new (double, double, double, double)[lines.Length];
            for (int i = 0; i < lines.Length; i++)
            {
                result[i] = (lines[i].Item0, lines[i].Item1, lines[i].Item2, lines[i].Item3);
            }
            return result;
        }
    }
}

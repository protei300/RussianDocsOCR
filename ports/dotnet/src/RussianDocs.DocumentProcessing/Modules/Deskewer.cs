using OpenCvSharp;
using RussianDocs.DocumentProcessing.Imaging;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// Removes residual tilt by scanning candidate angles and maximising the variance of the
/// row-ink profile.
///
/// <para>
/// A COARSE-TO-FINE search: a sparse scan across the whole range, then a dense scan around the coarse
/// winner. Verified in the reference to choose the same angles as a single dense scan while being
/// about 2.9x faster.
/// </para>
/// </summary>
public sealed class DocDeskewer
{
    private readonly double _angleRange;
    private readonly double _minAngle;
    private readonly double _scale;
    private readonly double[] _coarseAngles;
    private readonly double _fineHalfRange;
    private readonly int _fineCount;

    /// <summary>
    /// The parameters the PIPELINE uses.
    ///
    /// <para>
    /// Deliberately not the reference class's own defaults, which differ (its <c>angle_range</c> is
    /// 2.0). Reading only the class would give a working deskewer that chooses different angles.
    /// </para>
    /// </summary>
    public static DocDeskewer ForPipeline() => new(10.0, 101, 2.0, 0.4, 21);

    public DocDeskewer(double angleRange, int angleSteps, double minAngle, double scale,
        int coarseSteps)
    {
        _angleRange = angleRange;
        _minAngle = minAngle;
        _scale = scale;

        int coarse = Math.Clamp(coarseSteps, 3, angleSteps);
        _coarseAngles = Linspace(-angleRange, angleRange, coarse);

        double coarseStep = 2 * angleRange / (coarse - 1);
        double fullResolution = 2 * angleRange / (angleSteps - 1);
        _fineHalfRange = coarseStep;
        _fineCount = Math.Max(3, (int)Math.Round(2 * coarseStep / fullResolution) + 1);
    }

    /// <summary>
    /// Rotates the residual tilt out. Returns the corrected image and the angle applied.
    ///
    /// <para>
    /// Below <c>minAngle</c> the image is returned unchanged — a copy, so ownership is uniform. The
    /// threshold is load-bearing: it decides whether the canvas is rotated at all, and therefore
    /// whether every box downstream lands in the same place.
    /// </para>
    /// </summary>
    public (Image Deskewed, double Angle) Deskew(Image image)
    {
        double angle = FindAngle(image);
        if (Math.Abs(angle) < _minAngle)
        {
            return (image.Clone(), angle);
        }

        using Mat rotation = RotationMatrix(image.Width / 2.0, image.Height / 2.0, angle, 1.0);
        var dst = new Mat();
        Cv2.WarpAffine(image.Mat, dst, rotation, new Size(image.Width, image.Height),
            InterpolationFlags.Linear, BorderTypes.Replicate);
        return (Image.Wrap(dst), angle);
    }

    private double FindAngle(Image image)
    {
        using Image gray = Io.ToGray(image);

        int sh = Math.Max(1, (int)(gray.Height * _scale));
        int sw = Math.Max(1, (int)(gray.Width * _scale));
        using Image small = Io.Resize(gray, sw, sh, Interpolation.Area);
        (Image binary, double _) = Contours.ThresholdOtsu(small, invert: true);

        using (binary)
        {
            double cx = sw / 2.0, cy = sh / 2.0;
            double[] coarse = ScoreAngles(binary, sw, sh, cx, cy, _coarseAngles);
            int ci = Argmax(coarse);

            // **A winner at either END of the coarse scan means no tilt.** The true optimum lies
            // outside the search range, so refining around the edge would pick an arbitrary angle;
            // the reference bails to 0.0 and this early exit is what keeps the canvas unrotated.
            if (ci == 0 || ci == _coarseAngles.Length - 1)
            {
                return 0.0;
            }

            double best = _coarseAngles[ci];
            double lo = Math.Max(-_angleRange, best - _fineHalfRange);
            double hi = Math.Min(_angleRange, best + _fineHalfRange);
            double[] fineAngles = Linspace(lo, hi, _fineCount);
            double[] fine = ScoreAngles(binary, sw, sh, cx, cy, fineAngles);
            return fineAngles[Argmax(fine)];
        }
    }

    private static double[] ScoreAngles(Image binary, int sw, int sh, double cx, double cy,
        double[] angles)
    {
        var scores = new double[angles.Length];
        for (int i = 0; i < angles.Length; i++)
        {
            using Mat rotation = RotationMatrix(cx, cy, angles[i], 1.0);
            using var rotated = new Mat();
            // Nearest-neighbour and ZERO borders: the input is a binary mask, so interpolation would
            // invent grey values, and rotated-in area must contribute no ink.
            Cv2.WarpAffine(binary.Mat, rotated, rotation, new Size(sw, sh),
                InterpolationFlags.Nearest, BorderTypes.Constant, Scalar.All(0));
            scores[i] = Variance(RowSums(rotated));
        }
        return scores;
    }

    /// <summary>
    /// Per-row ink totals, as int64.
    ///
    /// <para>
    /// Exact integers, so this step contributes no float error at all — which is what makes the
    /// variance below the only numerically delicate part of the search.
    /// </para>
    /// </summary>
    private static long[] RowSums(Mat binary)
    {
        var sums = new long[binary.Rows];
        for (int y = 0; y < binary.Rows; y++)
        {
            long total = 0;
            for (int x = 0; x < binary.Cols; x++)
            {
                total += binary.Get<byte>(y, x);
            }
            sums[y] = total;
        }
        return sums;
    }

    /// <summary>
    /// numpy's TWO-PASS variance: subtract the mean, then average the squares.
    ///
    /// <para>
    /// **The one-pass form <c>E[x²] - E[x]²</c> is not equivalent here.** On values of order 255·W it
    /// loses about seven significant digits, which is enough to flip the argmax between two adjacent,
    /// nearly-equal angles — and the output is a discrete choice that rotates the image. Two passes,
    /// always.
    /// </para>
    /// </summary>
    private static double Variance(long[] values)
    {
        if (values.Length == 0)
        {
            return 0;
        }
        double mean = 0;
        foreach (long v in values)
        {
            mean += v;
        }
        mean /= values.Length;

        double acc = 0;
        foreach (long v in values)
        {
            double d = v - mean;
            acc += d * d;
        }
        return acc / values.Length;
    }

    /// <summary>First maximum, like <c>np.argmax</c>. Strict <c>&gt;</c> only.</summary>
    private static int Argmax(double[] values)
    {
        int best = 0;
        double bestValue = double.NegativeInfinity;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] > bestValue)
            {
                best = i;
                bestValue = values[i];
            }
        }
        return best;
    }

    private static double[] Linspace(double from, double to, int count)
    {
        if (count <= 1)
        {
            return [from];
        }
        var values = new double[count];
        double step = (to - from) / (count - 1);
        for (int i = 0; i < count; i++)
        {
            values[i] = from + step * i;
        }
        return values;
    }

    /// <summary>
    /// The rotation matrix, built BY HAND rather than through <c>GetRotationMatrix2D</c>.
    ///
    /// <para>
    /// D-08. The reference passes a FRACTIONAL centre — <c>(sw/2.0, sh/2.0)</c> — and some bindings
    /// only accept an integer point, so the centre gets truncated. Measured in the Go spike: the
    /// integer version shifted the variance array by 3.8e-3 relative, which is ABOVE the 1e-3 policy,
    /// and therefore capable of selecting a different angle. This formula was verified against
    /// OpenCV's to 1.6e-14.
    /// </para>
    /// </summary>
    private static Mat RotationMatrix(double cx, double cy, double angleDegrees, double scale)
    {
        double radians = angleDegrees * Math.PI / 180.0;
        double alpha = Math.Cos(radians) * scale;
        double beta = Math.Sin(radians) * scale;

        var m = new Mat(2, 3, MatType.CV_64FC1);
        m.Set(0, 0, alpha);
        m.Set(0, 1, beta);
        m.Set(0, 2, (1 - alpha) * cx - beta * cy);
        m.Set(1, 0, -beta);
        m.Set(1, 1, alpha);
        m.Set(1, 2, beta * cx + (1 - alpha) * cy);
        return m;
    }
}

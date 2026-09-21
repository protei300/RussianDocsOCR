using OpenCvSharp;
using RussianDocs.DocumentProcessing.Imaging;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// 3x3 homography arithmetic and the two OpenCV calls that produce/consume one:
/// <c>cv2.getPerspectiveTransform</c> (4 exact correspondences) and
/// <c>cv2.findHomography</c> with MAGSAC (many, noisy correspondences). Shared by
/// <see cref="PageRegistrar"/> and, later, <c>LineRefine</c>'s model matrix.
///
/// <para>
/// A plain <c>double[3,3]</c> rather than a wrapping type — every consumer either feeds it straight
/// to OpenCvSharp (which wants a <c>Mat</c> anyway) or does small linear algebra on it, and a type
/// with no invariants to protect does not earn its keep.
/// </para>
/// </summary>
public static class Homography
{
    public static double[,] Identity() =>
        new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

    public static double[,] Multiply(double[,] a, double[,] b)
    {
        var r = new double[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double sum = 0;
                for (int k = 0; k < 3; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                r[i, j] = sum;
            }
        }
        return r;
    }

    /// <summary>Closed-form 3x3 inverse via the adjugate. Returns null when the matrix is singular.</summary>
    public static double[,]? Invert(double[,] m)
    {
        double det =
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) -
            m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) +
            m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]);
        if (Math.Abs(det) < 1e-12)
        {
            return null;
        }
        double invDet = 1.0 / det;
        var r = new double[3, 3];
        r[0, 0] = (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) * invDet;
        r[0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * invDet;
        r[0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * invDet;
        r[1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * invDet;
        r[1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * invDet;
        r[1, 2] = (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) * invDet;
        r[2, 0] = (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]) * invDet;
        r[2, 1] = (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) * invDet;
        r[2, 2] = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * invDet;
        return r;
    }

    public static Point TransformPoint(double[,] h, Point p)
    {
        double x = h[0, 0] * p.X + h[0, 1] * p.Y + h[0, 2];
        double y = h[1, 0] * p.X + h[1, 1] * p.Y + h[1, 2];
        double w = h[2, 0] * p.X + h[2, 1] * p.Y + h[2, 2];
        return Math.Abs(w) < 1e-12 ? new Point(double.NaN, double.NaN) : new Point(x / w, y / w);
    }

    public static Point[] TransformPoints(double[,] h, IReadOnlyList<Point> pts) =>
        [.. pts.Select(p => TransformPoint(h, p))];

    /// <summary>
    /// <c>cv2.getPerspectiveTransform</c>: the exact homography mapping 4 source points onto 4
    /// destination points. Returns null on a degenerate correspondence.
    /// </summary>
    public static double[,]? Solve4(IReadOnlyList<Point> src, IReadOnlyList<Point> dst)
    {
        if (src.Count != 4 || dst.Count != 4)
        {
            throw new ArgumentException("page_registration: Solve4 needs exactly 4 points each side");
        }
        Point2f[] s = [.. src.Select(p => new Point2f((float)p.X, (float)p.Y))];
        Point2f[] d = [.. dst.Select(p => new Point2f((float)p.X, (float)p.Y))];
        try
        {
            using Mat m = Cv2.GetPerspectiveTransform(s, d);
            return ToArray(m);
        }
        catch (OpenCVException)
        {
            return null;
        }
    }

    /// <summary>
    /// <c>cv2.findHomography(..., cv2.USAC_MAGSAC, reproj, maxIters=10000, confidence=0.999)</c>.
    ///
    /// <para>
    /// <c>USAC_MAGSAC</c> has no named member in this OpenCvSharp build (checked: `HomographyMethods`
    /// only declares the pre-USAC values), so the raw OpenCV enum value is used directly — verified by
    /// a runtime call, not assumed: <c>opencv2/calib3d.hpp</c> numbers the USAC family
    /// 32 (USAC_DEFAULT) through 38 (USAC_MAGSAC), and a probe against a real template image
    /// (<c>intpassport_page2_a.png</c>, 521 SIFT keypoints) returned a non-empty homography and an
    /// inlier mask exactly the length of the input, using <c>(HomographyMethods)38</c>.
    /// </para>
    /// </summary>
    public const int UsacMagsac = 38;

    public static (double[,]? H, bool[] Inliers, bool Ok) FindMagsac(
        IReadOnlyList<Point> src, IReadOnlyList<Point> dst, double reprojPx,
        int maxIters = 10000, double confidence = 0.999)
    {
        if (src.Count < 4 || src.Count != dst.Count)
        {
            return (null, [], false);
        }
        Point2f[] s = [.. src.Select(p => new Point2f((float)p.X, (float)p.Y))];
        Point2f[] d = [.. dst.Select(p => new Point2f((float)p.X, (float)p.Y))];
        using var mask = new Mat();
        try
        {
            using Mat h = Cv2.FindHomography(InputArray.Create(s), InputArray.Create(d),
                (HomographyMethods)UsacMagsac, reprojPx, mask, maxIters, confidence);
            if (h.Empty())
            {
                return (null, [], false);
            }
            var inliers = new bool[src.Count];
            for (int i = 0; i < src.Count; i++)
            {
                inliers[i] = mask.At<byte>(i) != 0;
            }
            return (ToArray(h), inliers, true);
        }
        catch (OpenCVException)
        {
            return (null, [], false);
        }
    }

    private static double[,] ToArray(Mat m)
    {
        var r = new double[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                r[i, j] = m.At<double>(i, j);
            }
        }
        return r;
    }

    /// <summary>
    /// Warps <paramref name="src"/> through an arbitrary 3x3 homography — unlike
    /// <see cref="Contours.WarpPerspectiveQuad"/> (built from 4 destination corners only), every
    /// caller here needs the FULL matrix (composed from a coarse match, a chain prior, a page-margin
    /// shift, or a scale). <paramref name="replicate"/> selects <c>BORDER_REPLICATE</c> (every
    /// <c>page_registration.py</c> warp uses it) over the default constant/black border.
    /// </summary>
    public static Image WarpByHomography(Image src, double[,] h, int width, int height, bool replicate)
    {
        using Mat m = ToMat(h);
        var dst = new Mat();
        Cv2.WarpPerspective(src.Mat, dst, m, new Size(width, height), InterpolationFlags.Linear,
            replicate ? BorderTypes.Replicate : BorderTypes.Constant);
        return Image.Wrap(dst);
    }

    public static Mat ToMat(double[,] h)
    {
        var m = new Mat(3, 3, MatType.CV_64FC1);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                m.Set(i, j, h[i, j]);
            }
        }
        return m;
    }
}

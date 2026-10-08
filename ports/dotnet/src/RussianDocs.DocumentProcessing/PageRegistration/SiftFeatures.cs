using OpenCvSharp;
using OpenCvSharp.Features2D;

namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// SIFT keypoints and descriptors in an order every platform agrees on. Port of
/// <c>page_registration.detect_features</c>.
///
/// <para>
/// OpenCV orders its keypoints with <c>std::sort</c> (duplicate removal) and cuts an <c>nfeatures</c> budget with
/// <c>std::nth_element</c>; both leave ties in an order that depends on the C++ library OpenCV was built with.
/// MAGSAC samples matches by index, so the same keypoints in another order give another homography: on an STS of
/// the 2019 form the Windows wheel kept the Borders canvas (skew 0.0076) and the same code on Linux re-cut it from
/// the template (0.0118) - conformance D-07/D-08 (2026-10-08). So the detector runs WITHOUT a budget (the detector
/// must be created with <c>nFeatures: 0</c>), the keypoints are sorted here by response (descending) and then by
/// position, size, angle and octave, and the budget is cut from that order.
/// </para>
/// </summary>
public static class SiftFeatures
{
    /// <summary>
    /// The indices of the keypoints in the order <c>np.lexsort((octave, angle, size, x, y, -response))</c> gives:
    /// the primary key is the response, descending, then y, x, size, angle and octave, all ascending. The
    /// comparison is in double (the keys are float32 widened, as NumPy widens them) and STABLE; the budget, when
    /// positive, keeps the first that many.
    /// </summary>
    public static int[] Order(IReadOnlyList<KeyPoint> keypoints, int budget)
    {
        IEnumerable<int> order = Enumerable.Range(0, keypoints.Count)
            .OrderBy(i => -(double)keypoints[i].Response)
            .ThenBy(i => (double)keypoints[i].Pt.Y)
            .ThenBy(i => (double)keypoints[i].Pt.X)
            .ThenBy(i => (double)keypoints[i].Size)
            .ThenBy(i => (double)keypoints[i].Angle)
            .ThenBy(i => (double)keypoints[i].Octave);
        if (budget > 0)
        {
            order = order.Take(budget);
        }
        return [.. order];
    }

    /// <summary>
    /// Puts keypoints and descriptors (one row per keypoint) in <see cref="Order"/>; the descriptors follow their
    /// keypoints. The returned matrix is owned by the caller.
    /// </summary>
    public static (KeyPoint[] Keypoints, Mat Descriptors) Reorder(KeyPoint[] keypoints, Mat descriptors, int budget)
    {
        if (keypoints.Length == 0)
        {
            return ([], new Mat());
        }
        int[] order = Order(keypoints, budget);
        var rows = new Mat(order.Length, descriptors.Cols, descriptors.Type());
        for (int i = 0; i < order.Length; i++)
        {
            using Mat source = descriptors.Row(order[i]);
            using Mat target = rows.Row(i);
            source.CopyTo(target);
        }
        return ([.. order.Select(i => keypoints[i])], rows);
    }

    /// <summary>
    /// Detects on <paramref name="gray"/> (under <paramref name="mask"/>, 0 = skip, when given) and orders the
    /// result. With no keypoints the descriptors are an empty matrix, which <c>PageRegistrar</c> reads as "no
    /// features". The caller owns the returned matrix.
    /// </summary>
    public static (KeyPoint[] Keypoints, Mat Descriptors) Detect(SIFT sift, Mat gray, Mat? mask, int budget)
    {
        using var descriptors = new Mat();
        sift.DetectAndCompute(gray, mask, out KeyPoint[] keypoints, descriptors);
        return Reorder(keypoints, descriptors, budget);
    }
}

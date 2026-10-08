using OpenCvSharp;
using RussianDocs.DocumentProcessing.PageRegistration;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// Template matching does not depend on the order OpenCV hands keypoints over. Port of tests/test_sift_order.py.
///
/// <para>
/// OpenCV orders SIFT keypoints with <c>std::sort</c> and cuts its <c>nfeatures</c> budget with
/// <c>std::nth_element</c>; the order of ties depends on the C++ library it was built with. MAGSAC samples by
/// index, so the Windows wheel and Linux reached different decisions on one STS card (conformance D-07/D-08,
/// 2026-10-08). <see cref="SiftFeatures"/> sorts the keypoints itself and cuts the budget from that order. No
/// models needed.
/// </para>
/// </summary>
[TestFixture]
public class SiftOrderTests
{
    /// <summary>300 random keypoints plus 20 that tie on the response and differ only by position; row i of the descriptors belongs to keypoint i.</summary>
    private static (KeyPoint[] Keypoints, Mat Descriptors) Keypoints()
    {
        var rng = new Random(0);
        var kp = new List<KeyPoint>();
        for (int i = 0; i < 300; i++)
        {
            kp.Add(new KeyPoint((float)(rng.NextDouble() * 500), (float)(rng.NextDouble() * 500), 3f, 0f,
                (float)rng.NextDouble(), 0, -1));
        }
        for (int x = 0; x < 20; x++)
        {
            kp.Add(new KeyPoint(x, 10f, 3f, 0f, 0.5f, 0, -1));
        }
        var desc = new Mat(kp.Count, 4, MatType.CV_32FC1);
        for (int i = 0; i < kp.Count; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                desc.Set(i, j, (float)(i * 4 + j));
            }
        }
        return ([.. kp], desc);
    }

    private static T[] Shuffled<T>(T[] items, int seed)
    {
        var rng = new Random(seed);
        return [.. items.OrderBy(_ => rng.Next())];
    }

    [Test]
    public void TheOrderHandedOver_DoesNotMatter()
    {
        (KeyPoint[] kp, Mat desc) = Keypoints();
        using (desc)
        {
            string Signature(int seed)
            {
                // the same keypoints, with their descriptor rows, in another order
                int[] permutation = Shuffled(Enumerable.Range(0, kp.Length).ToArray(), seed);
                using var shuffled = new Mat(kp.Length, 4, MatType.CV_32FC1);
                for (int i = 0; i < permutation.Length; i++)
                {
                    using Mat from = desc.Row(permutation[i]);
                    using Mat to = shuffled.Row(i);
                    from.CopyTo(to);
                }
                (KeyPoint[] got, Mat gotDesc) = SiftFeatures.Reorder([.. permutation.Select(i => kp[i])], shuffled, 200); // the budget reaches into the tied group
                using (gotDesc)
                {
                    gotDesc.GetArray(out float[] rows);
                    return string.Join(";", got.Select(k => $"{k.Pt.X:R},{k.Pt.Y:R},{k.Response:R}"))
                        + "|" + string.Join(",", rows);
                }
            }

            string first = Signature(0);
            Assert.That(Enumerable.Range(1, 5).Select(Signature), Is.All.EqualTo(first));
        }
    }

    [Test]
    public void TheBudget_KeepsTheStrongest()
    {
        (KeyPoint[] kp, Mat desc) = Keypoints();
        using (desc)
        {
            (KeyPoint[] got, Mat gotDesc) = SiftFeatures.Reorder(Shuffled(kp, 1), desc, 50);
            using (gotDesc)
            {
                float[] responses = [.. got.Select(k => k.Response)];
                Assert.That(got, Has.Length.EqualTo(50));
                Assert.That(responses, Is.Ordered.Descending);
                Assert.That(responses.Min(), Is.GreaterThanOrEqualTo(kp.Select(k => k.Response).OrderDescending().ElementAt(49)));

            }
        }
    }

    [Test]
    public void TiesOnTheResponse_AreBrokenByPosition_YThenX()
    {
        (KeyPoint[] kp, Mat desc) = Keypoints();
        using (desc)
        {
            KeyPoint[] handed = Shuffled(kp, 3);
            int[] order = SiftFeatures.Order(handed, 0);
            var tied = order.Where(i => handed[i].Response == 0.5f && handed[i].Pt.Y == 10f).Select(i => handed[i].Pt.X).ToArray();
            Assert.That(tied, Has.Length.EqualTo(20));
            Assert.That(tied, Is.Ordered.Ascending, "equal responses and y: x ascending");
        }
    }

    [Test]
    public void TheDescriptorRowOfAKeypoint_TravelsWithIt()
    {
        (KeyPoint[] kp, Mat desc) = Keypoints();
        using (desc)
        {
            // keypoint 0..3 are distinct; give the input an order where they sit last, then check the rows
            int[] permutation = [.. Enumerable.Range(4, kp.Length - 4), 0, 1, 2, 3];
            using var shuffled = new Mat(kp.Length, 4, MatType.CV_32FC1);
            for (int i = 0; i < permutation.Length; i++)
            {
                using Mat from = desc.Row(permutation[i]);
                using Mat to = shuffled.Row(i);
                from.CopyTo(to);
            }
            (KeyPoint[] got, Mat gotDesc) = SiftFeatures.Reorder([.. permutation.Select(i => kp[i])], shuffled, 0);
            using (gotDesc)
            {
                gotDesc.GetArray(out float[] rows);
                for (int i = 0; i < got.Length; i++)
                {
                    int original = (int)rows[4 * i] / 4;
                    Assert.That(kp[original].Pt.X, Is.EqualTo(got[i].Pt.X));
                    Assert.That(kp[original].Response, Is.EqualTo(got[i].Response));
                }
            }
        }
    }
}

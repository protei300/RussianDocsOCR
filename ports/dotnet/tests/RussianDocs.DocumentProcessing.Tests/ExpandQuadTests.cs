using RussianDocs.DocumentProcessing.Imaging;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// <c>expand_quad</c> runs on a float32 array in the reference, so every step rounds to float32; a
/// double computation lands up to ~6e-5 px elsewhere and moves the warped canvas by whole grey levels.
/// </summary>
[TestFixture]
public class ExpandQuadTests
{
    private static readonly Point[] Quad =
        [new(103, 211), new(871, 197), new(889, 1304), new(95, 1312)];

    [Test]
    public void F32_EveryCoordinateIsAFloat32Value()
    {
        Point[] expanded = Geometry.ExpandQuadF32(Quad, 0.01);
        Assert.Multiple(() =>
        {
            foreach (Point p in expanded)
            {
                Assert.That(p.X, Is.EqualTo((double)(float)p.X));
                Assert.That(p.Y, Is.EqualTo((double)(float)p.Y));
            }
        });
    }

    [Test]
    public void F32_AgreesWithTheDoubleVersionToFloat32Precision_ButNotBitwise()
    {
        Point[] f32 = Geometry.ExpandQuadF32(Quad, 0.01);
        Point[] f64 = Geometry.ExpandQuad(Quad, 0.01);
        for (int i = 0; i < Quad.Length; i++)
        {
            Assert.That(f32[i].X, Is.EqualTo(f64[i].X).Within(1e-3));
            Assert.That(f32[i].Y, Is.EqualTo(f64[i].Y).Within(1e-3));
        }
        Assert.That(f32.Zip(f64).Any(t => t.First != t.Second), Is.True,
            "the point of the float32 version is that it rounds where double does not");
    }

    [Test]
    public void F32_NoMargin_ReturnsTheQuadUnchanged()
    {
        Assert.That(Geometry.ExpandQuadF32(Quad, 0), Is.EqualTo(Quad));
    }
}

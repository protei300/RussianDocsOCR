using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.PageRegistration;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// <see cref="PageRegistrar.QuadIou"/> / the shared Sutherland-Hodgman clip in
/// <see cref="QuadFit"/> — pins a real bug found while wiring page registration into the pipeline
/// (2026-09-21): the clip's half-plane test had the wrong sign for TL,TR,BR,BL ordering in IMAGE
/// coordinates (Y grows downward), so <c>QuadIou</c> returned 0 for almost every pair, including two
/// heavily-overlapping quads. That silently broke two things at once — <see cref="PageRegistrar"/>'s
/// registered-page-to-Borders-quad matching (a page's real Borders quad looked unrelated to it, so a
/// LEFTOVER quad from a different page got used instead — INTPASSPORT_2011's page 3 came out as a
/// second copy of page 2) and <see cref="QuadFit.FitQuadLines"/>'s own <c>MIN_IOU_WITH_INIT</c> gate
/// (the RANSAC-fitted quad could never be accepted, so every call silently fell back to the plain
/// polygon quad). Caught by exactly the check this class is: a synthetic case with a known answer.
/// </summary>
[TestFixture]
public class PageRegistrationGeometryTests
{
    [Test]
    public void QuadIou_TwoSquaresOverlappingByHalf_MatchesTheClosedFormAnswer()
    {
        // Two 100x100 axis-aligned squares, the second shifted right by 50: overlap 50x100=5000,
        // union 100*100*2-5000=15000, IoU = 5000/15000 = 1/3 exactly.
        Point[] a = [new(0, 0), new(100, 0), new(100, 100), new(0, 100)];
        Point[] b = [new(50, 0), new(150, 0), new(150, 100), new(50, 100)];

        Assert.That(PageRegistrar.QuadIou(a, b), Is.EqualTo(1.0 / 3.0).Within(1e-9));
    }

    [Test]
    public void QuadIou_IdenticalQuad_IsOne()
    {
        Point[] a = [new(10, 20), new(210, 25), new(205, 180), new(8, 175)];
        Assert.That(PageRegistrar.QuadIou(a, a), Is.EqualTo(1.0).Within(1e-9));
    }

    [Test]
    public void QuadIou_DisjointQuads_IsZero()
    {
        Point[] a = [new(0, 0), new(100, 0), new(100, 100), new(0, 100)];
        Point[] b = [new(0, 200), new(100, 200), new(100, 300), new(0, 300)];
        Assert.That(PageRegistrar.QuadIou(a, b), Is.EqualTo(0.0).Within(1e-9));
    }

    /// <summary>
    /// The exact case that exposed the bug: a registered page's own quad-in-image, against the two
    /// Borders quads it was meant to be told apart from. Kept as literal coordinates (not
    /// re-derived) because the bug was specifically about THIS shape of input, not general convexity.
    /// </summary>
    [Test]
    public void QuadIou_RealRegistrationCase_PicksTheOverlappingBordersQuad()
    {
        Point[] quad0 = [new(19, 4), new(511, 7), new(493, 348), new(18, 346)];
        Point[] quad1 = [new(25, 367), new(491, 372), new(518, 726), new(16, 714)];
        Point[] regQuad = [new(-1, -13), new(506, -2), new(498, 343), new(14, 336)];

        Assert.Multiple(() =>
        {
            Assert.That(PageRegistrar.QuadIou(quad0, regQuad), Is.GreaterThan(0.4),
                "regQuad visibly overlaps quad0 (both span roughly the same photo region)");
            Assert.That(PageRegistrar.QuadIou(quad1, regQuad), Is.EqualTo(0.0).Within(1e-9),
                "regQuad and quad1 are disjoint (opposite halves of the photo)");
        });
    }
}

using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// Per-class confidence and IOU overrides on <see cref="YoloDetector"/> — the "PerClassYOLODetector"
/// tag's <c>CLSPerClass</c>/<c>IOUPerClass</c> (postprocessing.py:509-593). TextFields/ONNX/model.json
/// carries <c>{"MRZ": 0.4}</c> / <c>{"MRZ": 0.6}</c>; these tests pin the general mechanism with
/// synthetic labels rather than depending on that specific model being loadable.
/// </summary>
[TestFixture]
public class DetectorPerClassTests
{
    private static Context NoResizeContext() => new() { Resize = false };

    [Test]
    public void ClsPerClass_LetsAnOverriddenLabelSurviveBelowTheSharedThreshold()
    {
        // One anchor, labels ["A", "MRZ"], shared cls 0.5, MRZ overridden down to 0.4 — exactly the
        // shipped TextFields override. Raw scores: A=0.30 (fails everything), MRZ=0.45 (fails the
        // shared 0.5 but passes its own 0.4).
        var withOverride = new YoloDetector(["A", "MRZ"], iou: 0.2, cls: 0.5, NmsMode.PerClass,
            clsPerClass: new Dictionary<string, double> { ["MRZ"] = 0.4 });
        NdArray output = NdArray.FromFloat32(
            [10f, 10f, 4f, 4f, 0.30f, 0.45f], 1, 6);

        List<Box> kept = withOverride.Decode(output, NoResizeContext());

        Assert.That(kept, Has.Count.EqualTo(1));
        Assert.That(kept[0].Label, Is.EqualTo("MRZ"));
        Assert.That(kept[0].Conf, Is.EqualTo(0.45).Within(1e-6));

        // Without the override the same row is dropped entirely — the shared threshold beats every
        // column. Proves the assertion above is exercising the override, not some other path.
        var withoutOverride = new YoloDetector(["A", "MRZ"], iou: 0.2, cls: 0.5, NmsMode.PerClass);
        Assert.That(withoutOverride.Decode(output, NoResizeContext()), Is.Empty);
    }

    /// <summary>
    /// The winning class is the argmax AFTER zeroing columns that fail their OWN threshold
    /// (postprocessing.py:557-566, conf_scores), not the argmax of the raw scores. A column with the
    /// highest raw score can lose if its own (overridden) bar is higher than what it scored, letting a
    /// lower raw score that clears its own bar win instead.
    /// </summary>
    [Test]
    public void ClsPerClass_WinnerIsArgmaxAfterZeroing_NotOfRawScores()
    {
        // A's own threshold is raised to 0.9; A's raw score (0.7) is the overall max but fails its own
        // bar. MRZ keeps the shared 0.5 and its raw score (0.6) clears it.
        var detector = new YoloDetector(["A", "MRZ"], iou: 0.2, cls: 0.5, NmsMode.PerClass,
            clsPerClass: new Dictionary<string, double> { ["A"] = 0.9 });
        NdArray output = NdArray.FromFloat32(
            [10f, 10f, 4f, 4f, 0.70f, 0.60f], 1, 6);

        List<Box> kept = detector.Decode(output, NoResizeContext());

        Assert.That(kept, Has.Count.EqualTo(1));
        Assert.That(kept[0].Label, Is.EqualTo("MRZ"), "A's raw score is higher, but it fails A's own (overridden) threshold");
        Assert.That(kept[0].Conf, Is.EqualTo(0.60).Within(1e-6));
    }

    /// <summary>
    /// The plain "YOLODetector" tag never sees per-class dictionaries (models.py:139-146 only reads
    /// them for "PerClassYOLODetector"), so its class-agnostic argmax must be untouched: the raw
    /// argmax wins even when it would differ from the fused per-column computation if a threshold
    /// override were present.
    /// </summary>
    [Test]
    public void NoOverrides_ArgmaxMatchesTheRawScores()
    {
        var detector = new YoloDetector(["A", "B"], iou: 0.2, cls: 0.5, NmsMode.ClassAgnostic);
        NdArray output = NdArray.FromFloat32([10f, 10f, 4f, 4f, 0.55f, 0.60f], 1, 6);

        List<Box> kept = detector.Decode(output, NoResizeContext());

        Assert.That(kept, Has.Count.EqualTo(1));
        Assert.That(kept[0].Label, Is.EqualTo("B"));
        Assert.That(kept[0].Conf, Is.EqualTo(0.60).Within(1e-6));
    }

    /// <summary>
    /// IOUPerClass (postprocessing.py:568-584, <c>iou_for</c>): two same-class boxes at IOU 0.25 are
    /// suppressed to one under the shared 0.2 threshold, but both survive once that class's NMS
    /// threshold is raised to 0.6 — the shipped MRZ override, which exists precisely so a tilted
    /// document's two MRZ lines stop suppressing each other.
    /// </summary>
    [Test]
    public void IouPerClass_RaisesTheNmsBarForOnlyThatClass()
    {
        // Two boxes, both class "MRZ" (single-label model for simplicity): 40x10 each, overlapping by
        // 4 in y for a 40x4 intersection over a 640 union -> IOU 0.25.
        float[] rows =
        [
            50f, 50f, 40f, 10f, 0.9f, // box A: x[30,70] y[45,55]
            50f, 56f, 40f, 10f, 0.9f, // box B: x[30,70] y[51,61]
        ];
        NdArray output = NdArray.FromFloat32(rows, 2, 5);

        var withoutOverride = new YoloDetector(["MRZ"], iou: 0.2, cls: 0.5, NmsMode.PerClass);
        Assert.That(withoutOverride.Decode(output, NoResizeContext()), Has.Count.EqualTo(1),
            "IOU 0.25 exceeds the shared 0.2 threshold, so one box suppresses the other");

        var withOverride = new YoloDetector(["MRZ"], iou: 0.2, cls: 0.5, NmsMode.PerClass,
            iouPerClass: new Dictionary<string, double> { ["MRZ"] = 0.6 });
        Assert.That(withOverride.Decode(output, NoResizeContext()), Has.Count.EqualTo(2),
            "IOU 0.25 is below the overridden 0.6 threshold, so both survive");
    }

    /// <summary>The class-agnostic NMS path must ignore <c>iouPerClass</c> entirely (only
    /// <c>PerClassYOLODetectorPostprocessing.nms_indices</c> reads it in the reference).</summary>
    [Test]
    public void IouPerClass_HasNoEffectUnderClassAgnosticMode()
    {
        float[] rows =
        [
            50f, 50f, 40f, 10f, 0.9f,
            50f, 56f, 40f, 10f, 0.9f,
        ];
        NdArray output = NdArray.FromFloat32(rows, 2, 5);

        var detector = new YoloDetector(["MRZ"], iou: 0.2, cls: 0.5, NmsMode.ClassAgnostic,
            iouPerClass: new Dictionary<string, double> { ["MRZ"] = 0.6 });

        Assert.That(detector.Decode(output, NoResizeContext()), Has.Count.EqualTo(1),
            "class-agnostic NMS always uses the shared threshold, override or not");
    }
}

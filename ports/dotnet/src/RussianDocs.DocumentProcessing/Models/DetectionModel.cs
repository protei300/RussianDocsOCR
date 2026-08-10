using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Preprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Models;

/// <summary>
/// A detector: one YOLO input, one detector output.
///
/// <para>
/// Built through the same three switches as everything else — <see cref="Loader"/> — rather than
/// assembling its pre/post pipeline by hand. The Go port initially did assemble it by hand, which
/// meant the dispatch design that MAPPING.md calls the portable core was bypassed by precisely the
/// models that use it.
/// </para>
/// </summary>
public sealed class DetectionModel : IDisposable
{
    private readonly Session _session;
    private readonly IPreprocessor _pre;
    private readonly YoloDetector _detector;

    public ModelConfig Config { get; }

    public DetectionModel(string dir, Device device, int threads)
    {
        Config = ModelConfig.Load(dir);
        _pre = Loader.NewPreprocessor(Config.Inputs[0]);

        // From the switch, then ONE cast — the NMS mode is decided there by the declared output
        // type, because TextFields declares PerClassYOLODetector while Words declares YOLODetector
        // and that difference is the whole distinction between the two.
        _detector = Loader.NewPostprocessor(Config.Outputs[0], Config.Dir) as YoloDetector
            ?? throw new InvalidDataException(
                $"models: {Config.Name} output 0 is not a detector ({Config.Outputs[0].Type})");
        _session = new Session(Config.ModelPath, device, threads);
    }

    /// <summary>Detects, with coordinates mapped back to the input image.</summary>
    public List<Box> Predict(Image image)
    {
        (NdArray tensor, Meta meta) = _pre.Apply(image);
        NdArray[] raw = _session.Run([tensor]);
        return _detector.Decode(raw[0], ContextOf(meta, resize: true));
    }

    internal static Context ContextOf(Meta meta, bool resize) => new()
    {
        Ratio = meta.Ratio,
        PadExtra = meta.PadExtra,
        PadLetter = meta.PadLetter,
        PaddedH = meta.PaddedH,
        PaddedW = meta.PaddedW,
        OrigH = meta.OrigH,
        OrigW = meta.OrigW,
        Resize = resize,
    };

    public void Dispose() => _session.Dispose();
}

/// <summary>
/// A segmentation model: one YOLO input, a detector output and a proto-mask output.
/// </summary>
public sealed class SegmentationModel : IDisposable
{
    private readonly Session _session;
    private readonly IPreprocessor _pre;
    private readonly YoloDetector _detector;
    private readonly YoloSegmentor _segmentor;

    public ModelConfig Config { get; }

    public SegmentationModel(string dir, Device device, int threads)
    {
        Config = ModelConfig.Load(dir);
        if (Config.Outputs.Count != 2)
        {
            throw new InvalidDataException(
                $"models: {Config.Name} expects 2 outputs (boxes, proto), got {Config.Outputs.Count}");
        }

        _pre = Loader.NewPreprocessor(Config.Inputs[0]);

        // NumpyOnly: the segmentation path needs RAW float coordinates and the mask coefficients.
        // Truncating and labelling first would discard the sub-pixel information the mask crop
        // depends on. The `with` copy keeps the switch as the single construction site.
        YoloDetector detector = Loader.NewPostprocessor(Config.Outputs[0], Config.Dir) as YoloDetector
            ?? throw new InvalidDataException(
                $"models: {Config.Name} output 0 is not a detector ({Config.Outputs[0].Type})");
        _detector = detector.WithNumpyOnly();

        _segmentor = Loader.NewPostprocessor(Config.Outputs[1], Config.Dir) as YoloSegmentor
            ?? throw new InvalidDataException(
                $"models: {Config.Name} output 1 is not a segmentor ({Config.Outputs[1].Type})");
        _session = new Session(Config.ModelPath, device, threads);
    }

    /// <summary>
    /// Detects and segments.
    ///
    /// <para>
    /// The segmentor is handed <c>PaddedH</c>/<c>PaddedW</c> — the size BEFORE the letterbox but
    /// AFTER any extra padding — because that is the space the mask is unpadded into. Passing the
    /// original dimensions instead produces a mask that is subtly the wrong shape.
    /// </para>
    /// </summary>
    public (List<Box> Boxes, List<Point[]> Segments) Predict(Image image)
    {
        (NdArray tensor, Meta meta) = _pre.Apply(image);
        NdArray[] raw = _session.Run([tensor]);

        List<Box> boxes = _detector.Decode(raw[0], DetectionModel.ContextOf(meta, resize: true));
        if (boxes.Count == 0)
        {
            return ([], []);
        }

        List<Point[]> segments = _segmentor.Segment(raw[1], boxes, meta.PadExtra,
            meta.PaddedH, meta.PaddedW);
        return (boxes, segments);
    }

    public void Dispose() => _session.Dispose();
}

using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Preprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Models;

/// <summary>
/// Builds a runnable model from its <c>model.json</c>.
///
/// <para>
/// **The three switches below are the most portable code in the project, and they must stay that
/// way.** No reflection, no attributes, no DI container, no self-registering initialisers: one
/// <c>switch</c> per factory, one construction expression per case, cases in the same ORDER as the
/// reference's <c>match</c> statements (recorded in MAPPING.md). A dispatch table built by scanning
/// assemblies is more idiomatic C# and would make the correspondence with the other three languages
/// unverifiable.
/// </para>
///
/// <para>
/// An unknown tag is an ERROR naming the tag (D-06). The reference falls through to <c>None</c> and
/// turns a typo into a null dereference three stages later.
/// </para>
/// </summary>
public static class Loader
{
    /// <summary>Preprocessor by the input's declared <c>Type</c>. Case order follows the reference.</summary>
    public static IPreprocessor NewPreprocessor(ModelInput input) => input.Type switch
    {
        "Classification" => new Classification(input),
        // Recognised but not implemented yet — wired so a port cannot quietly grow a different
        // behaviour for them, and so the reader sees what exists rather than guessing.
        "YOLO" => new Yolo(input),
        "YOLOOBB" => new NotImplementedPreprocessor("YOLOOBB"),
        "OCRv2" => new OcrV2(input),
        // "OCR" is the removed legacy 31x200 grayscale path. No shipped model.json declares it;
        // wired anyway, because an omitted case reads as an oversight.
        "OCR" => new NotImplementedPreprocessor("OCR"),
        _ => throw new InvalidDataException($"models: unknown input type \"{input.Type}\""),
    };

    /// <summary>Postprocessor by the output's declared <c>Type</c>.</summary>
    /// <param name="root">
    /// The repository root. Needed because OCRProbs resolves its ALLOWED charset from
    /// config/ocr_alphabets.json, which lives beside the library rather than beside the model — the
    /// Go port added the same parameter at this milestone and for the same reason.
    /// </param>
    public static IPostprocessor NewPostprocessor(ModelOutput output, string dir, string? root = null)
        => output.Type switch
    {
        // **The tag is MultiLabelClassification, and I got this wrong by guessing "MultiClass".**
        // Worth leaving a note: the D-06 error naming the unknown tag is what turned a wrong guess
        // into a one-line fix, instead of the reference's fall-through to None and a null
        // dereference two stages later. Read the artifacts, do not name the tags from memory.
        // BinaryClassification is a SIGMOID against a threshold, MultiLabelClassification is an
        // argmax. Both were briefly routed to MultiClass here, which silently returned the first
        // label for every binary output — see BinaryClass for what that cost.
        "BinaryClassification" => new BinaryClass(output.LabelsAsStrings(), output.Threshold ?? 0.5),
        "MultiLabelClassification" => new MultiClass(output),
        "Metric" => new Metric(
            Path.Combine(dir, ModelPaths.NormaliseSeparators(
                output.Centers ?? throw new InvalidDataException(
                    "models: Metric output has no Centers"))),
            output.Metric ?? "cosine"),
        // Routed through the switch even though the detection and segmentation models know exactly
        // what they need. The Go port initially built these two by hand and only noticed at M6 that
        // the dispatch design MAPPING.md calls the portable core was being bypassed by precisely the
        // models that use it — and a fourth port would have copied that verbatim.
        "YOLODetector" => new YoloDetector(output.LabelsAsStrings(), output.Iou ?? 0.45,
            output.Cls ?? 0.5, NmsMode.ClassAgnostic),
        "PerClassYOLODetector" => new YoloDetector(output.LabelsAsStrings(), output.Iou ?? 0.45,
            output.Cls ?? 0.5, NmsMode.PerClass),
        "YOLOOBBDetector" => new NotImplementedPostprocessor("YOLOOBBDetector"),
        "YOLOSegmentor" => new YoloSegmentor(output.MaskFilter ?? 0.5),
        "OCRProbs" => new OcrProbs(
            output.Alphabet ?? throw new InvalidDataException(
                "models: OCRProbs output declares no Alphabet"),
            root is null
                ? null
                : Alphabets.AllowedCharset(root, output.Script ?? "cyrillic", output.Country),
            output.BlankIndex ?? 0),
        _ => throw new InvalidDataException($"models: unknown output type \"{output.Type}\""),
    };

    /// <summary>
    /// The model wrapper by <c>ModelType</c>.
    ///
    /// <para>
    /// <c>UnifiedModel</c> covers everything the reference actually routes through it. The
    /// <c>"UnifedModel"</c> spelling is deliberate: it is the typo in the shipped
    /// <c>DocTypeAngles/model.json</c>, which worked in the reference only by falling through to a
    /// default. Accepting both is not politeness, it is the difference between loading the shipped
    /// artifact and not.
    /// </para>
    /// </summary>
    public static Model NewModel(ModelConfig config, Device device, int threads) =>
        config.ModelType switch
        {
            "UnifiedModel" or "UnifedModel" or "" => new Model(config, device, threads),
            _ => throw new InvalidDataException($"models: unknown ModelType \"{config.ModelType}\""),
        };

    /// <summary>Loads the config from a directory and builds the model.</summary>
    public static Model Load(string dir, Device device, int threads) =>
        NewModel(ModelConfig.Load(dir), device, threads);
}

/// <summary>
/// A model: one session, one preprocessor per input, one postprocessor per output.
/// </summary>
public sealed class Model : IDisposable
{
    private readonly Session _session;
    private readonly IPreprocessor[] _pre;
    private readonly IPostprocessor[] _post;

    public ModelConfig Config { get; }

    internal Model(ModelConfig config, Device device, int threads)
    {
        Config = config;
        _pre = [.. config.Inputs.Select(Loader.NewPreprocessor)];
        _post = [.. config.Outputs.Select(o => Loader.NewPostprocessor(o, config.Dir))];
        _session = new Session(config.ModelPath, device, threads);
    }

    /// <summary>
    /// Runs the model over one image and postprocesses every output.
    ///
    /// <para>
    /// Outputs are returned POSITIONALLY, matched to <c>Outputs[i]</c>. The session collects them by
    /// declared name, so the position here is the model's declaration order rather than whatever
    /// order ONNX Runtime happened to return — which matters for <c>DocTypeAngles</c>, where swapping
    /// two heads would silently produce a document type from an angle vector.
    /// </para>
    /// </summary>
    public IResult[] Predict(Image image)
    {
        // Single-input models only, which is every shipped artifact. A second input would need its
        // own preprocessor and a decision about which image feeds it; refusing is better than
        // guessing.
        if (_pre.Length != 1)
        {
            throw new InvalidDataException(
                $"models: {Config.Name} declares {_pre.Length} inputs, only 1 is supported");
        }

        (NdArray tensor, Meta meta) = _pre[0].Apply(image);
        NdArray[] raw = _session.Run([tensor]);

        if (raw.Length != _post.Length)
        {
            throw new InvalidDataException(
                $"models: {Config.Name} returned {raw.Length} outputs, config declares {_post.Length}");
        }

        var context = new Context
        {
            Ratio = meta.Ratio,
            PadExtra = meta.PadExtra,
            PadLetter = meta.PadLetter,
            PaddedH = meta.PaddedH,
            PaddedW = meta.PaddedW,
            OrigH = meta.OrigH,
            OrigW = meta.OrigW,
        };

        var results = new IResult[raw.Length];
        for (int i = 0; i < raw.Length; i++)
        {
            results[i] = _post[i].Apply(raw[i], context);
        }
        return results;
    }

    public void Dispose() => _session.Dispose();
}

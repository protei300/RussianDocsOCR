using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// Finds the registration stamps on the registration page of an internal passport (issue #27). Port of
/// <c>RegistrationStamps</c>.
///
/// <para>
/// A YOLO-OBB detector of the same family and input as <see cref="AddressLinesDetector"/>, run on the same
/// canvas. One oriented box per stamp, the whole impression with the signature and the unit code in it, classed
/// <c>registered</c> («Зарегистрирован», carries address lines), <c>deregistered</c> («Снят с регистрационного
/// учёта», carries none) or <c>unclear</c>. The stamp is LOCATED, not read: which address lines belong to it is
/// decided by the pipeline (a line lies inside its stamp).
/// </para>
/// </summary>
public sealed class RegistrationStamps : IDisposable
{
    /// <summary>The key in <c>models_path.yaml</c>, and the folder under <c>models/</c>.</summary>
    public const string ModelKey = "RegistrationStamps";

    /// <summary>Model label to the kind reported in the results (<c>KINDS</c>).</summary>
    public static readonly IReadOnlyDictionary<string, string> Kinds = new Dictionary<string, string>
    {
        ["Stamp_registered"] = "registered",
        ["Stamp_deregistered"] = "deregistered",
        ["Stamp_unclear"] = "unclear",
    };

    private readonly ObbDetectionModel _model;

    public RegistrationStamps(string root, IReadOnlyDictionary<string, string> paths, Device device, int threads)
        => _model = new ObbDetectionModel(
            Path.Combine(ModelPaths.Resolve(root, paths, ModelKey), "ONNX"), device, threads);

    /// <summary>
    /// True when this weight set carries the stamp detector. A set of models-v10 or older does not, and the
    /// reference then reads the address as before, without stamps - so the port asks BEFORE constructing.
    /// </summary>
    public static bool WeightsPresent(string root, IReadOnlyDictionary<string, string> paths)
    {
        if (!paths.ContainsKey(ModelKey))
        {
            return false;
        }
        string dir = Path.Combine(ModelPaths.Resolve(root, paths, ModelKey), "ONNX");
        return File.Exists(Path.Combine(dir, "model.json")) && File.Exists(Path.Combine(dir, "model.onnx"));
    }

    /// <summary>The oriented boxes of the stamps on the canvas.</summary>
    public List<ObbBox> Predict(Image canvas) => _model.Predict(canvas);

    public void Dispose() => _model.Dispose();
}

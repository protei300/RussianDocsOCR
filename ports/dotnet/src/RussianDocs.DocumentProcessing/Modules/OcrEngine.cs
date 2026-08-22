using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Preprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>Which OCR tier to load. Two artifacts per script, same graph shape.</summary>
public enum OcrTier
{
    Accurate,
    Fast,
}

/// <summary>
/// One OCR engine. **ONE type for both scripts**, with a <c>script</c> field.
///
/// <para>
/// D-11: the reference has <c>OCRCyrillic</c> and <c>OCRLatin</c> as separate classes, but they share
/// no state and override nothing — the only difference is which artifact they load and which
/// corrections they apply. Keeping two copies of the field lists in each of four languages is four
/// extra places for them to drift apart.
/// </para>
/// </summary>
public sealed class OcrEngine : IDisposable
{
    /// <summary>Cyrillic name fields, which get their leading dots stripped.</summary>
    private static readonly string[] RuNameFields =
    [
        "Last_name_ru", "First_name_ru", "Birth_place_ru", "Living_region_ru", "Middle_name_ru",
        "Issue_organization_ru",
    ];

    private static readonly string[] DateFields = ["Issue_date", "Expiration_date", "Birth_date"];

    /// <summary>
    /// The same normalization from the CYRILLIC engine, and the list is longer by the two parent
    /// birth dates of a birth certificate — which only ever reach this engine. NEITHER engine reads
    /// the printed separator of a digit date ('22.06.2010' comes back as '22/06/2010' from both), so
    /// the repair has to live on both routes; the Cyrillic one needs it since birth-certificate
    /// dates moved there for the 2018 blank, whose months are spelled out. Membership is by FIELD
    /// NAME, never by content: a series or a document number can hold eight digits too, and
    /// reformatting one as a date would be silent and wrong. Mirrors _DATE_FIELDS in ocr_cyrillic.py.
    /// </summary>
    private static readonly string[] CyrDateFields =
        ["Issue_date", "Birth_date", "Expiration_date", "Father_birth_date", "Mother_birth_date"];

    private readonly string _script;
    private readonly Session _session;
    private readonly IPreprocessor _pre;
    private readonly OcrProbs _decoder;

    public static OcrEngine Cyrillic(string root, IReadOnlyDictionary<string, string> paths,
        Device device, int threads, OcrTier tier) =>
        new(root, paths, device, threads, "cyrillic",
            tier == OcrTier.Accurate ? "OCRCyrillicAccurate" : "OCRCyrillicFast");

    public static OcrEngine Latin(string root, IReadOnlyDictionary<string, string> paths,
        Device device, int threads, OcrTier tier) =>
        new(root, paths, device, threads, "latin",
            tier == OcrTier.Accurate ? "OCRLatinAccurate" : "OCRLatinFast");

    private OcrEngine(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads, string script, string configKey)
    {
        _script = script;
        string dir = Path.Combine(ModelPaths.Resolve(root, paths, configKey), "ONNX");
        ModelConfig config = ModelConfig.Load(dir);

        _pre = Loader.NewPreprocessor(config.Inputs[0]);

        // From the switch, with `root` so it can resolve the ALLOWED charset from
        // ocr_alphabets.json. The model's FULL alphabet comes from model.json and the allowed subset
        // from that table — two different things, and passing the full alphabet as the allowed set
        // would disable masking with no error at all.
        _decoder = Loader.NewPostprocessor(config.Outputs[0], config.Dir, root) as OcrProbs
            ?? throw new InvalidDataException(
                $"modules: {configKey} output 0 is not an OCR decoder ({config.Outputs[0].Type})");

        _session = new Session(config.ModelPath, device, threads);
    }

    /// <summary>Decodes one word crop.</summary>
    public string Predict(Image word)
    {
        (NdArray tensor, Meta _) = _pre.Apply(word);
        NdArray[] raw = _session.Run([tensor]);
        return _decoder.Decode(raw[0]);
    }

    /// <summary>
    /// The per-field text corrections.
    ///
    /// <para>
    /// Dispatched on the SCRIPT first and the field name second, matching the reference. A Cyrillic
    /// engine never applies the date normaliser even to a date field, because dates are routed to the
    /// Latin engine — except in SNILS, where the month is a Russian word and the parity rule sends odd
    /// words to Cyrillic anyway.
    /// </para>
    /// </summary>
    public string FixErrors(string fieldType, string text)
    {
        if (_script == "cyrillic")
        {
            if (Array.IndexOf(CyrDateFields, fieldType) >= 0)
            {
                // Rewrites only when the text holds exactly eight digits, so a date spelled out in
                // words passes through — which is what lets one rule serve both birth-certificate
                // blanks and SNILS.
                return OcrCorrections.CheckDdmmyyyy(text);
            }
            if (fieldType == "Sex_ru")
            {
                return OcrCorrections.CheckRusSex(text);
            }
            return Array.IndexOf(RuNameFields, fieldType) >= 0
                ? OcrCorrections.StripEdgeDots(text)
                : text;
        }

        if (Array.IndexOf(DateFields, fieldType) >= 0)
        {
            // The reference wraps this in `except ValueError: return text`; CheckDdmmyyyy folds that
            // in by returning its input unchanged rather than raising.
            return OcrCorrections.CheckDdmmyyyy(text);
        }
        return fieldType switch
        {
            "Sex_en" => OcrCorrections.CheckEnSex(text),
            "Driver_class" => OcrCorrections.CheckDriverClass(text),
            _ => text,
        };
    }

    public void Dispose() => _session.Dispose();
}

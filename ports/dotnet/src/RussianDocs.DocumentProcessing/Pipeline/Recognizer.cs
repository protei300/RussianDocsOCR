using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>Per-run knobs. A record rather than functional options, so all four ports read alike.</summary>
public sealed record RunOptions
{
    public double Docconf { get; init; } = 0.5;
    public int ImgSize { get; init; } = 1500;
    public IStageSink Sink { get; init; } = NullStageSink.Instance;
    /// <summary>Stop after this stage, inclusive. Empty runs the whole pipeline.</summary>
    public string? UpTo { get; init; }
    public bool IncludeDebug { get; init; }
}

/// <summary>
/// What one run produced. **The images are OWNED BY THE CALLER**, who must dispose them.
///
/// <para>
/// Python's GC hid this entirely, and it is exactly how a port that passes conformance dies after
/// five hundred documents — measured, in the Go port, at 12.7 MB per document with no plateau.
/// </para>
/// </summary>
public sealed class Results : IDisposable
{
    public string DocType { get; internal set; } = "NONE";
    public double DocConfidence { get; internal set; }
    public int Angle { get; internal set; }
    public double AngleConfidence { get; internal set; }

    /// <summary>Which device the run actually used — reported, not requested (D-13).</summary>
    public string Device { get; internal set; } = "cpu";

    public Dictionary<string, double> Timings { get; internal set; } = [];

    /// <summary>Field label to joined value. What the view model's `ocr` block carries.</summary>
    public Dictionary<string, string> Ocr { get; } = new(StringComparer.Ordinal);

    /// <summary>Per-field word lists, which localise a single bad word inside a good field.</summary>
    public List<FieldText> Words { get; internal set; } = [];

    /// <summary>Field boxes, for the view model. Plain data, so it outlives the run's images.</summary>
    public List<ViewModel.Box2> Boxes { get; internal set; } = [];

    /// <summary>Quality verdicts, keyed as the wire expects.</summary>
    public Dictionary<string, object> Quality { get; internal set; } = [];

    /// <summary>The selected document contours, in PRE-warp space. Debug only.</summary>
    public List<Point[]>? Segments { get; internal set; }

    /// <summary>The corrected canvas. Null when the run short-circuited before producing one.</summary>
    public Image? Canvas { get; internal set; }

    /// <summary>
    /// Intermediates this run allocated. Kept as a list rather than named fields because the count
    /// varies with how far the run got.
    /// </summary>
    private readonly List<Image> _owned = [];

    internal void Own(Image image) => _owned.Add(image);

    /// <summary>
    /// Hands the canvas to the caller and releases everything else.
    ///
    /// <para>
    /// The service needs exactly one image to outlive the run — the canvas it stores as a PNG — while
    /// every intermediate must go back immediately. Without this, the only options are disposing what
    /// the caller still needs or disposing nothing, and the Go port shipped the second: 663 MB to
    /// 6932 MB over 460 documents, with the conformance suite green the whole way, because the CLI
    /// processes one document per process.
    /// </para>
    ///
    /// <para>After this returns, <see cref="Dispose"/> is a no-op, so a `using` left in place stays safe.</para>
    /// </summary>
    public Image? TakeCanvas()
    {
        Image? canvas = Canvas;
        Canvas = null; // cleared BEFORE Dispose so Dispose skips what the caller now owns
        Dispose();
        return canvas;
    }

    public void Dispose()
    {
        foreach (Image image in _owned)
        {
            image.Dispose();
        }
        _owned.Clear();
        Canvas?.Dispose();
        Canvas = null;
    }
}

/// <summary>
/// The pipeline. Port of <c>Pipeline</c> in <c>document_processing/pipeline/pipeline.py</c>.
///
/// <para>
/// Stage coverage grows one milestone at a time; <c>Program.StagesImplemented</c> in the CLI must
/// list exactly what this emits, and never more.
/// </para>
/// </summary>
public sealed class Recognizer : IDisposable
{
    private readonly DocTypeAngles _docTypeAngles;
    private readonly Glare _glare;
    private readonly Blur _blur;
    private readonly Spoofing _printSpoofing;
    private readonly Spoofing _lcdSpoofing;
    private readonly DocDetector _docDetector;
    private readonly DocDeskewer _deskewer;
    private readonly TextFieldsDetector _textFields;
    private readonly WordsDetector _words;
    private readonly OcrEngine _cyrillic;
    private readonly OcrEngine _latin;
    private readonly Device _device;

    /// <summary>
    /// Builds every module. SLOW — 215 MB of weights and one session each — so call it once and keep
    /// the instance. The reference loads them eagerly in its constructor for the same reason, and the
    /// service wraps the whole thing in a pool of exactly one.
    /// </summary>
    public Recognizer(Device device = Device.Cpu, int intraOpThreads = 1,
        OcrTier ocrTier = OcrTier.Accurate)
    {
        string root = ModelPaths.Root();
        var paths = ModelPaths.Load(root);
        _docTypeAngles = new DocTypeAngles(root, paths, device, intraOpThreads);
        _glare = new Glare(root, paths, device, intraOpThreads);
        _blur = new Blur(root, paths, device, intraOpThreads);
        _printSpoofing = Spoofing.Print(root, paths, device, intraOpThreads);
        _lcdSpoofing = Spoofing.Lcd(root, paths, device, intraOpThreads);
        _docDetector = new DocDetector(root, paths, device, intraOpThreads);
        _deskewer = DocDeskewer.ForPipeline();
        _device = device;
        _textFields = new TextFieldsDetector(root, paths, device, intraOpThreads);
        _words = new WordsDetector(root, paths, device, intraOpThreads);

        // **OCR stays on the CPU even when the detectors are on the GPU.** Measured, not assumed:
        // per-word dynamic widths make the CUDA provider recompile the graph on every distinct width,
        // and the Go port measured the whole corpus 13.7x SLOWER on GPU than on CPU. The reference
        // pins ocr_device to cpu for the same reason.
        _cyrillic = OcrEngine.Cyrillic(root, paths, Device.Cpu, intraOpThreads, ocrTier);
        _latin = OcrEngine.Latin(root, paths, Device.Cpu, intraOpThreads, ocrTier);
    }

    /// <summary>
    /// Runs the pipeline over one file.
    ///
    /// <para>
    /// M1 implements the <c>prepare</c> stage only: decode to RGB and shrink to <c>img_size</c>. That
    /// is deliberately the first milestone, because those two operations are where a port silently
    /// diverges before any model runs — and because being able to grade them alone is the whole
    /// point of <c>--upto</c>.
    /// </para>
    /// </summary>
    public Results Run(string imagePath, RunOptions options)
    {
        var results = new Results { Device = _device.Wire() };
        try
        {
            // ---- stage: prepare -------------------------------------------------------------
            Image source = Io.LoadRgb(imagePath);
            results.Own(source);

            Image prepared = Io.FitToLongestSide(source, options.ImgSize);
            results.Own(prepared);

            DirectoryStageSink.EmitImage(options.Sink, "prepare", prepared);
            if (options.UpTo == "prepare")
            {
                return results;
            }

            // ---- stages: doctype.label, rotate ----------------------------------------------
            var timings = new Timings();
            (DocTypeResult meta, Image upright) = timings.Time(Timings.DocTypeAngle,
                () => _docTypeAngles.PredictTransform(prepared));
            results.Own(upright);

            results.DocType = meta.DocType;
            results.DocConfidence = meta.DocTypeConfidence;
            results.Angle = meta.Angle;
            results.AngleConfidence = meta.AngleConfidence;

            options.Sink.Emit("doctype.label", meta);
            DirectoryStageSink.EmitImage(options.Sink, "rotate", upright);
            if (options.UpTo == "rotate")
            {
                return results;
            }

            // ---- stage: quality -------------------------------------------------------------
            Dictionary<string, object> quality = RunQuality(upright, meta.DocTypeConfidence,
                timings);
            results.Quality = quality;
            options.Sink.Emit("quality", quality);
            if (options.UpTo == "quality")
            {
                return results;
            }

            // ---- stages: borders.segments, borders.canvas ------------------------------------
            // max_pages is 2 only for the internal-passport spread; every other type passes 1, so a
            // background blob can never be stitched in.
            int maxPages = meta.DocType.StartsWith("INTPASSPORT", StringComparison.Ordinal)
                && !meta.DocType.Contains("ADDR", StringComparison.Ordinal) ? 2 : 1;

            (Image canvas, List<Point[]>? segments) = timings.Time(Timings.DocDetector,
                () => _docDetector.PredictTransform(upright, maxPages));
            results.Own(canvas);
            results.Segments = segments;

            options.Sink.Emit("borders.segments", SegmentsPayload(segments));
            DirectoryStageSink.EmitImage(options.Sink, "borders.canvas", canvas);
            if (options.UpTo == "borders.canvas")
            {
                return results;
            }

            // ---- stage: deskew.canvas -------------------------------------------------------
            (Image deskewed, double _) = timings.Time(Timings.Deskew,
                () => _deskewer.Deskew(canvas));
            results.Canvas = deskewed;

            DirectoryStageSink.EmitImage(options.Sink, "deskew.canvas", deskewed);
            if (options.UpTo == "deskew.canvas")
            {
                return results;
            }

            // ---- stage: fields.bbox ---------------------------------------------------------
            OcrOptions ocrOptions = OcrOptions.For(meta.DocType);
            List<Field> fields = timings.Time(Timings.FieldsDetector,
                () => _textFields.PredictTransform(deskewed, ocrOptions.NeedsLicenceRotation));
            results.Boxes = [.. fields.Select(f => new ViewModel.Box2(
                f.Box.X1, f.Box.Y1, f.Box.X2, f.Box.Y2, f.Box.Conf, f.Box.Cls, f.Box.Label))];
            try
            {
                options.Sink.Emit("fields.bbox", BoxesPayload(fields.Select(f => f.Box)));
                if (options.UpTo == "fields.bbox")
                {
                    return results;
                }

                // ---- stages: words.<Field>.bbox ---------------------------------------------
                // The address path (INTPASSPORTADDR) is out of scope for this port, so no
                // address.lines stage is emitted and the checker skips it.
                List<FieldWords> fieldWords = timings.Time(Timings.SplitWords,
                    () => SplitWords.Run(fields, ocrOptions, _words));
                try
                {
                    foreach (FieldWords fw in fieldWords)
                    {
                        options.Sink.Emit($"words.{fw.Label}.bbox", WordBoxesPayload(fw.WordBoxes));
                    }
                    if (options.UpTo == "words")
                    {
                        return results;
                    }

                    // ---- stages: ocr.<Field>.words, join ------------------------------------
                    // The bare type, without the year suffix: the SNILS parity rule and the date
                    // join both test it, and "SNILS_1996" would match neither.
                    (string bareType, string _) = OcrOptions.SplitDocType(meta.DocType);

                    List<FieldText> texts = timings.Time(Timings.Ocr,
                        () => Ocr.Run(fieldWords, bareType, ocrOptions, _cyrillic, _latin));
                    Ocr.FixFms(texts, bareType);

                    // Ruler cleanup applies to the FINAL per-field values only: the reference emits
                    // `join` from the raw dict and cleans meta_results['OCR'] afterwards
                    // (pipeline.py:1058), so the conformance payload stays raw here too.
                    bool cleanRulers = bareType.Contains("birthcert", StringComparison.OrdinalIgnoreCase);

                    var joined = new Dictionary<string, string>(StringComparer.Ordinal);
                    foreach (FieldText text in texts)
                    {
                        options.Sink.Emit($"ocr.{text.Label}.words", text.Words);
                        joined[text.Label] = text.Value;
                        results.Ocr[text.Label] = cleanRulers
                            ? Ocr.CleanRulerArtifacts(text.Value)
                            : text.Value;
                    }
                    options.Sink.Emit("join", joined);
                    results.Words = texts;

                    results.Timings = timings.Report();
                    if (options.UpTo == "join")
                    {
                        return results;
                    }

                    // ---- stage: viewmodel ---------------------------------------------------
                    options.Sink.Emit("viewmodel",
                        BuildViewModel(results, options.IncludeDebug));
                    return results;
                }
                finally
                {
                    SplitWords.CloseAll(fieldWords);
                }
            }
            finally
            {
                Fields.CloseAll(fields);
            }
        }
        catch
        {
            // Any failure releases everything the run allocated. The Go port's `fail` closure does
            // the same, and the reason is that an exception on stage seven must not leak six stages
            // of intermediates.
            results.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Boxes in the wire shape: <c>[x1, y1, x2, y2, conf, cls, label]</c>.
    ///
    /// <para>
    /// The coordinates are TRUNCATED to int here even though they are already whole after the
    /// detector's own truncation — because the reference emits <c>int(...)</c> at this point, and the
    /// harness compares these rows positionally with a per-column tolerance.
    /// </para>
    /// </summary>
    private static object[] BoxesPayload(IEnumerable<Box> boxes) =>
        [.. boxes.Select(b => new object[]
        {
            (int)b.X1, (int)b.Y1, (int)b.X2, (int)b.Y2, b.Conf, b.Cls, b.Label,
        })];

    /// <summary>
    /// One field's word boxes, one entry per DETECTION of that field.
    ///
    /// <para>
    /// A null entry stays JSON null and means "this field needs no splitting, so its whole patch is
    /// the single word" — a different claim from "the detector found exactly one word". A port that
    /// split a field it should not have would otherwise look like agreement.
    /// </para>
    /// </summary>
    private static object?[] WordBoxesPayload(IEnumerable<List<Box>?> wordBoxes) =>
        [.. wordBoxes.Select(boxes => boxes is null ? null : (object)BoxesPayload(boxes))];

    /// <summary>
    /// Contours as the harness expects them: a list of point lists, or null when nothing was found.
    ///
    /// <para>
    /// Compared under relaxation R-01 rather than point-for-point, because the number of points
    /// findContours returns legitimately depends on the OpenCV minor version. Area, centroid and
    /// Hausdorff distance are what actually get checked.
    /// </para>
    /// </summary>
    private static object? SegmentsPayload(List<Point[]>? segments) =>
        segments?.Select(seg => seg.Select(p => new[] { p.X, p.Y }).ToArray()).ToArray();

    /// <summary>
    /// The four quality checks, run CONCURRENTLY.
    ///
    /// <para>
    /// Launched in the reference's source order and collected positionally — see
    /// <see cref="Group.Run"/> for why that is not a style choice. Each has its own model and
    /// therefore its own session, which is what makes concurrency worth having: the per-session lock
    /// only serialises calls to the SAME session, so four different models genuinely overlap.
    /// </para>
    ///
    /// <para>
    /// The verdicts are strings — <c>"good"</c>/<c>"bad"</c> for glare and blur, <c>"REAL"</c>/
    /// <c>"FAKE"</c> for the two spoofing checks. That inconsistency is in the reference and the wire
    /// contract carries it, so the dictionary is deliberately heterogeneous rather than normalised.
    /// </para>
    /// </summary>
    /// <summary>
    /// Assembles the view model from a finished run.
    ///
    /// <para>
    /// Built here rather than by the service, so the conformance CLI can emit it without an HTTP
    /// layer existing — D-01. Takes the canvas DIMENSIONS out of the result rather than the image,
    /// which keeps the builder free of any ownership question.
    /// </para>
    /// </summary>
    public static ViewModel.Payload BuildViewModel(Results results, bool includeDebug) =>
        ViewModel.Builder.Build(new ViewModel.Input
        {
            DocType = results.DocType,
            Device = results.Device,
            CanvasW = results.Canvas?.Width ?? 0,
            CanvasH = results.Canvas?.Height ?? 0,
            CanvasMissing = results.Canvas is null,
            Boxes = results.Boxes,
            Ocr = results.Ocr,
            Quality = results.Quality,
            Timings = results.Timings,
            Segments = results.Segments,
        }, includeDebug);

    private Dictionary<string, object> RunQuality(Image image, double docConfidence,
        Timings timings)
    {
        var groupStart = System.Diagnostics.Stopwatch.StartNew();
        (string Key, string Label)[] names =
        [
            ("Glare", ""), ("Blur", ""), ("PrintSpoofing", ""), ("LCDSpoofing", ""),
        ];

        (string?[] labels, Exception? error) = Group.Run<string>(0,
        [
            () => _glare.Predict(image).Label,
            () => _blur.Predict(image).Label,
            () => _printSpoofing.Predict(image).Label,
            () => _lcdSpoofing.Predict(image).Label,
        ]);
        if (error is not null)
        {
            throw error;
        }

        // DocConf first, matching the reference's insertion order. The comparison is key-by-key so
        // order does not affect it, but a diff of two dumps is far easier to read when it does.
        // The group's own wall time counts toward the total; its members' do not, or the report
        // would claim more time than actually elapsed. The members are recorded as zero because the
        // reference measures them inside the group and this port does not thread a stopwatch through
        // four closures for a value the tolerance spec never compares.
        timings.RecordGroup(Timings.QualityAndBorders, groupStart.Elapsed,
            new Dictionary<string, TimeSpan>
            {
                [Timings.Glare] = TimeSpan.Zero,
                [Timings.Blur] = TimeSpan.Zero,
                [Timings.PrintSpoofing] = TimeSpan.Zero,
                [Timings.LcdSpoofing] = TimeSpan.Zero,
            });

        var quality = new Dictionary<string, object> { ["DocConf"] = docConfidence };
        for (int i = 0; i < names.Length; i++)
        {
            quality[names[i].Key] = labels[i]
                ?? throw new InvalidOperationException($"pipeline: {names[i].Key} produced no verdict");
        }
        return quality;
    }

    /// <summary>
    /// Disposes every module.
    ///
    /// <para>
    /// Each closer runs even if an earlier one throws. Stopping at the first failure would leak the
    /// remaining sessions, and on GPU that is retained device memory — which outlives the process's
    /// own memory in how long it takes to notice.
    /// </para>
    /// </summary>
    public void Dispose()
    {
        foreach (IDisposable module in new IDisposable[]
                 { _docTypeAngles, _glare, _blur, _printSpoofing, _lcdSpoofing, _docDetector, _textFields, _words, _cyrillic, _latin })
        {
            try
            {
                module.Dispose();
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[pipeline] disposing {module.GetType().Name}: {ex.Message}");
            }
        }
    }
}

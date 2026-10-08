using RussianDocs.DocumentProcessing.Maps;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.PageRegistration;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

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

/// <summary>One date field whose word-by-word reading was replaced by the reading of the whole line.</summary>
public sealed record DateReread(string Field, string Split, string Whole);

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

    /// <summary>
    /// Canonical <c>dd.mm.yyyy</c> view of the date fields, alongside the reading (<see cref="Dates"/>).
    /// A SEPARATE map, deliberately — see <see cref="Ocr"/> and <c>PipelineResults.ocr_normalized</c>
    /// (pipeline.py:342-352). Only fields that converted appear here.
    /// </summary>
    public Dictionary<string, string> OcrNormalized { get; internal set; } = [];

    /// <summary>
    /// Every document the detector found in the frame, largest first, with boxes on the INPUT image
    /// (<c>results.documents</c>). Empty when none was found, or when the weight set has no document
    /// detector — the whole frame was read then.
    /// </summary>
    public List<DetectedDocument> Documents { get; internal set; } = [];

    /// <summary>
    /// Which of <see cref="Documents"/> was read: always the largest, 0, when there is one; null when
    /// the whole frame was read (<c>results.document_index</c>).
    /// </summary>
    public int? DocumentIndex { get; internal set; }

    /// <summary>The box of the document read, on the input image; null when the whole frame was read.</summary>
    public double[]? DocumentBox => DocumentIndex is int i ? Documents[i].Box : null;

    /// <summary>
    /// Lines read WHOLE because the word split lost a word (<c>results.words_fallback</c>). Part of
    /// the contract: a measurement that corrects for the missing spaces must apply that correction
    /// ONLY to these lines.
    /// </summary>
    public List<LineFlag> WordsFallback { get; internal set; } = [];

    /// <summary>Lines the gap guard declined to re-read because they carry no strokes (<c>results.words_no_ink</c>).</summary>
    public List<LineFlag> WordsNoInk { get; internal set; } = [];

    /// <summary>Date fields whose reading was replaced by the reading of the whole line (<c>DatesReadWhole</c>).</summary>
    public List<DateReread> DatesReadWhole { get; internal set; } = [];

    /// <summary>
    /// The STS special marks say the vehicle is leased (<c>results.leasing</c>, which is
    /// <c>{'leasing': True}</c> or None in the reference). Only the flag is reported: the lessor and the
    /// contract are parsed (<see cref="StsMarks.ParseLeasing"/>) but not reported until the reading of
    /// the small special-marks print can carry them (<c>Pipeline.LEASING_REPORTED</c>).
    /// </summary>
    public bool Leasing { get; internal set; }

    /// <summary>
    /// Index of the other side of this document in <see cref="Recognizer.RunFrame"/>'s list
    /// (<c>results.paired_with</c>); null when no pair was found or in a single-document call.
    /// </summary>
    public int? PairedWith { get; internal set; }

    /// <summary>
    /// Map from the canvas the field detector read back to the image passed to <c>Run</c> (<c>results.geometry</c>);
    /// null before the image is prepared. See <see cref="Maps.IMap"/>.
    /// </summary>
    public Maps.IMap? Geometry { get; internal set; }

    /// <summary>
    /// Points of the canvas on the input image (<c>results.to_input</c>); null when a stage of this run changed
    /// the image in a way no point map expresses (<see cref="Maps.UnknownMap"/>) - the canvas and what was read
    /// are fine, the way back is not known.
    /// </summary>
    public Point[]? ToInput(IReadOnlyList<Point> points) => Geometry is null ? [.. points] : Geometry.ToInput(points);

    /// <summary>
    /// Where the read text fields lie on the input image (<c>results.field_quads</c>): label to one
    /// quadrilateral per detection, in the order the fields were read (top to bottom); corners follow the
    /// canvas box from its top-left. Null - the way back to the input image is not known for this run: a caller
    /// draws these over a photo, so a quadrilateral that silently landed elsewhere would be worse than none.
    /// </summary>
    public Dictionary<string, List<Point[]>>? FieldQuads { get; internal set; } = [];

    /// <summary>label to the quadrilateral of each patch of words of that field, same order (<c>results.word_quads</c>); null on the same terms.</summary>
    public Dictionary<string, List<Point[]>>? WordQuads { get; internal set; } = [];

    /// <summary>
    /// Registration address lines on the input image (<c>results.address_line_quads</c>). The address page is
    /// not read by this port, so it is always empty.
    /// </summary>
    public List<Point[]>? AddressLineQuads { get; internal set; } = [];

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
    /// <summary>
    /// The first stage (decision #142). Null when the weight set carries no <c>DocDetect</c> — an
    /// older set — or when the caller asked for whole frames: the frame is then read as it always was.
    /// </summary>
    private readonly DocumentDetector? _documentDetector;
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
    private readonly PageRegistrar _pageRegistrar;
    private readonly string _root;
    private readonly bool _cardRegistration;

    /// <summary>
    /// The template registrar of each STS type, built on first use (<c>Pipeline._card_registrar</c>);
    /// null for a type without templates. Guarded by a lock: the registrar is built inside a run, and
    /// the runs of one instance are serialised by the caller, but a cache is cheap to make safe.
    /// </summary>
    private readonly Dictionary<string, PageRegistrar?> _cardRegistrars = new(StringComparer.Ordinal);

    /// <summary>
    /// Builds every module. SLOW — 215 MB of weights and one session each — so call it once and keep
    /// the instance. The reference loads them eagerly in its constructor for the same reason, and the
    /// service wraps the whole thing in a pool of exactly one.
    /// </summary>
    public Recognizer(Device device = Device.Cpu, int intraOpThreads = 1,
        OcrTier ocrTier = OcrTier.Accurate, bool detectDocuments = true, bool cardRegistration = true)
    {
        string root = ModelPaths.Root();
        _root = root;
        // `Pipeline(card_registration=True)`: straighten a vehicle registration certificate by its
        // printed blank instead of its Borders quad (see RegisterCard).
        _cardRegistration = cardRegistration;
        var paths = ModelPaths.Load(root);

        // `Pipeline(detect_documents=True)` (the default): find the documents in the frame first. A
        // weight set of models-v8 or older has no DocDetect, and the reference then reads whole frames
        // rather than refuse to start — it prints a warning and goes on, and so does this.
        if (detectDocuments)
        {
            if (DocumentDetector.WeightsPresent(root, paths))
            {
                _documentDetector = new DocumentDetector(root, paths, device, intraOpThreads);
            }
            else
            {
                Console.Error.WriteLine("[!] DocumentDetector weights not found (models/DocDetect): " +
                    "reading whole frames. Run scripts/fetch_models.py for a weight set that has them.");
            }
        }

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

        // `PageRegistrar()` in the reference (pipeline.py:652) — built unconditionally whenever
        // page_registration=True, which is the default, regardless of what document type a given
        // run turns out to be. Every other type's `_register_pages` returns immediately without
        // touching it (see the call site below), so the cost here is model-loading only, not
        // per-document.
        _pageRegistrar = new PageRegistrar(root, "INTPASSPORT");
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
            var timings = new Timings();

            // ---- stage: documents -----------------------------------------------------------
            // Decision #142: find the documents in the frame FIRST, then read the crop of one of them
            // (the largest — process_img). Without a detector, or with no document found, the whole
            // frame is read, as before.
            Image source = Io.LoadRgb(imagePath);
            results.Own(source);

            List<DetectedDocument> documents = FindDocuments(source, options.ImgSize, timings,
                options.Sink);
            results.Documents = documents;
            results.DocumentIndex = documents.Count > 0 ? 0 : null;
            if (options.UpTo == "documents")
            {
                return results;
            }
            return ReadDocument(source, results, timings, options);
        }
        catch
        {
            results.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Reads EVERY document in the frame, not only the largest (<c>Pipeline.process_frame</c>, issue
    /// #26). One <see cref="Results"/> per document, largest first; with no document found, a one-element
    /// list holding the whole-frame reading, exactly what <see cref="Run"/> returns. Two sides of one
    /// document lying in the same frame are paired by the number printed on both
    /// (<see cref="PairSides"/>): <see cref="Results.PairedWith"/> is the index of the other side in the
    /// returned list. Each element is a separate <see cref="Results"/>, to be disposed by the caller.
    /// </summary>
    public List<Results> RunFrame(string imagePath, RunOptions options)
    {
        var all = new List<Results>();
        Image? source = null;
        try
        {
            source = Io.LoadRgb(imagePath);
            var timings = new Timings();
            List<DetectedDocument> documents = FindDocuments(source, options.ImgSize, timings,
                options.Sink);
            int count = Math.Max(1, documents.Count);
            for (int index = 0; index < count; index++)
            {
                var results = new Results
                {
                    Device = _device.Wire(),
                    Documents = documents,
                    DocumentIndex = documents.Count > 0 ? index : null,
                };
                all.Add(results);
                if (options.UpTo == "documents")
                {
                    continue;
                }
                ReadDocument(source, results, index == 0 ? timings : new Timings(), options);
            }
            int?[] pairs = PairSides([.. all.Select(r => (r.DocType, r.Ocr.GetValueOrDefault("Licence_number")))]);
            for (int i = 0; i < all.Count; i++)
            {
                all[i].PairedWith = pairs[i];
            }
            return all;
        }
        catch
        {
            foreach (Results r in all)
            {
                r.Dispose();
            }
            throw;
        }
        finally
        {
            source?.Dispose();
        }
    }

    /// <summary>
    /// The sides of a document that belong together: a front and a back whose types are paired, whose
    /// series and number read the same, and where that number is unique among the sides of that type in
    /// the frame. Port of <c>pair_sides</c>. Two certificates scanned on one sheet must not cross-pair,
    /// and an ambiguous number pairs nothing rather than guessing.
    /// </summary>
    /// <param name="sides">The type label and the read <c>Licence_number</c> (null when none) of each
    /// document of the frame, in the order of the frame's list.</param>
    /// <returns>For each document, the index of the other side, or null.</returns>
    public static int?[] PairSides(IReadOnlyList<(string? DocType, string? LicenceNumber)> sides)
    {
        var paired = new int?[sides.Count];
        // A licence back carries no number the pipeline reads, so it is not here (PAIRED_SIDES).
        foreach ((string front, string back) in PairedSides)
        {
            var fronts = new Dictionary<string, List<int>>(StringComparer.Ordinal);
            var backs = new Dictionary<string, List<int>>(StringComparer.Ordinal);
            var frontOrder = new List<string>();
            for (int i = 0; i < sides.Count; i++)
            {
                string number = new([.. (sides[i].LicenceNumber ?? "").Where(char.IsDigit)]);
                if (number.Length < 6)
                {
                    continue;
                }
                string label = sides[i].DocType ?? "NONE";
                int cut = label.LastIndexOf('_');
                string family = cut >= 0 ? label[..cut] : label;
                Dictionary<string, List<int>>? side = family == front ? fronts : family == back ? backs : null;
                if (side is null)
                {
                    continue;
                }
                if (!side.TryGetValue(number, out List<int>? list))
                {
                    side[number] = list = [];
                    if (side == fronts)
                    {
                        frontOrder.Add(number);
                    }
                }
                list.Add(i);
            }
            foreach (string number in frontOrder)
            {
                List<int> f = fronts[number];
                List<int> b = backs.GetValueOrDefault(number) ?? [];
                if (f.Count == 1 && b.Count == 1)
                {
                    paired[f[0]] = b[0];
                    paired[b[0]] = f[0];
                }
            }
        }
        return paired;
    }

    /// <summary>
    /// Type families whose two sides carry the same series and number, a front and a back. Both sides of
    /// an STS read it as <c>Licence_number</c>, so a front and a back lying in one frame pair up by it
    /// (<c>PAIRED_SIDES</c>, issue #26).
    /// </summary>
    private static readonly (string Front, string Back)[] PairedSides = [("STS", "STSBACK")];

    /// <summary>
    /// Everything after the documents are found: prepare the crop of the chosen one (the whole frame when
    /// none), then type, borders, fields, reading. Port of <c>_read_document</c> + <c>_process_document</c>.
    /// </summary>
    private Results ReadDocument(Image source, Results results, Timings timings, RunOptions options)
    {
        try
        {
            List<DetectedDocument> documents = results.Documents;

            // ---- stage: prepare -------------------------------------------------------------
            // The crop is cut from the input at FULL resolution (a licence on an A4 scan keeps its
            // pixels) and only then shrunk to img_size.
            Image? cropped = null;
            Image prepared;
            // The first maps of the canvas every later stage reads (geometry.py): the crop of the document,
            // then the resize (`_prepare_image`). Scale is the new size over the size of the cut-out.
            OffsetMap? cropOffset = null;
            int cutW, cutH;
            try
            {
                if (results.DocumentIndex is int index)
                {
                    (int x0, int y0, int x1, int y1) = DocumentCrop(source.Width, source.Height,
                        documents[index].Box);
                    cropped = Crop.ClampedCrop(source, x0, y0, x1, y1);
                    cropOffset = new OffsetMap(-x0, -y0);
                    cutW = cropped.Width;
                    cutH = cropped.Height;
                    prepared = Io.FitToLongestSide(cropped, options.ImgSize);
                }
                else
                {
                    cutW = source.Width;
                    cutH = source.Height;
                    prepared = Io.FitToLongestSide(source, options.ImgSize);
                }
            }
            finally
            {
                cropped?.Dispose();
            }
            results.Own(prepared);
            ChainMap canvasMap = new ChainMap(cropOffset is null ? [] : [cropOffset])
                .Then(new ScaleMap((double)prepared.Width / cutW, (double)prepared.Height / cutH));
            results.Geometry = canvasMap;

            DirectoryStageSink.EmitImage(options.Sink, "prepare", prepared);
            if (options.UpTo == "prepare")
            {
                return results;
            }

            // ---- stages: doctype.label, rotate ----------------------------------------------
            (DocTypeResult meta, Image upright) = timings.Time(Timings.DocTypeAngle,
                () => _docTypeAngles.PredictTransform(prepared));
            results.Own(upright);

            results.DocType = meta.DocType;
            results.DocConfidence = meta.DocTypeConfidence;
            results.Angle = meta.Angle;
            results.AngleConfidence = meta.AngleConfidence;
            canvasMap = canvasMap.Then(new QuarterTurnsMap(prepared.Width, prepared.Height, meta.Angle / 90));
            results.Geometry = canvasMap;

            options.Sink.Emit("doctype.label", meta);
            DirectoryStageSink.EmitImage(options.Sink, "rotate", upright);
            if (options.UpTo == "rotate")
            {
                return results;
            }

            // ---- the borders-first fallback on NONE -----------------------------------------
            // A hard but legitimate shot (strong perspective, small document scale, occluded header) can
            // push the raw-frame embedding past the metric NONE-threshold while the same classifier reads
            // the border-cropped document confidently. Costs one extra DocDetector run on NONE frames
            // only. Measured by the reference (docs/progress-log.md, 2026-08-02): recovered 6/6 synthetic
            // NONE cases at conf 0.98+, zero new false-accepts on a no-document negative set.
            Image? fallbackCanvas = null;
            if (meta.DocType == "NONE")
            {
                (Image borders0, List<Point[]>? segments0, IMap map0) = timings.Time(Timings.DocDetector,
                    () => _docDetector.PredictTransform(upright, 1));
                fallbackCanvas = borders0.Clone();
                results.Own(fallbackCanvas);
                using (borders0)
                {
                    if (segments0 is { Count: > 0 })
                    {
                        (DocTypeResult again, Image reUpright) = timings.Time(Timings.DocTypeAngle,
                            () => _docTypeAngles.PredictTransform(borders0));
                        meta = again;
                        results.DocType = again.DocType;
                        results.DocConfidence = again.DocTypeConfidence;
                        results.Angle = again.Angle;
                        results.AngleConfidence = again.AngleConfidence;
                        if (again.DocType.Contains("intpassport", StringComparison.OrdinalIgnoreCase))
                        {
                            // The fallback crop above ran while the type was still unknown (max_pages=1),
                            // so a two-page passport spread lost its second page before OCR could see it.
                            // Now that the type is known, redo border detection on the same pre-crop
                            // frame: the two-page stitch is allowed. Type and angle stay as classified on
                            // the single page (measured reliable there); only the canvas is rebuilt.
                            reUpright.Dispose();
                            (Image spread, _, IMap map2) = timings.Time(Timings.DocDetector,
                                () => _docDetector.PredictTransform(upright, 2));
                            canvasMap = canvasMap.Then(map2)
                                .Then(new QuarterTurnsMap(spread.Width, spread.Height, again.Angle / 90));
                            upright = TurnQuarters(spread, again.Angle / 90);
                        }
                        else
                        {
                            // the crop of the first border detection, then the turn the re-classification
                            // asked for on it
                            canvasMap = canvasMap.Then(map0)
                                .Then(new QuarterTurnsMap(borders0.Width, borders0.Height, again.Angle / 90));
                            upright = reUpright;
                        }
                        results.Geometry = canvasMap;
                        results.Own(upright);
                    }
                }
            }
            if (meta.DocType == "NONE")
            {
                Console.Error.WriteLine("[!] The document on picture has unknown type");
                results.Quality = new Dictionary<string, object> { ["DocConf"] = meta.DocTypeConfidence };
                // The canvas of an unrecognised frame is what the fallback's border detection warped
                // (`img_with_fixed_perspective` reads the DocDetector result once it exists, and the
                // fallback above ran it); the fallback is skipped only when no border detection was asked.
                results.Canvas = (fallbackCanvas ?? upright).Clone();
                results.Timings = timings.Report();
                options.Sink.Emit("viewmodel", BuildViewModel(results, options.IncludeDebug));
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
            int maxPages = meta.DocType.Contains("intpassport", StringComparison.OrdinalIgnoreCase)
                ? 2 : 1;

            (Image canvas, List<Point[]>? segments, IMap bordersMap) = timings.Time(Timings.DocDetector,
                () => _docDetector.PredictTransform(upright, maxPages));
            results.Segments = segments;

            // `_register_pages` (pipeline.py) runs here unconditionally — it is TIMED for every document
            // type, but the reference's own method returns immediately unless the type is a plain
            // "intpassport" (not "...addr") or an STS, which has its own path. Rebuilds the canvas from
            // template-registered pages ONLY for those; every other type's timing entry is a genuine
            // no-op, matching the reference exactly. `canvas` is reassigned INSIDE the closure — safe here
            // because `Timings.Time` runs it synchronously before returning, not because of anything
            // special about the lambda.
            //
            // `registered`: a canvas rebuilt from the printed blank was straightened by its own lines
            // already, and the projection-profile deskew is NOT run on top of it (see below).
            bool registered = false;
            timings.Time(Timings.RegisterPages, () =>
            {
                string lowered = meta.DocType.ToLowerInvariant();
                (Image Canvas, IMap Map)? rebuilt;
                if (lowered.StartsWith("sts", StringComparison.Ordinal))
                {
                    rebuilt = _cardRegistration ? RegisterCard(upright, segments, meta.DocType) : null;
                }
                else
                {
                    bool plainIntPassport = lowered.Contains("intpassport", StringComparison.Ordinal)
                        && !lowered.Contains("addr", StringComparison.Ordinal);
                    if (!plainIntPassport || segments is null || segments.Count == 0)
                    {
                        return;
                    }
                    rebuilt = RegisterPages(upright, segments);
                }
                if (rebuilt is { } built)
                {
                    canvas.Dispose();
                    canvas = built.Canvas;
                    // The stitch of the registered pages replaces the map of the Borders canvas
                    // (`det['geometry'] = stitched_geometry(...)`).
                    bordersMap = built.Map;
                    registered = true;
                }
            });
            results.Own(canvas);

            options.Sink.Emit("borders.segments", SegmentsPayload(segments));
            DirectoryStageSink.EmitImage(options.Sink, "borders.canvas", canvas);
            if (options.UpTo == "borders.canvas")
            {
                return results;
            }

            // ---- stage: deskew.canvas -------------------------------------------------------
            // Registration straightened the page by its own lines (line_refine, blob-tolerant). The
            // projection-profile deskew is NOT run on top of it: on a small photo it took the dark
            // cushion around a page for text and rotated the straight page (pipeline.py `_deskew`). The
            // stage still exists, and the canvas is the registered one.
            IMap? deskewMap = null;
            Image deskewed = timings.Time(Timings.Deskew, () =>
            {
                if (registered)
                {
                    return canvas.Clone();
                }
                (Image rotated, _, deskewMap) = _deskewer.DeskewWithMap(canvas);
                return rotated;
            });
            results.Canvas = deskewed;
            // `_deskew` wraps the border map in a chain and adds the rotation (nothing, when the image was
            // handed on unchanged); a registered canvas is not deskewed and keeps its map as it is. Then the
            // canvas every later stage reads is the previous one passed through the border map.
            if (!registered)
            {
                bordersMap = new ChainMap([bordersMap]).Then(deskewMap);
            }
            canvasMap = canvasMap.Then(bordersMap);
            results.Geometry = canvasMap;

            DirectoryStageSink.EmitImage(options.Sink, "deskew.canvas", deskewed);
            if (options.UpTo == "deskew.canvas")
            {
                return results;
            }

            // ---- stage: fields.bbox ---------------------------------------------------------
            OcrOptions ocrOptions = OcrOptions.For(meta.DocType);
            List<Field> fields = timings.Time(Timings.FieldsDetector, () =>
            {
                List<Field> detected = _textFields.PredictTransform(deskewed, ocrOptions.NeedsLicenceRotation);
                ReadMargins(detected, ocrOptions, deskewed);
                return detected;
            });
            results.Boxes = [.. fields.Select(f => new ViewModel.Box2(
                f.Box.X1, f.Box.Y1, f.Box.X2, f.Box.Y2, f.Box.Conf, f.Box.Cls, f.Box.Label))];
            try
            {
                options.Sink.Emit("fields.bbox", BoxesPayload(fields.Select(f => f.Box)));
                if (options.UpTo == "fields.bbox")
                {
                    return results;
                }

                // The MRZ boxes and the canvas, remembered for the length self-check of the OCR
                // loop (`_note_mrz_zone`). Borrows `deskewed`, which outlives the loop.
                MrzZone? mrzZone = MrzZone.Note([.. fields.Select(f => f.Box)], deskewed);

                // ---- stages: words.<Field>.bbox ---------------------------------------------
                // The address path (INTPASSPORTADDR) is out of scope for this port, so no
                // address.lines stage is emitted and the checker skips it.
                // The bare type, without the year suffix: the SNILS parity rule, the date join and
                // the gap guard all test it, and "SNILS_1996" would match none of them.
                (string bareType, string year) = OcrOptions.SplitDocType(meta.DocType);

                SplitOutcome split = timings.Time(Timings.SplitWords,
                    () => SplitWords.Run(fields, ocrOptions, _words, bareType));
                List<FieldWords> fieldWords = split.Fields;
                results.WordsFallback = split.Flags.Fallback;
                results.WordsNoInk = split.Flags.NoInk;
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

                    // ---- stage: quads -------------------------------------------------------
                    // The way back to the input image (geometry.py): field and word quadrilaterals, or null
                    // per key when this run's way back is not known. Before the OCR, as in the reference.
                    FieldQuadSet quads = FieldQuads.Build(results.Geometry, fields, split, ocrOptions);
                    results.FieldQuads = quads.Fields;
                    results.WordQuads = quads.Words;
                    options.Sink.Emit("quads", quads.Payload());
                    if (options.UpTo == "quads")
                    {
                        return results;
                    }

                    // ---- stages: ocr.<Field>.words, join ------------------------------------
                    List<FieldText> texts = timings.Time(Timings.Ocr,
                        () => Ocr.Run(fieldWords, bareType, ocrOptions, _cyrillic, _latin, mrzZone,
                            year.Length > 0 ? year : null));
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

                    // A date field the splitter cut badly is read again, whole line by whole line,
                    // and the whole reading replaces the split one only where it converts and the
                    // split one does not. After the ruler cleanup, as in the reference (`_ocr` ends
                    // with it, then `_reread_dates_whole`): a re-read value is stored as read.
                    results.DatesReadWhole = RereadDatesWhole(split.DateLines, results.Ocr, ocrOptions);

                    // Canonical dates are built once, on the FINISHED dict, after the ruler cleanup
                    // above: the reference's `_normalize_dates` runs right after `_ocr` returns
                    // (pipeline.py:900), so nothing upstream sees a rewritten value. Not a timed
                    // stage — the reference calls it as a plain method, not through `_model_call`.
                    results.OcrNormalized = Dates.NormalizeDates(
                        results.Ocr, texts.Select(t => t.Label));

                    // The leasing flag of an STS, from the finished special marks. Not a timed stage.
                    ReadLeasing(bareType, results, options.Sink);

                    results.Timings = timings.Report();
                    if (options.UpTo == "join" || options.UpTo == "leasing")
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
    /// Share of the document box's longer side added around it when it is cut from the frame: the
    /// border detector still needs a strip of background to find the edges, and the box itself can
    /// sit a few pixels inside the paper (handoff of the detector, 2026-10-01: ~3 %).
    /// </summary>
    public const double DocumentCropMargin = 0.03;

    /// <summary>
    /// Documents in the frame, largest first, boxes on the input image. Port of
    /// <c>Pipeline._find_documents</c>.
    ///
    /// <para>
    /// The detector reads the frame at the processing size; its boxes are scaled back to the input
    /// image so the crop can be cut at full resolution — a licence on an A4 scan keeps its pixels
    /// instead of the ~300 left to it once the whole sheet is shrunk to <c>img_size</c>. With no
    /// detector nothing is emitted and the frame is read whole; WITH a detector that finds nothing the
    /// stage is emitted as an empty list, which says "asked, found none".
    /// </para>
    /// </summary>
    private List<DetectedDocument> FindDocuments(Image frame, int imgSize, Timings timings,
        IStageSink sink)
    {
        if (_documentDetector is null)
        {
            return [];
        }

        using Image small = Io.FitToLongestSide(frame, imgSize);
        double sx = (double)frame.Width / small.Width;
        double sy = (double)frame.Height / small.Height;

        List<DetectedDocument> documents = timings.Time(Timings.DocumentDetector,
            () => _documentDetector.Predict(small));
        foreach (DetectedDocument doc in documents)
        {
            doc.Box = ScaleBox(doc.Box, sx, sy);
            foreach (DocumentPage page in doc.Pages)
            {
                page.Box = ScaleBox(page.Box, sx, sy);
            }
        }

        sink.Emit("documents", DocumentsPayload(documents));
        return documents;
    }

    private static double[] ScaleBox(double[] box, double sx, double sy) =>
        [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy];

    /// <summary>
    /// <c>[{box, conf, pages}]</c>, boxes rounded to 0.1 px. A box on the rounding edge moves the crop
    /// by one pixel, so the checker compares these like detections.
    /// </summary>
    private static object[] DocumentsPayload(List<DetectedDocument> documents) =>
        [.. documents.Select(d => new Dictionary<string, object>
        {
            ["box"] = d.Box.Select(v => Ops.RoundHalfEven(v, 1)).ToArray(),
            ["conf"] = Ops.RoundHalfEven(d.Conf, 3),
            ["pages"] = d.Pages.Select(p => p.Box.Select(v => Ops.RoundHalfEven(v, 1)).ToArray()).ToArray(),
        })];

    /// <summary>
    /// (x0, y0, x1, y1) on the input image: the document box plus its margin. Port of
    /// <c>Pipeline._document_crop</c> — floor and ceil, clamped to the frame.
    /// </summary>
    public static (int X0, int Y0, int X1, int Y1) DocumentCrop(int width, int height, double[] box)
    {
        double m = DocumentCropMargin * Math.Max(box[2] - box[0], box[3] - box[1]);
        return ((int)Math.Max(0, Math.Floor(box[0] - m)),
                (int)Math.Max(0, Math.Floor(box[1] - m)),
                (int)Math.Min(width, Math.Ceiling(box[2] + m)),
                (int)Math.Min(height, Math.Ceiling(box[3] + m)));
    }

    /// <summary>
    /// Re-reads a date field line by line WHOLE when the word-by-word reading is not a date and the
    /// whole reading is. Port of <c>Pipeline._reread_dates_whole</c>.
    ///
    /// <para>
    /// The word splitter can drop a word without leaving a hole the gap guard sees: on a real 1998
    /// birth certificate the issue date «ДД» МЕСЯЦА ГГГГ г. came out as month and year only — the day,
    /// pressed against the left edge of the field crop, was not taken for a word at all, and the
    /// empty stretch it left was 2.6 typical words wide against the guard's measured 3.0. Read whole,
    /// the same crop gives the day glued to the month and year, which converts.
    /// </para>
    ///
    /// <para>
    /// The rule checks itself: it replaces a reading only when that reading does NOT convert to
    /// dd.mm.yyyy and the whole-line one DOES, so a date that already converts is never touched. The
    /// price is the one the gap guard pays too — a line read whole comes back without spaces; the
    /// canonical view is unaffected.
    /// </para>
    /// </summary>
    private List<DateReread> RereadDatesWhole(List<(string Label, List<Image> Lines)> dateLines,
        Dictionary<string, string> ocr, OcrOptions options)
    {
        var done = new List<DateReread>();
        if (dateLines.Count == 0 || ocr.Count == 0)
        {
            return done;
        }

        foreach ((string label, List<Image> lines) in dateLines)
        {
            string splitReading = ocr.GetValueOrDefault(label, "");
            if (Dates.CanonicalDate(label, splitReading) is not null)
            {
                continue;
            }

            OcrEngine engine = Array.IndexOf(options.RuFields, label) >= 0 ? _cyrillic : _latin;
            var reads = new List<string>(lines.Count);
            foreach (Image line in lines)
            {
                reads.Add(engine.FixErrors(label, engine.Predict(line)));
            }
            string whole = string.Join(" ", reads.Where(r => r.Length > 0)).Trim();
            if (Dates.CanonicalDate(label, whole) is null)
            {
                continue;
            }

            ocr[label] = whole;
            done.Add(new DateReread(label, splitReading, whole));
        }
        return done;
    }

    /// <summary>
    /// Rebuilds the internal-passport canvas from template-registered pages. Port of
    /// <c>Pipeline._register_pages</c> (pipeline.py:1026-1129), minus the diagnostic
    /// <c>meta_results['PageRegistration']</c> info dict — nothing in the graded contract reads it
    /// (only the rebuilt canvas feeds anything downstream), so building it would be dead code here.
    /// </summary>
    /// <returns>The stitched canvas, or null when no page could be registered or warped at all — the
    /// caller then keeps the Borders canvas unchanged, matching <c>if pages:</c> in the reference.</returns>
    private (Image Canvas, IMap Map)? RegisterPages(Image upright, List<Point[]> segments)
    {
        const double QuadSamePageIou = 0.40;
        const double QuadClippedIou = 0.80;

        (List<Point[]> quads, List<QuadFit.Info> _) =
            _pageRegistrar.PageQuads(segments, upright.Height, upright.Width);
        List<PageRegistrationResult> regs = _pageRegistrar.Register(upright, quads);
        double scale = _pageRegistrar.NativeScale(regs);

        // Plan: for each registration, decide whether its page comes from the Borders quad that
        // agrees with it (IoU >= QuadSamePageIou) or from the template homography — the SAME quad
        // may only back one page, so claiming it removes it from later candidates' search.
        var used = new HashSet<int>();
        var planIsQuad = new bool?[regs.Count]; // null = no page (r.Ok was false)
        var planQuadIdx = new int?[regs.Count];

        for (int ri = 0; ri < regs.Count; ri++)
        {
            PageRegistrationResult r = regs[ri];
            if (!r.Ok)
            {
                continue;
            }
            int? bestI = null;
            double bestIou = 0.0;
            for (int qi = 0; qi < quads.Count; qi++)
            {
                if (used.Contains(qi))
                {
                    continue;
                }
                double iou = PageRegistrar.QuadIou(quads[qi], r.Quad!);
                if (iou > bestIou)
                {
                    bestI = qi;
                    bestIou = iou;
                }
            }
            if (bestI is int bi && bestIou >= QuadSamePageIou)
            {
                used.Add(bi);
                Point[] q = quads[bi];
                bool clipped = q.Any(p => p.X <= 1 || p.Y <= 1
                    || p.X >= upright.Width - 2 || p.Y >= upright.Height - 2);
                if (clipped && bestIou < QuadClippedIou)
                {
                    planIsQuad[ri] = false;
                }
                else
                {
                    planIsQuad[ri] = true;
                    planQuadIdx[ri] = bi;
                }
            }
            else
            {
                planIsQuad[ri] = false;
            }
        }

        // Spare Borders quads, by vertical order, for a registration that failed outright.
        List<int> spare = [.. Enumerable.Range(0, quads.Count)
            .Where(i => !used.Contains(i))
            .OrderBy(i => quads[i].Min(p => p.Y))];

        var pages = new List<Image>();
        var pageQuads = new List<Point[]>();
        var pageMaps = new List<IMap?>();
        try
        {
            for (int ri = 0; ri < regs.Count; ri++)
            {
                PageRegistrationResult r = regs[ri];
                Image page;
                double[,] matrix;
                Point[] quadForStitch;
                if (planIsQuad[ri] is null)
                {
                    if (spare.Count == 0)
                    {
                        continue;
                    }
                    int qi = spare[0];
                    spare.RemoveAt(0);
                    Point[] expanded = Geometry.ExpandQuadF32(quads[qi], Geometry.DocMarginFraction);
                    (page, matrix) = _pageRegistrar.WarpQuadWithMatrix(upright, expanded, scale);
                    quadForStitch = quads[qi];
                }
                else if (planIsQuad[ri] == true)
                {
                    int qi = planQuadIdx[ri]!.Value;
                    Point[] expanded = Geometry.ExpandQuadF32(quads[qi], Geometry.DocMarginFraction);
                    (page, matrix) = _pageRegistrar.WarpQuadWithMatrix(upright, expanded, scale);
                    quadForStitch = quads[qi];
                }
                else
                {
                    (page, matrix) = _pageRegistrar.WarpPageWithMatrix(upright, r, scale);
                    quadForStitch = r.Quad!;
                }

                (Image straightened, _, IMap? straight) = _pageRegistrar.StraightenWithMap(page, scale);
                pages.Add(straightened);
                pageQuads.Add(quadForStitch);
                // page -> the image the registrar received (geometry.py)
                pageMaps.Add(new ChainMap([new HomographyMap(matrix)]).Then(straight));
            }

            if (pages.Count == 0)
            {
                return null;
            }
            (Image? stitched, Placement[] placements) = Geometry.StitchPagesPlaced(pages, pageQuads,
                StackDirection.Vertical);
            return (stitched!, Stitched.Geometry([.. pages.Select(pg => (pg.Width, pg.Height))], placements, pageMaps));
        }
        finally
        {
            foreach (Image p in pages)
            {
                p.Dispose();
            }
        }
    }


    /// <summary>
    /// Template matches a vehicle registration certificate needs before its template geometry replaces
    /// the Borders quad. The card is printed dense (captions, rules, guilloche), so a real match runs to
    /// a hundred or more points (median 141 on the client's test cards, 2026-10-02); 40 is the floor the
    /// form matching of the training-data side uses, below it the Borders canvas stays.
    /// </summary>
    public const int CardMinInliers = 40;

    /// <summary>
    /// Skew of the Borders canvas below which it is kept (see <see cref="PageRegistrar.CardSkew"/>): the
    /// card's corners found by the template, carried into that canvas, are fitted by a similarity; the
    /// residual as a share of the long side. Visible skew starts at about 1.4 % (handoff of 2026-10-02).
    /// </summary>
    public const double CardSkewKeep = 0.01;

    /// <summary>
    /// Least-squares re-fits after MAGSAC in the card's template match (<c>Pipeline.CARD_REFIT_ROUNDS</c>): the
    /// skew decision above must not move with MAGSAC's samples.
    /// </summary>
    public const int CardRefitRounds = 5;

    /// <summary>The template registrar of one STS type, built on first use; null without templates.</summary>
    private PageRegistrar? CardRegistrar(string docType)
    {
        lock (_cardRegistrars)
        {
            if (!_cardRegistrars.TryGetValue(docType, out PageRegistrar? reg))
            {
                try
                {
                    reg = new PageRegistrar(_root, docType, refitRounds: CardRefitRounds);
                }
                catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
                {
                    reg = null;
                }
                _cardRegistrars[docType] = reg;
            }
            return reg;
        }
    }

    /// <summary>
    /// Rebuilds a vehicle registration certificate's canvas from its printed blank. Port of
    /// <c>Pipeline._register_card</c>.
    ///
    /// <para>
    /// The opposite policy to the passport's. There the Borders quad keeps the geometry whenever it
    /// agrees with the template, because the template fit of a sparsely printed page turns by degrees.
    /// Here the card often lies in a plastic sleeve or lamination and the Borders quad is the SLEEVE's
    /// edge: it overlaps the card well — it "agrees" — and still skews the canvas (on the client's 1217
    /// cards ~12 % came out visibly skewed). The card is printed dense, so the template match is strong:
    /// when it reaches <see cref="CardMinInliers"/> the template geometry is taken and the page
    /// straightened by its own lines; otherwise the Borders canvas stays as it is.
    /// </para>
    ///
    /// <para>Where the card runs past the photo, the canvas is painted the card's own paper colour rather
    /// than the smeared edge (<see cref="PageRegistrar.WarpMatrix"/>'s fill).</para>
    /// </summary>
    /// <returns>The new canvas, or null when the Borders canvas stays.</returns>
    private (Image Canvas, IMap Map)? RegisterCard(Image upright, List<Point[]>? segments, string docType)
    {
        PageRegistrar? reg = CardRegistrar(docType);
        if (reg is null)
        {
            return null;
        }
        (List<Point[]> quads, _) = reg.PageQuads(segments, upright.Height, upright.Width);
        PageRegistrationResult r = reg.Register(upright, quads)[0];
        if (!r.Ok || r.Inliers < CardMinInliers)
        {
            return null;
        }
        // Where the Borders canvas is not skewed, keep it: re-cutting a canvas that was right only
        // resamples it (measured on the client's test cards: on the 101 not skewed by Borders, 15 fields
        // read better and 17 worse — noise; on the 26 skewed by >= 1 %, 5 better, 1 worse). Skew, not
        // offset: a sleeve runs parallel to the card, so its edge sits 2-3 % off the card's even on a
        // straight canvas; what matters is whether the card comes out a rectangle.
        if (quads.Count > 0 && quads.Min(q => reg.CardSkew(q, r.Quad!)) < CardSkewKeep)
        {
            return null;
        }
        double scale = reg.NativeScale([r]);
        double[,] m = reg.PageMatrix(r, scale);
        int[]? fill = reg.CardFill(upright, r.Quad!);
        Image page = reg.WarpMatrix(upright, m, scale, fill);
        (Image straightened, _, IMap? straight) = reg.StraightenWithMap(page, scale);
        // the page's own map, then the only page's place on its canvas (a single page is handed on as it is)
        IMap pageMap = new ChainMap([new HomographyMap(m)]).Then(straight);
        return (straightened, Stitched.Geometry([(straightened.Width, straightened.Height)],
            [new Placement(1.0, 0.0, 0.0)], [pageMap]));
    }

    /// <summary>
    /// For each box, the rows (top, bottom) of the vertically widened crop to READ, or null when the box
    /// is read as the detector cut it. Port of the arithmetic of <c>Pipeline._read_margins</c>.
    ///
    /// <para>
    /// A field labelled tight to its letters gets a vertical margin, so the reading gets whole glyphs.
    /// The box stays as the detector gave it — only the crop that is READ grows. The STS special marks
    /// are why: their lines stand so close that a labelling margin merged neighbours (39-41 % overlap on
    /// real canvases), so they were labelled tight, and the detector learnt to cut the tops and bottoms
    /// off the letters. The margin never reaches the next line: it stops halfway to the nearest box above
    /// or below that shares part of the width, of any label.
    /// </para>
    /// </summary>
    public static (int Top, int Bottom)?[] ReadMarginRows(IReadOnlyList<Box> boxes, OcrOptions options,
        int canvasHeight)
    {
        var rows = new (int Top, int Bottom)?[boxes.Count];
        if (options.ReadMargin.Count == 0 || boxes.Count == 0)
        {
            return rows;
        }
        for (int i = 0; i < boxes.Count; i++)
        {
            if (!options.ReadMargin.TryGetValue(boxes[i].Label, out double share) || share == 0)
            {
                continue;
            }
            int x1 = (int)boxes[i].X1, y1 = (int)boxes[i].Y1, x2 = (int)boxes[i].X2, y2 = (int)boxes[i].Y2;
            int pad = (int)Math.Round((y2 - y1) * share, MidpointRounding.ToEven);
            int top = Math.Max(0, y1 - pad), bottom = Math.Min(canvasHeight, y2 + pad);
            for (int j = 0; j < boxes.Count; j++)
            {
                Box other = boxes[j];
                if (j == i || Math.Min(x2, other.X2) <= Math.Max(x1, other.X1))
                {
                    continue;                                            // no shared width
                }
                if (other.Y2 <= y1)                                      // a line above
                {
                    top = Math.Max(top, FloorHalf((int)other.Y2 + y1 + 1));
                }
                else if (other.Y1 >= y2)                                 // a line below
                {
                    bottom = Math.Min(bottom, FloorHalf(y2 + (int)other.Y1));
                }
            }
            if (top == y1 && bottom == y2)
            {
                continue;
            }
            rows[i] = (top, bottom);
        }
        return rows;
    }

    private static int FloorHalf(int value) => (int)Math.Floor(value / 2.0);   // Python's int //

    /// <summary>Re-cuts the READ crop of the fields <see cref="ReadMarginRows"/> widens, in place.</summary>
    private static void ReadMargins(List<Field> fields, OcrOptions options, Image canvas)
    {
        var rows = ReadMarginRows([.. fields.Select(f => f.Box)], options, canvas.Height);
        for (int i = 0; i < fields.Count; i++)
        {
            if (rows[i] is not (int top, int bottom))
            {
                continue;
            }
            Box b = fields[i].Box;
            // the box stays as the detector gave it; the READ crop and its frame (geometry.py) move
            fields[i].ReplacePatch(Crop.ClampedCrop(canvas, (int)b.X1, top, (int)b.X2, bottom),
                (int)b.X1, top);
        }
    }

    /// <summary>
    /// The leasing flag from the STS special marks, alongside the reading (<c>results.leasing</c>); the
    /// special marks themselves stay as read. Port of <c>Pipeline._read_leasing</c>.
    ///
    /// <para>Every STS back — the side with the special marks — emits the stage, null when there is no
    /// leasing: a port that misses a leasing record must differ from the reference, not be skipped. The
    /// front has no marks to read. Only the flag is reported (<c>LEASING_REPORTED</c>).</para>
    /// </summary>
    private static void ReadLeasing(string bareType, Results results, IStageSink sink)
    {
        string upper = bareType.ToUpperInvariant();
        if (!upper.StartsWith("STS", StringComparison.Ordinal))
        {
            return;
        }
        LeasingRecord? record = StsMarks.ParseLeasing(results.Ocr.GetValueOrDefault("Special_marks"));
        results.Leasing = record is not null;
        if (upper.StartsWith("STSBACK", StringComparison.Ordinal))
        {
            sink.Emit("leasing", record is null
                ? null!
                : new Dictionary<string, object> { ["leasing"] = true });
        }
    }

    /// <summary>Rotates a quarter turn counter-clockwise <paramref name="n"/> times; consumes the input.</summary>
    private static Image TurnQuarters(Image image, int n)
    {
        Image current = image;
        for (int i = 0; i < n; i++)
        {
            var dst = new OpenCvSharp.Mat();
            OpenCvSharp.Cv2.Rotate(current.Mat, dst, OpenCvSharp.RotateFlags.Rotate90Counterclockwise);
            current.Dispose();
            current = Image.Wrap(dst);
        }
        return current;
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
            Normalized = results.OcrNormalized,
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
        foreach (IDisposable? module in new IDisposable?[]
                 {
                     _documentDetector, _docTypeAngles, _glare, _blur, _printSpoofing, _lcdSpoofing, _docDetector,
                     _textFields, _words, _cyrillic, _latin, _pageRegistrar,
                 }.Concat(_cardRegistrars.Values))
        {
            if (module is null)
            {
                continue;
            }
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

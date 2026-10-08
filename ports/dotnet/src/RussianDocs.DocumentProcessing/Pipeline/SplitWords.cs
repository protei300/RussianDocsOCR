using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// One field's word crops. <see cref="WordBoxes"/> has one entry per DETECTION of the field, and a
/// null entry means "this field needed no splitting" — which is not the same as "the detector found
/// one word".
/// </summary>
public sealed class FieldWords : IDisposable
{
    public required string Label { get; init; }
    public List<Image> Patches { get; init; } = [];
    public List<List<Box>?> WordBoxes { get; init; } = [];

    /// <summary>
    /// Words per detection of the field, in reading order (<c>Pipeline._field_lines</c>): the join
    /// flattens the lines, and a word torn by a line break is found at their boundary
    /// (<see cref="StsMarks.GlueTornWords"/>).
    /// </summary>
    public List<int> LineWordCounts { get; init; } = [];

    public void Dispose()
    {
        foreach (Image patch in Patches)
        {
            patch.Dispose();
        }
        Patches.Clear();
    }
}

/// <summary>
/// One line of one field the gap guard acted on (or declined to). Mirrors the entries of
/// <c>PipelineResults.words_fallback</c> / <c>words_no_ink</c>.
/// </summary>
/// <param name="Field">The field label.</param>
/// <param name="Line">The ordinal of the line WITHIN its field, top to bottom.</param>
/// <param name="Gap">
/// The widest empty stretch in typical word widths; null stands for the reference's <c>None</c> —
/// no words were found at all, so the ratio has no denominator. Set on fallback entries.
/// </param>
/// <param name="Ink">The Laplacian variance, rounded to 2 places. Set on no-ink entries only.</param>
public sealed record LineFlag(string Field, int Line, double? Gap, double? Ink);

/// <summary>
/// What the gap guard reports. Part of the contract, not debug output: a measurement that corrects
/// for the missing spaces of a line read whole must apply that correction ONLY to these lines.
/// </summary>
public sealed class SplitFlags
{
    /// <summary>Lines read WHOLE because the word split lost a word (<c>words_fallback</c>).</summary>
    public List<LineFlag> Fallback { get; } = [];

    /// <summary>Lines left alone because they carry no strokes (<c>words_no_ink</c>).</summary>
    public List<LineFlag> NoInk { get; } = [];
}

/// <summary>Everything <see cref="SplitWords.Run"/> produces.</summary>
public sealed class SplitOutcome
{
    /// <summary>The per-field word crops, owned by the caller (<see cref="SplitWords.CloseAll"/>).</summary>
    public required List<FieldWords> Fields { get; init; }

    public required SplitFlags Flags { get; init; }

    /// <summary>
    /// The whole line crops of the date fields that were read word by word, kept for the whole-line
    /// date re-read: the split can drop a word without leaving a hole wide enough for the gap guard.
    /// In field first-detection order. **BORROWED** — these are the detected fields' own patches,
    /// valid until the caller disposes the fields.
    /// </summary>
    public required List<(string Label, List<Image> Lines)> DateLines { get; init; }

    /// <summary>
    /// The kept detections in reading order (top to bottom): which field each came from and the boxes its words
    /// were found at, in the field's own patch. What the quadrilaterals on the input image are built from.
    /// </summary>
    public required List<SplitDetection> Detections { get; init; }
}

/// <summary>
/// One kept detection of a field. <paramref name="WordBoxes"/> is null for a field that needed no splitting (its
/// whole patch is the single word) - not the same as a split that found no words, which is an empty list.
/// </summary>
/// <param name="FieldIndex">Index into the list of fields the split was given.</param>
public sealed record SplitDetection(int FieldIndex, string Label, List<Box>? WordBoxes);

public static class SplitWords
{
    /// <summary>
    /// WORDS_MAX_GAP: an empty stretch on a line wider than this many typical words means the split
    /// dropped a word, and the line is read whole instead. Measured over 157 documents, 815 fields:
    /// the widest gap on a line that really lost a long word has a median of 2.74 typical word
    /// widths, against 0.06 on intact lines — a factor of forty. The share of the line covered by
    /// word boxes was measured on the same run and rejected: it reads print DENSITY, not the
    /// integrity of the split (a correctly read internal-passport Licence_number sits at 0.64-0.75).
    /// Known limit: when the missing word left NO gap because a neighbour's box swallowed it,
    /// geometry cannot see it at all — 8 of 19 measured cases.
    /// </summary>
    public const double WordsMaxGap = 3.0;

    /// <summary>
    /// LINE_MIN_INK: below this much fine-grained detail (variance of the Laplacian) a line carries no
    /// strokes and the fallback above must NOT re-read it. "No word boxes" has a second cause besides
    /// a lost split: there IS no text (an anonymised, blurred name strip), and re-reading it whole
    /// manufactured «ЛВН». From a blur ladder over 612 lines: sharp text median 1590, mild blur 249,
    /// blur wider than the stroke 27; 100 sits in that trough. Known limit: an empty strip and a strip
    /// with VERY FAINT text look the same to this measure.
    /// </summary>
    public const double LineMinInk = 100.0;

    /// <summary>
    /// IoU above which a <c>&lt;name&gt;_ru</c> box and a <c>&lt;name&gt;_en</c> box are one line read as
    /// both fields. Real pairs sit apart (ru/en lines of an external passport overlap at 0.2-0.3, of a
    /// driving licence at up to 0.5); the duplicates seen on a retrained detector overlap at 0.97-1.00.
    /// </summary>
    public const double PairedDuplicateIou = 0.8;

    /// <summary>Disposes every word crop. Unconditional — see <see cref="Run"/>.</summary>
    public static void CloseAll(IEnumerable<FieldWords>? fieldWords)
    {
        if (fieldWords is null)
        {
            return;
        }
        foreach (FieldWords fw in fieldWords)
        {
            fw.Dispose();
        }
    }

    /// <summary>
    /// Turns detected fields into per-field word crops.
    ///
    /// <para>
    /// Fields that are not OCR fields for this document type are dropped, duplicates of the
    /// must-be-unique fields are dropped, and the rest are either split into words or passed through
    /// whole. The kept detections are ordered top to bottom by their vertical centre (a STABLE sort
    /// on that one key, as the reference's <c>list.sort</c>): a multi-line field is labelled one
    /// detection per line, and the per-label assembly below concatenates the words of its lines in
    /// list order.
    /// </para>
    ///
    /// <para>
    /// **A field can be detected TWICE and legitimately so** — the internal passport prints its
    /// series and number in two places — in which case the crops are concatenated under one label and
    /// the OCR results join. That is why <see cref="FieldWords.WordBoxes"/> is a list of lists.
    /// </para>
    ///
    /// <para>
    /// <paramref name="docType"/> is the BARE type ("SNILS", not "SNILS_1996"): the gap guard compares
    /// it with "SNILS", exactly as the OCR routing does.
    /// </para>
    /// </summary>
    public static SplitOutcome Run(List<Field> fields, OcrOptions options, WordsDetector words,
        string docType)
    {
        var flags = new SplitFlags();

        HashSet<int> drop = DuplicateFieldIndices(fields);
        drop.UnionWith(PairedDuplicateIndices(fields));

        // Stable by vertical centre. OrderBy is stable; List.Sort is not.
        List<int> kept = [.. Enumerable.Range(0, fields.Count)
            .Where(i => !drop.Contains(i) && options.IsOcrField(fields[i].Box.Label))
            .OrderBy(i => (fields[i].Box.Y1 + fields[i].Box.Y2) / 2)];

        var splitIndices = kept.Where(i => options.NeedsSplit(fields[i].Box.Label)).ToList();

        // Splitting runs CONCURRENTLY across fields, one task each, capped at 8 to match the
        // reference's ThreadPoolExecutor(max_workers=8). Results are collected positionally.
        var byIndex = new Dictionary<int, (List<Box> Boxes, List<Image> Patches)>();
        if (splitIndices.Count > 0)
        {
            var tasks = splitIndices
                .Select(i => (Func<(List<Box>, List<Image>)>)(() => words.PredictTransform(fields[i].Patch)))
                .ToList();

            (var results, Exception? error) = Group.Run(Group.MinLimit(8, splitIndices.Count), tasks);
            if (error is not null)
            {
                // The crops of the tasks that DID succeed are already allocated, and nothing
                // downstream will ever see them — releasing them here is the only chance. This is why
                // Group.Run returns partial results on error.
                foreach ((List<Box> _, List<Image> patches) in results.Where(r => r.Item2 is not null))
                {
                    foreach (Image patch in patches)
                    {
                        patch.Dispose();
                    }
                }
                throw error;
            }

            for (int k = 0; k < splitIndices.Count; k++)
            {
                byIndex[splitIndices[k]] = (results[k].Item1, results[k].Item2);
            }
        }

        // Detections whose word crops the gap guard replaced by the whole line.
        var readWhole = new HashSet<int>();
        try
        {
            ApplyGapGuard(fields, kept, byIndex, readWhole, flags, docType);
        }
        catch
        {
            foreach ((List<Box> _, List<Image> patches) in byIndex.Values)
            {
                foreach (Image patch in patches)
                {
                    patch.Dispose();
                }
            }
            throw;
        }

        // Whole lines of the date fields that were read word by word, kept for the whole-line date
        // re-read. SNILS is excluded for the same parity reason as the gap guard; a line the guard
        // already reads whole has nothing to add.
        var dateLines = new List<(string Label, List<Image> Lines)>();
        if (docType != "SNILS")
        {
            foreach (int i in kept)
            {
                string label = fields[i].Box.Label;
                if (!byIndex.ContainsKey(i)
                    || !label.Contains("date", StringComparison.OrdinalIgnoreCase)
                    || readWhole.Contains(i))
                {
                    continue;
                }
                int at = dateLines.FindIndex(d => d.Label == label);
                if (at < 0)
                {
                    dateLines.Add((label, [fields[i].Patch]));
                }
                else
                {
                    dateLines[at].Lines.Add(fields[i].Patch);
                }
            }
        }

        var output = new List<FieldWords>();
        var detections = new List<SplitDetection>(kept.Count);
        var position = new Dictionary<string, int>(StringComparer.Ordinal);
        try
        {
            foreach (int i in kept)
            {
                string label = fields[i].Box.Label;

                List<Image> patches;
                List<Box>? boxes;
                if (byIndex.TryGetValue(i, out (List<Box> Boxes, List<Image> Patches) split))
                {
                    patches = split.Patches;
                    boxes = split.Boxes;
                    // An empty detection yields an EMPTY word list, exactly as the reference does —
                    // it does NOT fall back to the whole patch. The fallback belongs to fields that
                    // were never split at all (and to lines the gap guard re-read whole, above).
                }
                else
                {
                    // CLONED, not borrowed. The reference aliases the field's own patch here and
                    // Python's GC makes that free; in a port, a borrowed Mat inside a list the caller
                    // disposes is a double free that surfaces only in bulk. One copy per unsplit
                    // field buys uniform ownership and removes the special case from CloseAll.
                    patches = [fields[i].Patch.Clone()];
                    boxes = null;
                }

                detections.Add(new SplitDetection(i, label, boxes));
                if (position.TryGetValue(label, out int at))
                {
                    output[at].LineWordCounts.Add(patches.Count);
                    output[at].Patches.AddRange(patches);
                    output[at].WordBoxes.Add(boxes);
                    continue;
                }
                position[label] = output.Count;
                output.Add(new FieldWords
                {
                    Label = label, Patches = patches, WordBoxes = [boxes], LineWordCounts = [patches.Count],
                });
            }
            return new SplitOutcome { Fields = output, Flags = flags, DateLines = dateLines, Detections = detections };
        }
        catch
        {
            CloseAll(output);
            throw;
        }
    }

    /// <summary>
    /// The gap guard (Pipeline._split_words). A hole on the line wider than a few typical words means
    /// the split dropped a word — measured twice: a 9 px crop where the detector returned NO words
    /// (the field vanished without a trace), and a line whose longest word («Тракторозаводский», 17
    /// characters) was the one missed. Reading the line whole recovers it; the price is that the
    /// engine emits no spaces, so the line comes back glued, which is why this is a fallback on a
    /// signal and not the default.
    ///
    /// <para>
    /// SNILS is excluded BY CONSTRUCTION, not by hoping the threshold spares it: there the engine is
    /// chosen by word-index parity, and a line read whole destroys the parity the routing depends on.
    /// </para>
    /// </summary>
    private static void ApplyGapGuard(List<Field> fields, List<int> kept,
        Dictionary<int, (List<Box> Boxes, List<Image> Patches)> byIndex, HashSet<int> readWhole,
        SplitFlags flags, string docType)
    {
        if (docType == "SNILS")
        {
            return;
        }

        // `kept` order is top-to-bottom, and a multi-line field collects its lines in that same
        // order later — so counting per label here gives the line's ordinal WITHIN its field, which
        // is what a reader of the flag can act on. The raw box index would be meaningless outside
        // this function.
        var seen = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (int i in kept)
        {
            string label = fields[i].Box.Label;
            int ordinal = seen.GetValueOrDefault(label, 0);
            seen[label] = ordinal + 1;
            if (!byIndex.TryGetValue(i, out (List<Box> Boxes, List<Image> Patches) split))
            {
                continue;
            }

            double gap = WidestGap(split.Boxes, fields[i].Patch.Width);
            if (gap > WordsMaxGap && split.Boxes.Count == 0)
            {
                // No boxes at all has two causes, and only one of them is a lost split: the other is
                // a line with nothing on it. Asked HERE only — where boxes were found the text is
                // there by definition, and the question would be noise.
                double ink = Ink.LaplacianVariance(fields[i].Patch);
                if (ink < LineMinInk)
                {
                    flags.NoInk.Add(new LineFlag(label, ordinal, null, Ops.RoundHalfEven(ink, 2)));
                    continue;
                }
            }
            if (gap > WordsMaxGap)
            {
                // The whole patch becomes the single word; the detected boxes (an empty or a holed
                // list) stay as the stage payload, exactly as the reference leaves word_bbox_by_idx
                // untouched. A CLONE, as everywhere the reference aliases the field's patch.
                foreach (Image patch in split.Patches)
                {
                    patch.Dispose();
                }
                byIndex[i] = (split.Boxes, [fields[i].Patch.Clone()]);
                readWhole.Add(i);
                flags.Fallback.Add(new LineFlag(label, ordinal,
                    double.IsPositiveInfinity(gap) ? null : Ops.RoundHalfEven(gap, 3), null));
            }
        }
    }

    /// <summary>
    /// The widest empty stretch on the line, in typical word widths. Port of
    /// <c>Pipeline._widest_gap</c>.
    ///
    /// <para>
    /// A dropped word leaves a hole about as wide as a word; evenly spaced printing does not, however
    /// wide the spacing. That is the whole reason this is a ratio to the line's OWN median word width
    /// instead of a share of the line. Edges count as gaps too — a word lost from the start or the end
    /// of a line leaves the hole at the border. Nothing found at all means the whole line is one hole,
    /// so the answer is +Infinity.
    /// </para>
    /// </summary>
    public static double WidestGap(List<Box>? wordBoxes, double lineWidth)
    {
        if (wordBoxes is null || wordBoxes.Count == 0)
        {
            return double.PositiveInfinity;
        }
        if (lineWidth == 0)
        {
            return 0.0;
        }

        // sorted() on (x1, x2) tuples: lexicographic, stable.
        List<(double A, double B)> spans = [.. wordBoxes
            .Select(b => (A: b.X1, B: b.X2))
            .OrderBy(s => s.A).ThenBy(s => s.B)];

        List<double> widths = [.. spans.Where(s => s.B > s.A).Select(s => s.B - s.A).Order()];
        if (widths.Count == 0)
        {
            return double.PositiveInfinity;
        }
        double typical = widths[widths.Count / 2];
        if (typical <= 0)
        {
            return double.PositiveInfinity;
        }

        var gaps = new List<double> { spans[0].A };   // empty stretch on the left
        double end = spans[0].B;
        foreach ((double a, double b) in spans.Skip(1))
        {
            gaps.Add(Math.Max(0.0, a - end));
            end = Math.Max(end, b);
        }
        gaps.Add(Math.Max(0.0, lineWidth - end));      // and on the right
        return gaps.Max() / typical;
    }

    /// <summary>
    /// Marks all but the highest-confidence detection of each must-be-unique field.
    ///
    /// <para>
    /// The internal passport prints its series and number — and the FMS code — twice, so the detector
    /// legitimately returns duplicates and OCR'ing both would read the same value twice.
    /// </para>
    ///
    /// <para>
    /// **Strict <c>&gt;</c>, so a confidence tie keeps the EARLIER detection.** That matches Python's
    /// <c>max()</c>, which returns the first maximum. Using <c>&gt;=</c> would keep the later one and
    /// pick a different crop on any tie.
    /// </para>
    /// </summary>
    private static HashSet<int> DuplicateFieldIndices(List<Field> fields)
    {
        string[] uniqueFields = ["Licence_number", "Issue_organisation_code"];
        var drop = new HashSet<int>();

        foreach (string label in uniqueFields)
        {
            var indices = Enumerable.Range(0, fields.Count)
                .Where(i => fields[i].Box.Label == label)
                .ToList();
            if (indices.Count <= 1)
            {
                continue;
            }

            int best = indices[0];
            foreach (int i in indices.Skip(1))
            {
                if (fields[i].Box.Conf > fields[best].Box.Conf)
                {
                    best = i;
                }
            }
            foreach (int i in indices.Where(i => i != best))
            {
                drop.Add(i);
            }
        }
        return drop;
    }

    /// <summary>
    /// Indices of the weaker box wherever a <c>&lt;name&gt;_ru</c> and a <c>&lt;name&gt;_en</c> box
    /// cover the same line. Port of <c>Pipeline._paired_duplicate_indices</c>.
    ///
    /// <para>
    /// NMS runs per class on purpose (the ru/en pairs must not suppress each other), so nothing else
    /// removes a line the detector labels as BOTH languages. Seen on a retrained TextFields: one line
    /// came out as Birth_place_ru at 0.92 and Birth_place_en at 0.62 on the same box, and the Latin
    /// engine read the Cyrillic line into Birth_place_en. The more confident label keeps the line; on
    /// a tie the <c>_ru</c> one does (<c>&gt;=</c>, as the reference).
    /// </para>
    /// </summary>
    private static HashSet<int> PairedDuplicateIndices(List<Field> fields)
    {
        static double Iou(Box a, Box b)
        {
            double w = Math.Min(a.X2, b.X2) - Math.Max(a.X1, b.X1);
            double h = Math.Min(a.Y2, b.Y2) - Math.Max(a.Y1, b.Y1);
            if (w <= 0 || h <= 0)
            {
                return 0.0;
            }
            double inter = w * h;
            return inter / ((a.X2 - a.X1) * (a.Y2 - a.Y1) + (b.X2 - b.X1) * (b.Y2 - b.Y1) - inter);
        }

        var drop = new HashSet<int>();
        for (int i = 0; i < fields.Count; i++)
        {
            string label = fields[i].Box.Label;
            if (!label.EndsWith("_ru", StringComparison.Ordinal))
            {
                continue;
            }
            string name = label[..^3];
            for (int j = 0; j < fields.Count; j++)
            {
                if (fields[j].Box.Label == name + "_en"
                    && Iou(fields[i].Box, fields[j].Box) > PairedDuplicateIou)
                {
                    drop.Add(fields[i].Box.Conf >= fields[j].Box.Conf ? j : i);
                }
            }
        }
        return drop;
    }
}

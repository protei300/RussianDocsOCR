using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Maps;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// One address line as read (an entry of <c>meta_results['Address_lines']</c>): <c>kind</c> is
/// <c>printed</c> or <c>handwritten</c>, <c>PHandwritten</c> is the classifier's raw P(handwritten), <c>Text</c>
/// is null for a handwritten line (OCR reads print only).
/// </summary>
public sealed class AddressLineMeta
{
    public string Kind { get; init; } = "printed";
    public double PHandwritten { get; init; }
    public string? Text { get; set; }

    /// <summary>
    /// Index of the stamp this line lies in (an index into the stamps, freshest first), or null for a line inside
    /// no stamp. Meaningful only when the stamp detector ran - see <see cref="AddressRead.StampsRan"/>.
    /// </summary>
    public int? Stamp { get; set; }
}

/// <summary>What <see cref="AddressReader.Read"/> produced.</summary>
public sealed class AddressRead
{
    /// <summary>Every detection of the line detector, in reading order - including a line whose patch came out empty (the view model's <c>obbox</c> list).</summary>
    public List<ObbBox> Detections { get; init; } = [];

    /// <summary>The lines as read, in reading order; a detection with an empty patch has no entry here.</summary>
    public List<AddressLineMeta> Lines { get; init; } = [];

    /// <summary>The oriented box of each entry of <see cref="Lines"/> (the reference's <c>_address_line_obboxes</c>).</summary>
    public List<ObbBox> LineBoxes { get; init; } = [];

    /// <summary>Each line on the input image; null - the way back to it is not known for this run.</summary>
    public List<Point[]>? LineQuads { get; init; }

    public bool HasHandwritten { get; init; }

    /// <summary>The lines joined by a newline (<c>OCR['Address']</c>); a handwritten one reads as the placeholder.</summary>
    public string Text { get; init; } = "";

    /// <summary>True once the stamp detector has run: only then does <c>address.lines</c> carry <c>stamp</c>.</summary>
    public bool StampsRan { get; set; }

    /// <summary>The registration stamps, freshest first; empty without a stamp detector.</summary>
    public List<RegistrationStamp> Stamps { get; set; } = [];
}

/// <summary>
/// The registration page (INTPASSPORTADDR): the address lines found, classified as printed or handwritten, split
/// into words and read. Port of <c>ReadingStage._address_lines</c> and <c>_route_word_ocr</c>.
/// </summary>
public static class AddressReader
{
    /// <summary>
    /// What a handwritten address line reads as (OCR is printed-only). ASCII brackets on purpose: the report gets
    /// printed to consoles that are still cp1251/cp866 on Windows.
    /// </summary>
    public const string HandwrittenPlaceholder = "[рукопись]";

    /// <summary>
    /// Reads the address page.
    ///
    /// <para>
    /// The oriented line detector runs directly on the perspective-fixed canvas (it handles the tilt itself, so
    /// no deskew is needed) and yields upright patches already in reading order. Each line is classified; a
    /// printed one is split into words and read word by word, a handwritten one is kept as a placeholder so the
    /// order and the structure of the lines are preserved. There is no batched GPU path here: this port reads
    /// every word on the CPU, which is the reference's own default.
    /// </para>
    /// </summary>
    /// <param name="toInput">Canvas points to the input image (<c>results.to_input</c>); null when unknown.</param>
    public static AddressRead Read(Image canvas, AddressLinesDetector lines, AddressTextKindClassifier textKind,
        WordsDetector words, OcrEngine cyrillic, OcrEngine latin,
        Func<IReadOnlyList<Point>, Point[]?> toInput)
    {
        (List<ObbBox> detections, List<Image> patches) = lines.PredictTransform(canvas);
        var wordImages = new List<Image>();    // owned here: cut by the word splitter
        try
        {
            var lineMeta = new List<AddressLineMeta>();
            var lineBoxes = new List<ObbBox>();
            List<Point[]>? lineQuads = [];
            var slots = new List<int>();                 // -1 handwritten, else an index into printedWords
            var printedWords = new List<List<Image>>();  // one list of word patches per printed line
            bool hasHandwritten = false;

            for (int i = 0; i < detections.Count; i++)
            {
                Image patch = patches[i];
                ObbBox box = detections[i];
                if (patch.IsEmpty)
                {
                    continue;
                }
                lineBoxes.Add(box);

                // The line on the input image, in the order of lineMeta. "Not known" is said once, for the whole
                // run (PipelineResults.to_input): once a quad is null, no later one is kept.
                Point[]? onCanvas = AddressLinesDetector.CropGeometry(box)
                    .ToInput(Corners.Of(0, 0, patch.Width, patch.Height));
                Point[]? quad = onCanvas is null ? null : toInput(onCanvas);
                if (quad is null)
                {
                    lineQuads = null;
                }
                else
                {
                    lineQuads?.Add(quad);
                }

                (string kind, double probability) = textKind.Predict(patch);
                if (kind == "handwritten")
                {
                    hasHandwritten = true;
                    lineMeta.Add(new AddressLineMeta { Kind = "handwritten", PHandwritten = probability });
                    slots.Add(-1);
                    continue;
                }

                // printed line: split into words (left to right); the reading happens below
                (_, List<Image> cut) = words.PredictTransform(patch);
                wordImages.AddRange(cut);
                List<Image> lineWords = cut.Count == 0 ? [patch] : cut;
                lineWords = [.. lineWords.Where(w => !w.IsEmpty)];
                slots.Add(printedWords.Count);
                printedWords.Add(lineWords);
                lineMeta.Add(new AddressLineMeta { Kind = "printed", PHandwritten = probability });
            }

            List<List<string>> textsByLine =
            [
                .. printedWords.Select(ws => ws.Select(w => RouteWordOcr(w, cyrillic, latin)).ToList()),
            ];

            var lineTexts = new List<string>();
            for (int i = 0; i < lineMeta.Count; i++)
            {
                if (slots[i] < 0)
                {
                    lineTexts.Add(HandwrittenPlaceholder);
                    continue;
                }
                // `[t.strip() for t in texts if t]`: an empty reading is dropped, a blank one is kept as ''
                string lineText = string.Join(" ",
                    textsByLine[slots[i]].Where(t => t.Length > 0).Select(t => t.Trim()));
                lineMeta[i].Text = lineText;
                lineTexts.Add(lineText);
            }

            return new AddressRead
            {
                Detections = detections,
                Lines = lineMeta,
                LineBoxes = lineBoxes,
                LineQuads = lineQuads,
                HasHandwritten = hasHandwritten,
                Text = string.Join("\n", lineTexts),
            };
        }
        finally
        {
            foreach (Image word in wordImages)
            {
                word.Dispose();
            }
            foreach (Image patch in patches)
            {
                patch.Dispose();
            }
        }
    }

    /// <summary>
    /// Picks the engine for an address word. Both the Cyrillic and the Latin engine read it, and the Latin
    /// reading is kept only when it is digit-dominated (a house, building or flat number): the Latin engine reads
    /// digits and Latin letters cleanly, so its digit-dominated output is the reliable sign of a numeric token.
    /// A Cyrillic word is taken from the Cyrillic engine (<c>_route_word_ocr</c>).
    /// </summary>
    public static string RouteWordOcr(Image word, OcrEngine cyrillic, OcrEngine latin)
    {
        string ru = cyrillic.Predict(word);
        string en = latin.Predict(word);
        return IsNumberToken(en) ? en : ru;
    }

    /// <summary>
    /// True if the Latin engine's output is digit-dominated: at least one digit, and digits not outnumbered by
    /// letters among the alphanumeric characters (<c>_is_number_token</c>).
    /// </summary>
    public static bool IsNumberToken(string? text)
    {
        int digits = 0, alnum = 0;
        foreach (char c in text ?? "")
        {
            if (char.IsLetterOrDigit(c))
            {
                alnum++;
                if (char.IsDigit(c))
                {
                    digits++;
                }
            }
        }
        if (alnum == 0)
        {
            return false;
        }
        int letters = alnum - digits;
        return digits >= 1 && digits >= letters;
    }
}

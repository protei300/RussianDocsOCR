using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// The machine-readable zone as the pipeline remembers it for the length self-check: the canvas and
/// the MRZ boxes, so one line can be re-cut and re-read when it comes out the wrong length. Port of
/// <c>Pipeline._note_mrz_zone</c> / <c>_read_mrz</c> and their constants.
///
/// <para>
/// Nothing is modified when the zone is built: the boxes the detector produced stay exactly as they
/// are, so <c>fields.bbox</c> and every other field are untouched. The canvas is BORROWED — it
/// belongs to the run — and must outlive this object.
/// </para>
/// </summary>
public sealed class MrzZone
{
    /// <summary>
    /// A line of a machine-readable zone is exactly 44 characters, and that is a rare luxury: the
    /// pipeline can tell that it read the line WRONG without being told, and try again. Anything
    /// shorter means the crop lost part of the line.
    /// </summary>
    public const int LineLength = 44;

    /// <summary>
    /// The MRZ is printed as ONE rectangle holding two lines, so both lines share the same
    /// horizontal span — but the detector does not know that. Measured over samples/: the two boxes
    /// of a zone start within 10 px of each other when the zone reads correctly and 90-182 px apart
    /// when it does not, and the characters outside the narrower box never reach the engine. That
    /// accounted for 23 of the 34 damaged lines; the engine was innocent.
    ///
    /// <para>
    /// So the zone's own span is the FIRST retry candidate, then the ladder widens further.
    /// Deliberately a candidate and not a rewrite: forcing every MRZ box to the union span fixed the
    /// external passports and damaged an internal one, because a wider crop can also pull in the page
    /// edge. Reading the detector's own crop first and widening only on a wrong length cannot lose a
    /// line that was already right. The ladder reaches 34 % of the span on each side because that is
    /// what the worst measured case needed; the crop is clamped to the canvas, so the last steps
    /// saturate instead of running away.
    /// </para>
    /// </summary>
    private static readonly double[] RetryGrowth = [0.0, 0.05, 0.10, 0.16, 0.24, 0.34];

    /// <summary>
    /// The zone's alphabet is closed: capitals, digits and the filler. A line cannot begin or end
    /// with anything else, so a stray '.' or '_' at an edge is the page border caught by the crop, not
    /// text. Trimming those — and ONLY at the edges — is what keeps a widened crop from turning a
    /// correct 44-character line into a 45-character one.
    /// </summary>
    private const string Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<";

    private readonly Image _canvas;
    private readonly List<int[]> _boxes;
    private readonly (int Left, int Right)? _span;

    private MrzZone(Image canvas, List<int[]> boxes, (int, int)? span)
    {
        _canvas = canvas;
        _boxes = boxes;
        _span = span;
    }

    /// <summary>
    /// Builds the zone from the field detections, or null when there is no MRZ. Boxes are kept top to
    /// bottom, which is the order the OCR loop walks the patches in — the retry has to know which box
    /// a patch came from.
    /// </summary>
    public static MrzZone? Note(IReadOnlyList<Box> detections, Image canvas)
    {
        // Stable by vertical centre, as list.sort.
        List<Box> mrz = [.. detections.Where(b => b.Label == "MRZ")
            .OrderBy(b => (b.Y1 + b.Y2) / 2)];
        if (mrz.Count == 0)
        {
            return null;
        }

        List<int[]> boxes = [.. mrz.Select(b => new[] { (int)b.X1, (int)b.Y1, (int)b.X2, (int)b.Y2 })];

        // The span of the zone as a whole: for a line whose own box is too narrow this is where the
        // missing characters are. Only from boxes that are line-shaped — a box several line-heights
        // tall is not a line, and its edges say nothing about where the line ends (measured: the one
        // such zone reads worse from a widened crop).
        List<int[]> lineShaped = [.. boxes.Where(b => b[2] - b[0] >= 10 * Math.Max(1, b[3] - b[1]))];
        (int, int)? span = null;
        if (lineShaped.Count > 1)
        {
            span = (lineShaped.Min(b => b[0]), lineShaped.Max(b => b[2]));
        }
        return new MrzZone(canvas, boxes, span);
    }

    /// <summary>
    /// Drops edge characters that cannot occur in a machine-readable zone. Only the ends, and only
    /// characters outside the zone's closed alphabet: the page border sometimes lands inside a crop and
    /// comes back as '.' or '_'. Anything inside the line is left alone — a wrong character there is a
    /// reading error, and hiding it would be worse than showing it.
    /// </summary>
    public static string TrimToAlphabet(string text)
    {
        int start = 0, end = text.Length;
        while (start < end && !Alphabet.Contains(text[start]))
        {
            start++;
        }
        while (end > start && !Alphabet.Contains(text[end - 1]))
        {
            end--;
        }
        return text[start..end];
    }

    /// <summary>
    /// Re-reads one MRZ line from a wider crop when it came out too short. The reading the detector's
    /// own crop produced is kept unless it is the wrong length; then the zone's full span is tried,
    /// then progressively wider crops, and the first result of exactly 44 characters wins. Falls back
    /// to the longest reading seen, which is still closer than the short one.
    ///
    /// <para>
    /// Never invents a line: it only re-reads a box the detector found, so a zone whose second line
    /// was never detected stays a one-line zone. That matters — "reading in" a missing line would turn
    /// a real miss into a plausible string.
    /// </para>
    /// </summary>
    public static string Read(MrzZone? zone, OcrEngine latin, int lineIndex, string text)
    {
        text = TrimToAlphabet(text);
        if (text.Length == LineLength)
        {
            return text;
        }
        if (zone is null || lineIndex >= zone._boxes.Count)
        {
            return text;
        }

        int width = zone._canvas.Width;
        int[] box = zone._boxes[lineIndex];
        string best = text;
        foreach (double growth in RetryGrowth)
        {
            (int left, int right) = zone._span ?? (box[0], box[2]);
            // int(round(...)): Python's round is half-to-even.
            int step = PyNum.RoundHalfEvenToInt((right - left) * growth);
            using Image crop = Crop.ClampedCrop(zone._canvas,
                Math.Max(0, left - step), box[1], Math.Min(width, right + step), box[3]);
            if (crop.Width == 0 || crop.Height == 0)
            {
                continue;
            }
            string candidate = TrimToAlphabet(latin.FixErrors("MRZ", latin.Predict(crop)));
            if (candidate.Length == LineLength)
            {
                return candidate;
            }
            if (candidate.Length > best.Length)
            {
                best = candidate;
            }
        }
        return best;
    }
}

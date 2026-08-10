using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;

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

    public void Dispose()
    {
        foreach (Image patch in Patches)
        {
            patch.Dispose();
        }
        Patches.Clear();
    }
}

public static class SplitWords
{
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
    /// whole.
    /// </para>
    ///
    /// <para>
    /// **A field can be detected TWICE and legitimately so** — the internal passport prints its
    /// series and number in two places — in which case the crops are concatenated under one label and
    /// the OCR results join. That is why <see cref="FieldWords.WordBoxes"/> is a list of lists.
    /// </para>
    /// </summary>
    public static List<FieldWords> Run(List<Field> fields, OcrOptions options, WordsDetector words)
    {
        HashSet<int> drop = DuplicateFieldIndices(fields);

        var kept = new List<int>();
        for (int i = 0; i < fields.Count; i++)
        {
            if (drop.Contains(i))
            {
                continue;
            }
            if (options.IsOcrField(fields[i].Box.Label))
            {
                kept.Add(i);
            }
        }

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

        var output = new List<FieldWords>();
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
                    // were never split at all.
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

                if (position.TryGetValue(label, out int at))
                {
                    output[at].Patches.AddRange(patches);
                    output[at].WordBoxes.Add(boxes);
                    continue;
                }
                position[label] = output.Count;
                output.Add(new FieldWords { Label = label, Patches = patches, WordBoxes = [boxes] });
            }
            return output;
        }
        catch
        {
            CloseAll(output);
            throw;
        }
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
}

using System.Diagnostics;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// Per-stage timings.
///
/// <para>
/// **The stage NAMES are part of the compared contract**, which is not obvious: the tolerance spec
/// says timings are never compared by VALUE, and that is true — but the comparator descends into
/// dictionaries before consulting its ignore list, so the SET OF KEYS is compared exactly.
/// </para>
///
/// <para>
/// The names come from the reference's private method names, leading underscore and all. Renaming
/// them to something idiomatic would be a breaking change, not a tidy-up.
/// </para>
/// </summary>
public sealed class Timings
{
    public const string DocTypeAngle = "_doctype_angle";
    public const string QualityAndBorders = "_quality_and_borders";
    public const string Glare = "_glare";
    public const string Blur = "_blur";
    public const string PrintSpoofing = "_print_spoofing";
    public const string LcdSpoofing = "_lcd_spoofing";
    public const string DocDetector = "_doc_detector";
    public const string Deskew = "_deskew";
    public const string FieldsDetector = "_fields_detector";
    public const string SplitWords = "_split_words";
    public const string Ocr = "_ocr";

    private readonly Dictionary<string, double> _stages = new(StringComparer.Ordinal);
    private readonly HashSet<string> _concurrent = new(StringComparer.Ordinal);

    public void Record(string stage, TimeSpan elapsed) =>
        _stages[stage] = Ops.RoundHalfEven(elapsed.TotalSeconds, 4);

    /// <summary>Times an action and records it under <paramref name="stage"/>.</summary>
    public T Time<T>(string stage, Func<T> action)
    {
        var watch = Stopwatch.StartNew();
        try
        {
            return action();
        }
        finally
        {
            // Recorded in a finally block, so a failing stage still reports how long it took before
            // failing — which is the timing an operator actually wants when something is slow.
            Record(stage, watch.Elapsed);
        }
    }

    public void Time(string stage, Action action) => Time(stage, () =>
    {
        action();
        return 0;
    });

    /// <summary>
    /// Records a concurrent group: its own wall time, plus each member's.
    ///
    /// <para>
    /// The members are marked concurrent so they do NOT contribute to the total — the group's wall
    /// time already covers them, and adding both would report more time than elapsed.
    /// </para>
    /// </summary>
    public void RecordGroup(string name, TimeSpan wall, IReadOnlyDictionary<string, TimeSpan> members)
    {
        Record(name, wall);
        foreach ((string stage, TimeSpan elapsed) in members)
        {
            Record(stage, elapsed);
            _concurrent.Add(stage);
        }
    }

    /// <summary>Every stage plus <c>total</c>, which sums only the non-concurrent ones.</summary>
    public Dictionary<string, double> Report()
    {
        var report = new Dictionary<string, double>(_stages, StringComparer.Ordinal);
        double total = _stages.Where(kv => !_concurrent.Contains(kv.Key)).Sum(kv => kv.Value);
        report["total"] = Ops.RoundHalfEven(total, 4);
        return report;
    }
}

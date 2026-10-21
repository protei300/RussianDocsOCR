using System.Globalization;

namespace RussianDocs.Service.Repositories;

/// <summary>
/// The recognition journal: one line per <c>POST /recognize</c> call, and nothing about the document.
///
/// <para>
/// A line says THAT something was recognised, how it ended and how long it took: id, time, outcome, HTTP
/// status, milliseconds. No filename, no document type, no size, no field, no caller, no client address —
/// the journal is the one record of an in-memory recognition, and it must be safe to show to anyone who
/// can see the service at all. The UI lists these lines but cannot open them: nothing is behind them.
/// </para>
///
/// <para>
/// **In process memory only**, capped, lost at restart — never the store, never a file: the point of the
/// mode it serves is that a recognition leaves nothing on disk, and a journal file would be the one
/// exception an auditor asks about first. Port of <c>service/repositories/recognitions.py</c>;
/// ports/MEMORY.md §4.
/// </para>
/// </summary>
public sealed class RecognitionJournal(int capacity)
{
    /// <summary>
    /// <c>recognised</c> = a document type was found; <c>not_recognised</c> = the image was read but no
    /// supported document was in it; <c>rejected</c> = refused before recognition (bad or oversized image);
    /// <c>busy</c> = no pipeline became free in time; <c>error</c> = anything else.
    /// </summary>
    public static readonly string[] Outcomes = ["recognised", "not_recognised", "rejected", "busy", "error"];

    /// <summary>One line. Exactly these fields — ports/MEMORY.md §4 lists them and nothing else.</summary>
    public sealed record Entry(int Id, DateTime At, string Outcome, int StatusCode, int ProcessingMs)
    {
        public Dictionary<string, object?> Public() => new(StringComparer.Ordinal)
        {
            ["id"] = Id,
            // Python's isoformat() of an aware UTC datetime, with "+00:00" spelled "Z": microseconds.
            ["at"] = At.ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'", CultureInfo.InvariantCulture),
            ["outcome"] = Outcome,
            ["status_code"] = StatusCode,
            ["processing_ms"] = ProcessingMs,
            ["stored"] = false, // the UI keys "no drill-down" off this
        };
    }

    private readonly object _gate = new();
    private readonly LinkedList<Entry> _entries = new(); // newest first
    private readonly int _capacity = Math.Max(1, capacity);
    private int _nextId = 1;

    public Entry Record(string outcome, int statusCode, int processingMs)
    {
        lock (_gate)
        {
            var entry = new Entry(_nextId++, DateTime.UtcNow, outcome, statusCode, Math.Max(0, processingMs));
            _entries.AddFirst(entry);
            while (_entries.Count > _capacity)
            {
                _entries.RemoveLast();
            }
            return entry;
        }
    }

    /// <summary>Newest first. Returns <c>(page, total)</c>.</summary>
    public (List<Entry> Page, int Total) Recent(int limit, int offset)
    {
        lock (_gate)
        {
            return (_entries.Skip(offset).Take(Math.Max(0, limit)).ToList(), _entries.Count);
        }
    }

    /// <summary>Per outcome, plus <c>total</c>. Every outcome is present, zero or not.</summary>
    public Dictionary<string, int> Counts()
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string outcome in Outcomes)
        {
            counts[outcome] = 0;
        }
        lock (_gate)
        {
            foreach (Entry entry in _entries)
            {
                counts[entry.Outcome] = counts.GetValueOrDefault(entry.Outcome) + 1;
            }
            counts["total"] = _entries.Count;
        }
        return counts;
    }
}

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;

namespace RussianDocs.Service.Logging;

/// <summary>One buffered record. The JSON shape is what the logs page reads.</summary>
public sealed record LogEntry
{
    [JsonPropertyName("ts")] public double Ts { get; init; }
    [JsonPropertyName("level")] public required string Level { get; init; }
    [JsonPropertyName("logger")] public required string Logger { get; init; }
    [JsonPropertyName("message")] public required string Message { get; init; }
    [JsonPropertyName("exc")] public string? Exc { get; init; }
}

/// <summary>
/// The in-memory log buffer behind <c>GET /logs</c>, plus the stdout writer.
///
/// <para>
/// **Two sinks, and the split matters**: stdout carries structured lines at the configured level,
/// while the ring buffer captures EVERYTHING regardless of that level. The reason is the operator
/// workflow — when something goes wrong you want the debug lines that were already emitted, and
/// raising the level afterwards cannot retrieve them.
/// </para>
///
/// <para>Port of <c>service/core/logging.py</c>.</para>
/// </summary>
public static class LogRing
{
    /// <summary>
    /// How many entries the ring holds.
    ///
    /// <para>
    /// 5000, matching the reference, so both services keep the same amount of history and an operator
    /// comparing them is not misled by one having forgotten more. Bounded on purpose: this is an
    /// in-memory diagnostic aid, not a log store, and an unbounded buffer in a long-running service is
    /// a slow leak nobody planned. At roughly 150 bytes per entry that is under a megabyte.
    /// </para>
    /// </summary>
    public const int Capacity = 5000;

    private static readonly Dictionary<string, int> LevelOrder = new(StringComparer.Ordinal)
    {
        ["DEBUG"] = 0, ["INFO"] = 1, ["WARN"] = 2, ["WARNING"] = 2, ["ERROR"] = 3,
        ["CRITICAL"] = 4,
    };

    // An array used circularly rather than a queue: fixed allocation, no per-entry garbage, and the
    // read path is a single pass.
    private static readonly object Gate = new();
    private static readonly LogEntry?[] Entries = new LogEntry?[Capacity];
    private static int _next;
    private static bool _filled;

    internal static void Add(LogEntry entry)
    {
        lock (Gate)
        {
            Entries[_next] = entry;
            _next = (_next + 1) % Entries.Length;
            if (_next == 0)
            {
                _filled = true;
            }
        }
    }

    /// <summary>Entries NEWEST FIRST.</summary>
    private static List<LogEntry> Snapshot()
    {
        lock (Gate)
        {
            int count = _filled ? Entries.Length : _next;
            var output = new List<LogEntry>(count);
            for (int i = 0; i < count; i++)
            {
                int index = (_next - 1 - i + Entries.Length * 2) % Entries.Length;
                if (Entries[index] is { } entry)
                {
                    output.Add(entry);
                }
            }
            return output;
        }
    }

    /// <summary>
    /// The most recent entries, optionally filtered.
    ///
    /// <para>
    /// <paramref name="level"/> is a MINIMUM severity, not an exact match: asking for warnings should
    /// show errors too, which is what an operator means by "show me warnings".
    /// </para>
    /// </summary>
    public static List<LogEntry> Recent(int n, string level, string search)
    {
        int floor = LevelOrder.GetValueOrDefault(level.ToUpperInvariant(), 0);
        string needle = search.ToLowerInvariant();

        var output = new List<LogEntry>(n);
        foreach (LogEntry entry in Snapshot())
        {
            if (LevelOrder.TryGetValue(entry.Level, out int rank) && rank < floor)
            {
                continue;
            }
            if (needle.Length > 0 &&
                !entry.Message.ToLowerInvariant().Contains(needle, StringComparison.Ordinal))
            {
                continue;
            }
            output.Add(entry);
            if (output.Count >= n)
            {
                break;
            }
        }
        return output;
    }

    public static LogLevel ParseLevel(string name) => name.Trim().ToUpperInvariant() switch
    {
        "DEBUG" => LogLevel.Debug,
        "WARNING" or "WARN" => LogLevel.Warning,
        "ERROR" => LogLevel.Error,
        "CRITICAL" => LogLevel.Critical,
        _ => LogLevel.Information,
    };

    /// <summary>The names the Python service uses, so one log pipeline can ingest either.</summary>
    public static string LevelName(LogLevel level) => level switch
    {
        LogLevel.Trace or LogLevel.Debug => "DEBUG",
        LogLevel.Information => "INFO",
        LogLevel.Warning => "WARNING",
        LogLevel.Error => "ERROR",
        _ => "CRITICAL",
    };
}

/// <summary>
/// The logger provider: writes one JSON object per line to stdout, and every record to the ring.
///
/// <para>
/// Hand-written rather than a logging package, for the same reason the JWT is: it is forty lines, and
/// it keeps the service's dependency list at what the library genuinely needs. It also makes the
/// two-sink rule visible in one place instead of spread across provider configuration.
/// </para>
/// </summary>
public sealed class RingLoggerProvider(LogLevel stdoutLevel) : ILoggerProvider
{
    /// <summary>
    /// Serialised because two threads writing partial lines to the same stream interleave them, and
    /// an interleaved JSON line is worse than a dropped one — a log parser rejects the whole file.
    /// </summary>
    private static readonly object WriteGate = new();

    public ILogger CreateLogger(string categoryName) =>
        new RingLogger(categoryName, stdoutLevel, WriteGate);

    public void Dispose() { }

    private sealed class RingLogger(string category, LogLevel stdoutLevel, object writeGate)
        : ILogger
    {
        /// <summary>
        /// Unconditionally TRUE: the buffer wants every level. The stdout level is applied in
        /// <see cref="Log{TState}"/>, so stdout stays at the configured verbosity while the ring
        /// keeps everything.
        /// </summary>
        public bool IsEnabled(LogLevel level) => level != LogLevel.None;

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public void Log<TState>(LogLevel level, EventId eventId, TState state, Exception? error,
            Func<TState, Exception?, string> formatter)
        {
            string message = formatter(state, error);
            string levelName = LogRing.LevelName(level);
            LogRing.Add(new LogEntry
            {
                Ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
                Level = levelName,
                Logger = category,
                Message = message,
                Exc = error?.ToString(),
            });

            if (level < stdoutLevel)
            {
                return;
            }

            var line = new StringBuilder();
            line.Append('{');
            line.Append("\"timestamp\":").Append(JsonSerializer.Serialize(
                // Quoted T and Z, for the reason spelled out on NullableUtcConverter.Pattern.
                DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'",
                    CultureInfo.InvariantCulture)));
            line.Append(",\"level\":").Append(JsonSerializer.Serialize(levelName));
            line.Append(",\"logger\":").Append(JsonSerializer.Serialize(category));
            line.Append(",\"message\":").Append(JsonSerializer.Serialize(message));
            if (error is not null)
            {
                line.Append(",\"exc\":").Append(JsonSerializer.Serialize(error.ToString()));
            }
            line.Append('}');

            lock (writeGate)
            {
                Console.Out.WriteLine(line.ToString());
            }
        }
    }
}

using System.Text.Json;
using System.Text.Json.Serialization;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// Receives per-stage intermediates for the conformance probe.
///
/// <para>
/// Production passes <see cref="NullStageSink"/>, which costs nothing and changes no behaviour.
/// That matters: the intermediates must NOT be threaded through return values, because that would
/// alter the very code the ports are meant to copy.
/// </para>
/// </summary>
public interface IStageSink
{
    void Emit(string stage, object payload);
}

/// <summary>The production sink. Does nothing, on purpose.</summary>
public sealed class NullStageSink : IStageSink
{
    public static readonly NullStageSink Instance = new();
    public void Emit(string stage, object payload) { }
}

/// <summary>Wraps an array so the sink knows to write `.npy` rather than JSON.</summary>
public sealed record ArrayPayload(NdArray Array);

/// <summary>
/// Writes one file per stage into a directory, plus <c>stages.json</c> as an ordered index.
///
/// <para>
/// File naming and the index shape are fixed by <c>conformance/spec/stages.md</c> and must match the
/// other ports exactly — the checker reads them.
/// </para>
/// </summary>
public sealed class DirectoryStageSink(string root, string? upTo = null) : IStageSink
{
    private readonly List<StageEntry> _index = [];
    private bool _stopped;

    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
    };

    public int Count => _index.Count;

    public void Emit(string stage, object payload)
    {
        if (_stopped)
        {
            return;
        }

        // Stage names contain dots but never separators; taking the file name defends the dump
        // directory against a stage name that somehow acquires one.
        string safe = Path.GetFileName(stage);

        StageEntry entry;
        if (payload is ArrayPayload array)
        {
            string file = safe + ".npy";
            Npy.Save(Path.Combine(root, file), array.Array);
            entry = new StageEntry(stage, file, "npy", DtypeName(array.Array.Dtype), array.Array.Shape);
        }
        else
        {
            string file = safe + ".json";
            Directory.CreateDirectory(root);
            File.WriteAllText(Path.Combine(root, file),
                JsonSerializer.Serialize(payload, Options) + "\n");
            entry = new StageEntry(stage, file, "json", null, null);
        }
        _index.Add(entry);

        // `--upto` stops AFTER the named stage, inclusive. This is what makes a milestone verifiable
        // before the pipeline is finished, and it is the reason the harness exists at all.
        if (!string.IsNullOrEmpty(upTo) && stage == upTo)
        {
            _stopped = true;
        }
    }

    /// <summary>Convenience for image stages: copy pixels out and emit as an array.</summary>
    public static void EmitImage(IStageSink sink, string stage, Image image) =>
        sink.Emit(stage, new ArrayPayload(image.ToArray()));

    /// <summary>Writes the index. Call once, at the end.</summary>
    public void Close()
    {
        Directory.CreateDirectory(root);
        var payload = new Dictionary<string, object?>
        {
            ["stages"] = _index,
        };
        File.WriteAllText(Path.Combine(root, "stages.json"),
            JsonSerializer.Serialize(payload, Options) + "\n");
    }

    private static string DtypeName(Dtype dtype) => dtype switch
    {
        Dtype.Float32 => "<f4",
        Dtype.Float64 => "<f8",
        Dtype.UInt8 => "|u1",
        Dtype.Int64 => "<i8",
        Dtype.Unicode => "<U",
        _ => "?",
    };

    private sealed record StageEntry(
        [property: JsonPropertyName("stage")] string Stage,
        [property: JsonPropertyName("file")] string File,
        [property: JsonPropertyName("kind")] string Kind,
        [property: JsonPropertyName("dtype")] string? Dtype,
        [property: JsonPropertyName("shape")] int[]? Shape);
}

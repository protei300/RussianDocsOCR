using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.ViewModel;

namespace RussianDocs.Conform;

/// <summary>
/// The conformance CLI. Implements conformance/spec/cli.md and nothing else.
///
/// <para>
/// **stdout carries only the payload.** Every log line, warning and progress message goes to
/// stderr. This is not style: the checker parses stdout as JSON, and one stray line makes the
/// output look like a serialisation bug. The reference CLI has to redirect the Python library's
/// own prints for exactly the same reason.
/// </para>
///
/// <para>
/// Exit codes are the contract: 0 ran, 2 not implemented (the checker SKIPS rather than failing),
/// 3 input error, 1 crash. The 2-versus-1 distinction is what lets a port under construction say
/// "not yet" without being scored as broken — it is why M2 can be green while M6 does not exist.
/// </para>
/// </summary>
internal static class Program
{
    /// <summary>
    /// Stages this port can emit, in pipeline order. The checker skips everything not claimed
    /// here, which is the mechanism that makes a partial port gradeable.
    ///
    /// <para>
    /// Grows one milestone at a time. **An entry added before the stage actually works is worse
    /// than a missing one**: the checker would then compare a stage the port never emits and
    /// report a failure whose real cause is this list. The Go port hit the mirror image — a stage
    /// the REFERENCE forgot to claim was silently skipped in its own self-check, and reported
    /// PASS while grading 26 stages out of 44.
    /// </para>
    /// </summary>
    private static readonly string[] StagesImplemented = ["prepare", "doctype.label", "rotate", "quality", "borders.segments", "borders.canvas", "deskew.canvas", "fields.bbox", "words.<Field>.bbox", "ocr.<Field>.words", "join", "viewmodel"];

    /// <summary>
    /// Pinned to 1 for every conformance run.
    ///
    /// <para>
    /// ONNX Runtime splits its CPU reductions across threads, so a different thread count legitimately
    /// shifts results by about 1e-6. That is well inside the float tolerance, but it is enough to flip
    /// an argmax on two near-equal scores — which is an exact-match failure with no float anywhere
    /// near it. The sidecar records the value so a future divergence can be checked against it.
    /// </para>
    /// </summary>
    private const int IntraOpThreads = 1;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        // Every wire name is written by hand on the DTOs. No naming policy: the ~60 names in the
        // view model must match the reference byte for byte, and a policy that gets 59 of them
        // right is worse than none because the one it misses looks like a typo somewhere else.
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
    };

    private static int Main(string[] args)
    {
        // Invariant culture, belt and braces alongside InvariantGlobalization in
        // Directory.Build.props. CONVENTIONS §6.16: a ru-RU machine would otherwise serialise
        // 0,904 where the contract requires 0.904 — silently, and only on that machine.
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
        Console.OutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

        try
        {
            if (args.Length == 0)
            {
                Usage();
                return 3;
            }

            return args[0] switch
            {
                "info" => CmdInfo(),
                "recognize" => CmdRecognize(ParseFlags(args)),
                "probe" => CmdProbe(ParseFlags(args)),
                "soak" => CmdSoak(args),
                _ => UnknownCommand(args[0]),
            };
        }
        catch (FlagException ex)
        {
            Console.Error.WriteLine($"[conform] {ex.Message}");
            return 3;
        }
        catch (NotImplementedException ex)
        {
            // Deliberately distinct from a crash: the checker must skip, not fail.
            Console.Error.WriteLine($"[conform] not implemented: {ex.Message}");
            return 2;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[conform] crashed: {ex}");
            return 1;
        }
    }

    private static int UnknownCommand(string name)
    {
        Console.Error.WriteLine($"[conform] unknown command \"{name}\"");
        Usage();
        return 3;
    }

    private static void Usage() => Console.Error.WriteLine("""
        usage:
          rdocs-conform info
          rdocs-conform recognize --image <path> [--device cpu|gpu] [--ocr accurate|fast]
                                  [--img-size N] [--docconf F] [--include-debug]
          rdocs-conform probe --image <path> --dump-dir <dir> [--upto <stage>] [same flags]
          rdocs-conform soak --dir <samples> [--rounds N] [--device cpu|gpu]
        """);

    /// <summary>Describes this implementation. Shape fixed by spec/cli.md.</summary>
    private static int CmdInfo()
    {
        // Probing the native halves here is deliberate: `info` is the cheapest command, so if
        // OpenCV's or ONNX Runtime's native library fails to load, it surfaces as a readable
        // version string rather than midway through a pipeline stage.
        var payload = new Dictionary<string, object?>
        {
            ["port"] = "dotnet",
            ["language"] = $".NET {Environment.Version}",
            ["versions"] = new Dictionary<string, string?>
            {
                // `runtime` is the LANGUAGE runtime, not ONNX Runtime — the two were briefly the
                // same expression here, which made the sidecar claim .NET 1.21.0.
                ["runtime"] = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
                ["onnxruntime"] = Probes.OnnxRuntimeVersion(),
                ["opencv"] = Probes.OpenCvVersion(),
            },
            ["device"] = "cpu",
            // OCR stays on the CPU even when the detectors do not — measured at 13.7x slower on GPU
            // in the Go port, because per-word dynamic widths make the CUDA provider recompile the
            // graph per width. Reported separately so a glance at nvidia-smi does not become a bug
            // report.
            ["ocr_device"] = "cpu",
            ["providers"] = Probes.AvailableProviders(),
            ["model_format"] = "ONNX",
            ["ocr_mode"] = "accurate",
            ["stages_implemented"] = StagesImplemented,
            // An environment variable rather than a baked-in constant, so the binary stays
            // reproducible and a stale value shows up as empty instead of as the wrong commit.
            ["commit"] = Environment.GetEnvironmentVariable("RDOCS_COMMIT") ?? "",
        };
        Console.Out.WriteLine(JsonSerializer.Serialize(payload, JsonOptions));
        return 0;
    }

    /// <summary>
    /// Emits the view model on stdout and nothing else.
    ///
    /// <para>
    /// Shares the pipeline pass with <see cref="CmdProbe"/> rather than repeating it. Two separate
    /// passes could diverge, and then a golden would disagree with a live run for a reason that is not
    /// a behaviour change at all — the reference's own `regen` reuses probe and recognize for exactly
    /// this reason.
    /// </para>
    /// </summary>
    private static int CmdRecognize(Flags flags)
    {
        RequireImage(flags);

        using var recognizer = new Recognizer(DeviceNames.Parse(flags.Device), IntraOpThreads,
            flags.Ocr == "fast" ? OcrTier.Fast : OcrTier.Accurate);
        using Results results = recognizer.Run(flags.Image!, new RunOptions
        {
            Docconf = flags.Docconf,
            ImgSize = flags.ImgSize,
            IncludeDebug = flags.IncludeDebug,
        });

        Payload payload = Recognizer.BuildViewModel(results, flags.IncludeDebug);
        Console.Out.WriteLine(JsonSerializer.Serialize(payload, JsonOptions));
        return 0;
    }

    private static int CmdProbe(Flags flags)
    {
        RequireImage(flags);
        if (string.IsNullOrEmpty(flags.DumpDir))
        {
            throw new FlagException("probe requires --dump-dir");
        }

        var sink = new DirectoryStageSink(flags.DumpDir, flags.UpTo);
        using var recognizer = new Recognizer(DeviceNames.Parse(flags.Device), IntraOpThreads,
            flags.Ocr == "fast" ? OcrTier.Fast : OcrTier.Accurate);
        try
        {
            using Results results = recognizer.Run(flags.Image!, new RunOptions
            {
                Docconf = flags.Docconf,
                ImgSize = flags.ImgSize,
                Sink = sink,
                UpTo = flags.UpTo,
                IncludeDebug = flags.IncludeDebug,
            });
        }
        catch (NotImplementedException) when (sink.Count > 0)
        {
            // Stages BEFORE the unimplemented one were emitted and are gradeable, so this is a
            // success: the checker skips what `stages_implemented` does not claim. Exiting 2 here
            // would throw away work the port actually did — which is the difference between "M1 is
            // green" and "the port is broken".
            Console.Error.WriteLine("[conform] stopped at the first unimplemented stage");
        }
        finally
        {
            // The index must be written even on the partial path, or the checker sees a dump
            // directory with files and no way to read them.
            sink.Close();
        }
        return 0;
    }

    /// <summary>
    /// Pushes a whole directory of documents through ONE Recognizer, several times, reporting RSS
    /// between rounds.
    ///
    /// <para>
    /// **This is the check the conformance harness structurally cannot perform.** It runs one document
    /// per process, so a path that never releases its intermediates passes every stage and still dies
    /// in production — measured in the Go port at 12.7 MB per document, unbounded, with the suite green
    /// throughout.
    /// </para>
    ///
    /// <para>
    /// A leak and an allocator plateau look identical in a single measurement. They differ only in the
    /// SHAPE of the curve across rounds, which is why the corpus is repeated rather than measured once.
    /// GC.Collect is forced between rounds because .NET makes this HARDER than Go rather than easier: a
    /// Mat has a finalizer, so a missed Dispose becomes delayed rather than permanent, and without
    /// forcing a collection the curve would measure collection lag instead of retention.
    /// </para>
    ///
    /// <para>
    /// RSS is read from the OS, not from GC.GetTotalMemory: OpenCV Mats and ONNX Runtime arenas live in
    /// NATIVE memory, which the managed heap counters cannot see at all. The Go port had the same trap
    /// with runtime.MemStats.
    /// </para>
    /// </summary>
    private static int CmdSoak(string[] args)
    {
        string dir = "samples";
        int rounds = 4;
        string device = "cpu";
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--dir": dir = Next(args, ref i, args[i]); break;
                case "--rounds": rounds = Int(Next(args, ref i, args[i]), "--rounds"); break;
                case "--device": device = OneOf(Next(args, ref i, args[i]), "cpu", "gpu"); break;
                default: throw new FlagException($"unknown flag {args[i]}");
            }
        }

        string[] files = Directory.GetFiles(dir, "*.jpg", SearchOption.AllDirectories)
            .Where(f => Path.GetDirectoryName(f) != Path.GetFullPath(dir))
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToArray();
        if (files.Length == 0)
        {
            throw new FlagException($"no *.jpg found under {dir}");
        }

        using var recognizer = new Recognizer(DeviceNames.Parse(device), IntraOpThreads);
        Console.Out.WriteLine($"ready, rss={Rss()} MB   ({files.Length} documents, {rounds} rounds)");

        for (int round = 1; round <= rounds; round++)
        {
            int failed = 0;
            foreach (string file in files)
            {
                try
                {
                    using Results results = recognizer.Run(file, new RunOptions());
                    // Force the view model too: it is what the service builds, and building it is
                    // where the Go port's leak actually lived.
                    _ = Recognizer.BuildViewModel(results, includeDebug: false);
                }
                catch (Exception ex)
                {
                    failed++;
                    Console.Error.WriteLine($"[soak] {Path.GetFileName(file)}: {ex.Message}");
                }
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            Console.Out.WriteLine(
                $"round {round}: {round * files.Length} docs cumulative, rss={Rss()} MB, " +
                $"failed={failed}");
        }
        return 0;
    }

    private static long Rss()
    {
        using var self = System.Diagnostics.Process.GetCurrentProcess();
        self.Refresh();
        return self.WorkingSet64 / (1024 * 1024);
    }

    /// <summary>
    /// A missing or unreadable image is exit 3, not 1. The checker fails the case either way, but
    /// the distinction records whether the port or the input was at fault.
    /// </summary>
    private static void RequireImage(Flags flags)
    {
        if (string.IsNullOrEmpty(flags.Image))
        {
            throw new FlagException("--image is required");
        }
        if (!File.Exists(flags.Image))
        {
            throw new FlagException($"image not found: {flags.Image}");
        }
    }

    private sealed record Flags
    {
        public string? Image { get; init; }
        public string? DumpDir { get; init; }
        public string? UpTo { get; init; }
        public string Device { get; init; } = "cpu";
        public string Ocr { get; init; } = "accurate";
        public int ImgSize { get; init; } = 1500;
        public double Docconf { get; init; } = 0.5;
        public bool IncludeDebug { get; init; }
    }

    /// <summary>
    /// Hand-rolled flag parsing, on purpose.
    ///
    /// <para>
    /// A parser library would be more idiomatic and would also hide the contract: these flag names
    /// and defaults are shared with the Go and Python CLIs and must stay greppable side by side.
    /// Same reason the reference parses its query filters explicitly instead of leaning on
    /// FastAPI's magic — logic that lives only inside a framework's declarations does not survive
    /// a port.
    /// </para>
    /// </summary>
    private static Flags ParseFlags(string[] args)
    {
        var flags = new Flags();
        for (int i = 1; i < args.Length; i++)
        {
            string arg = args[i];
            flags = arg switch
            {
                "--include-debug" => flags with { IncludeDebug = true },
                "--image" => flags with { Image = Next(args, ref i, arg) },
                "--dump-dir" => flags with { DumpDir = Next(args, ref i, arg) },
                "--upto" => flags with { UpTo = Next(args, ref i, arg) },
                "--device" => flags with { Device = OneOf(Next(args, ref i, arg), "cpu", "gpu") },
                "--ocr" => flags with { Ocr = OneOf(Next(args, ref i, arg), "accurate", "fast") },
                "--img-size" => flags with { ImgSize = Int(Next(args, ref i, arg), arg) },
                "--docconf" => flags with { Docconf = Dbl(Next(args, ref i, arg), arg) },
                _ => throw new FlagException($"unknown flag {arg}"),
            };
        }
        return flags;
    }

    private static string Next(string[] args, ref int i, string flag) =>
        i + 1 < args.Length ? args[++i] : throw new FlagException($"{flag} needs a value");

    private static string OneOf(string value, params string[] allowed) =>
        Array.IndexOf(allowed, value) >= 0
            ? value
            : throw new FlagException($"expected one of {string.Join('|', allowed)}, got {value}");

    // Invariant parsing, for the same reason as invariant formatting: `--docconf 0.5` must not
    // depend on the machine's decimal separator.
    private static int Int(string value, string flag) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n)
            ? n
            : throw new FlagException($"{flag} expects an integer, got {value}");

    private static double Dbl(string value, string flag) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double d)
            ? d
            : throw new FlagException($"{flag} expects a number, got {value}");

    private sealed class FlagException(string message) : Exception(message);
}

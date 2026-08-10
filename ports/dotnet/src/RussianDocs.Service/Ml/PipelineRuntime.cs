using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Pipeline;
using RussianDocs.DocumentProcessing.ViewModel;
using RussianDocs.Service.Errors;
using Image = RussianDocs.DocumentProcessing.Imaging.Image;
// Aliased because ASP.NET Core's Microsoft.AspNetCore.Http.Results is in scope everywhere in this
// project, and `Results` unqualified would silently mean the wrong one.
using PipelineResults = RussianDocs.DocumentProcessing.Pipeline.Results;

namespace RussianDocs.Service.Ml;

/// <summary>
/// Owns and safely calls the recognition pipeline.
///
/// <para>
/// **This is the reference part of the reference project.** Everything below encodes a rule that is
/// easy to get wrong and expensive to debug, and each was verified against the library rather than
/// inferred from documentation. The Python original states ten; the ones that survive into .NET, and
/// the two that do NOT, are laid out here.
/// </para>
///
/// <list type="number">
/// <item>
/// <b>A pipeline instance is not re-entrant.</b> In Python <c>process_img</c> rebinds
/// <c>self.results</c> and <c>self.ocr_options</c> on every call, so two concurrent calls on one
/// instance silently return each other's fields — no crash, no reproduction in single-user testing,
/// corrupted data under load. THIS PORT DOES NOT HAVE THAT BUG: <c>Run</c> holds its state in locals
/// and returns it. The lease is kept anyway, for rules 3 and 9, and because removing it would make
/// this port differ structurally from the reference and from Go.
/// </item>
/// <item>
/// <b>The per-session CUDA lock does not help with (1).</b> It serialises individual ONNX
/// <c>Run()</c> calls on GPU; it fixes device wedging, not re-entrancy. Different problem, different
/// scope.
/// </item>
/// <item>
/// <b>Transform the result before releasing the lease.</b> In Python <c>results</c> IS
/// <c>pipeline.results</c>, and the next call replaces it. <see cref="Recognise"/> does the whole
/// read-and-convert inside the lease for that reason — and the signature makes it structural rather
/// than a rule to remember, because <see cref="Use{T}"/> is the only way in.
/// </item>
/// <item>
/// <b>The library's own warmup cannot report failure</b> (it swallows exceptions into a print),
/// which is why warmup here calls the ordinary path and lets the error surface.
/// </item>
/// <item>
/// <b>Warmup needs a REAL document.</b> A synthetic grey frame classifies as 'NONE' and
/// short-circuits before the border, field and OCR stages, warming perhaps a fifth of the graph. It
/// must be an ANONYMISED repository sample — warmup re-reads the file at every start, so pointing it
/// at a real document is a data-handling error, not just a taste one.
/// </item>
/// <item>
/// <b>The library prints to stdout</b> in Python, which would corrupt a JSON log stream. Not
/// applicable here: this port logs through <c>ILogger</c> and prints nothing.
/// </item>
/// <item>
/// <b>A listed CUDA provider does not mean a working GPU</b>, and in a container without
/// <c>--gpus</c> the provider SEGFAULTS instead of erroring. Hence the device-node probe gating the
/// attempt — see <see cref="DeviceResolution.GpuVisible"/>.
/// </item>
/// <item>
/// <b>GPU does not mean GPU OCR.</b> The detectors run on CUDA while the OCR engines stay on CPU,
/// because per-word dynamic widths are far slower on CUDA — measured at 13.7x end-to-end in the Go
/// port. <see cref="RuntimeInfo"/> reports <c>device</c> and <c>ocr_device</c> SEPARATELY so the
/// status page can say so instead of claiming "GPU active".
/// </item>
/// <item>
/// <b>Models load eagerly and cost 215 MB.</b> Twelve sessions per instance; a second instance on
/// one card is also a second CUDA context. Hence a pool of size 1.
/// </item>
/// <item>
/// <b>Only this namespace touches the library from the service side.</b> That keeps the rest of the
/// service testable without 215 MB of models and bounds the work of porting the service again.
/// </item>
/// </list>
/// </summary>
public sealed class PipelineRuntime(ILogger log) : IDisposable
{
    /// <summary>
    /// How long a caller waits for a free pipeline before giving up.
    ///
    /// <para>
    /// SHORT ON PURPOSE: a queued job that cannot get a pipeline should go back on the queue and
    /// surface as "degraded", not block a worker indefinitely.
    /// </para>
    /// </summary>
    public static readonly TimeSpan LeaseTimeout = TimeSpan.FromSeconds(5);

    public const string StateInitializing = "initializing";
    public const string StateReady = "ready";
    public const string StateError = "error";

    private readonly object _gate = new();
    private RuntimeInfo _info = new() { State = StateInitializing, Providers = [] };

    /// <summary>
    /// The pool: a counting semaphore plus a bag of instances.
    ///
    /// <para>
    /// A semaphore rather than a plain lock because the lease needs a TIMEOUT, which a monitor cannot
    /// express, and because "wait for one of N" is exactly what a semaphore is. The Go port uses a
    /// buffered channel for the same reason; Kotlin will use a semaphore plus a queue.
    /// </para>
    /// </summary>
    private readonly ConcurrentBag<Instance> _pool = [];

    private SemaphoreSlim? _available;
    private int _poolSize;

    /// <summary>
    /// One recognition pipeline.
    ///
    /// <para>
    /// A type rather than a bare set of modules so the pool has something to hold, and so disposal has
    /// one place to release twelve sessions.
    /// </para>
    /// </summary>
    public sealed class Instance(Device device, string modelFormat, string ocrMode,
        Recognizer pipeline) : IDisposable
    {
        public Device Device { get; } = device;

        /// <summary>
        /// Separate from <see cref="Device"/> — rule 8. A separate value rather than a derived one so
        /// the status page can report the two independently, which is the whole point: an operator
        /// looking at nvidia-smi and seeing idle OCR needs the service to have said so first.
        /// </summary>
        public Device OcrDevice { get; } = DocumentProcessing.Inference.Device.Cpu;

        public string ModelFormat { get; } = modelFormat;
        public string OcrMode { get; } = ocrMode;
        internal Recognizer Pipeline { get; } = pipeline;

        public void Dispose() => Pipeline.Dispose();
    }

    /// <summary>Configuration for <see cref="Init"/>.</summary>
    public sealed record Options
    {
        /// <summary>auto | cpu | gpu</summary>
        public string ComputeDevice { get; init; } = "auto";

        public string ModelFormat { get; init; } = "ONNX";
        public string OcrMode { get; init; } = "accurate";
        public string WarmupImage { get; init; } = "";
        public int PoolSize { get; init; } = 1;

        /// <summary>Locates <c>samples/</c> for the warmup fallback.</summary>
        public string? RepoRoot { get; init; }
    }

    /// <summary>A snapshot, with the live pool counts filled in.</summary>
    public RuntimeInfo Info()
    {
        RuntimeInfo snapshot;
        lock (_gate)
        {
            snapshot = _info;
        }
        return snapshot with
        {
            PoolSize = _poolSize,
            PoolAvailable = _available?.CurrentCount ?? 0,
            // Copied so a caller cannot mutate the published list.
            Providers = snapshot.Providers.ToArray(),
        };
    }

    private void Set(Func<RuntimeInfo, RuntimeInfo> mutate)
    {
        lock (_gate)
        {
            _info = mutate(_info);
        }
    }

    public bool IsReady => Info().State == StateReady;

    /// <summary>
    /// Builds the pipelines, warms them, and publishes what actually happened.
    ///
    /// <para>
    /// Blocking and slow. **NEVER THROWS for a recognition failure**: a failure is recorded as
    /// <c>State == "error"</c> so the service can still serve its status page and explain itself
    /// rather than refusing to start. A service that will not boot cannot tell you why it will not
    /// boot.
    /// </para>
    /// </summary>
    public RuntimeInfo Init(Options opts)
    {
        // Providers are OBSERVED, not advertised. CPU is always real; CUDA is added below only if a
        // session actually builds on it. The reference reports what the library advertises, which is
        // the very list rule 7 says cannot be trusted.
        string[] providers = ["CPUExecutionProvider"];
        Set(i => i with
        {
            Providers = providers,
            ModelFormat = opts.ModelFormat,
            OcrMode = opts.OcrMode,
            RequestedDevice = opts.ComputeDevice,
        });

        // TWO INDEPENDENT CONDITIONS, and both are required. The provider list says the GPU build is
        // present; GpuVisible says a device was actually passed through. With the first true and the
        // second false, building a CUDA session TERMINATES the process instead of returning an error.
        bool hasDevice = DeviceResolution.GpuVisible();

        string wanted = opts.ComputeDevice;
        switch (opts.ComputeDevice)
        {
            case "auto":
                wanted = hasDevice ? "gpu" : "cpu";
                break;
            case "gpu" when !hasDevice:
                log.LogError(
                    "[RUNTIME] compute_device=gpu but no GPU is visible to this process — " +
                    "refusing to attempt CUDA, which would TERMINATE the process rather than " +
                    "fail cleanly. Using CPU. In Docker, pass --gpus all.");
                wanted = "cpu";
                break;
        }

        Device[] attempts = wanted == "gpu" ? [Device.Gpu, Device.Cpu] : [Device.Cpu];

        string? sample = FindWarmupSample(opts.WarmupImage, opts.RepoRoot);
        if (sample is null)
        {
            log.LogWarning("[RUNTIME] no warmup sample found; the first real document will pay " +
                           "the cold-start cost");
        }

        int poolSize = Math.Max(1, opts.PoolSize);
        Exception? lastError = null;

        for (int idx = 0; idx < attempts.Length; idx++)
        {
            Device attempt = attempts[idx];
            log.LogInformation("[RUNTIME] building pipeline on {Device} ({Format}, ocr={Ocr})",
                attempt.Wire(), opts.ModelFormat, opts.OcrMode);
            var started = Stopwatch.StartNew();

            var built = new List<Instance>(poolSize);
            Exception? buildError = null;
            for (int n = 0; n < poolSize; n++)
            {
                try
                {
                    // intraOpThreads: 0 leaves the thread count to ONNX Runtime — the opposite of
                    // the conformance CLI, which pins it to 1. Pinning exists so a thread-count
                    // change cannot shift a reduction by ~1e-6 and flip an argmax; the service has no
                    // goldens to match and wants the throughput.
                    built.Add(new Instance(attempt, opts.ModelFormat, opts.OcrMode,
                        new Recognizer(attempt, intraOpThreads: 0,
                            ocrTier: OcrTierOf(opts.OcrMode))));
                }
                catch (Exception ex)
                {
                    buildError = ex;
                    break;
                }
            }

            if (buildError is not null)
            {
                // Partial builds are released before falling back: leaving them would hold a CUDA
                // context that the CPU attempt then competes with.
                foreach (Instance instance in built)
                {
                    instance.Dispose();
                }
                lastError = buildError;
                string next = idx + 1 < attempts.Length
                    ? $"falling back to {attempts[idx + 1].Wire()}"
                    : "no fallback left";
                log.LogError("[RUNTIME] pipeline init FAILED on {Device}: {Error} — {Next}",
                    attempt.Wire(), buildError.Message, next);
                continue;
            }

            int loadMs = (int)started.ElapsedMilliseconds;
            log.LogInformation("[RUNTIME] {Count} instance(s) constructed on {Device} in {Ms} ms",
                built.Count, attempt.Wire(), loadMs);

            int? warmupMs = null;
            if (sample is not null)
            {
                int total = 0, warmed = 0;
                foreach (Instance instance in built)
                {
                    try
                    {
                        total += Warm(instance, sample);
                        warmed++;
                    }
                    catch (Exception ex)
                    {
                        // A failed warmup is LOGGED, not fatal: the pipeline is built and works; the
                        // first document just pays the cold cost. The reference could not even report
                        // this — its warmup swallows the exception into a print.
                        log.LogWarning("[RUNTIME] warmup failed: {Error}", ex.Message);
                    }
                }
                if (warmed > 0)
                {
                    warmupMs = total / warmed;
                }
            }

            _available = new SemaphoreSlim(built.Count, built.Count);
            foreach (Instance instance in built)
            {
                _pool.Add(instance);
            }
            _poolSize = built.Count;

            Instance first = built[0];
            bool fellBack = wanted == "gpu" && attempt == Device.Cpu;
            // Recorded only now: the session built, so CUDA is not merely installed but working.
            string[] observed = attempt == Device.Gpu
                ? ["CUDAExecutionProvider", .. providers]
                : providers;

            Set(i => i with
            {
                State = StateReady,
                Providers = observed,
                Device = first.Device.Wire(),
                OcrDevice = first.OcrDevice.Wire(),
                FellBack = fellBack,
                LoadMs = loadMs,
                WarmupMs = warmupMs,
                Error = null,
            });
            log.LogInformation(
                "[RUNTIME] ready: device={Device} ocr_device={Ocr} load_ms={Load} instances={N}",
                first.Device.Wire(), first.OcrDevice.Wire(), loadMs, built.Count);
            if (fellBack)
            {
                log.LogError("[RUNTIME] GPU was requested but only CPU worked — check " +
                             "CUDA/cuDNN. Recognition will be slower.");
            }
            return Info();
        }

        log.LogError("[RUNTIME] recognition unavailable; the service will start and accept " +
                     "uploads, but every document will fail: {Error}", lastError?.Message);
        string message = lastError?.Message ?? "unknown error";
        Set(i => i with { State = StateError, Error = message });
        return Info();
    }

    private static OcrTier OcrTierOf(string mode) =>
        mode == "fast" ? OcrTier.Fast : OcrTier.Accurate;

    /// <summary>Drops the pipelines.</summary>
    public void Dispose()
    {
        int drained = 0;
        while (_pool.TryTake(out Instance? instance))
        {
            instance.Dispose();
            drained++;
        }
        _available?.Dispose();
        _available = null;
        Set(i => i with { State = StateInitializing, Device = null, OcrDevice = null });
        log.LogInformation("[RUNTIME] released {Count} pipeline instance(s)", drained);
    }

    /// <summary>
    /// Runs <paramref name="body"/> with exclusive access to one instance.
    ///
    /// <para>
    /// **The only way to reach a pipeline.** A higher-order function rather than Acquire/Release,
    /// deliberately: it makes rule 3 — transform the result BEFORE releasing — structurally enforced
    /// instead of a comment somebody deletes. Python expresses the same thing as a context manager,
    /// Go as a callback, Kotlin as an inline lambda.
    /// </para>
    ///
    /// <para>
    /// Throws <see cref="ErrorKind.RuntimeNotReady"/> before the models finish loading and
    /// <see cref="ErrorKind.PipelineBusy"/> if none becomes free within the timeout. Both are
    /// TRANSIENT, so the caller requeues rather than failing the job.
    /// </para>
    /// </summary>
    public T Use<T>(TimeSpan timeout, Func<Instance, T> body)
    {
        RuntimeInfo info = Info();
        switch (info.State)
        {
            case StateError:
                throw ServiceException.NotReady(
                    $"Recognition runtime failed to start: {info.Error}");
            case StateReady:
                break;
            default:
                throw ServiceException.NotReady("Recognition runtime is still loading models");
        }

        SemaphoreSlim? available = _available;
        if (available is null || !available.Wait(timeout))
        {
            throw ServiceException.Busy(
                $"No pipeline became available within {timeout.TotalSeconds:0.#}s");
        }

        Instance? instance = null;
        try
        {
            if (!_pool.TryTake(out instance))
            {
                // Cannot happen while the semaphore and the bag agree, and worth saying rather than
                // dereferencing null: the two are only ever changed together in Init and Dispose.
                throw ServiceException.Busy("Pipeline pool is inconsistent");
            }
            return body(instance);
        }
        finally
        {
            // Returned in a finally so an exception in the body cannot leak the instance and wedge
            // the pool at zero available — which would look exactly like the hang the lease timeout
            // exists to report.
            if (instance is not null)
            {
                _pool.Add(instance);
            }
            available.Release();
        }
    }

    /// <summary>The per-document knobs.</summary>
    public sealed record RecogniseOptions
    {
        public bool IncludeDebug { get; init; }
        public double Docconf { get; init; } = 0.5;
        public int ImgSize { get; init; } = 1500;
        public TimeSpan LeaseTimeout { get; init; }
    }

    /// <summary>What <see cref="Recognise"/> produces.</summary>
    public sealed record RecogniseResult
    {
        public required Payload ViewModel { get; init; }

        /// <summary>
        /// The corrected canvas, RGB and OWNED BY THE CALLER, who must dispose it. <c>null</c> when
        /// the document short-circuited as unrecognised.
        /// </summary>
        public Image? Canvas { get; init; }
    }

    /// <summary>
    /// Processes one document. The whole public surface of this type.
    ///
    /// <para>
    /// The canvas is RGB and the encoder writes BGR, so the artifact layer converts before writing —
    /// see <c>Artifacts.SaveCanvas</c>. Getting that wrong swaps red and blue in every stored
    /// document, and on a passport it looks plausible enough to ship unnoticed.
    /// </para>
    /// </summary>
    public RecogniseResult Recognise(string imagePath, RecogniseOptions opts)
    {
        TimeSpan timeout = opts.LeaseTimeout > TimeSpan.Zero ? opts.LeaseTimeout : LeaseTimeout;

        return Use(timeout, instance =>
        {
            using PipelineResults results = instance.Pipeline.Run(imagePath, new RunOptions
            {
                Docconf = opts.Docconf,
                ImgSize = opts.ImgSize,
                IncludeDebug = opts.IncludeDebug,
            });

            // Built INSIDE the lease — rule 3. Structural here rather than remembered, because there
            // is no way to hold `results` past the lambda.
            Payload payload = Recognizer.BuildViewModel(results, opts.IncludeDebug);

            // **TakeCanvas, not a bare field read.** The canvas must outlive the run, but every other
            // image the run allocated must not. In the Go port, reading the field and returning left
            // the intermediates — the fully decoded original among them — alive forever: 663 MB ->
            // 4018 MB across 230 documents, growing without bound. Nothing in the conformance suite
            // could catch it, because the CLI disposes its Results after a single document.
            Image? canvas = results.TakeCanvas();

            // The canvas ESCAPES the lease deliberately and is now owned by the caller: it is a
            // standalone Mat, not a view into pipeline state, so unlike the reference's `results` it
            // is safe to read after release.
            return new RecogniseResult { ViewModel = payload, Canvas = canvas };
        });
    }

    /// <summary>Pays the cold-start cost once, up front.</summary>
    private static int Warm(Instance instance, string sample)
    {
        var started = Stopwatch.StartNew();
        using PipelineResults results = instance.Pipeline.Run(sample,
            new RunOptions { Docconf = 0.5, ImgSize = 1500 });
        return (int)started.ElapsedMilliseconds;
    }

    /// <summary>
    /// Resolves the configured image, else picks one from <c>samples/</c>.
    ///
    /// <para>
    /// **Only anonymised repository samples are eligible.** Warmup re-reads this file at every start,
    /// so a real document here would be read on every boot of every deployment — which is why the
    /// fallback searches <c>samples/</c> and never the data directory.
    /// </para>
    /// </summary>
    private string? FindWarmupSample(string configured, string? repoRoot)
    {
        if (configured.Length > 0)
        {
            if (File.Exists(configured))
            {
                return configured;
            }
            log.LogWarning("[RUNTIME] configured warmup image does not exist: {Path}", configured);
        }
        if (repoRoot is null)
        {
            return null;
        }
        // A fixed preference order rather than "the first file found": the chosen sample decides
        // which parts of the graph get warmed, and a directory listing order is not a decision
        // anybody made.
        foreach (string relative in new[]
                 {
                     Path.Combine("samples", "INTPASSPORT_2011", "12_CR_INTPASSPORT_2011.jpg"),
                     Path.Combine("samples", "DL_2011", "1_CR_DL_2010.jpg"),
                 })
        {
            string candidate = Path.Combine(repoRoot, relative);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        return null;
    }
}

/// <summary>
/// What the recognition runtime actually ended up doing.
///
/// <para>
/// Reported verbatim by <c>GET /status</c>. An operator needs the REAL answer, not the configured
/// intent, because the two differ whenever a GPU was asked for and not obtained.
/// </para>
/// </summary>
public sealed record RuntimeInfo
{
    [JsonPropertyName("state")] public required string State { get; init; }
    [JsonPropertyName("providers")] public required IReadOnlyList<string> Providers { get; init; }

    /// <summary>What the detectors use. <c>ocr_device</c> differs from it BY DESIGN — rule 8.</summary>
    [JsonPropertyName("device")] public string? Device { get; init; }

    [JsonPropertyName("ocr_device")] public string? OcrDevice { get; init; }
    [JsonPropertyName("model_format")] public string? ModelFormat { get; init; }
    [JsonPropertyName("ocr_mode")] public string? OcrMode { get; init; }
    [JsonPropertyName("requested_device")] public string? RequestedDevice { get; init; }

    /// <summary>
    /// Records that a GPU was requested and CPU was used. The single most useful field on the status
    /// page, because it is the difference between "slow" and "broken".
    /// </summary>
    [JsonPropertyName("fell_back")] public bool FellBack { get; init; }

    [JsonPropertyName("load_ms")] public int? LoadMs { get; init; }
    [JsonPropertyName("warmup_ms")] public int? WarmupMs { get; init; }
    [JsonPropertyName("library_version")] public string? LibraryVersion { get; init; }
    [JsonPropertyName("error")] public string? Error { get; init; }
    [JsonPropertyName("pool_size")] public int PoolSize { get; init; }
    [JsonPropertyName("pool_available")] public int PoolAvailable { get; init; }
}

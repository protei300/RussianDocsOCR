using System.Diagnostics;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using RussianDocs.DocumentProcessing.ViewModel;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Settings;
using RussianDocs.Service.Store;
using Image = RussianDocs.DocumentProcessing.Imaging.Image;

namespace RussianDocs.Service.Worker;

/// <summary>
/// Live progress for one document.
///
/// <para>
/// **Two honest steps rather than five interpolated ones.** The pipeline exposes no progress
/// callbacks, so a finer breakdown would be theatre — and at ~0.4 s per document a five-segment
/// animated bar is a lie the user can see through. <c>recognizing</c> self-calibrates from real
/// completions instead of using a constant.
/// </para>
/// </summary>
public sealed record Progress
{
    [JsonPropertyName("step")] public required string Step { get; init; }
    [JsonPropertyName("label")] public required string Label { get; init; }
    [JsonPropertyName("pct")] public double Pct { get; init; }
    [JsonPropertyName("eta_sec")] public double EtaSec { get; init; }
    [JsonPropertyName("queue_position")] public int? QueuePosition { get; init; }
}

/// <summary>
/// The background recognition loop.
///
/// <para>
/// One task pulls queued documents and runs them through the pipeline. Several choices look arbitrary
/// until something breaks, so the reasoning is attached:
/// </para>
///
/// <list type="bullet">
/// <item><b>Event-driven, not fixed-interval polling.</b> Recognition takes ~0.4 s; a ten-second poll
/// would dominate the latency a user perceives. Uploads signal the loop, and a two-second timeout is
/// only a safety net for anything that enqueues without signalling.</item>
///
/// <item><b>A dedicated concurrency bound, not the ambient one.</b> Python's default executor sizes
/// itself to min(32, cpu+4), and twenty threads racing for one pipeline lease is not useful. Here the
/// bound is structural: ONE drain loop, so the invariant holds by construction rather than by pool
/// sizing.</item>
///
/// <item><b>A timeout cannot kill work already inside the library.</b> This is the sharpest edge. In
/// Python <c>asyncio.wait_for</c> cancels the coroutine while the executor thread keeps running
/// <c>process_img</c> and keeps holding its lease; a .NET <c>Task</c> running synchronous native code
/// cannot be aborted either. The job is marked failed and the loop moves on; the lease is released
/// when that task finally finishes. Later jobs then get <c>PipelineBusy</c> — a BOUNDED wait — and
/// requeue rather than blocking forever. A genuinely hung ONNX call needs a process restart, and the
/// container's restart policy is the last line of defence.</item>
///
/// <item><b>Transient versus deterministic failures.</b> Retrying a corrupt JPEG forever is as wrong
/// as giving up on a CUDA hiccup, so only transient failures consume a retry.</item>
/// </list>
///
/// <para>Port of <c>service/worker.py</c>.</para>
/// </summary>
public sealed class RecognitionWorker
{
    /// <summary>
    /// The fallback interval. Normal flow is driven by the wake signal; this only catches anything
    /// that enqueues without signalling.
    /// </summary>
    private static readonly TimeSpan QueuePoll = TimeSpan.FromSeconds(2);

    private sealed record StepConfig(string Label, double PctStart, double PctEnd, double Duration);

    private static readonly Dictionary<string, StepConfig> StepConfigs = new(StringComparer.Ordinal)
    {
        ["loading"] = new("Loading models", 0, 90, 20.0),
        ["recognizing"] = new("Recognising document", 5, 95, 0.6),
    };

    private readonly IDocumentStore _db;
    private readonly PipelineRuntime _runtime;
    private readonly Config.Settings _cfg;
    private readonly SettingsRepository _settings;
    private readonly ILogger _log;

    /// <summary>
    /// The "there may be work" signal.
    ///
    /// <para>
    /// A flag, not a queue: many uploads collapse into one wake-up and no producer can ever block on
    /// a busy loop. The Go port uses a capacity-1 channel with a non-blocking send for the same
    /// reason.
    /// </para>
    /// </summary>
    private readonly SemaphoreSlim _wake = new(0, 1);

    /// <summary>Gates the drain loop until model loading has finished (or failed).</summary>
    private readonly TaskCompletionSource _runtimeReady =
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    private readonly object _gate = new();
    private readonly Dictionary<int, (string Step, long StartedTicks)> _processing = [];
    private double _durationEma = 0.6;

    public RecognitionWorker(IDocumentStore db, PipelineRuntime runtime, Config.Settings cfg,
        SettingsRepository settings, ILogger log)
    {
        _db = db;
        _runtime = runtime;
        _cfg = cfg;
        _settings = settings;
        _log = log;
    }

    /// <summary>
    /// Wakes the drain loop. Called by the upload and reprocess endpoints.
    ///
    /// <para>
    /// Non-blocking: if a wake is already pending the signal is dropped, which is correct because the
    /// loop rescans the queue from scratch anyway.
    /// </para>
    /// </summary>
    public void NotifyNewWork()
    {
        try
        {
            if (_wake.CurrentCount == 0)
            {
                _wake.Release();
            }
        }
        catch (SemaphoreFullException)
        {
            // Raced with another notifier; a wake is pending either way, which is all that matters.
        }
    }

    /// <summary>The current EMA, used by the status page.</summary>
    public double AverageDurationSec()
    {
        lock (_gate)
        {
            return _durationEma;
        }
    }

    private void SetStep(int id, string step)
    {
        lock (_gate)
        {
            _processing[id] = (step, Stopwatch.GetTimestamp());
        }
    }

    private void ClearStep(int id)
    {
        lock (_gate)
        {
            _processing.Remove(id);
        }
    }

    private void RecordDuration(double seconds)
    {
        lock (_gate)
        {
            _durationEma = 0.7 * _durationEma + 0.3 * Math.Max(seconds, 0.05);
        }
    }

    /// <summary>
    /// Live progress, or <c>null</c> when the document is not being processed.
    ///
    /// <para>
    /// <c>null</c> is a real answer, not an error: the endpoint returns 200 with a JSON null for a
    /// document that is queued or finished, and a 404 there would make the SPA treat a completed
    /// document as missing.
    /// </para>
    /// </summary>
    public Progress? DocumentProgress(int id)
    {
        string step;
        long startedTicks;
        double ema;
        lock (_gate)
        {
            if (!_processing.TryGetValue(id, out (string Step, long StartedTicks) state))
            {
                return null;
            }
            (step, startedTicks) = state;
            ema = _durationEma;
        }

        StepConfig cfg = StepConfigs.GetValueOrDefault(step, StepConfigs["recognizing"]);
        double duration = step == "recognizing" ? ema : cfg.Duration;
        double elapsed = Stopwatch.GetElapsedTime(startedTicks).TotalSeconds;
        // Capped below 1: a bar that reaches 100 % and then waits is worse than one that stalls at 95
        // and then jumps, because the first looks broken.
        double fraction = Math.Min(elapsed / Math.Max(duration, 0.05), 0.95);
        double pct = cfg.PctStart + fraction * (cfg.PctEnd - cfg.PctStart);

        return new Progress
        {
            Step = step,
            Label = cfg.Label,
            Pct = Round1(pct),
            EtaSec = Round1(Math.Max(0, duration - elapsed)),
        };
    }

    private static double Round1(double v) => Math.Round(v * 10) / 10;

    /// <summary>
    /// Launches the runtime initialisation and the drain loop.
    ///
    /// <para>
    /// **Runtime init runs on its OWN task and startup does not wait for it**: 215 MB of sessions plus
    /// a warmup document take seconds, and blocking startup would delay <c>/health</c> and fight
    /// Docker's healthcheck. Uploads are accepted immediately and wait in the queue, which is exactly
    /// what the async design is for.
    /// </para>
    /// </summary>
    public void Start(CancellationToken ct)
    {
        int recovered = Documents.ResetStaleProcessing(_db);
        if (recovered > 0)
        {
            _log.LogInformation("[WORKER] requeued {Count} document(s) left mid-processing",
                recovered);
        }

        // Long-running so the scheduler gives each its own thread rather than a pool slot: both block
        // for seconds at a time inside native code, and starving the pool would stall the HTTP side.
        _ = Task.Factory.StartNew(() => InitRuntime(), ct, TaskCreationOptions.LongRunning,
            TaskScheduler.Default);
        _ = Task.Factory.StartNew(() => DrainLoop(ct), ct, TaskCreationOptions.LongRunning,
            TaskScheduler.Default);
    }

    private void InitRuntime()
    {
        try
        {
            string device = SettingsSchema.TypedString("compute_device",
                _settings.SettingValue(_db, "compute_device"), _cfg.ComputeDevice);
            string mode = SettingsSchema.TypedString("ocr_mode",
                _settings.SettingValue(_db, "ocr_mode"), _cfg.OcrMode);

            RuntimeInfo info = _runtime.Init(new PipelineRuntime.Options
            {
                ComputeDevice = device,
                ModelFormat = _cfg.ModelFormat,
                OcrMode = mode,
                WarmupImage = _cfg.WarmupImage,
                PoolSize = _cfg.PipelinePoolSize,
                RepoRoot = _cfg.RepoRoot(),
            });
            if (info.State != PipelineRuntime.StateReady)
            {
                _log.LogError("[WORKER] recognition runtime failed to start: {Error}", info.Error);
            }
        }
        finally
        {
            // Released EITHER WAY: with a broken runtime the drain loop still needs to run, so queued
            // documents fail with a clear message instead of sitting in 'queued' forever with no
            // explanation.
            _runtimeReady.TrySetResult();
            NotifyNewWork();
        }
    }

    private void DrainLoop(CancellationToken ct)
    {
        _log.LogInformation("[WORKER] drain loop started");
        try
        {
            _runtimeReady.Task.Wait(ct);
        }
        catch (OperationCanceledException)
        {
            return;
        }

        while (!ct.IsCancellationRequested)
        {
            int? id = Documents.NextQueued(_db);
            if (id is null)
            {
                try
                {
                    // Waits for a signal OR the fallback tick.
                    _wake.Wait(QueuePoll, ct);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                continue;
            }
            ProcessDocument(id.Value, ct);
        }
        _log.LogInformation("[WORKER] drain loop stopped");
    }

    /// <summary>The result of one recognition attempt.</summary>
    private sealed record Outcome(PipelineRuntime.RecogniseResult? Result, int ElapsedMs,
        Exception? Error);

    /// <summary>
    /// Waits for abandoned recognition work and releases what it produced.
    ///
    /// <para>
    /// Called when the timeout fires or the service is shutting down. The recognition itself CANNOT be
    /// cancelled — synchronous native code has no kill — so it will finish, and its canvas is an
    /// unmanaged Mat. A missed Dispose here becomes a full canvas held until the finalizer thread
    /// happens to reach it, so this is the only place that frees it deterministically.
    /// </para>
    ///
    /// <para>
    /// It also releases the pipeline lease as a side effect of the work completing, which is why a
    /// subsequent job gets <c>PipelineBusy</c> (a bounded wait) rather than blocking forever.
    /// </para>
    /// </summary>
    private void Reap(int id, Task<Outcome> work) => _ = work.ContinueWith(finished =>
    {
        Outcome got = finished.Result;
        got.Result?.Canvas?.Dispose();
        _log.LogWarning(
            "[WORKER] abandoned recognition for document {Id} finished after {Ms} ms; its canvas " +
            "was released ({Error})", id, got.ElapsedMs, got.Error?.Message ?? "no error");
    }, TaskScheduler.Default);

    /// <summary>
    /// Claims one document and recognises it.
    ///
    /// <para>
    /// The claim is a status transition, and re-reading the record first is what makes it safe: between
    /// the queue scan and here the document may have been deleted or claimed, so a record that is no
    /// longer <c>queued</c> is skipped rather than processed twice.
    /// </para>
    /// </summary>
    private void ProcessDocument(int id, CancellationToken ct)
    {
        Document? record = Documents.GetById(_db, id);
        if (record is null || record.Status != DocumentStatus.Queued)
        {
            return;
        }
        record = Documents.UpdateStatus(_db, record, DocumentStatus.Processing, null, null);

        TimeSpan timeout = TimeSpan.FromSeconds(SettingsSchema.TypedInt("job_timeout_sec",
            _settings.SettingValue(_db, "job_timeout_sec"), _cfg.JobTimeoutSec));
        int maxRetries = SettingsSchema.TypedInt("max_retries",
            _settings.SettingValue(_db, "max_retries"), _cfg.MaxRetries);
        double docconf = SettingsSchema.TypedFloat("docconf",
            _settings.SettingValue(_db, "docconf"), _cfg.Docconf);
        int imgSize = SettingsSchema.TypedInt("img_size",
            _settings.SettingValue(_db, "img_size"), _cfg.ImgSize);

        SetStep(id, "recognizing");

        // The recognition runs on its own task so the timeout can be observed. It CANNOT be cancelled
        // — see the type note.
        Task<Outcome> work = Task.Factory.StartNew(() =>
        {
            var started = Stopwatch.StartNew();
            try
            {
                PipelineRuntime.RecogniseResult result = Recognise(id, docconf, imgSize);
                return new Outcome(result, (int)started.ElapsedMilliseconds, null);
            }
            catch (Exception ex)
            {
                return new Outcome(null, (int)started.ElapsedMilliseconds, ex);
            }
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

        Outcome got;
        bool completed;
        try
        {
            completed = work.Wait(timeout, ct);
        }
        catch (OperationCanceledException)
        {
            // Shutdown. Same reasoning as the timeout path: the task is still running and its result
            // must be released rather than left to a finalizer, which is not a schedule anybody can
            // reason about.
            Reap(id, work);
            ClearStep(id);
            return;
        }

        if (!completed)
        {
            got = new Outcome(null, 0,
                new TimeoutException($"Recognition exceeded {timeout.TotalSeconds:0.#}s"));
            // **The abandoned work still produces a canvas, and somebody has to free it.** The
            // recognition cannot be cancelled, so it will finish and deliver a result nothing is
            // reading any more — and that result owns an unmanaged Mat. Without this reaper every
            // timed-out document holds a full canvas until a finalizer runs, which shows up only in
            // bulk and looks like a slow leak rather than a timeout.
            Reap(id, work);
        }
        else
        {
            got = work.Result;
        }
        ClearStep(id);

        if (got.Error is not null)
        {
            HandleFailure(id, got.Error, maxRetries);
            return;
        }

        using Image? canvas = got.Result!.Canvas;
        RecordDuration(got.ElapsedMs / 1000.0);

        record = Documents.GetById(_db, id);
        if (record is null)
        {
            return; // deleted while we were recognising
        }

        if (canvas is not null)
        {
            try
            {
                Artifacts.SaveCanvas(_db, id, canvas);
                Artifacts.SaveThumbnail(_db, id, canvas, 96);
            }
            catch (Exception ex)
            {
                // A missing preview must NOT fail an otherwise good recognition: the fields are the
                // product, the picture is a convenience.
                _log.LogError("[WORKER] canvas or thumbnail write failed for {Id}: {Error}", id,
                    ex.Message);
            }
        }

        Payload payload = got.Result.ViewModel;
        JsonNode node;
        try
        {
            node = SearchText.ToNode(payload);
        }
        catch (Exception ex)
        {
            HandleFailure(id, ex, maxRetries);
            return;
        }

        try
        {
            Documents.SaveResult(_db, record, node,
                SearchText.Build(record.Filename, payload), got.ElapsedMs);
        }
        catch (Exception ex)
        {
            _log.LogError("[WORKER] cannot save result for {Id}: {Error}", id, ex.Message);
            return;
        }

        _log.LogInformation("[WORKER] done: doc={Id} ms={Ms} type={Type} fields={Fields}",
            id, got.ElapsedMs, payload.DocType, payload.Fields.Count);
    }

    /// <summary>Loads the stored original and runs the pipeline.</summary>
    private PipelineRuntime.RecogniseResult Recognise(int id, double docconf, int imgSize)
    {
        if (Artifacts.OpenArtifact(_db, id, "original") is not { } artifact)
        {
            // Deterministic, so never retried: this is the symptom of the upload race that
            // Documents.ReserveId exists to prevent, and retrying would only hide it.
            throw ServiceException.Unreadable($"Document {id} has no stored original");
        }
        return _runtime.Recognise(artifact.Path, new PipelineRuntime.RecogniseOptions
        {
            Docconf = docconf,
            ImgSize = imgSize,
        });
    }

    /// <summary>
    /// Classifies the error and either requeues or fails the document.
    ///
    /// <para>
    /// **Only TRANSIENT failures consume a retry.** A corrupt JPEG fails immediately and forever,
    /// because the same bytes will fail the same way and a retry loop on them starves the queue.
    /// </para>
    /// </summary>
    private void HandleFailure(int id, Exception error, int maxRetries)
    {
        (string code, bool transient) = Classify(error);
        _log.LogWarning("[WORKER] document {Id} failed: code={Code} transient={Transient} {Error}",
            id, code, transient, error.Message);

        Document? record = Documents.GetById(_db, id);
        if (record is null)
        {
            return;
        }
        string message = error.Message;
        if (transient && record.RetryCount < maxRetries)
        {
            int nextRetry = record.RetryCount + 1;
            Documents.Update(_db, record, d =>
            {
                d.Status = DocumentStatus.Queued;
                d.RetryCount = nextRetry;
                d.Error = message;
                d.ErrorCode = code;
                d.StartedAt = null;
            });
            NotifyNewWork();
            return;
        }
        Documents.UpdateStatus(_db, record, DocumentStatus.Failed, message, code);
    }

    /// <summary>
    /// Maps an error to a machine-readable code and a retry decision.
    ///
    /// <para>
    /// The CODE is separate from the message because the UI is English while a message may not be, and
    /// because a client should branch on a stable token rather than on prose.
    /// </para>
    /// </summary>
    private static (string Code, bool Transient) Classify(Exception error) => error switch
    {
        ServiceException { Kind: ErrorKind.PipelineBusy } => ("pipeline_busy", true),
        ServiceException { Kind: ErrorKind.RuntimeNotReady } => ("runtime_not_ready", true),
        ServiceException { Kind: ErrorKind.ImageUnreadable } => ("image_unreadable", false),
        TimeoutException => ("timeout", true),
        // UNKNOWN IS NOT TRANSIENT. The safe direction: an unrecognised error retried forever stops
        // the queue making progress, and nothing in the log says why.
        _ => ("error", false),
    };
}

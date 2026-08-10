// Package worker is the background recognition loop.
//
// One goroutine pulls queued documents and runs them through the pipeline. Several choices
// look arbitrary until something breaks, so the reasoning is attached:
//
//   - **Event-driven, not fixed-interval polling.** Recognition takes ~0.4 s; a ten-second
//     poll would dominate the latency a user perceives. Uploads signal the loop, and a
//     two-second timeout is only a safety net for anything that enqueues without signalling.
//
//   - **A dedicated concurrency bound, not the ambient one.** Python's default executor
//     sizes itself to min(32, cpu+4), and twenty threads racing for one pipeline lease is
//     not useful. Here the bound is structural: ONE drain goroutine, so the invariant holds
//     by construction rather than by pool sizing.
//
//   - **A timeout cannot kill work already inside the library.** This is the sharpest edge.
//     In Python `asyncio.wait_for` cancels the coroutine while the executor thread keeps
//     running `process_img` and keeps holding its lease; in Go a goroutine cannot be
//     cancelled from outside either. The job is marked failed and the loop moves on; the
//     lease is released when that goroutine finally finishes. Later jobs then get
//     ErrPipelineBusy — a BOUNDED wait — and requeue rather than blocking forever. A
//     genuinely hung ONNX call needs a process restart, and the container's restart policy
//     is the last line of defence.
//
//   - **Transient versus deterministic failures.** Retrying a corrupt JPEG forever is as
//     wrong as giving up on a CUDA hiccup, so only transient failures consume a retry.
//
// Port of service/worker.py.
package worker

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/runtime"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/settingsschema"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// QueuePoll is the fallback interval. Normal flow is driven by the wake signal; this only
// catches anything that enqueues without signalling.
const QueuePoll = 2 * time.Second

// Progress reports two honest steps rather than five interpolated ones.
//
// The pipeline exposes no progress callbacks, so a finer breakdown would be theatre — and at
// ~0.4 s per document a five-segment animated bar is a lie the user can see through.
// `recognizing` self-calibrates from real completions instead of using a constant.
type Progress struct {
	Step          string  `json:"step"`
	Label         string  `json:"label"`
	Pct           float64 `json:"pct"`
	EtaSec        float64 `json:"eta_sec"`
	QueuePosition *int    `json:"queue_position"`
}

type stepConfig struct {
	label            string
	pctStart, pctEnd float64
	duration         float64
}

var stepConfigs = map[string]stepConfig{
	"loading":     {"Loading models", 0, 90, 20.0},
	"recognizing": {"Recognising document", 5, 95, 0.6},
}

// Worker owns the drain loop.
type Worker struct {
	db  store.DocumentStore
	rt  *runtime.Runtime
	cfg config.Settings

	// wake is a capacity-1 channel used as a "there may be work" signal.
	//
	// Capacity 1 and a non-blocking send: the signal is a FLAG, not a queue, so many
	// uploads collapse into one wake-up and no producer can ever block on a busy loop.
	wake chan struct{}
	// runtimeReady gates the drain loop until model loading has finished (or failed).
	runtimeReady chan struct{}
	readyOnce    sync.Once

	mu sync.Mutex
	// processing tracks the live step per document, for the progress endpoint.
	processing map[int]stepState
	// durationEMA tracks real completion times so the ETA follows reality.
	durationEMA float64
}

type stepState struct {
	step    string
	started time.Time
}

func New(db store.DocumentStore, rt *runtime.Runtime, cfg config.Settings) *Worker {
	return &Worker{
		db: db, rt: rt, cfg: cfg,
		wake:         make(chan struct{}, 1),
		runtimeReady: make(chan struct{}),
		processing:   map[int]stepState{},
		durationEMA:  0.6,
	}
}

// NotifyNewWork wakes the drain loop. Called by the upload and reprocess endpoints.
//
// Non-blocking: if a wake is already pending the signal is dropped, which is correct because
// the loop rescans the queue from scratch anyway.
func (w *Worker) NotifyNewWork() {
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

// AverageDurationSec is the current EMA, used by the status page.
func (w *Worker) AverageDurationSec() float64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.durationEMA
}

func (w *Worker) setStep(id int, step string) {
	w.mu.Lock()
	w.processing[id] = stepState{step: step, started: time.Now()}
	w.mu.Unlock()
}

func (w *Worker) clearStep(id int) {
	w.mu.Lock()
	delete(w.processing, id)
	w.mu.Unlock()
}

func (w *Worker) recordDuration(seconds float64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.durationEMA = 0.7*w.durationEMA + 0.3*math.Max(seconds, 0.05)
}

// DocumentProgress returns live progress, or nil when the document is not being processed.
//
// nil is a real answer, not an error: the endpoint returns 200 with a JSON null for a
// document that is queued or finished, and a 404 there would make the SPA treat a completed
// document as missing.
func (w *Worker) DocumentProgress(id int) *Progress {
	w.mu.Lock()
	state, ok := w.processing[id]
	ema := w.durationEMA
	w.mu.Unlock()
	if !ok {
		return nil
	}

	cfg, found := stepConfigs[state.step]
	if !found {
		cfg = stepConfigs["recognizing"]
	}
	duration := cfg.duration
	if state.step == "recognizing" {
		duration = ema
	}
	elapsed := time.Since(state.started).Seconds()
	// Capped below 1: a bar that reaches 100 % and then waits is worse than one that
	// stalls at 95 and then jumps, because the first looks broken.
	fraction := math.Min(elapsed/math.Max(duration, 0.05), 0.95)
	pct := cfg.pctStart + fraction*(cfg.pctEnd-cfg.pctStart)

	return &Progress{
		Step:   state.step,
		Label:  cfg.label,
		Pct:    round1(pct),
		EtaSec: round1(math.Max(0, duration-elapsed)),
	}
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }

// Start launches the runtime initialisation and the drain loop.
//
// Runtime init runs in its OWN goroutine and startup does not wait for it: 215 MB of
// sessions plus a warmup document take seconds, and blocking startup would delay /health and
// fight Docker's healthcheck. Uploads are accepted immediately and wait in the queue, which
// is exactly what the async design is for.
func (w *Worker) Start(ctx context.Context) {
	recovered := repo.ResetStaleProcessing(w.db)
	if recovered > 0 {
		slog.Info("[WORKER] requeued documents left mid-processing", "count", recovered)
	}

	go w.initRuntime(ctx)
	go w.drainLoop(ctx)
}

// Stop releases the pipelines. The loops end with the context.
func (w *Worker) Stop() {
	w.rt.Shutdown()
	slog.Info("[WORKER] stopped")
}

func (w *Worker) initRuntime(ctx context.Context) {
	// Released EITHER WAY, in a defer: with a broken runtime the drain loop still needs to
	// run, so queued documents fail with a clear message instead of sitting in 'queued'
	// forever with no explanation.
	defer w.readyOnce.Do(func() { close(w.runtimeReady) })

	device := settingsschema.TypedString("compute_device",
		repo.SettingValue(w.db, w.cfg, "compute_device"), w.cfg.ComputeDevice)
	mode := settingsschema.TypedString("ocr_mode",
		repo.SettingValue(w.db, w.cfg, "ocr_mode"), w.cfg.OcrMode)

	info := w.rt.Init(runtime.Options{
		ComputeDevice: device,
		ModelFormat:   w.cfg.ModelFormat,
		OcrMode:       mode,
		WarmupImage:   w.cfg.WarmupImage,
		PoolSize:      w.cfg.PipelinePoolSize,
		RepoRoot:      w.cfg.RepoRoot(),
	})
	if info.State != runtime.StateReady {
		slog.Error("[WORKER] recognition runtime failed to start", "err", info.Error)
	}
	w.NotifyNewWork()
}

func (w *Worker) drainLoop(ctx context.Context) {
	slog.Info("[WORKER] drain loop started")
	select {
	case <-w.runtimeReady:
	case <-ctx.Done():
		return
	}

	for {
		if ctx.Err() != nil {
			slog.Info("[WORKER] drain loop stopped")
			return
		}
		id, ok := repo.NextQueued(w.db)
		if !ok {
			// Waits for a signal OR the fallback tick. Draining the channel first would
			// race with a producer; select on both instead.
			select {
			case <-w.wake:
			case <-time.After(QueuePoll):
			case <-ctx.Done():
			}
			continue
		}
		w.processDocument(ctx, id)
	}
}

// outcome is the result of one recognition attempt.
//
// At package scope rather than inside processDocument because `reap` needs it: the abandoned
// work has to be drained and freed from somewhere the timeout path can reach.
type outcome struct {
	res     *runtime.Result
	elapsed int
	err     error
}

// reap waits for abandoned recognition work and releases what it produced.
//
// Called when the timeout fires or the service is shutting down. The recognition itself CANNOT
// be cancelled — a goroutine has no kill — so it will finish, and its canvas is an unmanaged Mat
// that Go's garbage collector will not free. This is the only place that frees it.
//
// It also releases the pipeline lease as a side effect of the work completing, which is why a
// subsequent job gets ErrPipelineBusy (a bounded wait) rather than blocking forever.
func reap(id int, done <-chan outcome) {
	got := <-done
	if got.res != nil && got.res.HasCanvas {
		_ = got.res.Canvas.Close()
	}
	slog.Warn("[WORKER] abandoned recognition finished; its canvas was released",
		"doc", id, "ms", got.elapsed, "err", got.err)
}

// processDocument claims one document and recognises it.
//
// The claim is a status transition, and re-reading the record first is what makes it safe:
// between the queue scan and here the document may have been deleted or claimed, so a
// record that is no longer `queued` is skipped rather than processed twice.
func (w *Worker) processDocument(ctx context.Context, id int) {
	record := repo.GetByID(w.db, id)
	if record == nil || record.Status != model.StatusQueued {
		return
	}
	record, err := repo.UpdateStatus(w.db, record, model.StatusProcessing, nil, nil)
	if err != nil {
		slog.Error("[WORKER] cannot claim document", "doc", id, "err", err)
		return
	}

	timeout := time.Duration(settingsschema.TypedInt("job_timeout_sec",
		repo.SettingValue(w.db, w.cfg, "job_timeout_sec"), w.cfg.JobTimeoutSec)) * time.Second
	maxRetries := settingsschema.TypedInt("max_retries",
		repo.SettingValue(w.db, w.cfg, "max_retries"), w.cfg.MaxRetries)
	docconf := settingsschema.TypedFloat("docconf",
		repo.SettingValue(w.db, w.cfg, "docconf"), w.cfg.Docconf)
	imgSize := settingsschema.TypedInt("img_size",
		repo.SettingValue(w.db, w.cfg, "img_size"), w.cfg.ImgSize)

	w.setStep(id, "recognizing")

	// The recognition runs in its own goroutine so the timeout can be observed. It CANNOT
	// be cancelled — see the package note. The channel is buffered so that goroutine can
	// always finish and exit even after the timeout path has moved on; an unbuffered
	// channel would leak it forever.
	done := make(chan outcome, 1)
	go func() {
		started := time.Now()
		res, err := w.recognise(id, docconf, imgSize)
		done <- outcome{res: res, elapsed: int(time.Since(started).Milliseconds()), err: err}
	}()

	var got outcome
	select {
	case got = <-done:
	case <-time.After(timeout):
		got = outcome{err: fmt.Errorf("recognition exceeded %s", timeout)}
		// **The abandoned work still produces a canvas, and somebody has to free it.**
		// The recognition cannot be cancelled, so it will finish and deliver a Result into
		// the buffered channel that nothing is reading any more — and that Result owns an
		// unmanaged Mat. Without this reaper every timed-out document leaks a full canvas,
		// which is precisely the failure mode that only shows up in bulk and looks like a
		// slow memory leak rather than a timeout.
		go reap(id, done)
	case <-ctx.Done():
		// Same reasoning on shutdown: the goroutine is still running and its result must be
		// released rather than left to the garbage collector, which does not free Mats.
		go reap(id, done)
		w.clearStep(id)
		return
	}
	w.clearStep(id)

	if got.err != nil {
		w.handleFailure(id, got.err, maxRetries)
		return
	}
	defer func() {
		if got.res.HasCanvas {
			_ = got.res.Canvas.Close()
		}
	}()

	w.recordDuration(float64(got.elapsed) / 1000)

	record = repo.GetByID(w.db, id)
	if record == nil {
		return // deleted while we were recognising
	}
	if got.res.HasCanvas {
		if _, _, _, err := repo.SaveCanvas(w.db, id, got.res.Canvas); err != nil {
			// A missing preview must NOT fail an otherwise good recognition: the fields
			// are the product, the picture is a convenience.
			slog.Error("[WORKER] canvas write failed", "doc", id, "err", err)
		} else if _, err := repo.SaveThumbnail(w.db, id, got.res.Canvas, 96); err != nil {
			slog.Error("[WORKER] thumbnail write failed", "doc", id, "err", err)
		}
	}

	payload, err := toMap(got.res.ViewModel)
	if err != nil {
		w.handleFailure(id, err, maxRetries)
		return
	}
	searchText := buildSearchText(record.Filename, got.res.ViewModel)
	if _, err := repo.SaveResult(w.db, record, payload, searchText, got.elapsed); err != nil {
		slog.Error("[WORKER] cannot save result", "doc", id, "err", err)
		return
	}

	slog.Info("[WORKER] done", "doc", id, "ms", got.elapsed,
		"type", got.res.ViewModel.DocType, "fields", len(got.res.ViewModel.Fields))
}

// recognise loads the stored original and runs the pipeline.
func (w *Worker) recognise(id int, docconf float64, imgSize int) (*runtime.Result, error) {
	path, _, ok := repo.OpenArtifact(w.db, id, "original")
	if !ok {
		// Deterministic, so never retried: this is the symptom of the upload race that
		// repo.ReserveID exists to prevent, and retrying would only hide it.
		return nil, fmt.Errorf("%w: document %d has no stored original", errs.ErrImageUnreadable, id)
	}
	return w.rt.Recognise(path, runtime.RecogniseOptions{
		Docconf: docconf,
		ImgSize: imgSize,
	})
}

// handleFailure classifies the error and either requeues or fails the document.
//
// Only TRANSIENT failures consume a retry. A corrupt JPEG fails immediately and forever,
// because the same bytes will fail the same way and a retry loop on them starves the queue.
func (w *Worker) handleFailure(id int, err error, maxRetries int) {
	code, transient := classify(err)
	slog.Warn("[WORKER] document failed", "doc", id, "code", code,
		"transient", transient, "err", err)

	record := repo.GetByID(w.db, id)
	if record == nil {
		return
	}
	msg := err.Error()
	if transient && record.RetryCount < maxRetries {
		repo.Update(w.db, record, func(d *model.Document) {
			d.Status = model.StatusQueued
			d.RetryCount = record.RetryCount + 1
			d.Error = &msg
			d.ErrorCode = &code
			d.StartedAt = model.Never()
		})
		w.NotifyNewWork()
		return
	}
	if _, err := repo.UpdateStatus(w.db, record, model.StatusFailed, &msg, &code); err != nil {
		slog.Error("[WORKER] cannot mark document failed", "doc", id, "err", err)
	}
}

// classify maps an error to a machine-readable code and a retry decision.
//
// The CODE is separate from the message because the UI is English while a message may not
// be, and because a client should branch on a stable token rather than on prose.
func classify(err error) (string, bool) {
	switch {
	case errors.Is(err, errs.ErrPipelineBusy):
		return "pipeline_busy", true
	case errors.Is(err, errs.ErrRuntimeNotReady):
		return "runtime_not_ready", true
	case errors.Is(err, errs.ErrImageUnreadable):
		return "image_unreadable", false
	default:
		// UNKNOWN IS NOT TRANSIENT. The safe direction: an unrecognised error retried
		// forever stops the queue making progress, and nothing in the log says why.
		return "error", false
	}
}

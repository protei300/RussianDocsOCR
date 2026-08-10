// Package runtime owns and safely calls the recognition pipeline.
//
// **This is the reference part of the reference project.** Everything below encodes a rule
// that is easy to get wrong and expensive to debug, and each was verified against the
// library rather than inferred from documentation. The Python original states ten; the ones
// that survive into Go, and the two that do NOT, are laid out here.
//
//  1. **A pipeline instance is not re-entrant.** In Python `process_img` rebinds
//     `self.results` and `self.ocr_options` on every call, so two concurrent calls on one
//     instance silently return each other's fields — no crash, no reproduction in
//     single-user testing, corrupted data under load. THE GO PORT DOES NOT HAVE THIS BUG:
//     `run` holds its state in locals and returns it. The lease is kept anyway, for rules
//     3 and 9, and because removing it would make the .NET and Kotlin ports — which will
//     wrap stateful objects — differ structurally from this one.
//
//  2. **The per-session CUDA mutex does not help with (1).** It serialises individual ONNX
//     Run() calls on GPU; it fixes device wedging, not re-entrancy. Different problem,
//     different scope. See docproc/inference.
//
//  3. **Transform the result before releasing the lease.** In Python `results` IS
//     `pipeline.results`, and the next call replaces it. Recognise below does the whole
//     read-and-convert inside the lease for that reason — and the Go signature makes it
//     structural rather than a rule to remember, because Use() is the only way in.
//
//  4. **The library's own warmup cannot report failure** (it swallows exceptions into a
//     print), which is why warmup here calls the ordinary path and returns its error.
//     D-03.
//
//  5. **Warmup needs a REAL document.** A synthetic grey frame classifies as 'NONE' and
//     short-circuits before the border, field and OCR stages, warming perhaps a fifth of
//     the graph. It must be an ANONYMISED repository sample — warmup re-reads the file at
//     every start, so pointing it at a real document is a data-handling error, not just a
//     taste one.
//
//  6. **The library prints to stdout** in Python, which would corrupt a JSON log stream.
//     Not applicable here: the Go port logs through slog and prints nothing.
//
//  7. **A listed CUDA provider does not mean a working GPU**, and in a container without
//     --gpus the provider SEGFAULTS instead of erroring. Hence the device-node probe
//     gating the attempt — see docproc/inference.GpuVisible.
//
//  8. **GPU does not mean GPU OCR.** The detectors run on CUDA while the OCR engines stay
//     on CPU, because per-word dynamic widths are far slower on CUDA — measured at 13.7x
//     end-to-end for this port (M8). Info reports Device and OcrDevice SEPARATELY so the
//     status page can say so instead of claiming "GPU active".
//
//  9. **Models load eagerly and cost 215 MB.** Twelve sessions per instance; a second
//     instance on one card is also a second CUDA context. Hence a pool of size 1.
//
//  10. **Only this package touches the library from the service side.** That keeps the rest
//     of the service testable without 215 MB of models and bounds the work of porting the
//     service again. Enforced by review, and by the import graph: nothing under svc/
//     imports docproc/ except here.
package runtime

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/pipeline"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/viewmodel"
)

// LeaseTimeout is how long a caller waits for a free pipeline before giving up.
//
// SHORT ON PURPOSE: a queued job that cannot get a pipeline should go back on the queue and
// surface as "degraded", not block a worker indefinitely.
const LeaseTimeout = 5 * time.Second

// State values for Info.
const (
	StateInitializing = "initializing"
	StateReady        = "ready"
	StateError        = "error"
)

// Info is what the recognition runtime actually ended up doing.
//
// Reported verbatim by GET /status. An operator needs the REAL answer, not the configured
// intent, because the two differ whenever a GPU was asked for and not obtained.
type Info struct {
	State     string   `json:"state"`
	Providers []string `json:"providers"`
	// Device is what the detectors use; OcrDevice differs from it BY DESIGN — rule 8.
	Device          *string `json:"device"`
	OcrDevice       *string `json:"ocr_device"`
	ModelFormat     *string `json:"model_format"`
	OcrMode         *string `json:"ocr_mode"`
	RequestedDevice *string `json:"requested_device"`
	// FellBack records that a GPU was requested and CPU was used. The single most useful
	// field on the status page, because it is the difference between "slow" and "broken".
	FellBack       bool    `json:"fell_back"`
	LoadMs         *int    `json:"load_ms"`
	WarmupMs       *int    `json:"warmup_ms"`
	LibraryVersion *string `json:"library_version"`
	Error          *string `json:"error"`
	PoolSize       int     `json:"pool_size"`
	PoolAvailable  int     `json:"pool_available"`
}

// Options configure init.
type Options struct {
	ComputeDevice string // auto | cpu | gpu
	ModelFormat   string
	OcrMode       string
	WarmupImage   string
	PoolSize      int
	// RepoRoot locates samples/ for the warmup fallback.
	RepoRoot string
}

// Runtime is the pipeline pool plus the published Info.
type Runtime struct {
	mu   sync.Mutex
	info Info

	// pool is a BUFFERED CHANNEL used as a semaphore holding the instances.
	//
	// A channel rather than a mutex because the lease needs a TIMEOUT, which a mutex
	// cannot express, and because "wait for one of N" is exactly what a buffered channel
	// is. This is the one place in the port where a channel is part of the design
	// (CONVENTIONS §1 permits it here and nowhere else): it maps onto
	// BlockingCollection in C# and a Semaphore-plus-queue in Kotlin.
	pool chan *Instance

	poolSize int
}

// Instance is one recognition pipeline.
//
// A struct rather than a bare set of modules so the pool has something to hold, and so
// Close has one place to release twelve sessions.
type Instance struct {
	device inference.Device
	// ocrDevice is separate from device — rule 8.
	ocrDevice   inference.Device
	modelFormat string
	ocrMode     string

	pipe *pipeline.Recognizer
}

func (i *Instance) Close() error { return i.pipe.Close() }

// newInstance builds one pipeline.
//
// ocrDevice is pinned to CPU regardless of `device` — rule 8. It is a separate field rather
// than a derived value so the status page can report the two independently, which is the
// whole point: an operator looking at nvidia-smi and seeing idle OCR needs the service to
// have said so first.
func newInstance(device inference.Device, format, mode string) (*Instance, error) {
	pipe, err := pipeline.NewRecognizer(pipeline.RecognizerOptions{
		ModelFormat: format,
		Device:      device,
		OcrDevice:   inference.CPU,
		OcrTier:     mode,
	})
	if err != nil {
		return nil, err
	}
	return &Instance{
		device:      device,
		ocrDevice:   inference.CPU,
		modelFormat: format,
		ocrMode:     mode,
		pipe:        pipe,
	}, nil
}

// New returns an unstarted runtime. Init does the slow work.
func New() *Runtime {
	return &Runtime{info: Info{State: StateInitializing, Providers: []string{}}}
}

// Info returns a snapshot, with the live pool counts filled in.
func (r *Runtime) Info() Info {
	r.mu.Lock()
	out := r.info
	r.mu.Unlock()
	out.PoolSize = r.poolSize
	if r.pool != nil {
		out.PoolAvailable = len(r.pool)
	}
	// Copied so a caller cannot mutate the published slice.
	out.Providers = append([]string(nil), out.Providers...)
	return out
}

func (r *Runtime) set(mutate func(*Info)) {
	r.mu.Lock()
	mutate(&r.info)
	r.mu.Unlock()
}

// IsReady reports whether recognition can be attempted.
func (r *Runtime) IsReady() bool { return r.Info().State == StateReady }

// Init builds the pipelines, warms them, and publishes what actually happened.
//
// Blocking and slow. NEVER RETURNS AN ERROR for a recognition failure: a failure is recorded
// as State=="error" so the service can still serve its status page and explain itself rather
// than refusing to start. A service that will not boot cannot tell you why it will not boot.
func (r *Runtime) Init(opts Options) Info {
	// Providers are OBSERVED, not advertised — see D-13. CPU is always real; CUDA is added
	// below only if a session actually builds on it. The reference reports what the library
	// advertises, which is the very list rule 7 says cannot be trusted.
	providers := []string{inference.ProviderCPU}
	format, mode := opts.ModelFormat, opts.OcrMode
	requested := opts.ComputeDevice
	r.set(func(i *Info) {
		i.Providers = providers
		i.ModelFormat = &format
		i.OcrMode = &mode
		i.RequestedDevice = &requested
	})

	// TWO INDEPENDENT CONDITIONS, and both are required. The provider list says the GPU
	// build is present; GpuVisible says a device was actually passed through. With the
	// first true and the second false, building a CUDA session terminates the process
	// instead of returning an error.
	hasDevice := inference.GpuVisible()

	wanted := opts.ComputeDevice
	switch opts.ComputeDevice {
	case "auto":
		wanted = "cpu"
		if hasDevice {
			wanted = "gpu"
		}
	case "gpu":
		if !hasDevice {
			slog.Error("[RUNTIME] compute_device=gpu but no GPU is visible to this process " +
				"— refusing to attempt CUDA, which would TERMINATE the process rather than " +
				"fail cleanly. Using CPU. In Docker, pass --gpus all.")
			wanted = "cpu"
		}
	}

	attempts := []inference.Device{inference.CPU}
	if wanted == "gpu" {
		attempts = []inference.Device{inference.GPU, inference.CPU}
	}

	sample := findWarmupSample(opts.WarmupImage, opts.RepoRoot)
	if sample == "" {
		slog.Warn("[RUNTIME] no warmup sample found; the first real document will pay the " +
			"cold-start cost")
	}

	poolSize := opts.PoolSize
	if poolSize < 1 {
		poolSize = 1
	}

	var lastErr error
	for idx, attempt := range attempts {
		slog.Info("[RUNTIME] building pipeline", "device", attempt, "format", format, "ocr", mode)
		started := time.Now()

		built := make([]*Instance, 0, poolSize)
		buildErr := error(nil)
		for n := 0; n < poolSize; n++ {
			inst, err := newInstance(attempt, format, mode)
			if err != nil {
				buildErr = err
				break
			}
			built = append(built, inst)
		}
		if buildErr != nil {
			// Partial builds are released before falling back: leaving them would hold a
			// CUDA context that the CPU attempt then competes with.
			for _, inst := range built {
				_ = inst.Close()
			}
			lastErr = buildErr
			remaining := "no fallback left"
			if idx+1 < len(attempts) {
				remaining = fmt.Sprintf("falling back to %s", attempts[idx+1])
			}
			slog.Error("[RUNTIME] pipeline init FAILED", "device", attempt, "err", buildErr,
				"next", remaining)
			continue
		}

		loadMs := int(time.Since(started).Milliseconds())
		slog.Info("[RUNTIME] instances constructed", "count", len(built), "device", attempt,
			"load_ms", loadMs)

		var warmupMs *int
		if sample != "" {
			total := 0
			for _, inst := range built {
				ms, err := warm(inst, sample)
				if err != nil {
					// A failed warmup is LOGGED, not fatal: the pipeline is built and
					// works; the first document just pays the cold cost. This is D-03 —
					// the reference could not even report this.
					slog.Warn("[RUNTIME] warmup failed", "err", err)
					continue
				}
				total += ms
			}
			avg := total / len(built)
			warmupMs = &avg
		}

		r.pool = make(chan *Instance, len(built))
		for _, inst := range built {
			r.pool <- inst
		}
		r.poolSize = len(built)

		first := built[0]
		dev, ocrDev := string(first.device), string(first.ocrDevice)
		observed := providers
		if attempt == inference.GPU {
			// Recorded only now: the session built, so CUDA is not merely installed but
			// working. This is the whole point of D-13.
			observed = append([]string{inference.ProviderCUDA}, providers...)
		}
		fellBack := wanted == "gpu" && attempt == inference.CPU
		lm := loadMs
		r.set(func(i *Info) {
			i.State = StateReady
			i.Providers = observed
			i.Device = &dev
			i.OcrDevice = &ocrDev
			i.FellBack = fellBack
			i.LoadMs = &lm
			i.WarmupMs = warmupMs
			i.Error = nil
		})
		slog.Info("[RUNTIME] ready", "device", dev, "ocr_device", ocrDev,
			"load_ms", loadMs, "instances", len(built))
		if fellBack {
			slog.Error("[RUNTIME] GPU was requested but only CPU worked — check CUDA/cuDNN. " +
				"Recognition will be slower.")
		}
		return r.Info()
	}

	slog.Error("[RUNTIME] recognition unavailable; the service will start and accept "+
		"uploads, but every document will fail", "attempts", attempts, "err", lastErr)
	msg := fmt.Sprint(lastErr)
	r.set(func(i *Info) {
		i.State = StateError
		i.Error = &msg
	})
	return r.Info()
}

// Shutdown drops the pipelines.
func (r *Runtime) Shutdown() {
	drained := 0
	if r.pool != nil {
		for {
			select {
			case inst := <-r.pool:
				_ = inst.Close()
				drained++
			default:
				goto done
			}
		}
	}
done:
	r.set(func(i *Info) {
		i.State = StateInitializing
		i.Device = nil
		i.OcrDevice = nil
	})
	slog.Info("[RUNTIME] released pipeline instances", "count", drained)
}

// Use runs fn with exclusive access to one instance.
//
// **The only way to reach a pipeline.** A higher-order function rather than
// Acquire/Release, deliberately: it makes rule 3 — transform the result BEFORE releasing —
// structurally enforced instead of a comment somebody deletes. Python expresses the same
// thing as a context manager, C# as a using block, Kotlin as an inline lambda.
//
// Returns ErrRuntimeNotReady before the models finish loading and ErrPipelineBusy if none
// becomes free within the timeout. Both are TRANSIENT, so the caller requeues rather than
// failing the job.
func (r *Runtime) Use(timeout time.Duration, fn func(*Instance) error) error {
	info := r.Info()
	switch info.State {
	case StateError:
		return fmt.Errorf("%w: recognition runtime failed to start: %s",
			errs.ErrRuntimeNotReady, deref(info.Error))
	case StateReady:
	default:
		return fmt.Errorf("%w: recognition runtime is still loading models",
			errs.ErrRuntimeNotReady)
	}

	select {
	case inst := <-r.pool:
		// Returned in a defer so a panic in fn cannot leak the instance and wedge the
		// pool at zero available — which would look exactly like the hang the lease
		// timeout exists to report.
		defer func() { r.pool <- inst }()
		return fn(inst)
	case <-time.After(timeout):
		return fmt.Errorf("%w: no pipeline became available within %s",
			errs.ErrPipelineBusy, timeout)
	}
}

// RecogniseOptions are the per-document knobs.
type RecogniseOptions struct {
	IncludeDebug bool
	Docconf      float64
	ImgSize      int
	LeaseTimeout time.Duration
}

// Result is what Recognise produces.
type Result struct {
	ViewModel viewmodel.Payload
	// Canvas is the corrected canvas, RGB and OWNED BY THE CALLER, who must Close it.
	// Nil when the document short-circuited as unrecognised.
	Canvas imaging.Image
	// HasCanvas distinguishes a missing canvas from a zero-sized one.
	HasCanvas bool
}

// Recognise processes one document. The whole public surface of this package.
//
// The canvas is RGB and the encoder writes BGR, so the artifact layer converts before
// writing — see repo.SaveCanvas. Getting that wrong swaps red and blue in every stored
// document, and on a passport it looks plausible enough to ship unnoticed.
func (r *Runtime) Recognise(imagePath string, opts RecogniseOptions) (*Result, error) {
	timeout := opts.LeaseTimeout
	if timeout <= 0 {
		timeout = LeaseTimeout
	}

	var out *Result
	err := r.Use(timeout, func(inst *Instance) error {
		res, err := inst.pipe.Run(imagePath, pipeline.RunOptions{
			Docconf: opts.Docconf,
			ImgSize: opts.ImgSize,
		})
		if err != nil {
			return err
		}
		// Built INSIDE the lease — rule 3. Structural here rather than remembered,
		// because there is no way to hold `res` past the closure.
		// The view-model input is assembled HERE rather than by the library, so docproc
		// never imports viewmodel: the dependency runs one way only (D-01 puts viewmodel on
		// the library SIDE, not inside docproc).
		in := viewmodel.Input{
			DocType:       res.DocType,
			Device:        string(inst.device),
			CanvasMissing: !res.HasCanvas,
			Boxes:         res.Boxes,
			Ocr:           res.Ocr,
			Quality:       res.Quality,
			Timings:       res.Timings,
			Segments:      res.Segments,
		}
		if res.HasCanvas {
			in.CanvasW, in.CanvasH = res.Canvas.Width(), res.Canvas.Height()
		}
		// **TakeCanvas, not a bare field read.** The canvas must outlive the run, but every
		// other image the run allocated must not: reading res.Canvas and returning left the
		// intermediates — the fully decoded original among them — alive forever. Measured
		// before this call existed: 663 MB -> 4018 MB across 230 documents, growing without
		// bound. Nothing in the conformance suite could catch it, because the CLI closes its
		// Results after a single document.
		canvas, hasCanvas := res.TakeCanvas()
		out = &Result{
			ViewModel: viewmodel.Build(in, opts.IncludeDebug),
			Canvas:    canvas,
			HasCanvas: hasCanvas,
		}
		// The canvas ESCAPES the lease deliberately and is now owned by the caller: it is a
		// standalone Mat, not a view into pipeline state, so unlike the reference's
		// `results` it is safe to read after release.
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// warm pays the cold-start cost once, up front.
func warm(inst *Instance, sample string) (int, error) {
	started := time.Now()
	res, err := inst.pipe.Run(sample, pipeline.RunOptions{Docconf: 0.5, ImgSize: 1500})
	if err != nil {
		return 0, err
	}
	res.Close()
	return int(time.Since(started).Milliseconds()), nil
}

// findWarmupSample resolves the configured image, else picks one from samples/.
//
// **Only anonymised repository samples are eligible.** Warmup re-reads this file at every
// start, so a real document here would be read on every boot of every deployment — which is
// why the fallback searches samples/ and never the data directory.
func findWarmupSample(configured, repoRoot string) string {
	if configured != "" {
		if isFile(configured) {
			return configured
		}
		slog.Warn("[RUNTIME] configured warmup image does not exist", "path", configured)
	}
	if repoRoot == "" {
		return ""
	}
	// A fixed preference order rather than "the first file found": the chosen sample
	// decides which parts of the graph get warmed, and a directory listing order is not a
	// decision anybody made.
	for _, rel := range []string{
		filepath.Join("samples", "INTPASSPORT_2011", "12_CR_INTPASSPORT_2011.jpg"),
		filepath.Join("samples", "DL_2011", "1_CR_DL_2010.jpg"),
	} {
		candidate := filepath.Join(repoRoot, rel)
		if isFile(candidate) {
			return candidate
		}
	}
	return ""
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

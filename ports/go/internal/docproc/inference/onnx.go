// Package inference wraps ONNX Runtime. It is the only package that imports the
// onnxruntime binding, mirroring processing/inference.py.
//
// Two behaviours here are load-bearing and neither is obvious from the Python code's
// shape — both are documented at their call sites below:
//
//   - every input is cast to the dtype the SESSION DECLARES, not unconditionally to
//     float32 (the detectors want float32, the v2 OCR nets want uint8);
//   - a per-session mutex is held around Run(), but ONLY when the session is on the
//     GPU. Concurrent Run() on one CUDA session must be serialised; CPU sessions are
//     left unlocked so the pipeline's parallel groups keep their speedup.
package inference

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Device selects the execution provider. String constants rather than an iota enum,
// per CONVENTIONS §1: the same spellings appear on the command line and in the
// service's settings, and three languages serialise integer enums three ways.
type Device string

const (
	CPU Device = "cpu"
	GPU Device = "gpu"
)

var (
	envOnce sync.Once
	envErr  error
	envPath string
)

// InitEnvironment loads the ONNX Runtime shared library and creates the global
// environment. Idempotent; safe to call from every constructor.
//
// The library is located by ORT_DLL (or ORT_SO), else by the platform default name
// on the loader's search path. It must be **1.21.x**: onnxruntime_go v1.19.0 vendors
// ORT_API_VERSION 21, and the C API is backward compatible only one way — a newer
// library serves older requests, never the reverse. Against a mismatched library the
// failure is `GetApi` returning NULL, which surfaces as an initialisation error rather
// than anything about versions, so the hint is attached here.
func InitEnvironment() error {
	envOnce.Do(func() {
		envPath = os.Getenv("ORT_DLL")
		if envPath == "" {
			envPath = os.Getenv("ORT_SO")
		}
		if envPath != "" {
			ort.SetSharedLibraryPath(envPath)
		} else if runtime.GOOS != "windows" {
			// On Windows the binding already defaults to "onnxruntime.dll".
			ort.SetSharedLibraryPath("libonnxruntime.so")
		}
		if err := ort.InitializeEnvironment(); err != nil {
			envErr = fmt.Errorf("inference: initialise ONNX Runtime (%s): %w\n"+
				"  the library must be 1.21.x — onnxruntime_go v1.19.0 vendors "+
				"ORT_API_VERSION 21, and an older library makes GetApi() return NULL;\n"+
				"  set ORT_DLL/ORT_SO to a matching build", displayPath(envPath), err)
		}
	})
	return envErr
}

func displayPath(p string) string {
	if p == "" {
		return "default search path"
	}
	return p
}

// DestroyEnvironment releases the global environment. Call once at process exit.
func DestroyEnvironment() error { return ort.DestroyEnvironment() }

// Version reports the loaded runtime's version, or "" before initialisation.
func Version() string {
	if envErr != nil {
		return ""
	}
	if !ort.IsInitialized() {
		return ""
	}
	return ort.GetVersion()
}

// IOInfo describes one model input or output.
type IOInfo struct {
	Name  string
	Dims  []int64
	Dtype tensor.DType
}

// Session is one loaded model.
type Session struct {
	path    string
	device  Device
	inputs  []IOInfo
	outputs []IOInfo

	sess *ort.DynamicAdvancedSession

	// Guards Run() on a GPU session only. Concurrent Run() on one CUDA session
	// wedges or thrashes the device: measured in the spike at 8 goroutines x 300
	// calls taking 6.6 s with this lock and over 600 s without it, with the GPU
	// pinned at 100 %. Python needed the same lock for the same reason
	// (processing/inference.py). CPU sessions stay unlocked so the pipeline's
	// five-way quality group keeps its parallelism.
	mu *sync.Mutex
}

// Open loads a model and prepares a session.
//
// threads applies to CPU sessions only; pass 0 to leave the runtime's default. The
// conformance harness pins it to 1 on both sides, because ORT's CPU reductions
// partition by thread and a different count shifts results by ~1e-6 — inside the
// float tolerance, but enough to flip an argmax on a near-tie.
func Open(path string, device Device, threads int) (*Session, error) {
	if err := InitEnvironment(); err != nil {
		return nil, err
	}

	rawIn, rawOut, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, fmt.Errorf("inference: inspect %s: %w", path, err)
	}
	inputs := make([]IOInfo, 0, len(rawIn))
	inNames := make([]string, 0, len(rawIn))
	for _, i := range rawIn {
		dt, err := dtypeOf(i.DataType)
		if err != nil {
			return nil, fmt.Errorf("inference: %s input %q: %w", path, i.Name, err)
		}
		inputs = append(inputs, IOInfo{Name: i.Name, Dims: i.Dimensions, Dtype: dt})
		inNames = append(inNames, i.Name)
	}
	outputs := make([]IOInfo, 0, len(rawOut))
	outNames := make([]string, 0, len(rawOut))
	for _, o := range rawOut {
		dt, err := dtypeOf(o.DataType)
		if err != nil {
			return nil, fmt.Errorf("inference: %s output %q: %w", path, o.Name, err)
		}
		outputs = append(outputs, IOInfo{Name: o.Name, Dims: o.Dimensions, Dtype: dt})
		outNames = append(outNames, o.Name)
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("inference: session options: %w", err)
	}
	defer opts.Destroy()

	if device == GPU {
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("inference: CUDA provider options: %w", err)
		}
		defer cuda.Destroy()
		if err := cuda.Update(map[string]string{"device_id": "0"}); err != nil {
			return nil, fmt.Errorf("inference: CUDA provider options: %w", err)
		}
		// A listed provider never proved a working GPU. This call is the honest
		// probe: it fails loudly when the provider library cannot load, which is how
		// a CUDA/cuDNN version mismatch shows up (an "Error 127" from the loader).
		if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
			return nil, fmt.Errorf("inference: enable CUDA: %w", err)
		}
	} else if threads > 0 {
		if err := opts.SetIntraOpNumThreads(threads); err != nil {
			return nil, err
		}
		if err := opts.SetInterOpNumThreads(threads); err != nil {
			return nil, err
		}
	}

	sess, err := ort.NewDynamicAdvancedSession(path, inNames, outNames, opts)
	if err != nil {
		return nil, fmt.Errorf("inference: open %s: %w", path, err)
	}

	s := &Session{path: path, device: device, inputs: inputs, outputs: outputs, sess: sess}
	if device == GPU {
		s.mu = &sync.Mutex{}
	}
	return s, nil
}

func (s *Session) Close() error      { return s.sess.Destroy() }
func (s *Session) Inputs() []IOInfo  { return s.inputs }
func (s *Session) Outputs() []IOInfo { return s.outputs }
func (s *Session) Device() Device    { return s.device }
func (s *Session) Path() string      { return s.path }

// Run executes the model. Inputs are given in the model's declared input order.
//
// Each input is cast to the dtype the SESSION DECLARES — never unconditionally to
// float32. `ModelInference._np_dtype_for` does the same, and it matters: the
// detectors declare float32 while the v2 OCR nets declare uint8, and a mismatched
// buffer is rejected by the binding.
func (s *Session) Run(in []*tensor.Array) ([]*tensor.Array, error) {
	if len(in) != len(s.inputs) {
		return nil, fmt.Errorf("inference: %s wants %d input(s), got %d",
			s.path, len(s.inputs), len(in))
	}

	values := make([]ort.Value, len(in))
	for i, a := range in {
		v, err := toOrtValue(a, s.inputs[i].Dtype)
		if err != nil {
			return nil, fmt.Errorf("inference: %s input %q: %w", s.path, s.inputs[i].Name, err)
		}
		values[i] = v
	}
	defer func() {
		for _, v := range values {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()

	// nil outputs => ONNX Runtime allocates and reports the shape it chose. Required
	// for the OCR nets, whose output length varies with the input width; harmless and
	// uniform for the fixed-shape models.
	outs := make([]ort.Value, len(s.outputs))

	if s.mu != nil {
		// Held around Run() ONLY. The result is read after release, exactly as the
		// Python library does — holding it longer would serialise the post-processing
		// too, for no safety gain.
		s.mu.Lock()
	}
	err := s.sess.Run(values, outs)
	if s.mu != nil {
		s.mu.Unlock()
	}
	if err != nil {
		return nil, fmt.Errorf("inference: %s run: %w", s.path, err)
	}

	result := make([]*tensor.Array, len(outs))
	for i, v := range outs {
		if v == nil {
			return nil, fmt.Errorf("inference: %s produced no output %d", s.path, i)
		}
		arr, convErr := fromOrtValue(v)
		_ = v.Destroy()
		if convErr != nil {
			return nil, fmt.Errorf("inference: %s output %q: %w",
				s.path, s.outputs[i].Name, convErr)
		}
		result[i] = arr
	}
	return result, nil
}

func dtypeOf(t ort.TensorElementDataType) (tensor.DType, error) {
	switch t {
	case ort.TensorElementDataTypeFloat:
		return tensor.Float32, nil
	case ort.TensorElementDataTypeDouble:
		return tensor.Float64, nil
	case ort.TensorElementDataTypeUint8:
		return tensor.Uint8, nil
	case ort.TensorElementDataTypeInt64:
		return tensor.Int64, nil
	default:
		return "", fmt.Errorf("unsupported element type %v", t)
	}
}

func toOrtValue(a *tensor.Array, want tensor.DType) (ort.Value, error) {
	shape := ort.NewShape(int64sOf(a.Shape)...)
	switch want {
	case tensor.Float32:
		data, err := a.AsFloat32()
		if err != nil {
			return nil, err
		}
		return ort.NewTensor(shape, data)
	case tensor.Uint8:
		data, err := a.AsUint8()
		if err != nil {
			return nil, err
		}
		return ort.NewTensor(shape, data)
	case tensor.Int64:
		if a.Dtype != tensor.Int64 {
			return nil, fmt.Errorf("cannot feed %s where int64 is declared", a.Dtype)
		}
		return ort.NewTensor(shape, a.I64)
	default:
		return nil, fmt.Errorf("cannot build an input of declared type %s", want)
	}
}

func fromOrtValue(v ort.Value) (*tensor.Array, error) {
	shape := intsOf(v.GetShape())
	switch t := v.(type) {
	case *ort.Tensor[float32]:
		data := make([]float32, len(t.GetData()))
		copy(data, t.GetData())
		return tensor.Float32Of(shape, data)
	case *ort.Tensor[uint8]:
		data := make([]uint8, len(t.GetData()))
		copy(data, t.GetData())
		return tensor.Uint8Of(shape, data)
	case *ort.Tensor[int64]:
		data := make([]int64, len(t.GetData()))
		copy(data, t.GetData())
		return tensor.Int64Of(shape, data)
	default:
		return nil, fmt.Errorf("unsupported output type %T", v)
	}
}

func int64sOf(shape []int) []int64 {
	out := make([]int64, len(shape))
	for i, v := range shape {
		out[i] = int64(v)
	}
	return out
}

func intsOf(shape ort.Shape) []int {
	out := make([]int, len(shape))
	for i, v := range shape {
		out[i] = int(v)
	}
	return out
}

// ProviderCUDA and ProviderCPU are the provider names used in status output.
//
// String constants rather than a list from the binding, because onnxruntime_go v1.19.0 does
// not expose GetAvailableProviders at all — and that absence turns out to be a feature. See
// D-13: this port reports the providers it has OBSERVED WORKING rather than the ones the
// library advertises, precisely because the advertised list is the thing that cannot be
// trusted (svc/runtime rule 7: CUDA is listed whenever the GPU build is installed, including
// when cuDNN is missing and every session silently falls back to CPU).
const (
	ProviderCUDA = "CUDAExecutionProvider"
	ProviderCPU  = "CPUExecutionProvider"
)

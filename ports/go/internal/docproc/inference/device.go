package inference

import (
	"fmt"
	"os"
	"runtime"
)

// GpuVisible reports whether a GPU is actually reachable from this process.
//
// This exists to close a hole in "just try to build the session and catch the error".
// In a container started WITHOUT --gpus, onnxruntime's CUDA provider does not return an
// error — it SEGFAULTS. The process dies at exit 139 with no stack and no chance to fall
// back, so no amount of error handling around the attempt can save it. Verified in
// exactly that setup on the Python side; the Go binding calls the same C library through
// cgo, where a segfault is even less recoverable.
//
// The probe is therefore EVIDENCE-FIRST: attempt CUDA only when something says a device
// exists. Device nodes are the check, because they are the thing that is missing in the
// failing case, they cost two stat calls, and neither can crash the process.
//
// NVML is deliberately NOT used here, unlike the Python service. Loading nvml.dll /
// libnvidia-ml.so through cgo to answer a question two stat calls already answer would
// add a shared-library dependency to the LIBRARY layer, and on Windows the device-node
// path does not exist at all — so the two probes would not even agree on which platform
// they serve. The service layer may add richer reporting on top; the library keeps the
// cheap, crash-proof version.
func GpuVisible() bool {
	if runtime.GOOS == "windows" {
		// Windows has no device nodes to stat. The driver is reachable whenever the
		// runtime library loads, and the CUDA-provider segfault this function guards
		// against is a Linux-container failure mode, so the honest answer here is
		// "cannot tell from here, let the attempt decide" — and the attempt is safe on
		// Windows because the provider returns an error rather than dying.
		return true
	}
	for _, node := range []string{"/dev/nvidiactl", "/dev/dxg", "/dev/nvidia0"} {
		if _, err := os.Stat(node); err == nil {
			return true
		}
	}
	return false
}

// Resolve picks the device to run on and says why.
//
// Port of the attempt loop in service/ml/runtime.py. The rule it encodes, which is easy
// to get wrong in both directions:
//
//   - a REQUESTED gpu is an attempt, not a guarantee. It falls back to CPU, because a
//     service that refuses to start on a machine without a working GPU is worse than one
//     that runs slower;
//   - a requested cpu is honoured exactly, with no probing at all. "Give me CPU" is
//     sometimes a correctness requirement — the conformance goldens are CPU — and
//     silently upgrading it would be a bug.
//
// probe is the session constructor; it is called at most once per candidate, and its
// error is returned in `note` rather than swallowed, because "GPU was requested and we
// are on CPU" must be explainable from a log alone.
func Resolve(requested Device, probe func(Device) error) (Device, string, error) {
	if requested != GPU {
		return CPU, "cpu requested", nil
	}

	if !GpuVisible() {
		if err := probe(CPU); err != nil {
			return "", "", err
		}
		return CPU, "gpu requested but no device is visible to this process; " +
			"CUDA was NOT attempted, because without a device the provider can " +
			"terminate the process instead of returning an error", nil
	}

	if err := probe(GPU); err == nil {
		return GPU, "gpu requested and available", nil
	} else if cpuErr := probe(CPU); cpuErr != nil {
		// Both failed: report the CPU error, since that is the one that leaves the
		// caller with nothing at all.
		return "", "", fmt.Errorf("inference: neither device usable; cpu: %w (gpu: %v)", cpuErr, err)
	} else {
		return CPU, fmt.Sprintf("gpu requested but unusable, fell back to cpu: %v", err), nil
	}
}

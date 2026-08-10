// Package sysinfo reports host CPU, memory, disk and GPU for the status page.
//
// **The field names here are fixed by the SHARED FRONTEND, not chosen.** `web/` is reused
// unchanged by every port, so `web/src/views/pages/status/Index.vue` is the contract: it reads
// `server.cpu_pct`, `server.cpu_name`, `server.ram_used_gb` and the rest by name. An earlier
// version of this port returned a thinner, more Go-shaped block on the reasoning that pulling
// in a dependency to render a CPU gauge was a poor trade — and the status page rendered
// completely empty. The lesson is worth writing down: when a UI is shared, the UI owns the
// wire format.
//
// Every probe is INDIVIDUALLY GUARDED and degrades to a zero value. A service that cannot
// describe its own host must still recognise documents, so nothing in here may return an error
// that reaches a caller.
//
// Port of the `_server_stats` / `_gpu_stats` helpers in service/api/status.py.
package sysinfo

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Server is the host block. JSON names match the frontend exactly.
type Server struct {
	CPUPct      float64 `json:"cpu_pct"`
	CPUName     string  `json:"cpu_name"`
	CPUCores    int     `json:"cpu_cores"`
	CPUThreads  int     `json:"cpu_threads"`
	RAMUsedGB   float64 `json:"ram_used_gb"`
	RAMTotalGB  float64 `json:"ram_total_gb"`
	DiskUsedGB  float64 `json:"disk_used_gb"`
	DiskTotalGB float64 `json:"disk_total_gb"`
}

// GPU is the accelerator block, or nil when there is no GPU to describe.
type GPU struct {
	Name           string  `json:"name"`
	UtilizationPct int     `json:"utilization_pct"`
	VRAMUsedGB     float64 `json:"vram_used_gb"`
	VRAMTotalGB    float64 `json:"vram_total_gb"`
	TemperatureC   int     `json:"temperature_c"`
}

// ReadServer collects the host block.
//
// Implemented per platform in sysinfo_windows.go and sysinfo_unix.go, using x/sys rather than
// a metrics library: the four numbers wanted here are four syscalls, and the alternative
// (gopsutil) is a large dependency whose portability matters to a project that ships a
// container.
func ReadServer() Server {
	s := Server{
		CPUName:    cpuName(),
		CPUCores:   physicalCores(),
		CPUThreads: logicalCores(),
		CPUPct:     cpuPercent(),
	}
	if used, total, ok := memory(); ok {
		s.RAMUsedGB = gb(used)
		s.RAMTotalGB = gb(total)
	}
	if used, total, ok := disk(); ok {
		s.DiskUsedGB = gb(used)
		s.DiskTotalGB = gb(total)
	}
	return s
}

// ReadGPU queries the GPU through `nvidia-smi`, returning nil when there is none.
//
// **Why a subprocess rather than NVML.** NVML is the proper API and the Python service uses it
// through pynvml. Reaching it from Go means dynamic loading: a LazyDLL dance on Windows and
// dlopen through cgo on Linux — two platform-specific code paths, one of which could not be
// tested on the machine this was written on, for information that is purely diagnostic and
// polled by one page. `nvidia-smi` ships with the driver, is present in the CUDA runtime
// images, and its CSV output is stable across driver generations.
//
// The cost is real and bounded: one process spawn per status request, with a hard timeout. If
// that ever becomes a problem the fix is a cached value with a TTL, not NVML.
//
// Absence is NOT an error. No GPU, no driver, or a CPU-only container all mean nil, and the
// status page then shows the compute block alone — which is the part that answers whether the
// GPU is actually being used.
func ReadGPU() *GPU {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}

	// The first line only: a multi-GPU host reports one row per device, and the pipeline
	// pins device_id 0.
	line := strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0])
	if line == "" {
		return nil
	}
	parts := strings.Split(line, ",")
	if len(parts) < 5 {
		return nil
	}
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}

	// memory.* is reported in MiB with `nounits`.
	usedMiB := atof(parts[2])
	totalMiB := atof(parts[3])
	return &GPU{
		Name:           parts[0],
		UtilizationPct: atoi(parts[1]),
		VRAMUsedGB:     round1(usedMiB * 1024 * 1024 / 1e9),
		VRAMTotalGB:    round1(totalMiB * 1024 * 1024 / 1e9),
		TemperatureC:   atoi(parts[4]),
	}
}

// gb converts bytes to gigabytes at one decimal.
//
// DECIMAL gigabytes (1e9), matching the Python service, so the two report the same number for
// the same machine. Not GiB — a status page saying 32.0 GB for a 32 GB stick is what an
// operator expects, whatever the pedantically correct unit is.
func gb(bytes uint64) float64 { return round1(float64(bytes) / 1e9) }

func round1(v float64) float64 { return float64(int(v*10+0.5)) / 10 }

func atoi(s string) int {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0
	}
	return v
}

func atof(s string) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0
	}
	return v
}

// trimSpaces collapses internal runs of whitespace and trims the ends.
//
// Both the registry and /proc/cpuinfo pad CPU names, sometimes in the middle: the raw value can
// be "Intel(R) Core(TM) i7-13700K   CPU @ 3.40GHz", which renders with a visible gap.
func trimSpaces(s string) string { return strings.Join(strings.Fields(s), " ") }

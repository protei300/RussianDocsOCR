package sysinfo

import "testing"

// The host block must be POPULATED, not merely present. This is the test the earlier version
// of the status handler would have failed: it returned a well-formed object with none of the
// fields the shared frontend reads, and the page rendered empty.
func TestReadServerIsPopulated(t *testing.T) {
	s := ReadServer()

	if s.CPUName == "" {
		t.Error("cpu_name is empty; the status page shows a blank row")
	}
	if s.CPUThreads <= 0 {
		t.Errorf("cpu_threads = %d", s.CPUThreads)
	}
	if s.CPUCores <= 0 {
		t.Errorf("cpu_cores = %d; the physical-core probe returned nothing", s.CPUCores)
	}
	if s.CPUCores > s.CPUThreads {
		t.Errorf("cpu_cores %d > cpu_threads %d, which is impossible",
			s.CPUCores, s.CPUThreads)
	}
	if s.RAMTotalGB <= 0 {
		t.Errorf("ram_total_gb = %v", s.RAMTotalGB)
	}
	if s.RAMUsedGB <= 0 || s.RAMUsedGB > s.RAMTotalGB {
		t.Errorf("ram_used_gb = %v against a total of %v", s.RAMUsedGB, s.RAMTotalGB)
	}
	if s.DiskTotalGB <= 0 {
		t.Errorf("disk_total_gb = %v", s.DiskTotalGB)
	}
	if s.DiskUsedGB < 0 || s.DiskUsedGB > s.DiskTotalGB {
		t.Errorf("disk_used_gb = %v against a total of %v", s.DiskUsedGB, s.DiskTotalGB)
	}
	// A rate, so it needs two samples; 0 is legitimate on a quiet machine but a value
	// outside 0..100 means the idle/total arithmetic is wrong — which is exactly the
	// mistake of adding idle to a kernel time that already includes it.
	if s.CPUPct < 0 || s.CPUPct > 100 {
		t.Errorf("cpu_pct = %v, outside 0..100", s.CPUPct)
	}
}

// ReadGPU returns nil rather than erroring when there is nothing to describe, and that is a
// supported outcome: a CPU-only host and a container without --gpus both land here.
func TestReadGPUIsNilOrCoherent(t *testing.T) {
	g := ReadGPU()
	if g == nil {
		t.Skip("no GPU visible to this process — nil is the correct answer")
	}
	if g.Name == "" {
		t.Error("a GPU was reported with no name")
	}
	if g.VRAMTotalGB <= 0 {
		t.Errorf("vram_total_gb = %v", g.VRAMTotalGB)
	}
	if g.VRAMUsedGB < 0 || g.VRAMUsedGB > g.VRAMTotalGB {
		t.Errorf("vram_used_gb = %v against a total of %v", g.VRAMUsedGB, g.VRAMTotalGB)
	}
	if g.UtilizationPct < 0 || g.UtilizationPct > 100 {
		t.Errorf("utilization_pct = %d", g.UtilizationPct)
	}
	// A plausible-range check rather than an exact one: an idle card reads in the thirties
	// and a loaded one in the eighties, but zero means the field was never parsed.
	if g.TemperatureC <= 0 || g.TemperatureC > 120 {
		t.Errorf("temperature_c = %d, outside a plausible range", g.TemperatureC)
	}
}

// Decimal gigabytes (1e9), matching the Python service, so both report the same number for the
// same machine.
func TestGbUsesDecimalGigabytes(t *testing.T) {
	if got := gb(1_000_000_000); got != 1.0 {
		t.Errorf("gb(1e9) = %v, want 1.0", got)
	}
	if got := gb(137_100_000_000); got != 137.1 {
		t.Errorf("gb(137.1e9) = %v, want 137.1", got)
	}
}

// Padding shows up mid-string in both the registry and /proc/cpuinfo, which renders as a
// visible gap on the status page.
func TestTrimSpacesCollapsesInternalRuns(t *testing.T) {
	got := trimSpaces("  Intel(R) Core(TM) i7-13700K   CPU @ 3.40GHz  ")
	want := "Intel(R) Core(TM) i7-13700K CPU @ 3.40GHz"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

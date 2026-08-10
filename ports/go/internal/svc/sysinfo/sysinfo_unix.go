//go:build !windows

package sysinfo

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

// memory reads /proc/meminfo.
//
// USED is total minus AVAILABLE, not total minus free. MemFree excludes the page cache, so on a
// machine that has just loaded 215 MB of models "used" would look like nearly all of RAM.
// MemAvailable is the kernel's own estimate of what a new allocation could get, which is the
// number an operator means.
func memory() (used, total uint64, ok bool) {
	fields := parseKeyedKB("/proc/meminfo")
	total = fields["MemTotal"]
	if total == 0 {
		return 0, 0, false
	}
	avail, hasAvail := fields["MemAvailable"]
	if !hasAvail {
		// Pre-3.14 kernels and some containers do not publish it.
		avail = fields["MemFree"] + fields["Buffers"] + fields["Cached"]
	}
	if avail > total {
		avail = total
	}
	return total - avail, total, true
}

// disk reports the filesystem holding "/", matching the Python service's choice.
func disk() (used, total uint64, ok bool) {
	var stat unix.Statfs_t
	if err := unix.Statfs("/", &stat); err != nil {
		return 0, 0, false
	}
	blockSize := uint64(stat.Bsize)
	total = stat.Blocks * blockSize
	// Bavail, NOT Bfree: Bfree includes the root-reserved blocks an ordinary process cannot
	// use, and the service runs as a non-root user.
	free := stat.Bavail * blockSize
	if total == 0 || free > total {
		return 0, 0, false
	}
	return total - free, total, true
}

// cpuPercent samples /proc/stat twice. See the note in the Windows file on why an interval is
// unavoidable.
func cpuPercent() float64 {
	idle1, total1, ok1 := procStat()
	if !ok1 {
		return 0
	}
	time.Sleep(150 * time.Millisecond)
	idle2, total2, ok2 := procStat()
	if !ok2 {
		return 0
	}
	dTotal := float64(total2 - total1)
	if dTotal <= 0 {
		return 0
	}
	busy := dTotal - float64(idle2-idle1)
	return round1(busy / dTotal * 100)
}

// procStat returns (idle+iowait, every field summed) from the aggregate "cpu" line.
//
// iowait counts as IDLE: the CPU executes nothing during it, and treating it as busy makes a
// disk-bound service look CPU-bound.
func procStat() (idle, total uint64, ok bool) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "cpu ") {
			continue
		}
		for i, field := range strings.Fields(line)[1:] {
			v, err := strconv.ParseUint(field, 10, 64)
			if err != nil {
				continue
			}
			total += v
			if i == 3 || i == 4 { // idle, iowait
				idle += v
			}
		}
		return idle, total, total > 0
	}
	return 0, 0, false
}

// cpuName reads the model name from /proc/cpuinfo. The alternative keys cover ARM hosts, where
// there is no "model name" line.
func cpuName() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "Unknown CPU"
	}
	for _, line := range strings.Split(string(data), "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		switch strings.TrimSpace(key) {
		case "model name", "Hardware", "Processor":
			if v := trimSpaces(value); v != "" {
				return v
			}
		}
	}
	return "Unknown CPU"
}

func logicalCores() int { return runtime.NumCPU() }

// physicalCores counts DISTINCT (physical id, core id) pairs.
//
// Summing the "cpu cores" lines would multiply by the hyperthread count, and counting unique
// core ids alone breaks on a dual-socket machine, where both sockets number their cores from
// zero.
func physicalCores() int {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return 0
	}
	seen := map[string]bool{}
	physical, core := "", ""
	flush := func() {
		if physical != "" || core != "" {
			seen[physical+"/"+core] = true
			physical, core = "", ""
		}
	}
	for _, line := range strings.Split(string(data), "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			flush() // a blank line ends one processor block
			continue
		}
		switch strings.TrimSpace(key) {
		case "physical id":
			physical = strings.TrimSpace(value)
		case "core id":
			core = strings.TrimSpace(value)
		}
	}
	flush()
	return len(seen)
}

// parseKeyedKB reads "Key: 1234 kB" lines, returning BYTES.
func parseKeyedKB(path string) map[string]uint64 {
	out := map[string]uint64{}
	data, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(data), "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		fields := strings.Fields(value)
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		out[strings.TrimSpace(key)] = v * 1024
	}
	return out
}

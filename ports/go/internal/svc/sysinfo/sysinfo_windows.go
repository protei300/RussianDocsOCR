//go:build windows

package sysinfo

import (
	"runtime"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

var (
	kernel32               = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalMemoryStatus = kernel32.NewProc("GlobalMemoryStatusEx")
	procGetSystemTimes     = kernel32.NewProc("GetSystemTimes")
	procGetDiskFreeSpaceEx = kernel32.NewProc("GetDiskFreeSpaceExW")
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

func memory() (used, total uint64, ok bool) {
	var status memoryStatusEx
	status.Length = uint32(unsafe.Sizeof(status))
	ret, _, _ := procGlobalMemoryStatus.Call(uintptr(unsafe.Pointer(&status)))
	if ret == 0 {
		return 0, 0, false
	}
	return status.TotalPhys - status.AvailPhys, status.TotalPhys, true
}

// disk reports the volume holding the SYSTEM drive, matching the Python service's choice of
// C:\ on Windows. Not the data directory: the question the status page answers is "is this
// host running out of space", and on a single-volume machine they are the same anyway.
func disk() (used, total uint64, ok bool) {
	path, err := windows.UTF16PtrFromString(`C:\`)
	if err != nil {
		return 0, 0, false
	}
	var free, totalBytes, totalFree uint64
	ret, _, _ := procGetDiskFreeSpaceEx.Call(
		uintptr(unsafe.Pointer(path)),
		uintptr(unsafe.Pointer(&free)),
		uintptr(unsafe.Pointer(&totalBytes)),
		uintptr(unsafe.Pointer(&totalFree)),
	)
	if ret == 0 {
		return 0, 0, false
	}
	return totalBytes - totalFree, totalBytes, true
}

// cpuPercent samples system times twice over a short interval.
//
// CPU utilisation is a RATE, so it cannot be read instantaneously — it needs two samples. The
// 150 ms interval is copied from the Python service so both report comparable numbers, and it
// is the reason a status request takes at least that long.
func cpuPercent() float64 {
	idle1, total1, ok1 := systemTimes()
	if !ok1 {
		return 0
	}
	time.Sleep(150 * time.Millisecond)
	idle2, total2, ok2 := systemTimes()
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

func systemTimes() (idle, total uint64, ok bool) {
	var idleTime, kernelTime, userTime windows.Filetime
	ret, _, _ := procGetSystemTimes.Call(
		uintptr(unsafe.Pointer(&idleTime)),
		uintptr(unsafe.Pointer(&kernelTime)),
		uintptr(unsafe.Pointer(&userTime)),
	)
	if ret == 0 {
		return 0, 0, false
	}
	toU64 := func(f windows.Filetime) uint64 {
		return uint64(f.HighDateTime)<<32 | uint64(f.LowDateTime)
	}
	// kernelTime INCLUDES idle, which is the trap here: total is kernel+user, and
	// subtracting idle from that gives busy. Adding idle in separately double-counts it and
	// caps utilisation near 50 % on an idle machine.
	return toU64(idleTime), toU64(kernelTime) + toU64(userTime), true
}

// cpuName reads the marketing name from the registry — "13th Gen Intel(R) Core(TM) i7-13700K"
// rather than the family/model/stepping triple in PROCESSOR_IDENTIFIER, which is what an
// operator actually recognises.
func cpuName() string {
	key, err := registry.OpenKey(registry.LOCAL_MACHINE,
		`HARDWARE\DESCRIPTION\System\CentralProcessor\0`, registry.QUERY_VALUE)
	if err != nil {
		return "Unknown CPU"
	}
	defer key.Close()
	name, _, err := key.GetStringValue("ProcessorNameString")
	if err != nil {
		return "Unknown CPU"
	}
	return trimSpaces(name)
}

func logicalCores() int { return runtime.NumCPU() }

// physicalCores counts cores rather than threads, via the processor-relationship table.
//
// Reported separately from threads because the difference is the first thing to look at when
// throughput is half what was expected: this project's pipeline is thread-hungry, and 8 cores
// presented as 16 threads is not 16 cores of capacity.
// x/sys/windows does not wrap GetLogicalProcessorInformationEx, so it is called directly.
// Only the first two fields of each variable-length record are needed — the relationship tag
// and the record size — so the union that follows them is deliberately not declared.
const relationProcessorCore = 0

type processorRelationshipHeader struct {
	Relationship uint32
	Size         uint32
}

func physicalCores() int {
	proc := kernel32.NewProc("GetLogicalProcessorInformationEx")

	// A first call with a nil buffer to learn the required size — the documented two-step
	// pattern. It is EXPECTED to fail with ERROR_INSUFFICIENT_BUFFER, so only `size` matters.
	var size uint32
	proc.Call(uintptr(relationProcessorCore), 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return 0
	}

	buf := make([]byte, size)
	ret, _, _ := proc.Call(uintptr(relationProcessorCore),
		uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)))
	if ret == 0 {
		return 0
	}

	count, offset := 0, uint32(0)
	for offset+uint32(unsafe.Sizeof(processorRelationshipHeader{})) <= size {
		header := (*processorRelationshipHeader)(unsafe.Pointer(&buf[offset]))
		if header.Size == 0 {
			break // defensive: a zero size would loop forever
		}
		count++
		offset += header.Size
	}
	return count
}

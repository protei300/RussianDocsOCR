package modules

import (
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
)

// TestSoakConcurrentSession hammers ONE session from eight goroutines.
//
// This is the only test of the thing the per-session mutex exists for, and it has to be
// long to be worth anything: the Python library's failure mode was a CUDA
// `cudaErrorIllegalAddress` after roughly 200 calls, so a short run proves nothing. The Go
// failure mode measured in the spike is different and arguably worse — it does not crash,
// it DEGRADES by at least 90x, which from the outside is indistinguishable from a hang.
//
// Off by default and gated on RDOCS_SOAK=1, matching the library's RUN_QUALITY convention:
// it needs the model artifacts, it takes minutes, and on GPU it monopolises the device.
//
//	RDOCS_SOAK=1 RDOCS_SOAK_DEVICE=gpu go test ./internal/docproc/modules -run Soak
func TestSoakConcurrentSession(t *testing.T) {
	if os.Getenv("RDOCS_SOAK") != "1" {
		t.Skip("set RDOCS_SOAK=1 to run (needs models; takes minutes)")
	}

	device := inference.CPU
	if os.Getenv("RDOCS_SOAK_DEVICE") == "gpu" {
		device = inference.GPU
	}

	root, err := config.ModelsRoot()
	if err != nil {
		t.Fatal(err)
	}
	paths, err := config.LoadModelPaths(root)
	if err != nil {
		t.Fatal(err)
	}
	words, err := NewWordsDetector(root, paths, "ONNX", device, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer words.Close()

	// A synthetic patch of the shape a real field crop has. The point is the CALL
	// pattern, not the content: what is being tested is whether eight goroutines can
	// share one session, which does not depend on there being real text in the image.
	patch := imaging.NewFilled(48, 320, 200, 200, 200)
	defer patch.Close()

	const (
		goroutines       = 8
		callsPerRoutine  = 375 // 8 x 375 = 3000, ~500 documents at 6 split fields each
		perCallSoftLimit = 2 * time.Second
	)

	start := time.Now()
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < callsPerRoutine; i++ {
				callStart := time.Now()
				_, crops, err := words.PredictTransform(patch)
				for j := range crops {
					_ = crops[j].Close()
				}
				if err != nil {
					errs <- fmt.Errorf("goroutine %d call %d: %w", id, i, err)
					return
				}
				// A per-call ceiling rather than only a total: the degradation mode is
				// gradual, and catching it at the call that crossed the line says far more
				// than a total that is merely large.
				if took := time.Since(callStart); took > perCallSoftLimit {
					errs <- fmt.Errorf("goroutine %d call %d took %v, over the %v ceiling: "+
						"this is the contention-degradation signature the per-session mutex "+
						"exists to prevent", id, i, took, perCallSoftLimit)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}

	total := time.Since(start)
	calls := goroutines * callsPerRoutine
	t.Logf("%s: %d calls on one session from %d goroutines in %v (%.2f ms/call)",
		device, calls, goroutines, total.Round(time.Millisecond),
		float64(total.Microseconds())/float64(calls)/1000)
}

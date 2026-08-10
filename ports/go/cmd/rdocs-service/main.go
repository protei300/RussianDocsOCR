// Command rdocs-service is the HTTP service.
//
// Startup order matters and is not arbitrary:
//
//  1. logging, so everything after it is captured;
//  2. config, so a bad value fails before anything expensive;
//  3. the store, wiping first if configured — that has to precede anything that reads it;
//  4. the worker, which starts model loading in the BACKGROUND and returns immediately;
//  5. the HTTP listener, which is therefore serving within milliseconds.
//
// The service accepts uploads while the models are still loading. That is the entire point of
// the queue, and it is why /health reports OK during the fifteen seconds startup takes —
// gating health on the runtime would make Docker kill the container before it finished
// booting.
//
// Port of service/main.py.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/api"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	svclog "github.com/protei300/RussianDocsOCR/ports/go/internal/svc/logging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/runtime"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/seed"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/worker"
)

func main() {
	addr := flag.String("addr", ":8003", "listen address")
	// --healthcheck turns the binary into its own health probe, so the image needs no curl
	// and no shell. A HEALTHCHECK that depends on tools installed purely to run it is a
	// larger image and one more thing to keep patched.
	health := flag.Bool("healthcheck", false, "probe a running instance and exit 0 or 1")
	flag.Parse()

	if *health {
		os.Exit(healthcheck(*addr))
	}

	if err := run(*addr); err != nil {
		// Logged AND printed: if logging setup itself failed, the log line goes nowhere.
		slog.Error("[MAIN] fatal", "err", err)
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

// healthcheck probes the local instance. Exit 0 healthy, 1 otherwise.
//
// It talks to LOOPBACK rather than to the configured address, because --addr is a BIND
// spec: ":8003" means every interface, and using it verbatim as a URL host does not
// resolve. Only the port is taken from it.
func healthcheck(addr string) int {
	port := addr
	if i := strings.LastIndex(addr, ":"); i >= 0 {
		port = addr[i+1:]
	}
	client := &http.Client{Timeout: 4 * time.Second}
	resp, err := client.Get("http://127.0.0.1:" + port + "/health")
	if err != nil {
		fmt.Fprintln(os.Stderr, "healthcheck:", err)
		return 1
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Fprintln(os.Stderr, "healthcheck: HTTP", resp.StatusCode)
		return 1
	}
	return 0
}

func run(addr string) error {
	cfg, cfgErr := config.Load()
	// Logging is installed even when the config failed, so the error below is captured in
	// the ring buffer that /logs serves.
	svclog.Setup(cfg.LogLevel)
	if cfgErr != nil {
		return cfgErr
	}

	slog.Info("[MAIN] starting", "version", cfg.GitCommit,
		"data_dir", cfg.DataDir, "device", cfg.ComputeDevice)

	// **The data directory must live OUTSIDE the repository.** It holds uploaded documents,
	// which are personal data; the default is relative, so a deployment that leaves it
	// unset gets a directory next to the binary rather than inside a source tree.
	dataDir, err := filepath.Abs(cfg.DataDir)
	if err != nil {
		return fmt.Errorf("resolve data dir: %w", err)
	}

	if cfg.DataWipeOnStart {
		// Deliberate, and the reason "ephemeral" is a true statement: `docker restart`
		// keeps the writable layer, so having no volume is not enough on its own.
		size, err := store.Wipe(dataDir)
		if err != nil {
			slog.Warn("[MAIN] could not wipe the data directory", "err", err)
		} else {
			slog.Info("[MAIN] wiped data directory on startup",
				"mb", size/(1024*1024), "dir", dataDir)
		}
	}

	db, err := store.Open(dataDir)
	if err != nil {
		return err
	}

	// Resolved and logged HERE rather than at first use, because a generated key that
	// nobody ever sees is a service nobody can call. The masked/unmasked decision is in the
	// repository layer; this is only the log line.
	authCfg := auth.Config{
		Pin: cfg.AuthPin, JwtSecret: cfg.JwtSecret, JwtAlgorithm: cfg.JwtAlgorithm,
		JwtExpireMinutes: cfg.JwtExpireMinutes, DefaultApiKey: cfg.DefaultApiKey,
	}
	if key, generated, err := auth.ResolveDefaultKey(authCfg); err != nil {
		return fmt.Errorf("resolve default api key: %w", err)
	} else if generated {
		slog.Warn("[MAIN] DEFAULT_API_KEY was not set; generated one for this process only",
			"key", key)
	} else {
		slog.Info("[MAIN] using DEFAULT_API_KEY from the environment")
	}

	if cfg.JwtSecret == config.Defaults().JwtSecret {
		slog.Warn("[MAIN] JWT_SECRET is the built-in default — set it before exposing " +
			"this service to anything you care about")
	}
	if db.IsEphemeral() {
		slog.Warn("[MAIN] storage is TEMPORARY: everything is lost on restart. " +
			"Set a database connection string for anything real.")
	}

	repoRoot := cfg.RepoRoot()

	// Seeded BEFORE the worker starts, so the drain loop never sees a half-inserted fixture.
	// A negative SEED_SAMPLES disables it; 0 means all available.
	if cfg.SeedSamples >= 0 {
		if n := seed.IfEmpty(db, repoRoot, cfg.SeedSamples); n > 0 {
			slog.Info("[BOOT] seeded sample documents from pre-computed results",
				"count", n, "dir", seed.Dir(repoRoot))
		}
	}

	rt := runtime.New()
	wk := worker.New(db, rt, cfg)

	// The worker's context is cancelled on shutdown, which is what stops the drain loop.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	wk.Start(ctx)

	webRoot := api.FindWebRoot(repoRoot)
	if webRoot == "" {
		slog.Warn("[MAIN] no frontend build found; the API works but there is no UI",
			"repo_root", repoRoot)
	} else {
		slog.Info("[MAIN] serving frontend", "dir", webRoot)
	}

	server := &http.Server{
		Addr:    addr,
		Handler: api.NewServer(db, rt, wk, cfg, webRoot).Handler(),
		// Read and write timeouts are deliberately ASYMMETRIC. Reads are bounded because a
		// slow upload holds a connection; writes are generous because a 100 KB result blob
		// over a slow link is legitimate. Neither covers recognition, which happens in the
		// worker and not in a request.
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       5 * time.Minute,
		WriteTimeout:      5 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}

	errCh := make(chan error, 1)
	go func() {
		slog.Info("[MAIN] listening", "addr", addr)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		// **D-12: this path may not run on GPU.** The CUDA provider is reported to overwrite
		// Go's signal handlers, in which case SIGTERM is never delivered and the container
		// is killed after the grace period instead. Unverified here — Windows has no
		// SIGTERM — and to be checked in the Linux image; the recipe is in DEVIATIONS.
		slog.Info("[MAIN] shutdown signal received")
	}

	// The grace period is shorter than Docker's default ten seconds, so the service always
	// finishes on its own terms rather than being killed mid-write.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Warn("[MAIN] graceful shutdown timed out", "err", err)
	}
	wk.Stop()
	slog.Info("[MAIN] stopped")
	return nil
}

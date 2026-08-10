// Package logging installs the process logger and keeps a ring buffer for GET /logs.
//
// Two sinks, and the split matters: stdout carries structured lines at the configured level,
// while the ring buffer captures EVERYTHING regardless of that level. The reason is the
// operator workflow — when something goes wrong you want the debug lines that were already
// emitted, and raising the level afterwards cannot retrieve them.
//
// Port of service/core/logging.py.
package logging

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"
)

// Capacity is how many entries the ring holds.
//
// 5000, matching the reference, so both services keep the same amount of history and an
// operator comparing them is not misled by one having forgotten more. Bounded on purpose: this
// is an in-memory diagnostic aid, not a log store, and an unbounded buffer in a long-running
// service is a slow leak nobody planned.
//
// At roughly 150 bytes per entry that is under a megabyte, which is worth it — the entries you
// want are the ones already emitted before you thought to look.
const Capacity = 5000

// Entry is one buffered record. The JSON shape is what the logs page reads.
type Entry struct {
	Ts      float64 `json:"ts"`
	Level   string  `json:"level"`
	Logger  string  `json:"logger"`
	Message string  `json:"message"`
	Exc     *string `json:"exc"`
}

var levelOrder = map[string]int{
	"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3, "CRITICAL": 4,
}

// ring is the buffer. A slice used circularly rather than a linked list: fixed allocation,
// no per-entry garbage, and the read path is a single pass.
type ring struct {
	mu      sync.Mutex
	entries []Entry
	next    int
	filled  bool
}

var buffer = &ring{entries: make([]Entry, Capacity)}

func (r *ring) add(e Entry) {
	r.mu.Lock()
	r.entries[r.next] = e
	r.next = (r.next + 1) % len(r.entries)
	if r.next == 0 {
		r.filled = true
	}
	r.mu.Unlock()
}

// snapshot returns entries NEWEST FIRST.
func (r *ring) snapshot() []Entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	count := r.next
	if r.filled {
		count = len(r.entries)
	}
	out := make([]Entry, 0, count)
	for i := 0; i < count; i++ {
		idx := (r.next - 1 - i + len(r.entries)*2) % len(r.entries)
		out = append(out, r.entries[idx])
	}
	return out
}

// bufferHandler is a slog.Handler that fans out to the ring and to a delegate.
//
// A handler rather than a hook, because slog has no hooks — and this is the only way to
// capture at DEBUG while the delegate filters at the configured level.
type bufferHandler struct {
	delegate slog.Handler
	group    string
}

// Enabled is unconditionally TRUE: the buffer wants every level. The delegate applies its own
// level in Handle, so stdout stays at the configured verbosity.
func (h *bufferHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *bufferHandler) Handle(ctx context.Context, rec slog.Record) error {
	entry := Entry{
		Ts:      float64(rec.Time.UnixNano()) / 1e9,
		Level:   levelName(rec.Level),
		Logger:  h.group,
		Message: rec.Message,
	}
	// Attributes are folded into the message rather than kept structured, because the logs
	// page renders one line per entry and a nested object there is unreadable. The
	// structured form still reaches stdout through the delegate.
	var extra strings.Builder
	rec.Attrs(func(a slog.Attr) bool {
		extra.WriteString(" ")
		extra.WriteString(a.Key)
		extra.WriteString("=")
		extra.WriteString(a.Value.String())
		return true
	})
	if extra.Len() > 0 {
		entry.Message += extra.String()
	}
	buffer.add(entry)

	if h.delegate.Enabled(ctx, rec.Level) {
		return h.delegate.Handle(ctx, rec)
	}
	return nil
}

func (h *bufferHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &bufferHandler{delegate: h.delegate.WithAttrs(attrs), group: h.group}
}

func (h *bufferHandler) WithGroup(name string) slog.Handler {
	return &bufferHandler{delegate: h.delegate.WithGroup(name), group: name}
}

func levelName(l slog.Level) string {
	switch {
	case l < slog.LevelInfo:
		return "DEBUG"
	case l < slog.LevelWarn:
		return "INFO"
	case l < slog.LevelError:
		return "WARNING"
	default:
		return "ERROR"
	}
}

// Setup installs the logger. Call once, before anything logs.
func Setup(level string) {
	stdout := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: parseLevel(level),
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			// Renamed to match the Python service's field names, so one log pipeline can
			// ingest either implementation without a second parser.
			switch a.Key {
			case slog.TimeKey:
				a.Key = "timestamp"
				a.Value = slog.StringValue(a.Value.Time().UTC().Format(time.RFC3339))
			case slog.LevelKey:
				a.Key = "level"
				a.Value = slog.StringValue(levelName(a.Value.Any().(slog.Level)))
			case slog.MessageKey:
				a.Key = "message"
			}
			return a
		},
	})
	slog.SetDefault(slog.New(&bufferHandler{delegate: stdout, group: "service"}))
}

func parseLevel(name string) slog.Level {
	switch strings.ToUpper(strings.TrimSpace(name)) {
	case "DEBUG":
		return slog.LevelDebug
	case "WARNING", "WARN":
		return slog.LevelWarn
	case "ERROR":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// Entries returns the most recent entries, optionally filtered.
//
// `level` is a MINIMUM severity, not an exact match: asking for warnings should show errors
// too, which is what an operator means by "show me warnings".
func Entries(n int, level, search string) []Entry {
	floor := 0
	if v, ok := levelOrder[strings.ToUpper(level)]; ok {
		floor = v
	}
	needle := strings.ToLower(search)

	out := make([]Entry, 0, n)
	for _, e := range buffer.snapshot() {
		if rank, ok := levelOrder[e.Level]; ok && rank < floor {
			continue
		}
		if needle != "" && !strings.Contains(strings.ToLower(e.Message), needle) {
			continue
		}
		out = append(out, e)
		if len(out) >= n {
			break
		}
	}
	return out
}

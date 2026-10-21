package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"log/slog"
	"math/rand"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	svclog "github.com/protei300/RussianDocsOCR/ports/go/internal/svc/logging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/runtime"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/worker"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/viewmodel"
)

// POST /recognize and PROCESSING_MODE=memory: nothing about a document is kept
// (ports/MEMORY.md §7).
//
// Every test here checks a WAY something could be kept, not just that the route answers: files
// under DATA_DIR, temporary files during the body parse, log lines, the journal, and the buffers
// themselves — which must be zeroed, not merely dropped. The pipeline is replaced by a fake so
// the suite runs without the models; the real one is exercised by conformance/memory_contract.py.

const (
	recognizeKey = "rdk_test_contract_key"
	// What the fake "recognises", and what the client calls the file: neither may reach a log.
	secretText   = "ИВАНОВ ЛЕНИНА 17"
	secretName   = "Ivanov_passport_scan.jpg"
	fakeDocType  = "INTPASSPORT_2011"
	fakePNGBytes = "\x89PNG fake canvas bytes"
)

// fakeRecogniser stands for the pipeline: it records what it was given and keeps a handle on
// every buffer, so a test can look at them after the request.
type fakeRecogniser struct {
	pixels  []imaging.Image // shared handles on the decoded images it got
	png     [][]byte
	timeout time.Duration
	fail    error
}

func (f *fakeRecogniser) recognise(rgb imaging.Image, opts runtime.ImageOptions) (viewmodel.Payload, []byte, error) {
	f.pixels = append(f.pixels, rgb.Share())
	f.timeout = opts.LeaseTimeout
	if f.fail != nil {
		return viewmodel.Payload{}, nil, f.fail
	}
	docType, base, era, device := fakeDocType, "INTPASSPORT", "2011", "cpu"
	w, h := 10, 8
	vm := viewmodel.Payload{
		DocType: &docType, DocTypeBase: &base, DocTypeEra: &era, Recognised: true, Device: &device,
		Canvas:     viewmodel.Canvas{Width: &w, Height: &h},
		CoordSpace: "canvas",
		Boxes:      []viewmodel.Box{},
		Fields:     []viewmodel.Field{},
		Ocr:        map[string]string{"Address": secretText},
		Quality:    map[string]any{"DocConf": 0.97, "Blur": "good"},
		Timings:    map[string]float64{},
	}
	var out []byte
	if opts.WithCanvas {
		out = []byte(fakePNGBytes)
		f.png = append(f.png, out)
	}
	return vm, out, nil
}

func (f *fakeRecogniser) close() {
	for _, p := range f.pixels {
		_ = p.Close()
	}
}

func newRecognizeServer(t *testing.T, mutate func(*config.Settings)) (*Server, *fakeRecogniser, string) {
	t.Helper()
	svclog.Setup("DEBUG")
	dataDir := t.TempDir()
	db, err := store.Open(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.DefaultApiKey = recognizeKey
	if mutate != nil {
		mutate(&cfg)
	}
	rt := runtime.New()
	s := NewServer(db, rt, worker.New(db, rt, cfg), cfg, "")
	fake := &fakeRecogniser{}
	s.recogniseImage = fake.recognise
	t.Cleanup(fake.close)
	return s, fake, dataDir
}

// pngImage is a real, decodable image: noise (incompressible, so it can be made large) or flat.
func pngImage(t *testing.T, w, h int, noise bool) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(1))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.RGBA{200, 200, 200, 255}
			if noise {
				c = color.RGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255}
			}
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func postRaw(s *Server, path string, body []byte, contentType string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", path, bytes.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("X-API-Key", recognizeKey)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

func multipartBody(t *testing.T, field, filename string, data []byte) ([]byte, string) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	part, err := mw.CreateFormFile(field, filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes(), mw.FormDataContentType()
}

func filesUnder(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			out = append(out, path)
		}
		return nil
	})
	return out
}

func allZero(b []byte) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
}

// logSince is every message in the log ring written after the marker, newest first.
func logSince(marker string) string {
	var b strings.Builder
	for _, e := range svclog.Entries(5000, "", "") {
		if strings.Contains(e.Message, marker) {
			break
		}
		b.WriteString(e.Message)
		b.WriteString("\n")
	}
	return b.String()
}

// markLog writes a unique line that logSince stops at.
func markLog(t *testing.T) string {
	marker := fmt.Sprintf("[TEST] start %s %d", t.Name(), time.Now().UnixNano())
	slog.Info(marker)
	return marker
}

// --- the route -------------------------------------------------------------------------------

func TestRecognizeAnswersAndStoresNothing(t *testing.T) {
	s, _, dataDir := newRecognizeServer(t, nil)
	marker := markLog(t)
	before := filesUnder(t, dataDir)

	rec := postRaw(s, Prefix+"/recognize", pngImage(t, 32, 24, false), "image/png")
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	if rec.Header().Get("Cache-Control") != "no-store" {
		t.Errorf("Cache-Control %q, want no-store", rec.Header().Get("Cache-Control"))
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	want := []string{"stored", "doc_type", "doc_type_base", "doc_type_era", "recognised", "doc_conf",
		"device", "canvas", "coord_space", "coord_space_note", "boxes", "fields", "ocr", "address",
		"quality", "timings", "processing_ms"}
	if len(body) != len(want) {
		t.Errorf("answer has %d keys, want %d: %v", len(body), len(want), body)
	}
	for _, k := range want {
		if _, ok := body[k]; !ok {
			t.Errorf("answer lacks %q", k)
		}
	}
	if body["stored"] != false || body["recognised"] != true || body["doc_conf"] != 0.97 {
		t.Errorf("stored/recognised/doc_conf: %v %v %v", body["stored"], body["recognised"], body["doc_conf"])
	}
	if _, ok := body["quality"].(map[string]any)["DocConf"]; ok {
		t.Error("DocConf belongs in doc_conf, not in quality")
	}
	if _, ok := body["canvas"].(map[string]any)["image_png_base64"]; ok {
		t.Error("the canvas image came back without ?images=1")
	}
	if body["ocr"].(map[string]any)["Address"] != secretText {
		t.Errorf("ocr: %v", body["ocr"])
	}

	if after := filesUnder(t, dataDir); len(after) != len(before) {
		t.Errorf("a recognition left a file under DATA_DIR: %v -> %v", before, after)
	}
	if total := s.journal.Counts()["total"]; total != 1 {
		t.Errorf("journal total %d, want 1", total)
	}
	logs := logSince(marker)
	if strings.Contains(logs, secretText) || strings.Contains(logs, "INTPASSPORT") {
		t.Errorf("the log holds document data:\n%s", logs)
	}
	var lines []string
	for _, ln := range strings.Split(logs, "\n") {
		if strings.HasPrefix(ln, "[API] recognize") {
			lines = append(lines, ln)
		}
	}
	if len(lines) != 1 || !strings.HasPrefix(lines[0], "[API] recognize: outcome=recognised status=200 ms=") ||
		len(strings.Fields(strings.TrimPrefix(lines[0], "[API] recognize:"))) != 3 {
		t.Errorf("want exactly one 'outcome= status= ms=' line, got %q", lines)
	}
}

// §7.1 — a multipart body past any spool threshold goes through without a temporary file. The
// temp directory is pointed at an empty one and must stay empty: r.ParseMultipartForm /
// r.FormFile would write the part there (mime/multipart.Reader.ReadForm -> os.CreateTemp).
func TestMultipartIsSplitInMemoryNotSpooled(t *testing.T) {
	tmp := t.TempDir()
	for _, name := range []string{"TMP", "TEMP", "TMPDIR"} {
		t.Setenv(name, tmp)
	}
	if os.TempDir() != tmp {
		t.Fatalf("could not redirect the temp directory: %s", os.TempDir())
	}
	s, fake, _ := newRecognizeServer(t, nil)
	marker := markLog(t)

	big := pngImage(t, 1000, 800, true)
	if len(big) < 2*1024*1024 {
		t.Fatalf("the image is %d bytes, want >= 2 MB", len(big))
	}
	body, contentType := multipartBody(t, "file", secretName, big)
	rec := postRaw(s, Prefix+"/recognize", body, contentType)
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	if len(fake.pixels) != 1 || fake.pixels[0].Width() != 1000 {
		t.Errorf("the part did not reach the recogniser intact")
	}
	if files := filesUnder(t, tmp); len(files) != 0 {
		t.Errorf("parsing the upload created temporary files: %v", files)
	}
	if strings.Contains(logSince(marker), secretName) {
		t.Error("the filename reached the log")
	}
}

// §7.2 — the image bytes, the decoded pixels and the PNG are all zeros after the answer.
func TestBuffersAreZeroedAfterTheAnswer(t *testing.T) {
	for _, multi := range []bool{false, true} {
		t.Run(fmt.Sprintf("multipart=%v", multi), func(t *testing.T) {
			s, fake, _ := newRecognizeServer(t, nil)
			var images [][]byte
			inner := s.recogniseBytes
			s.recogniseBytes = func(image []byte, opts runtime.ImageOptions) (viewmodel.Payload, []byte, error) {
				images = append(images, image)
				return inner(image, opts)
			}

			body, contentType := pngImage(t, 64, 48, true), "image/png"
			if multi {
				body, contentType = multipartBody(t, "file", secretName, body)
			}
			rec := postRaw(s, Prefix+"/recognize?images=1", body, contentType)
			if rec.Code != http.StatusOK {
				t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
			}
			var answer struct {
				Canvas map[string]any `json:"canvas"`
			}
			_ = json.Unmarshal(rec.Body.Bytes(), &answer)
			b64, _ := answer.Canvas["image_png_base64"].(string)
			if got, _ := base64.StdEncoding.DecodeString(b64); string(got) != fakePNGBytes {
				t.Errorf("image_png_base64 decodes to %q", got)
			}

			if len(images) != 1 || len(images[0]) == 0 || !allZero(images[0]) {
				t.Error("the image bytes were not zeroed")
			}
			if len(fake.pixels) != 1 || !fake.pixels[0].IsAllZero() {
				t.Error("the decoded pixels were not zeroed")
			}
			if len(fake.png) != 1 || !allZero(fake.png[0]) {
				t.Error("the canvas PNG was not zeroed")
			}
		})
	}
}

func TestBadInputIsRefusedAndJournalled(t *testing.T) {
	s, _, _ := newRecognizeServer(t, func(c *config.Settings) { c.MaxUploadMB = 1 })
	jpegHead := []byte{0xff, 0xd8, 0xff, 0xe0}
	// A form with a plain field only: no part named 'file', no part with a filename.
	var form bytes.Buffer
	mw := multipart.NewWriter(&form)
	_ = mw.WriteField("note", "hello")
	_ = mw.Close()
	other, otherType := form.Bytes(), mw.FormDataContentType()
	for _, c := range []struct {
		name        string
		body        []byte
		contentType string
		status      int
		detail      string
	}{
		{"empty", nil, "image/jpeg", 400, "Empty upload"},
		{"pdf", []byte("%PDF-1.7 not an image"), "application/pdf", 415,
			"PDF is not supported — send a JPEG, PNG, WEBP, BMP or TIFF image"},
		{"text", []byte("just some text"), "text/plain", 415, "Unsupported file type — expected an image"},
		{"undecodable", append(append([]byte{}, jpegHead...), make([]byte, 100)...), "image/jpeg", 422,
			"The image could not be decoded — it may be corrupt"},
		{"too large", append(append([]byte{}, jpegHead...), make([]byte, 2*1024*1024)...), "image/jpeg", 413,
			"Image exceeds the upload limit"},
		{"no boundary", []byte("x"), "multipart/form-data", 400, "Malformed multipart body"},
		{"no image part", other, otherType, 400, "No image in the multipart body — send it as the 'file' part"},
	} {
		rec := postRaw(s, Prefix+"/recognize", c.body, c.contentType)
		if rec.Code != c.status || detailOf(rec) != c.detail {
			t.Errorf("%s: %d %q, want %d %q", c.name, rec.Code, detailOf(rec), c.status, c.detail)
		}
		items, _ := s.journal.Recent(1, 0)
		if len(items) != 1 || items[0].Outcome != "rejected" || items[0].StatusCode != c.status {
			t.Errorf("%s: journal %+v", c.name, items)
		}
	}
}

// A Content-Length over the cap is refused before a byte of the body is read.
func TestDeclaredOversizeIsRefusedUnread(t *testing.T) {
	s, _, _ := newRecognizeServer(t, func(c *config.Settings) { c.MaxUploadMB = 1 })
	body := &countingReader{}
	req := httptest.NewRequest("POST", Prefix+"/recognize", body)
	req.ContentLength = 50 * 1024 * 1024
	req.Header.Set("Content-Type", "image/jpeg")
	req.Header.Set("X-API-Key", recognizeKey)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge || body.read != 0 {
		t.Errorf("HTTP %d after reading %d bytes; want 413 and nothing read", rec.Code, body.read)
	}
}

func TestBusyPipelineIs503WithRetryAfter(t *testing.T) {
	s, fake, _ := newRecognizeServer(t, func(c *config.Settings) { c.RecognizeWaitSec = 7 })
	fake.fail = fmt.Errorf("%w: no pipeline", errs.ErrPipelineBusy)
	rec := postRaw(s, Prefix+"/recognize", pngImage(t, 16, 16, false), "image/png")
	if rec.Code != http.StatusServiceUnavailable || rec.Header().Get("Retry-After") != "5" {
		t.Errorf("HTTP %d Retry-After %q", rec.Code, rec.Header().Get("Retry-After"))
	}
	if fake.timeout != 7*time.Second {
		t.Errorf("lease timeout %s, want RECOGNIZE_WAIT_SEC=7s", fake.timeout)
	}
	if items, _ := s.journal.Recent(1, 0); items[0].Outcome != "busy" {
		t.Errorf("journal outcome %q", items[0].Outcome)
	}

	// Not ready yet: the real runtime, never initialised.
	s.recogniseImage = s.rt.RecogniseImage
	rec = postRaw(s, Prefix+"/recognize", pngImage(t, 16, 16, false), "image/png")
	if rec.Code != http.StatusServiceUnavailable || rec.Header().Get("Retry-After") != "10" {
		t.Errorf("not ready: HTTP %d Retry-After %q", rec.Code, rec.Header().Get("Retry-After"))
	}
}

type secretError struct{}

func (secretError) Error() string { return "failed while reading " + secretText }

func TestUnexpectedErrorIs500AndLogsOnlyTheType(t *testing.T) {
	s, fake, _ := newRecognizeServer(t, nil)
	marker := markLog(t)
	fake.fail = secretError{}
	rec := postRaw(s, Prefix+"/recognize", pngImage(t, 16, 16, false), "image/png")
	if rec.Code != http.StatusInternalServerError || strings.Contains(rec.Body.String(), secretText) {
		t.Errorf("HTTP %d %s", rec.Code, rec.Body)
	}
	logs := logSince(marker)
	if strings.Contains(logs, secretText) || !strings.Contains(logs, "secretError") {
		t.Errorf("want the type and not the message in the log:\n%s", logs)
	}
	if items, _ := s.journal.Recent(1, 0); items[0].Outcome != "error" || items[0].StatusCode != 500 {
		t.Errorf("journal %+v", items[0])
	}
}

func TestJournalHoldsOutcomeAndTimeOnly(t *testing.T) {
	s, _, _ := newRecognizeServer(t, nil)
	postRaw(s, Prefix+"/recognize", pngImage(t, 16, 16, false), "image/png")
	req := httptest.NewRequest("GET", Prefix+"/recognitions", nil)
	req.Header.Set("X-API-Key", recognizeKey)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	var body struct {
		Items  []map[string]any `json:"items"`
		Total  int              `json:"total"`
		Counts map[string]int   `json:"counts"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Total != 1 || body.Counts["recognised"] != 1 || body.Counts["total"] != 1 {
		t.Errorf("journal: %s", rec.Body)
	}
	keys := []string{"id", "at", "outcome", "status_code", "processing_ms", "stored"}
	if len(body.Items) != 1 || len(body.Items[0]) != len(keys) {
		t.Fatalf("items: %v", body.Items)
	}
	for _, k := range keys {
		if _, ok := body.Items[0][k]; !ok {
			t.Errorf("journal line lacks %q", k)
		}
	}
}

// --- memory mode -----------------------------------------------------------------------------

// countingReader is a request body that records whether anybody read it.
type countingReader struct{ read int }

func (c *countingReader) Read(p []byte) (int, error) {
	if c.read > 64*1024*1024 {
		return 0, io.EOF
	}
	for i := range p {
		p[i] = 'x'
	}
	c.read += len(p)
	return len(p), nil
}

// §7.4 — in memory mode every /documents route is 404 before auth and before the body is read.
func TestMemoryModeShortCircuitsDocumentRoutes(t *testing.T) {
	s, _, dataDir := newRecognizeServer(t, func(c *config.Settings) { c.ProcessingMode = "memory" })
	before := filesUnder(t, dataDir)

	for _, c := range []struct{ method, path string }{
		{"GET", "/documents"}, {"GET", "/documents/1"}, {"GET", "/documents/1/image/canvas"},
		{"GET", "/documents/1/progress"}, {"DELETE", "/documents/1"}, {"POST", "/documents/purge"},
		{"POST", "/documents/1/reprocess"},
	} {
		// No credential at all: the answer comes before authentication.
		rec := call(s, c.method, Prefix+c.path, "", "")
		if rec.Code != http.StatusNotFound || !strings.Contains(detailOf(rec), "PROCESSING_MODE=memory") {
			t.Errorf("%s %s: %d %q", c.method, c.path, rec.Code, detailOf(rec))
		}
	}

	// An oversized upload: answered without a single byte of it read.
	body := &countingReader{}
	req := httptest.NewRequest("POST", Prefix+"/documents", body)
	req.Header.Set("Content-Type", "multipart/form-data; boundary=x")
	req.Header.Set("X-API-Key", recognizeKey)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound || body.read != 0 {
		t.Errorf("POST /documents: HTTP %d after reading %d bytes; want 404 and nothing read",
			rec.Code, body.read)
	}

	if rec := postRaw(s, Prefix+"/recognize", pngImage(t, 16, 16, false), "image/png"); rec.Code != 200 {
		t.Errorf("/recognize in memory mode: %d %s", rec.Code, rec.Body)
	}
	if after := filesUnder(t, dataDir); len(after) != len(before) {
		t.Errorf("memory mode wrote a file: %v -> %v", before, after)
	}

	var cfg map[string]any
	_ = json.Unmarshal(call(s, "GET", Prefix+"/auth/config", "", "").Body.Bytes(), &cfg)
	if cfg["processing_mode"] != "memory" {
		t.Errorf("auth/config processing_mode %v", cfg["processing_mode"])
	}
	var status struct {
		Processing struct {
			Mode         string         `json:"mode"`
			Reason       *string        `json:"reason"`
			Recognitions map[string]int `json:"recognitions"`
		} `json:"processing"`
	}
	_ = json.Unmarshal(call(s, "GET", Prefix+"/status", sessionToken(t, s), "").Body.Bytes(), &status)
	if status.Processing.Mode != "memory" || status.Processing.Reason != nil ||
		status.Processing.Recognitions["recognised"] != 1 {
		t.Errorf("status.processing %+v", status.Processing)
	}
}

func TestStoredModeServesDocumentsAndReportsIt(t *testing.T) {
	s, _, _ := newRecognizeServer(t, nil)
	if rec := call(s, "GET", Prefix+"/documents", sessionToken(t, s), ""); rec.Code != http.StatusOK {
		t.Errorf("GET /documents in stored mode: %d", rec.Code)
	}
	var cfg map[string]any
	_ = json.Unmarshal(call(s, "GET", Prefix+"/auth/config", "", "").Body.Bytes(), &cfg)
	if cfg["processing_mode"] != "stored" {
		t.Errorf("auth/config processing_mode %v", cfg["processing_mode"])
	}
}

func TestMistypedModeMeansMemoryAndSaysWhy(t *testing.T) {
	s, _, _ := newRecognizeServer(t, func(c *config.Settings) { c.ProcessingMode = "memroy" })
	if rec := call(s, "GET", Prefix+"/documents", sessionToken(t, s), ""); rec.Code != http.StatusNotFound {
		t.Errorf("GET /documents: %d", rec.Code)
	}
	var status struct {
		Processing struct {
			Mode   string  `json:"mode"`
			Reason *string `json:"reason"`
		} `json:"processing"`
	}
	_ = json.Unmarshal(call(s, "GET", Prefix+"/status", sessionToken(t, s), "").Body.Bytes(), &status)
	if status.Processing.Mode != "memory" || status.Processing.Reason == nil ||
		!strings.Contains(*status.Processing.Reason, "memroy") {
		t.Errorf("status.processing %+v", status.Processing)
	}
}

// A viewer may read the journal but not recognise; an API key may recognise; nobody is 401.
func TestRecognizeRoles(t *testing.T) {
	s := newModeServer(t, "users")
	fake := &fakeRecogniser{}
	s.recogniseImage = fake.recognise
	t.Cleanup(fake.close)
	_, viewer := s.testAccount(t, "vera", "viewer", false)
	_, operator := s.testAccount(t, "otto", "operator", false)
	img := pngImage(t, 16, 16, false)

	send := func(token, key string) int {
		req := httptest.NewRequest("POST", Prefix+"/recognize", bytes.NewReader(img))
		req.Header.Set("Content-Type", "image/png")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		if key != "" {
			req.Header.Set("X-API-Key", key)
		}
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)
		return rec.Code
	}
	if code := send(viewer, ""); code != http.StatusForbidden {
		t.Errorf("viewer recognised: %d", code)
	}
	if code := send(operator, ""); code != http.StatusOK {
		t.Errorf("operator: %d", code)
	}
	if code := send("", "rdk_test_contract_key"); code != http.StatusOK {
		t.Errorf("API key: %d", code)
	}
	if code := send("", ""); code != http.StatusUnauthorized {
		t.Errorf("anonymous: %d", code)
	}
	if rec := call(s, "GET", Prefix+"/recognitions", viewer, ""); rec.Code != http.StatusOK {
		t.Errorf("viewer reading the journal: %d", rec.Code)
	}
	// Refusals by the guard are not journalled: only the two admitted requests are there.
	if total := s.journal.Counts()["total"]; total != 2 {
		t.Errorf("journal total %d, want 2", total)
	}
}

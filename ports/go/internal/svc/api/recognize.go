package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/recognitions"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/runtime"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/settingsschema"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/viewmodel"
)

// Recognition in memory: one image in, one answer out, nothing kept (ports/MEMORY.md §3-§5).
//
// POST /recognize reads the image into a buffer, decodes it, recognises it inside the request,
// answers, and zeroes what it owns. No record, no file, no artifact, no queue; one log line and
// one journal line with the outcome and the time — nothing about the document. Available in
// both processing modes; in PROCESSING_MODE=memory it is the only way a document gets in.
//
// What "nothing kept" takes, because each item below was a real way to keep something:
//
//   - **The body is never spooled to disk.** r.ParseMultipartForm / r.FormFile write every part
//     past their memory limit to a temporary file, so a passport photo would touch the disk
//     before any code of ours ran. This handler reads r.Body itself into ONE buffer of known
//     size, with a hard cap, and splits a multipart body with multipart.NewReader over those
//     bytes — Reader.NextPart does not spool, Reader.ReadForm does and is not used.
//   - **Buffers we own are zeroed**, not just dropped: the body, the image bytes, the decoded
//     pixels (and the decoder's BGR intermediate), the canvas PNG, the base64 of it and the
//     serialised answer. The pipeline's own images are zeroed inside the lease (runtime rule 11).
//   - **Strings cannot be wiped.** The recognised text lives in Go strings, which are immutable;
//     it is released and the memory reused, not overwritten. The README says so.
//   - **Nothing about the document reaches a log**: not the filename, the type, the size, a
//     library message or an error text — only the outcome, and for a failure the error's type.
//   - **The answer is not cacheable**: Cache-Control: no-store.
//
// Port of service/api/recognize.py.

// multipartFraming is what the cap allows on top of MAX_UPLOAD_MB for the multipart envelope.
const multipartFraming = 64 * 1024

// refusal is a /recognize answer decided before or around recognition: status, detail, outcome.
type refusal struct {
	status  int
	detail  string
	outcome string
	headers map[string]string
}

func (r *refusal) Error() string { return r.detail }

func rejected(status int, detail string) *refusal {
	return &refusal{status: status, detail: detail, outcome: recognitions.Rejected}
}

// wipe zeroes a byte buffer over its whole capacity.
func wipe(buf []byte) {
	clear(buf[:cap(buf)])
}

// readCapped reads the whole body into one buffer, refusing more than max bytes.
//
// io.ReadAll is not used: it grows its buffer by reallocation, and every reallocation leaves the
// previous copy of the document in the heap. With a Content-Length the buffer is allocated once
// at that size; without one (chunked), every grown buffer's predecessor is zeroed first.
func readCapped(body io.Reader, declared int64, max int64) ([]byte, *refusal) {
	tooLarge := rejected(http.StatusRequestEntityTooLarge, "Image exceeds the upload limit")
	if declared > max {
		return nil, tooLarge
	}
	size := int64(64 * 1024)
	if declared >= 0 {
		size = declared + 1 // +1: a body longer than it declared is noticed, not truncated
	}
	buf := make([]byte, 0, size)
	for {
		if len(buf) == cap(buf) {
			if int64(len(buf)) > max {
				wipe(buf)
				return nil, tooLarge
			}
			grown := make([]byte, len(buf), 2*cap(buf))
			copy(grown, buf)
			wipe(buf)
			buf = grown
		}
		n, err := body.Read(buf[len(buf):cap(buf)])
		buf = buf[:len(buf)+n]
		if int64(len(buf)) > max {
			wipe(buf)
			return nil, tooLarge
		}
		if errors.Is(err, io.EOF) {
			return buf, nil
		}
		if err != nil {
			// A connection that broke mid-upload: nothing to recognise.
			wipe(buf)
			return nil, rejected(http.StatusBadRequest, "Could not read the request body")
		}
	}
}

// imageFromMultipart copies the first part named "file", or the first part with a filename,
// into its own buffer. The body buffer is the caller's to zero; this copy is zeroed later too.
// The filename is looked at only to choose the part and kept nowhere.
func imageFromMultipart(body []byte, boundary string) ([]byte, *refusal) {
	noImage := rejected(http.StatusBadRequest,
		"No image in the multipart body — send it as the 'file' part")
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		part, err := reader.NextPart()
		if err != nil {
			return nil, noImage // io.EOF: no such part; anything else: malformed
		}
		if part.FormName() != "file" && part.FileName() == "" {
			continue
		}
		// The part cannot be longer than the body, so one allocation of that size holds it —
		// no reallocation, no stray copy. The unused tail is zero already.
		image := make([]byte, len(body))
		n, err := io.ReadFull(part, image)
		if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
			wipe(image)
			return nil, noImage
		}
		return image[:n], nil
	}
}

// boundaryOf is the multipart boundary of a Content-Type, or "" when there is none.
func boundaryOf(contentType string) string {
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return ""
	}
	return params["boundary"]
}

// queryBool reads a boolean query parameter the way pydantic does: absent is the default, the
// usual spellings either way, anything else 422 in pydantic's own shape (see params.go).
func queryBool(q url.Values, name string, def bool) (bool, error) {
	if !q.Has(name) {
		return def, nil
	}
	raw := q.Get(name)
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "on", "t", "true", "y", "yes":
		return true, nil
	case "0", "off", "f", "false", "n", "no":
		return false, nil
	}
	return false, &paramError{paramErrorItem{
		Type:  "bool_parsing",
		Loc:   []any{"query", name},
		Msg:   "Input should be a valid boolean, unable to interpret input",
		Input: raw,
	}}
}

// finish writes the journal line and the one log line of a recognition: outcome, status, time.
func (s *Server) finish(outcome string, status int, started time.Time) int {
	ms := int(time.Since(started).Milliseconds())
	s.journal.Record(outcome, status, ms)
	// The one log line of an in-memory recognition. Nothing else — exactly three tokens.
	slog.Info(fmt.Sprintf("[API] recognize: outcome=%s status=%d ms=%d", outcome, status, ms))
	return ms
}

// decodeAndRecognise is the default recogniseBytes: decode, recognise, zero. Every buffer it
// owns is zeroed in a defer, also when recognition fails, panics or is abandoned by a timeout.
func (s *Server) decodeAndRecognise(image []byte, opts runtime.ImageOptions) (viewmodel.Payload, []byte, error) {
	defer wipe(image)
	rgb, err := imaging.DecodeRGBScrubbed(image)
	if err != nil {
		if errors.Is(err, imaging.ErrNotDecoded) {
			return viewmodel.Payload{}, nil, rejected(http.StatusUnprocessableEntity,
				"The image could not be decoded — it may be corrupt")
		}
		return viewmodel.Payload{}, nil, err
	}
	defer rgb.ZeroAndClose()
	// The encoded bytes are not needed once decoded.
	wipe(image)
	return s.recogniseImage(rgb, opts)
}

type recognised struct {
	vm  viewmodel.Payload
	png []byte
	err error
}

// handleRecognize is POST /recognize.
func (s *Server) handleRecognize(w http.ResponseWriter, r *http.Request, _ *Identity) {
	// Parsed before anything is journalled: in the reference a bad query parameter is FastAPI's
	// own 422, raised before the handler body runs.
	withImages, err := queryBool(r.URL.Query(), "images", false)
	if err != nil {
		writeError(w, err)
		return
	}

	started := time.Now()
	limit := s.cfg.MaxUploadBytes()
	var body, image, png []byte
	handedOff := false // the image belongs to the recognition goroutine from then on
	defer func() {
		wipe(body)
		if !handedOff {
			wipe(image)
		}
		wipe(png)
	}()

	refuse := func(rf *refusal) {
		s.finish(rf.outcome, rf.status, started)
		for k, v := range rf.headers {
			w.Header().Set(k, v)
		}
		writeJSON(w, rf.status, errorBody{Detail: rf.detail})
	}

	var rf *refusal
	body, rf = readCapped(r.Body, r.ContentLength, limit+multipartFraming)
	if rf != nil {
		refuse(rf)
		return
	}
	contentType := r.Header.Get("Content-Type")
	if strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		boundary := boundaryOf(contentType)
		if boundary == "" {
			refuse(rejected(http.StatusBadRequest, "Malformed multipart body"))
			return
		}
		image, rf = imageFromMultipart(body, boundary)
		wipe(body)
		if rf != nil {
			refuse(rf)
			return
		}
	} else {
		image, body = body, nil
	}

	switch {
	case len(image) == 0:
		refuse(rejected(http.StatusBadRequest, "Empty upload"))
		return
	case int64(len(image)) > limit:
		refuse(rejected(http.StatusRequestEntityTooLarge,
			fmt.Sprintf("File exceeds the %d MB limit", s.cfg.MaxUploadMB)))
		return
	case repo.IsPDF(image):
		refuse(rejected(http.StatusUnsupportedMediaType,
			"PDF is not supported — send a JPEG, PNG, WEBP, BMP or TIFF image"))
		return
	}
	if _, _, ok := repo.SniffImage(image); !ok {
		refuse(rejected(http.StatusUnsupportedMediaType, "Unsupported file type — expected an image"))
		return
	}

	wait := time.Duration(s.cfg.RecognizeWaitSec) * time.Second
	opts := runtime.ImageOptions{
		Docconf: settingsschema.TypedFloat("docconf",
			repo.SettingValue(s.db, s.cfg, "docconf"), s.cfg.Docconf),
		ImgSize: settingsschema.TypedInt("img_size",
			repo.SettingValue(s.db, s.cfg, "img_size"), s.cfg.ImgSize),
		WithCanvas:   withImages,
		LeaseTimeout: wait,
	}
	jobTimeout := time.Duration(settingsschema.TypedInt("job_timeout_sec",
		repo.SettingValue(s.db, s.cfg, "job_timeout_sec"), s.cfg.JobTimeoutSec)) * time.Second

	// Its own goroutine, so the timeout can be observed; it cannot be cancelled (a goroutine
	// has no kill), so on a timeout it finishes alone and zeroes what it produced itself.
	done := make(chan recognised, 1)
	handedOff = true
	go func(image []byte) {
		var out recognised
		defer func() {
			if p := recover(); p != nil {
				out = recognised{err: fmt.Errorf("panic: %T", p)}
			}
			done <- out
		}()
		out.vm, out.png, out.err = s.recogniseBytes(image, opts)
	}(image)

	var got recognised
	select {
	case got = <-done:
	case <-time.After(jobTimeout + wait):
		go func() { wipe((<-done).png) }()
		refuse(&refusal{status: http.StatusGatewayTimeout, detail: "Recognition took too long",
			outcome: recognitions.Error})
		return
	}
	png = got.png

	if got.err != nil {
		var own *refusal
		switch {
		case errors.As(got.err, &own):
			refuse(own)
		case errors.Is(got.err, errs.ErrPipelineBusy):
			refuse(&refusal{status: http.StatusServiceUnavailable,
				detail:  "All recognition pipelines are busy — retry shortly",
				outcome: recognitions.Busy, headers: map[string]string{"Retry-After": "5"}})
		case errors.Is(got.err, errs.ErrRuntimeNotReady):
			refuse(&refusal{status: http.StatusServiceUnavailable,
				detail:  "The recognition runtime is not ready yet",
				outcome: recognitions.Busy, headers: map[string]string{"Retry-After": "10"}})
		default:
			s.finish(recognitions.Error, http.StatusInternalServerError, started)
			// The TYPE only: an error's text carries paths and recognised values.
			slog.Warn(fmt.Sprintf("[API] recognize failed: %T", got.err))
			writeJSON(w, http.StatusInternalServerError, errorBody{Detail: "Recognition failed"})
		}
		return
	}

	ms := int(time.Since(started).Milliseconds())
	answer, err := s.recognitionAnswer(got.vm, png, ms)
	if err != nil {
		s.finish(recognitions.Error, http.StatusInternalServerError, started)
		slog.Warn(fmt.Sprintf("[API] recognize failed: %T", err))
		writeJSON(w, http.StatusInternalServerError, errorBody{Detail: "Recognition failed"})
		return
	}
	defer wipe(answer)
	outcome := recognitions.NotRecognised
	if got.vm.Recognised {
		outcome = recognitions.Recognised
	}
	s.finish(outcome, http.StatusOK, started)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(answer)
}

// recognitionAnswer serialises the 200 answer (ports/MEMORY.md §3.4): the stored document's
// detail without the row, the URLs or anything that points at a stored thing, plus stored=false.
// The same view model the worker stores, through the same JSON, so the two cannot drift apart.
// Returned as bytes the caller zeroes after writing.
func (s *Server) recognitionAnswer(vm viewmodel.Payload, png []byte, ms int) ([]byte, error) {
	raw, err := json.Marshal(vm)
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	err = json.Unmarshal(raw, &payload)
	wipe(raw)
	if err != nil {
		return nil, err
	}

	canvas := map[string]any{}
	if c, ok := payload["canvas"].(map[string]any); ok {
		for k, v := range c {
			canvas[k] = v
		}
	}
	var b64 []byte
	if png != nil {
		b64 = make([]byte, base64.StdEncoding.EncodedLen(len(png)))
		base64.StdEncoding.Encode(b64, png)
		defer wipe(b64)
		// RawMessage: the base64 goes into the answer without becoming a string first.
		canvas["image_png_base64"] = json.RawMessage(append(append([]byte{'"'}, b64...), '"'))
	}

	// Same split as a stored document's row: DocConf on its own, the rest as quality.
	var docConf any
	quality := map[string]any{}
	if q, ok := payload["quality"].(map[string]any); ok {
		for k, v := range q {
			if k == "DocConf" {
				docConf = v
				continue
			}
			quality[k] = v
		}
	}
	answer := map[string]any{
		"stored":           false,
		"doc_type":         payload["doc_type"],
		"doc_type_base":    payload["doc_type_base"],
		"doc_type_era":     payload["doc_type_era"],
		"recognised":       payload["recognised"],
		"doc_conf":         docConf,
		"device":           payload["device"],
		"canvas":           canvas,
		"coord_space":      payload["coord_space"],
		"coord_space_note": payload["coord_space_note"],
		"boxes":            orEmptyList(payload["boxes"]),
		"fields":           orEmptyList(payload["fields"]),
		"ocr":              orEmptyMap(payload["ocr"]),
		"quality":          quality,
		"address":          payload["address"],
		"timings":          orEmptyMap(payload["timings"]),
		"processing_ms":    ms,
	}
	out, err := json.Marshal(answer)
	if raw, ok := canvas["image_png_base64"].(json.RawMessage); ok {
		wipe(raw)
	}
	return out, err
}

// handleRecognitions is GET /recognitions: the journal — when, how it ended, how long. Nothing
// else exists to show; these lines cannot be opened.
func (s *Server) handleRecognitions(w http.ResponseWriter, r *http.Request, _ *Identity) {
	q := r.URL.Query()
	limit, err := queryInt(q, "limit", 100, 1, 1000)
	if err != nil {
		writeError(w, err)
		return
	}
	offset, err := queryInt(q, "offset", 0, 0, 0)
	if err != nil {
		writeError(w, err)
		return
	}
	items, total := s.journal.Recent(limit, offset)
	writeJSON(w, http.StatusOK, map[string]any{
		"items":  items,
		"total":  total,
		"counts": s.journal.Counts(),
	})
}

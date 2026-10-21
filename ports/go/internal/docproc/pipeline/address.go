package pipeline

import (
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// The registration page of the internal passport (INTPASSPORTADDR): the address lines, read one
// by one, and the registration stamps they belong to. Port of pipeline/reading.py::
// ReadingStage._address_lines and _route_word_ocr / _is_number_token (the lines) and
// pipeline/stamps.py::StampsStage (the stamps); conformance stages address.lines (9),
// address.stamps (9a) and the address_lines part of quads (10a).

// AddressHandwrittenPlaceholder is ADDRESS_HW_PLACEHOLDER: what a handwritten line reads as
// (the OCR is printed-only). ASCII brackets on purpose, see the reference.
const AddressHandwrittenPlaceholder = "[рукопись]"

// SpreadAspect is StampsStage.SPREAD_ASPECT: a canvas wider than this many heights is a spread
// of two pages, split down the middle.
const SpreadAspect = 1.2

// AddressLine is one entry of meta_results['Address_lines'].
type AddressLine struct {
	// Kind is "printed" or "handwritten"; PHandwritten the classifier's raw score.
	Kind         string
	PHandwritten float64
	// Text is nil for a handwritten line (JSON null), the words joined by a space otherwise.
	Text *string
	// Stamp is the index into AddressResult.Stamps of the stamp the line lies in; nil when it lies
	// in none. Only meaningful when StampsRan.
	Stamp *int
}

// Stamp is one entry of meta_results['RegistrationStamps'], freshest first.
type Stamp struct {
	Kind       string
	Confidence float64
	// Quad is the stamp on the input image; nil when the way back is not known.
	Quad *geometry.Quad
	// Lines are indices into AddressResult.Lines; Address their text joined by a newline.
	Lines   []int
	Address string
	// Cancels is the index of the registration a deregistered stamp cancels; nil otherwise.
	Cancels *int
	// Current is true on the registration nothing cancels.
	Current bool
}

// AddressResult is everything the address path produced.
type AddressResult struct {
	// Detections are all the boxes of the line detector (meta_results['AddressLinesDetector']
	// ['obbox']), including those whose patch came out empty and were skipped. Lines is one entry
	// per box that was kept, LineBoxes their boxes in the same order: the two lists differ in
	// length exactly when a patch was skipped, which is what the view model's `aligned` says.
	Detections []postprocess.ObbBox
	LineBoxes  []postprocess.ObbBox
	Lines      []AddressLine

	// Text is OCR['Address'] (the lines joined by a newline, a handwritten one as the
	// placeholder); empty when there was no line. HasHandwritten is OCR['Address_has_handwritten'].
	Text           string
	HasHandwritten bool

	// LineQuads are the lines on the input image, in the order of Lines; nil when the way back
	// is not known (said once for the run).
	LineQuads []geometry.Quad
	QuadsOK   bool

	// StampsRan is false when the weight set has no stamp detector: Stamps is then nil and the
	// lines carry no stamp.
	StampsRan bool
	Stamps    []Stamp
}

// IsNumberToken is _is_number_token: true if the Latin engine's output is digit-dominated (a
// house, building or flat number) - at least one digit, and digits not outnumbered by letters
// among the alphanumeric characters.
func IsNumberToken(text string) bool {
	alnum, digits := 0, 0
	for _, c := range text {
		if unicode.IsLetter(c) || unicode.IsNumber(c) {
			alnum++
			if unicode.IsDigit(c) {
				digits++
			}
		}
	}
	if alnum == 0 {
		return false
	}
	letters := alnum - digits
	return digits >= 1 && digits >= letters
}

// routeWordOCR is _route_word_ocr: both engines read the word, and the Latin reading is kept
// only when it is digit-dominated, the Cyrillic one otherwise.
func (r *Recognizer) routeWordOCR(word imaging.Image) (string, error) {
	ru, err := r.cyr.Predict(word)
	if err != nil {
		return "", err
	}
	en, err := r.lat.Predict(word)
	if err != nil {
		return "", err
	}
	if IsNumberToken(en) {
		return en, nil
	}
	return ru, nil
}

// readAddress is _address_lines: the oriented line detector on the canvas, each line cut out
// upright, classified printed or handwritten; a printed line is split into words and read, a
// handwritten one is kept as a placeholder so the order of the lines is kept.
// scrub zeroes the line and word crops before they are released (RunOptions.Scrub).
func (r *Recognizer) readAddress(canvas imaging.Image, toInput func([]geometry.Point) ([]geometry.Point, bool),
	scrub bool) (*AddressResult, error) {
	boxes, patches, err := r.addrLines.PredictTransform(canvas)
	if err != nil {
		return nil, err
	}
	defer func() {
		for i := range patches {
			if scrub {
				patches[i].Zero()
			}
			_ = patches[i].Close()
		}
	}()
	res := &AddressResult{Detections: boxes, QuadsOK: true}

	type slot struct {
		handwritten bool
		words       []imaging.Image
		owned       []imaging.Image // the crops of the word detector, closed at the end
	}
	var slots []slot
	defer func() {
		for _, s := range slots {
			for i := range s.owned {
				if scrub {
					s.owned[i].Zero()
				}
				_ = s.owned[i].Close()
			}
		}
	}()

	for i, patch := range patches {
		if patch.Empty() {
			continue
		}
		ob := boxes[i]
		res.LineBoxes = append(res.LineBoxes, ob)

		// The line on the input image, in the order of Lines. "Not known" is said once for the
		// whole run.
		crop := modules.CropGeometry(ob)
		onCanvas, _ := crop.ToInput(geometry.Corners(0, 0, float64(patch.Width()), float64(patch.Height())))
		if quad, ok := toInput(onCanvas); !ok {
			res.QuadsOK = false
		} else if res.QuadsOK {
			res.LineQuads = append(res.LineQuads, toQuad(quad))
		}

		kind, prob, err := r.addrKind.Predict(patch)
		if err != nil {
			return nil, err
		}
		if kind == "handwritten" {
			res.HasHandwritten = true
			res.Lines = append(res.Lines, AddressLine{Kind: "handwritten", PHandwritten: prob})
			slots = append(slots, slot{handwritten: true})
			continue
		}
		// A printed line: split into words, left to right; the reading happens below.
		_, crops, err := r.words.PredictTransform(patch)
		if err != nil {
			return nil, err
		}
		s := slot{owned: crops}
		for _, w := range crops {
			if !w.Empty() {
				s.words = append(s.words, w)
			}
		}
		if len(crops) == 0 {
			s.words = []imaging.Image{patch} // the whole line, borrowed
		}
		slots = append(slots, s)
		res.Lines = append(res.Lines, AddressLine{Kind: "printed", PHandwritten: prob})
	}
	if !res.QuadsOK {
		res.LineQuads = nil
	}

	var lineTexts []string
	for i, s := range slots {
		if s.handwritten {
			lineTexts = append(lineTexts, AddressHandwrittenPlaceholder)
			continue
		}
		var words []string
		for _, w := range s.words {
			t, err := r.routeWordOCR(w)
			if err != nil {
				return nil, err
			}
			if t != "" {
				words = append(words, strings.TrimSpace(t))
			}
		}
		text := strings.Join(words, " ")
		res.Lines[i].Text = &text
		lineTexts = append(lineTexts, text)
	}
	res.Text = strings.Join(lineTexts, "\n")
	return res, nil
}

// addressLinesPayload renders the `address.lines` stage: one object per line, `stamp` only when
// the stamp detector ran.
func addressLinesPayload(a *AddressResult) []any {
	out := make([]any, 0, len(a.Lines))
	for _, l := range a.Lines {
		m := map[string]any{"kind": l.Kind, "p_handwritten": l.PHandwritten, "text": nil}
		if l.Text != nil {
			m["text"] = *l.Text
		}
		if a.StampsRan {
			if l.Stamp != nil {
				m["stamp"] = *l.Stamp
			} else {
				m["stamp"] = nil
			}
		}
		out = append(out, m)
	}
	return out
}

// addressStampsPayload renders the `address.stamps` stage; null when the weight set has no
// stamp detector.
func addressStampsPayload(a *AddressResult) any {
	if !a.StampsRan {
		return nil
	}
	out := make([]any, 0, len(a.Stamps))
	for _, s := range a.Stamps {
		m := map[string]any{
			"kind": s.Kind, "confidence": s.Confidence, "quad": nil,
			"lines": append([]int{}, s.Lines...), "address": s.Address,
			"cancels": nil, "current": s.Current,
		}
		if s.Quad != nil {
			m["quad"] = quadPayload(*s.Quad)
		}
		if s.Cancels != nil {
			m["cancels"] = *s.Cancels
		}
		out = append(out, m)
	}
	return out
}

func quadPayload(q Quad) [][2]float64 {
	out := make([][2]float64, 4)
	for i, p := range q {
		out[i] = [2]float64{p.X, p.Y}
	}
	return out
}

// readStamps is _registration_stamps: find the stamps on the address canvas and tie the address
// lines to them. A line belongs to the stamp its centre lies in (measured on the 19 held-out
// canvases of the issue #27 author: all 113 lines lie inside a stamp); a line inside no stamp
// keeps `stamp: nil`. A deregistered stamp holds no address - it is tied to the registration it
// cancels, and the registration nothing cancels is the current one.
func (r *Recognizer) readStamps(canvas imaging.Image, a *AddressResult,
	toInput func([]geometry.Point) ([]geometry.Point, bool)) error {

	if r.stamps == nil {
		return nil
	}
	a.StampsRan = true
	obboxes, err := r.stamps.Predict(canvas)
	if err != nil {
		return err
	}
	quads := make([][]geometry.Point, len(obboxes))
	for i, ob := range obboxes {
		quads[i] = obboxQuad(ob)
	}
	order := orderStamps(quads, canvas.Width(), canvas.Height())
	position := make(map[int]int, len(order))
	for k, det := range order {
		position[det] = k
	}

	a.Stamps = make([]Stamp, 0, len(order))
	for _, det := range order {
		kind, ok := modules.StampKinds[obboxes[det].Label]
		if !ok {
			kind = "unclear"
		}
		s := Stamp{Kind: kind, Confidence: obboxes[det].Conf, Lines: []int{}}
		if onInput, ok := toInput(quads[det]); ok {
			q := toQuad(onInput)
			s.Quad = &q
		}
		a.Stamps = append(a.Stamps, s)
	}
	for li, ob := range a.LineBoxes {
		if li >= len(a.Lines) {
			break
		}
		det := containingStamp(geometry.Point{X: ob.Cx, Y: ob.Cy}, quads)
		a.Lines[li].Stamp = nil
		if det >= 0 {
			k := position[det]
			a.Lines[li].Stamp = &k
			a.Stamps[k].Lines = append(a.Stamps[k].Lines, li)
		}
	}
	for k := range a.Stamps {
		var parts []string
		for _, li := range a.Stamps[k].Lines {
			if a.Lines[li].Kind == "handwritten" {
				parts = append(parts, AddressHandwrittenPlaceholder)
			} else if a.Lines[li].Text != nil {
				parts = append(parts, *a.Lines[li].Text)
			} else {
				parts = append(parts, "")
			}
		}
		a.Stamps[k].Address = strings.Join(parts, "\n")
	}
	tieCancellations(a.Stamps)
	return nil
}

// orderStamps is _order_stamps: detection indices, freshest stamp first. The page is filled in
// order - on a spread the left page before the right one, on a page top to bottom - and the
// freshest stamp is the last one filled. Two columns of stamps on one page are not covered (x
// only breaks ties).
func orderStamps(quads [][]geometry.Point, width, height int) []int {
	spread := float64(width) > SpreadAspect*float64(height)
	type key struct {
		page   int
		cy, cx float64
	}
	keys := make([]key, len(quads))
	idx := make([]int, len(quads))
	for k, q := range quads {
		// np.mean over four points: a plain left-to-right sum, then the division.
		cx := (((q[0].X + q[1].X) + q[2].X) + q[3].X) / 4
		cy := (((q[0].Y + q[1].Y) + q[2].Y) + q[3].Y) / 4
		page := 0
		if spread && cx >= float64(width)/2 {
			page = 1
		}
		keys[k] = key{page, cy, cx}
		idx[k] = k
	}
	// sorted(...) is stable; the list is then reversed whole, as [::-1] does.
	sort.SliceStable(idx, func(a, b int) bool {
		ka, kb := keys[idx[a]], keys[idx[b]]
		if ka.page != kb.page {
			return ka.page < kb.page
		}
		if ka.cy != kb.cy {
			return ka.cy < kb.cy
		}
		return ka.cx < kb.cx
	})
	for i, j := 0, len(idx)-1; i < j; i, j = i+1, j-1 {
		idx[i], idx[j] = idx[j], idx[i]
	}
	return idx
}

// tieCancellations is _tie_cancellations: set Cancels on every deregistered stamp and Current on
// the registration in force. stamps is freshest first. A deregistered stamp is put right after
// the registration it cancels, so it cancels the nearest registration filled before it - the
// next registered one in this order; one that opens the canvas cancels a registration on a page
// not in the photo and keeps nil. The current registration is the freshest one nothing cancels.
func tieCancellations(stamps []Stamp) {
	cancelled := map[int]bool{}
	for k := range stamps {
		if stamps[k].Kind != "deregistered" {
			continue
		}
		for j := k + 1; j < len(stamps); j++ {
			if stamps[j].Kind == "registered" {
				jj := j
				stamps[k].Cancels = &jj
				cancelled[j] = true
				break
			}
		}
	}
	for k := range stamps {
		if stamps[k].Kind == "registered" && !cancelled[k] {
			stamps[k].Current = true
			break
		}
	}
}

// obboxQuad is _obbox_quad: the stamp's corners on the canvas, clockwise from the top-left
// corner of its upright patch - the same convention as the address-line quads.
func obboxQuad(ob postprocess.ObbBox) []geometry.Point {
	crop := modules.CropGeometry(ob)
	wi := max(1, int(math.RoundToEven(ob.W)))
	hi := max(1, int(math.RoundToEven(ob.H)))
	pts, _ := crop.ToInput(geometry.Corners(0, 0, float64(wi), float64(hi)))
	return pts
}

// containingStamp is _containing_stamp: the index of the stamp whose quad holds the point;
// when quads overlap, the one whose centre is nearest. -1 when no quad holds it.
func containingStamp(p geometry.Point, quads [][]geometry.Point) int {
	best, bestD := -1, 0.0
	for k, q := range quads {
		allNonNeg, allNonPos := true, true
		for i := 0; i < 4; i++ {
			next := q[(i+1)%4]
			ex, ey := next.X-q[i].X, next.Y-q[i].Y
			rx, ry := p.X-q[i].X, p.Y-q[i].Y
			cross := ex*ry - ey*rx
			if !(cross >= 0) {
				allNonNeg = false
			}
			if !(cross <= 0) {
				allNonPos = false
			}
		}
		if !allNonNeg && !allNonPos {
			continue
		}
		cx := (((q[0].X + q[1].X) + q[2].X) + q[3].X) / 4
		cy := (((q[0].Y + q[1].Y) + q[2].Y) + q[3].Y) / 4
		d := math.Hypot(p.X-cx, p.Y-cy)
		if best < 0 || d < bestD {
			best, bestD = k, d
		}
	}
	return best
}

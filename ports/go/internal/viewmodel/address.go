package viewmodel

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/pipeline"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// The address block: present only for INTPASSPORTADDR, null for every other type. Port of
// transform.py::_build_address.

// Address is present only for INTPASSPORTADDR.
type Address struct {
	// Aligned is false when the geometry and text lists desynchronised. A consumer must
	// then suppress the overlay rather than caption boxes with the wrong text.
	Aligned bool          `json:"aligned"`
	Lines   []AddressLine `json:"lines"`
}

type AddressLine struct {
	ID           string   `json:"id"`
	Kind         *string  `json:"kind"`
	Text         *string  `json:"text"`
	PHandwritten *float64 `json:"p_handwritten"`
	Obbox        *Obbox   `json:"obbox"`
}

type Obbox struct {
	Cx *float64 `json:"cx"`
	Cy *float64 `json:"cy"`
	W  *float64 `json:"w"`
	H  *float64 `json:"h"`
	// AngleRad is RADIANS at six decimals, not the wire default of four.
	AngleRad *float64 `json:"angle_rad"`
	Conf     *float64 `json:"conf"`
	Label    *string  `json:"label"`
}

// AddressLineInput is one entry of the library's Address_lines: the verdict on one line.
type AddressLineInput struct {
	Kind         string
	PHandwritten float64
	// Text is nil for a handwritten line.
	Text *string
}

// buildAddress is _build_address: the oriented boxes of the line detector with the
// printed-or-handwritten verdicts.
//
// The two source lists can desynchronise: the library skips a line whose patch came out empty
// without recording a slot, so box i and line i stop lining up. Zipping them blindly would
// caption boxes with the wrong text, so when the lengths disagree `aligned` is false, the
// geometry is dropped and only the text kept - a client that respects the flag then suppresses
// the overlay rather than draw a confident lie. Nil when there is neither a box nor a line.
func buildAddress(obboxes []postprocess.ObbBox, lines []AddressLineInput) *Address {
	if len(obboxes) == 0 && len(lines) == 0 {
		return nil
	}
	aligned := len(obboxes) == len(lines)
	out := make([]AddressLine, 0, len(lines))
	for i, l := range lines {
		entry := AddressLine{
			ID:           fmt.Sprintf("o%d", i),
			Kind:         str(l.Kind),
			Text:         l.Text,
			PHandwritten: num(l.PHandwritten),
		}
		if aligned {
			row := obboxes[i]
			entry.Obbox = &Obbox{
				Cx: num(row.Cx), Cy: num(row.Cy), W: num(row.W), H: num(row.H),
				AngleRad: numAt(row.Angle, anglePrecision),
				Conf:     num(row.Conf),
				Label:    str(row.Label),
			}
		}
		out = append(out, entry)
	}
	return &Address{Aligned: aligned, Lines: out}
}

// AddressFrom turns what the pipeline's address path produced into the view-model input: all the
// boxes of the line detector and the verdict on each kept line. Nothing for another document type.
func AddressFrom(a *pipeline.AddressResult) ([]postprocess.ObbBox, []AddressLineInput) {
	if a == nil {
		return nil, nil
	}
	lines := make([]AddressLineInput, 0, len(a.Lines))
	for _, l := range a.Lines {
		lines = append(lines, AddressLineInput{Kind: l.Kind, PHandwritten: l.PHandwritten, Text: l.Text})
	}
	return a.Detections, lines
}

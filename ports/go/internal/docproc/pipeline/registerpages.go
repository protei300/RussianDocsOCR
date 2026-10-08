package pipeline

import (
	"math"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
)

// Port of the internal-passport page pipeline in pipeline.py: Pipeline._register_pages
// (template-registered canvas rebuild) and Pipeline._fields_from_pages /
// Pipeline._deskew's per-page branch (postraničnaya podača - detect/deskew each page of
// the spread separately, since the field detector's 640x640 input gives a stitched
// spread only half of itself per page). This file is the ORCHESTRATION layer; the
// matching/geometry engine lives in modules.PageRegistrar.

// Thresholds from pipeline.py:25-26 (module-level constants there, not on
// PageRegistrar - they decide which geometry SOURCE to trust, which is a pipeline
// policy, not a registration-engine one).
const (
	quadSamePageIOU = 0.40
	quadClippedIOU  = 0.80
)

// RegisteredPages is what registerPages produces: the rebuilt canvas plus the pages it
// was stitched from (for the per-page field detector) - or ok=false when nothing could
// be registered at all, in which case the caller keeps the plain Borders canvas.
type RegisteredPages struct {
	Canvas     imaging.Image
	Pages      []imaging.Image
	PageQuads  [][]imaging.Point
	Placements []imaging.PagePlacement
	// Geometry is the map from Canvas back to the image the registrar received (geometry.py), and
	// PageGeometries the per-page maps (page -> that image): REPLACES the border stage's, as the
	// reference'"'"'s _register_pages does.
	Geometry       geometry.Geometry
	PageGeometries []geometry.Geometry
}

// registerPages is Pipeline._register_pages. img is the upright photo (same frame
// DocDetector ran on); segments are the Borders contours DocDetector selected.
func (r *Recognizer) registerPages(img imaging.Image, segments [][]imaging.Point) (RegisteredPages, bool) {
	reg := r.pageRegistrar
	quads, _ := reg.PageQuads(segments, img.Height(), img.Width())
	regs := reg.Register(img, quads)
	scale := reg.NativeScale(regs)

	type step struct {
		kind    string // "none" | "quad" | "template"
		quadIdx int
		reg     modules.PageRegistration
	}
	used := make(map[int]bool)
	plan := make([]step, len(regs))
	for i, rr := range regs {
		if !rr.OK() {
			plan[i] = step{kind: "none"}
			continue
		}
		bestI, bestIoU := -1, 0.0
		for qi, q := range quads {
			if used[qi] {
				continue
			}
			if iou := modules.QuadIoU(q, rr.Quad); iou > bestIoU {
				bestI, bestIoU = qi, iou
			}
		}
		if bestI >= 0 && bestIoU >= quadSamePageIOU {
			used[bestI] = true
			clipped := quadTouchesFrame(quads[bestI], img.Width(), img.Height())
			if clipped && bestIoU < quadClippedIOU {
				plan[i] = step{kind: "template", reg: rr}
			} else {
				plan[i] = step{kind: "quad", quadIdx: bestI}
			}
		} else {
			plan[i] = step{kind: "template", reg: rr}
		}
	}

	var spare []int
	for qi := range quads {
		if !used[qi] {
			spare = append(spare, qi)
		}
	}
	sort.SliceStable(spare, func(a, b int) bool {
		return quadMinY(quads[spare[a]]) < quadMinY(quads[spare[b]])
	})

	var pages []imaging.Image
	var pageQuads [][]imaging.Point
	var pageGeos []geometry.Geometry
	for i := range regs {
		st := plan[i]
		var page imaging.Image
		var quadForPage []imaging.Point
		var M [3][3]float64
		have := true
		switch st.kind {
		case "none":
			if len(spare) == 0 {
				have = false
				break
			}
			qi := spare[0]
			spare = spare[1:]
			expanded := imaging.ExpandQuadF32(quads[qi], imaging.DocMarginFrac)
			page, M = reg.WarpQuadMatrix(img, expanded, scale)
			quadForPage = quads[qi]
		case "quad":
			expanded := imaging.ExpandQuadF32(quads[st.quadIdx], imaging.DocMarginFrac)
			page, M = reg.WarpQuadMatrix(img, expanded, scale)
			quadForPage = quads[st.quadIdx]
		default: // "template"
			page = reg.WarpPage(img, st.reg, scale)
			M = reg.PageMatrix(st.reg, scale)
			quadForPage = st.reg.Quad
		}
		if !have {
			continue
		}
		straightened, _, straight := reg.StraightenWithGeometry(page, scale)
		pages = append(pages, straightened)
		pageQuads = append(pageQuads, quadForPage)
		// page -> the image the registrar received (geometry.py)
		pageGeos = append(pageGeos, geometry.Chain{Maps: []geometry.Geometry{geometry.NewHomography(M)}}.Then(straight))
	}

	if len(pages) == 0 {
		return RegisteredPages{}, false
	}
	if len(pages) < 2 {
		// One page: StitchPages returns it AS the canvas (no copy) - nothing
		// downstream needs Pages for fewer than 2 pages, so hand it the original.
		w, h := pages[0].Width(), pages[0].Height()
		canvas, placements, ok := imaging.StitchPages(pages, pageQuads, imaging.StackVertical)
		if !ok {
			return RegisteredPages{}, false
		}
		geo := geometry.StitchedGeometry([]geometry.PlacedPage{{W: w, H: h, Scale: placements[0].Scale,
			DX: placements[0].DX, DY: placements[0].DY, Geo: pageGeos[0]}})
		return RegisteredPages{Canvas: canvas, Geometry: geo, PageGeometries: pageGeos}, true
	}
	// 2+ pages: StitchPages CONSUMES (Closes) every page it is handed, but
	// fieldsFromPages needs the (already straightened) pages to survive past this
	// call - see DocDetector.PredictTransform's identical fix for why skipping this
	// clone is a double-free, not a leak.
	stitchInput := make([]imaging.Image, len(pages))
	placed := make([]geometry.PlacedPage, len(pages))
	for i, p := range pages {
		stitchInput[i] = p.Clone()
		placed[i] = geometry.PlacedPage{W: p.Width(), H: p.Height(), Geo: pageGeos[i]}
	}
	canvas, placements, ok := imaging.StitchPages(stitchInput, pageQuads, imaging.StackVertical)
	if !ok {
		for _, p := range pages {
			_ = p.Close()
		}
		return RegisteredPages{}, false
	}
	for i := range placed {
		placed[i].Scale, placed[i].DX, placed[i].DY = placements[i].Scale, placements[i].DX, placements[i].DY
	}
	return RegisteredPages{Canvas: canvas, Pages: pages, PageQuads: pageQuads, Placements: placements,
		Geometry: geometry.StitchedGeometry(placed), PageGeometries: pageGeos}, true
}

// quadTouchesFrame is _register_pages' `clipped` check: any corner within 1px of the
// left/top edge or within 2px of the right/bottom edge.
func quadTouchesFrame(q []imaging.Point, w, h int) bool {
	for _, p := range q {
		if p.X <= 1 || p.Y <= 1 || p.X >= float64(w)-2 || p.Y >= float64(h)-2 {
			return true
		}
	}
	return false
}

func quadMinY(q []imaging.Point) float64 {
	m := q[0].Y
	for _, p := range q[1:] {
		if p.Y < m {
			m = p.Y
		}
	}
	return m
}

// fieldsFromPages is Pipeline._fields_from_pages: the text-field detector runs on each
// page separately, and its boxes are moved onto the stitched canvas by that page's
// (scale, dx, dy) placement. rotateLicence matches PredictTransform's own flag - the
// reference rotates the Licence_number PATCH only after the fact, but that is
// per-field and order-independent, so doing it inside each page's own detection call
// is equivalent (see modules.TextFieldsDetector.PredictTransform).
func (r *Recognizer) fieldsFromPages(pages []imaging.Image, placements []imaging.PagePlacement,
	rotateLicence bool) ([]modules.Field, []fieldFrame, error) {

	var all []modules.Field
	var frames []fieldFrame
	for i, page := range pages {
		fields, err := r.fields.PredictTransform(page, rotateLicence)
		if err != nil {
			modules.FieldsClose(all)
			return nil, nil, err
		}
		pl := placements[i]
		// The frame is written as the stages that make the PATCH out of the CANVAS (geometry.go
		// reads a chain that way round): take the page off its place on the canvas, undo its resize,
		// cut at the box.
		placedW, placedH := geometry.PlacedSize(page.Width(), page.Height(), pl.Scale)
		unplace := []geometry.Geometry{
			geometry.Offset{DX: -pl.DX, DY: -pl.DY},
			geometry.Scale{SX: float64(page.Width()) / float64(placedW), SY: float64(page.Height()) / float64(placedH)},
		}
		for _, f := range fields {
			frame := singleCanvasFrame(f, rotateLicence && f.Box.Label == "Licence_number")
			frame.Map = geometry.Chain{Maps: append(append([]geometry.Geometry(nil), unplace...), frame.Map)}
			frames = append(frames, frame)
			f.Box.X1 = math.RoundToEven(f.Box.X1*pl.Scale + pl.DX)
			f.Box.Y1 = math.RoundToEven(f.Box.Y1*pl.Scale + pl.DY)
			f.Box.X2 = math.RoundToEven(f.Box.X2*pl.Scale + pl.DX)
			f.Box.Y2 = math.RoundToEven(f.Box.Y2*pl.Scale + pl.DY)
			all = append(all, f)
		}
	}
	return all, frames, nil
}

// deskewedPages is what deskewPages produces besides the canvas.
type deskewedPages struct {
	Canvas     imaging.Image
	Pages      []imaging.Image
	Placements []imaging.PagePlacement
	// Geometry is the stitched canvas -> the image the border stage received; PageGeometries the
	// per-page maps: each page's own map followed by its own turn (geometry.py).
	Geometry       geometry.Geometry
	PageGeometries []geometry.Geometry
}

// deskewPages is the `pages` branch of Pipeline._deskew: each page of a (non
// template-registered) spread is deskewed on its own and the canvas is re-stitched
// from the deskewed pages - the projection-profile deskew works on TEXT LINES, and a
// spread's two pages routinely sit at different angles. pageGeos are the maps of the pages
// before the turn (DocDetector's), or nil entries.
func (r *Recognizer) deskewPages(pages []imaging.Image, quads [][]imaging.Point,
	pageGeos []geometry.Geometry) (deskewedPages, bool) {

	desk := make([]imaging.Image, len(pages))
	turned := make([]geometry.Geometry, len(pages))
	for i, p := range pages {
		d, _, turn, err := r.deskewer.DeskewWithGeometry(p)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = desk[j].Close()
			}
			return deskewedPages{}, false
		}
		desk[i] = d
		var own geometry.Chain
		if i < len(pageGeos) && pageGeos[i] != nil {
			own = geometry.Chain{Maps: []geometry.Geometry{pageGeos[i]}}
		}
		turned[i] = own.Then(turn)
	}
	// StitchPages CONSUMES (Closes) every page it is handed once there are 2+ - which
	// this always is here (the caller only calls deskewPages for a spread) - but the
	// caller needs `desk` (the per-page deskewed images) to survive for the field
	// detector, so StitchPages gets CLONES. See DocDetector.PredictTransform's
	// identical fix for why skipping this is a double-free.
	stitchInput := make([]imaging.Image, len(desk))
	for i, d := range desk {
		stitchInput[i] = d.Clone()
	}
	canvas, placements, ok := imaging.StitchPages(stitchInput, quads, imaging.StackAuto)
	if !ok {
		for _, d := range desk {
			_ = d.Close()
		}
		return deskewedPages{}, false
	}
	placed := make([]geometry.PlacedPage, len(desk))
	for i, d := range desk {
		placed[i] = geometry.PlacedPage{W: d.Width(), H: d.Height(), Scale: placements[i].Scale,
			DX: placements[i].DX, DY: placements[i].DY, Geo: turned[i]}
	}
	return deskewedPages{Canvas: canvas, Pages: desk, Placements: placements,
		Geometry: geometry.StitchedGeometry(placed), PageGeometries: turned}, true
}

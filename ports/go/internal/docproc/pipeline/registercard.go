package pipeline

import (
	"errors"
	"io/fs"
	"math"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
)

// Port of Pipeline._register_card and its helpers: a vehicle registration certificate
// (STS_*, STSBACK_*) straightened by its printed blank (4d8c2535).

// CardMinInliers is Pipeline.CARD_MIN_INLIERS: the template matches a vehicle registration
// certificate needs before its template geometry replaces the Borders quad. The card is
// printed dense (captions, rules, guilloche), so a real match runs to a hundred or more
// points (median 141 on the client's test cards, 2026-10-02); 40 is the floor the form
// matching of the training-data side uses, below it the Borders canvas stays.
const CardMinInliers = 40

// CardSkewKeep is Pipeline.CARD_SKEW_KEEP: the skew of the Borders canvas below which it is
// kept (see cardSkew): the card's corners found by the template, carried into that canvas,
// are fitted by a similarity; the residual as a share of the long side. The same measure the
// training-data side used on the client's canvases; visible skew starts at about 1.4 %
// (handoff of 2026-10-02).
const CardSkewKeep = 0.01

// isCardType is `doc_type.startswith('sts')` in Pipeline._register_pages, on the full
// label ('STS_1996', 'STSBACK_2019').
func isCardType(label string) bool {
	return strings.HasPrefix(strings.ToLower(label), "sts")
}

// cardRegistrar is Pipeline._card_registrar: the template registrar of one STS type, built
// on first use; nil when the type has no templates. Built lazily because only an STS ever
// needs it and each build runs SIFT over every reference of the type; guarded because a
// Recognizer may be called concurrently (the registrar itself, like the passport one, is
// used the way its OpenCV objects allow).
func (r *Recognizer) cardRegistrar(label string) (*modules.PageRegistrar, error) {
	r.cardMu.Lock()
	defer r.cardMu.Unlock()
	if r.cardRegs == nil {
		r.cardRegs = map[string]*modules.PageRegistrar{}
	}
	if reg, seen := r.cardRegs[label]; seen {
		return reg, nil
	}
	reg, err := modules.NewPageRegistrar(r.opts.Root, label, 6000)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			r.cardRegs[label] = nil
			return nil, nil
		}
		return nil, err
	}
	reg.RefitRounds = CardRefitRounds
	r.cardRegs[label] = reg
	return reg, nil
}

// cardSkew is Pipeline._card_skew: how far the Borders canvas bends the card out of a
// rectangle, as a share of the card's long side.
//
// The card's corners found by the template (cardQuad, on the photo) are carried into the
// canvas the Borders quad would give; a similarity (rotation, uniform scale, shift) is fitted
// to them by least median of squares against the ideal rectangle, and the RMS residual is
// the skew. 1.0 when no similarity is found.
//
// float32 where the reference is: the carried corners and the ideal rectangle are float32
// arrays, the fit and the residual are float64 (a float32 array @ a float64 matrix).
func cardSkew(reg *modules.PageRegistrar, bordersQuad, cardQuad []imaging.Point) float64 {
	M, ok := reg.QuadMatrix(bordersQuad, 1.0)
	if !ok {
		return 1.0
	}
	found := imaging.PerspectiveTransformF32(M, cardQuad)
	if len(found) != 4 {
		return 1.0
	}
	m := float64(reg.Margin())
	w, h := float64(reg.PageW()), float64(reg.PageH())
	ideal := []imaging.Point{{X: m, Y: m}, {X: m + w, Y: m}, {X: m + w, Y: m + h}, {X: m, Y: m + h}}
	A, ok := imaging.EstimateAffinePartialLMEDS(ideal, found)
	if !ok {
		return 1.0
	}
	var sum float64
	for i, p := range ideal {
		fx := p.X*A[0][0] + p.Y*A[0][1] + A[0][2]
		fy := p.X*A[1][0] + p.Y*A[1][1] + A[1][2]
		dx, dy := fx-found[i].X, fy-found[i].Y
		sum += dx*dx + dy*dy
	}
	return math.Sqrt(sum/4) / math.Max(w, h)
}

// registerCard is Pipeline._register_card: rebuild a vehicle registration certificate's
// canvas from its printed blank. ok is false when the Borders canvas stays as it is.
//
// The opposite policy to the passport's. There the Borders quad keeps the geometry whenever
// it agrees with the template, because the template fit of a sparsely printed page turns by
// degrees. Here the card often lies in a plastic sleeve or lamination and the Borders quad
// is the SLEEVE's edge: it overlaps the card well - it "agrees" - and still skews the canvas
// (on the client's 1217 cards ~12 % came out visibly skewed, 2026-10-02). The card is printed
// dense, so the template match is strong: when it reaches CardMinInliers the template
// geometry is taken and the page straightened by its own lines; otherwise the Borders canvas
// stays as it is.
//
// Where the card runs past the photo, the canvas is painted the card's own paper colour
// rather than the smeared edge (see PageRegistrar.WarpMatrix's fill).
//
// The returned canvas is the straightened page itself (stitch_pages of one page returns it);
// no per-page list is kept, since only a spread is ever read page by page.
func (r *Recognizer) registerCard(img imaging.Image, label string,
	segments [][]imaging.Point) (canvas imaging.Image, geo geometry.Geometry, ok bool, err error) {

	reg, err := r.cardRegistrar(label)
	if err != nil || reg == nil {
		return imaging.Image{}, nil, false, err
	}
	quads, _ := reg.PageQuads(segments, img.Height(), img.Width())
	rr := reg.Register(img, quads)[0]
	if !rr.OK() || rr.Inliers < CardMinInliers {
		return imaging.Image{}, nil, false, nil
	}
	// Where the Borders canvas is not skewed, keep it: re-cutting a canvas that was right
	// only resamples it (measured on the client's test cards: on the 101 not skewed by
	// Borders, 15 fields read better and 17 worse - noise; on the 26 skewed by >= 1 %, 5
	// better, 1 worse). Skew, not offset: a sleeve runs parallel to the card, so its edge
	// sits 2-3 % off the card's even on a straight canvas; what matters is whether the card
	// comes out a rectangle.
	if len(quads) > 0 {
		least := math.Inf(1)
		for _, q := range quads {
			least = math.Min(least, cardSkew(reg, q, rr.Quad))
		}
		if least < CardSkewKeep {
			return imaging.Image{}, nil, false, nil
		}
	}
	scale := reg.NativeScale([]modules.PageRegistration{rr})
	M := reg.PageMatrix(rr, scale)
	var fill *[3]uint8
	if colour, has := imaging.PolygonMedianColour(img, imaging.RoundPolygon(rr.Quad)); has {
		fill = &colour
	}
	page := reg.WarpMatrix(img, M, scale, fill)
	straightened, _, straight := reg.StraightenWithGeometry(page, scale)
	// photo -> page, then the straightening: det['"'"'geometry'"'"'] = stitched_geometry([page], placements,
	// [page_geometry]) with the one placement stitch_pages gives a single page (1, 0, 0)
	pageGeo := geometry.Chain{Maps: []geometry.Geometry{geometry.NewHomography(M)}}.Then(straight)
	geo = geometry.StitchedGeometry([]geometry.PlacedPage{{W: straightened.Width(), H: straightened.Height(),
		Scale: 1.0, Geo: pageGeo}})
	return straightened, geo, true, nil
}

// CardRefitRounds is Pipeline.CARD_REFIT_ROUNDS: least-squares re-fits after MAGSAC in the card's
// template match (PageRegistrar.RefitRounds): the skew decision (CardSkewKeep) must not move with
// MAGSAC's samples. The passport path keeps 0 - there the template only locates the page, and the
// re-fit cost 5 of 100 exact fields on the 1997 passports (measured by the reference, 2026-10-08).
const CardRefitRounds = 5

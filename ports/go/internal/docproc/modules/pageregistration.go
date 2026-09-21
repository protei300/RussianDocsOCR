package modules

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Port of pipeline_modules/page_registration/page_registration.py: aligns the photo
// of a booklet page (internal passport page 2 or 3) against a canonical, person-free
// template of the printed blank, via SIFT + MAGSAC homography, refined in the
// template's own frame. See that file's module docstring for the four-step method;
// this file mirrors PageTemplate and PageRegistrar directly, constant for constant.
//
// Two things are NOT ported because the pipeline never exercises them at the
// defaults this port targets (PageRegistrar() with use_ecc=False): the dense ECC
// polish (_ecc, _gradient) and everything that feeds only it. A future caller that
// wants use_ecc=True would need to add it; nothing here silently pretends to.
const (
	minCoarseInliers      = 8
	minRefinedInliers     = 18
	minRefinedInliersChain = 10
	minSpreadPxChain      = 140.0
	minSpreadPx           = 120.0
	ratioTest             = 0.8
	coarseReprojPx        = 4.0
	chainGapFrac          = 0.03
	quadDilateFrac        = 0.10
	maxCoarseCandidates   = 2
	goodCoarseInliers     = 20
	pageMarginFrac        = 0.03
)

var refineReprojPx = [2]float64{3.0, 2.0}
var refineRadiusPx = [2]float64{45.0, 15.0}
var chainRadiusPx = [2]float64{90.0, 15.0}

// pageTemplate is one reference image of a canonical page (PageTemplate in Python).
type pageTemplate struct {
	name          string
	gray          imaging.Image
	width, height int
	kp            []imaging.Point
	desc          [][]float32
	corners       [4]imaging.Point // {0,0},{w,0},{w,h},{0,h}
}

func (t *pageTemplate) quadInImage(H [3][3]float64) []imaging.Point {
	Hinv, ok := imaging.InvertH(H)
	if !ok {
		return nil
	}
	out := make([]imaging.Point, 4)
	for i, c := range t.corners {
		out[i] = imaging.TransformPoint(Hinv, c)
	}
	return out
}

// PageRegistration is the result for one canonical page (PageRegistration in Python).
// H maps photo pixels to canonical page pixels; nil means the page could not be
// registered.
type PageRegistration struct {
	Name    string
	H       *[3][3]float64
	Inliers int
	Method  string
	Quad    []imaging.Point
	Ref     int
}

func (r PageRegistration) OK() bool { return r.H != nil }

type registrarPage struct {
	name string
	refs []*pageTemplate
}

// PageRegistrar registers the pages of one document type against its templates.
// Port of the PageRegistrar class, defaults matching PageRegistrar() in pipeline.py:
// nfeatures=6000, use_ecc=False, line_quads=True, line_refine=True, line_dewarp=True.
type PageRegistrar struct {
	sift    *imaging.SiftDetector
	matcher *imaging.L2Matcher

	pages []registrarPage
	pageW, pageH int
	margin       int
	outW, outH   int

	LineQuads  bool
	LineRefine bool
	LineDewarp bool
}

type templateMeta struct {
	DocType string `json:"doc_type"`
	Pages   []struct {
		Name string `json:"name"`
		Refs []struct {
			Image string `json:"image"`
			Mask  string `json:"mask"`
		} `json:"refs"`
	} `json:"pages"`
}

// NewPageRegistrar loads templates/<docType>.json (lowercase) from
// document_processing/pipeline_modules/page_registration/templates under root, and
// builds SIFT features for every reference page.
func NewPageRegistrar(root, docType string, nfeatures int) (*PageRegistrar, error) {
	dir := filepath.Join(root, "document_processing", "pipeline_modules", "page_registration", "templates")
	metaPath := filepath.Join(dir, strings.ToLower(docType)+".json")
	raw, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, fmt.Errorf("modules: PageRegistrar: %w", err)
	}
	var meta templateMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, fmt.Errorf("modules: PageRegistrar: %s: %w", metaPath, err)
	}
	if len(meta.Pages) == 0 {
		return nil, fmt.Errorf("modules: PageRegistrar: %s has no pages", metaPath)
	}

	reg := &PageRegistrar{
		sift:       imaging.NewSiftDetector(nfeatures),
		matcher:    imaging.NewL2Matcher(),
		LineQuads:  true,
		LineRefine: true,
		LineDewarp: true,
	}
	for _, p := range meta.Pages {
		var refs []*pageTemplate
		for _, r := range p.Refs {
			tpl, err := loadPageTemplate(p.Name, filepath.Join(dir, r.Image), filepath.Join(dir, r.Mask), reg.sift)
			if err != nil {
				reg.Close()
				return nil, err
			}
			refs = append(refs, tpl)
		}
		reg.pages = append(reg.pages, registrarPage{name: p.Name, refs: refs})
	}
	first := reg.pages[0].refs[0]
	reg.pageW, reg.pageH = first.width, first.height
	reg.margin = int(math.Round(pageMarginFrac * float64(reg.pageW)))
	reg.outW = reg.pageW + 2*reg.margin
	reg.outH = reg.pageH + 2*reg.margin
	return reg, nil
}

func loadPageTemplate(name, imagePath, maskPath string, sift *imaging.SiftDetector) (*pageTemplate, error) {
	img, err := imaging.LoadRGB(imagePath)
	if err != nil {
		return nil, fmt.Errorf("modules: page template %s: %w", imagePath, err)
	}
	defer img.Close()
	gray := imaging.ToGray(img)

	maskImg, err := imaging.LoadRGB(maskPath)
	if err != nil {
		gray.Close()
		return nil, fmt.Errorf("modules: page template mask %s: %w", maskPath, err)
	}
	maskGray := imaging.ToGray(maskImg)
	maskImg.Close()
	defer maskGray.Close()
	maskBin := imaging.ThresholdFixed(maskGray, 127)

	kp, desc := sift.DetectAndComputeMasked(gray, maskBin)
	maskBin.Close()

	w, h := gray.Width(), gray.Height()
	return &pageTemplate{
		name: name, gray: gray, width: w, height: h, kp: kp, desc: desc,
		corners: [4]imaging.Point{{X: 0, Y: 0}, {X: float64(w), Y: 0},
			{X: float64(w), Y: float64(h)}, {X: 0, Y: float64(h)}},
	}, nil
}

func (r *PageRegistrar) Close() error {
	for _, p := range r.pages {
		for _, t := range p.refs {
			_ = t.gray.Close()
		}
	}
	if r.sift != nil {
		_ = r.sift.Close()
	}
	if r.matcher != nil {
		_ = r.matcher.Close()
	}
	return nil
}

func (r *PageRegistrar) PageNames() []string {
	out := make([]string, len(r.pages))
	for i, p := range r.pages {
		out[i] = p.name
	}
	return out
}

func (r *PageRegistrar) PageW() int { return r.pageW }
func (r *PageRegistrar) PageH() int { return r.pageH }

// ------------------------------------------------------------------ matching

// matchResult is _match's return, minus the ecc placeholder (always 0, use_ecc=False).
type matchResult struct {
	H       [3][3]float64
	Inliers int
	Pts     []imaging.Point // inlier TEMPLATE points
	OK      bool
}

func median(vals []float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return 0.5 * (s[n/2-1] + s[n/2])
}

// match is PageRegistrar._match: template -> photo/page matches, MAGSAC homography.
// kp/desc are the SEARCH SPACE (photo keypoints, or a page-frame re-detection); prior
// (with hasPrior) and radius (with hasRadius) enable guided matching.
func (r *PageRegistrar) match(tpl *pageTemplate, kp []imaging.Point, desc [][]float32,
	reproj float64, radius float64, hasRadius bool, prior [3][3]float64, hasPrior bool) matchResult {

	if len(desc) < 8 {
		return matchResult{Inliers: len(desc)}
	}
	knn := r.matcher.KnnMatch2(tpl.desc, desc) // query=template, train=photo/page
	var srcPhoto, dstTpl []imaging.Point
	for i, ms := range knn {
		if len(ms) != 2 {
			continue
		}
		if ms[0].Distance < ratioTest*ms[1].Distance {
			srcPhoto = append(srcPhoto, kp[ms[0].TrainIdx])
			dstTpl = append(dstTpl, tpl.pts()[i])
		}
	}
	if len(srcPhoto) < 8 {
		return matchResult{Inliers: len(srcPhoto)}
	}
	if hasPrior && hasRadius {
		pred := imaging.TransformPoints(prior, srcPhoto)
		disp := make([]imaging.Point, len(pred))
		for i := range pred {
			disp[i] = imaging.Point{X: pred[i].X - dstTpl[i].X, Y: pred[i].Y - dstTpl[i].Y}
		}
		var roughX, roughY []float64
		for _, d := range disp {
			if math.Hypot(d.X, d.Y) < 2.5*radius {
				roughX = append(roughX, d.X)
				roughY = append(roughY, d.Y)
			}
		}
		if len(roughX) >= 6 {
			mx, my := median(roughX), median(roughY)
			for i := range disp {
				disp[i].X -= mx
				disp[i].Y -= my
			}
		}
		var keptSrc, keptDst []imaging.Point
		for i, d := range disp {
			if math.Hypot(d.X, d.Y) < radius {
				keptSrc = append(keptSrc, srcPhoto[i])
				keptDst = append(keptDst, dstTpl[i])
			}
		}
		srcPhoto, dstTpl = keptSrc, keptDst
		if len(srcPhoto) < 8 {
			return matchResult{Inliers: len(srcPhoto)}
		}
	}
	H, mask, ok := imaging.FindHomographyMagsac(srcPhoto, dstTpl, reproj, 10000, 0.999)
	if !ok {
		return matchResult{}
	}
	var inl []imaging.Point
	cnt := 0
	for i, keep := range mask {
		if keep && i < len(dstTpl) {
			cnt++
			inl = append(inl, dstTpl[i])
		}
	}
	return matchResult{H: H, Inliers: cnt, Pts: inl, OK: true}
}

// tpl.pts() as a method-like accessor (template keypoints as plain points).
func (t *pageTemplate) pts() []imaging.Point { return t.kp }

// featuresInQuad restricts a keypoint/descriptor set to those inside quad grown by
// QUAD_DILATE_FRAC, matching _features_in_quad. Rounds each point the way the
// reference does (np.round then clip to the image) before the inside test.
func featuresInQuad(kp []imaging.Point, desc [][]float32, quad []imaging.Point, w, h int) ([]imaging.Point, [][]float32) {
	cx, cy := 0.0, 0.0
	for _, p := range quad {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(len(quad))
	cy /= float64(len(quad))
	grown := make([]imaging.Point, len(quad))
	for i, p := range quad {
		grown[i] = imaging.Point{X: cx + (p.X-cx)*(1+2*quadDilateFrac), Y: cy + (p.Y-cy)*(1+2*quadDilateFrac)}
	}
	var outKP []imaging.Point
	var outDesc [][]float32
	for i, p := range kp {
		px := clampF(math.Round(p.X), 0, float64(w-1))
		py := clampF(math.Round(p.Y), 0, float64(h-1))
		if pointInConvex(imaging.Point{X: px, Y: py}, grown) {
			outKP = append(outKP, p)
			outDesc = append(outDesc, desc[i])
		}
	}
	return outKP, outDesc
}

func pointInConvex(p imaging.Point, quad []imaging.Point) bool {
	n := len(quad)
	sign := 0.0
	for i := 0; i < n; i++ {
		a, b := quad[i], quad[(i+1)%n]
		cross := (b.X-a.X)*(p.Y-a.Y) - (b.Y-a.Y)*(p.X-a.X)
		if cross == 0 {
			continue
		}
		s := 1.0
		if cross < 0 {
			s = -1.0
		}
		if sign == 0 {
			sign = s
		} else if s != sign {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------- refinement

func spreadOK(pts []imaging.Point, minPx float64) bool {
	if len(pts) < 4 {
		return false
	}
	minX, maxX := pts[0].X, pts[0].X
	minY, maxY := pts[0].Y, pts[0].Y
	for _, p := range pts[1:] {
		minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
		minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
	}
	return maxX-minX >= minPx && maxY-minY >= minPx
}

// refine is PageRegistrar._refine (ECC branch dropped: use_ecc is always false here).
func (r *PageRegistrar) refine(gray imaging.Image, tpl *pageTemplate, H [3][3]float64,
	radii [2]float64, minInliers int) (out [3][3]float64, inliers int, pts []imaging.Point, ok bool) {

	page := imaging.WarpByHomography(gray, H, tpl.width, tpl.height, true)
	defer page.Close()
	kp, desc := r.sift.DetectAndCompute(page)

	prior := imaging.Identity3()
	for i := 0; i < 2; i++ {
		m := r.match(tpl, kp, desc, refineReprojPx[i], radii[i], true, prior, true)
		if !m.OK || m.Inliers < minInliers {
			return H, 0, nil, false
		}
		prior, inliers, pts = m.H, m.Inliers, m.Pts
	}
	return imaging.MulH(prior, H), inliers, pts, true
}

// ---------------------------------------------------------------- validation

func (r *PageRegistrar) saneQuad(quad []imaging.Point, imgW, imgH int, refArea float64, hasRefArea bool) bool {
	if quad == nil {
		return false
	}
	for _, p := range quad {
		if math.IsNaN(p.X) || math.IsInf(p.X, 0) || math.IsNaN(p.Y) || math.IsInf(p.Y, 0) {
			return false
		}
	}
	if !isConvex(quad) {
		return false
	}
	area := contourArea(quad)
	hw := float64(imgH) * float64(imgW)
	if area < 0.02*hw || area > 1.5*hw {
		return false
	}
	q := imaging.OrderPoints(quad)
	wt := 0.5 * (dist(q[1], q[0]) + dist(q[2], q[3]))
	ht := 0.5 * (dist(q[3], q[0]) + dist(q[2], q[1]))
	if ht < 1e-3 {
		return false
	}
	ratio := (wt / ht) / (float64(r.pageW) / float64(r.pageH))
	if !(ratio > 0.6 && ratio < 1.7) {
		return false
	}
	if hasRefArea && refArea > 0 {
		f := area / refArea
		if !(f > 0.4 && f < 2.5) {
			return false
		}
	}
	return true
}

func (r *PageRegistrar) accept(gray imaging.Image, tpl *pageTemplate, H [3][3]float64,
	radii [2]float64, refArea float64, hasRefArea bool, chain bool) *PageRegistration {

	if !r.saneQuad(tpl.quadInImage(H), gray.Width(), gray.Height(), refArea, hasRefArea) {
		return nil
	}
	minInl := minRefinedInliers
	minSpread := minSpreadPx
	if chain {
		minInl = minRefinedInliersChain
		minSpread = minSpreadPxChain
	}
	Hr, inl, pts, ok := r.refine(gray, tpl, H, radii, minInl)
	if !ok || inl < minInl || !spreadOK(pts, minSpread) {
		return nil
	}
	quad := tpl.quadInImage(Hr)
	if !r.saneQuad(quad, gray.Width(), gray.Height(), refArea, hasRefArea) {
		return nil
	}
	return &PageRegistration{Name: tpl.name, H: &Hr, Inliers: inl, Method: "x", Quad: quad}
}

// ------------------------------------------------------------------- driver

// Register registers every template page in img (upright RGB photo). quads are
// optional Borders page quads (photo pixels) used as coarse search regions and, when
// nothing else matches, as a geometric prior. Port of PageRegistrar.register.
func (r *PageRegistrar) Register(img imaging.Image, quads [][]imaging.Point) []PageRegistration {
	gray := imaging.ToGray(img)
	defer gray.Close()
	kp, desc := r.sift.DetectAndCompute(gray)

	type quadFeat struct {
		kp   []imaging.Point
		desc [][]float32
	}
	quadFeats := make([]quadFeat, len(quads))
	for i, q := range quads {
		qk, qd := featuresInQuad(kp, desc, q, gray.Width(), gray.Height())
		quadFeats[i] = quadFeat{qk, qd}
	}

	results := make([]PageRegistration, len(r.pages))
	for i, p := range r.pages {
		results[i] = PageRegistration{Name: p.name}
	}
	taken := map[int]bool{}
	pending := make([]int, len(r.pages))
	for i := range pending {
		pending[i] = i
	}

	for round := 0; round < 2; round++ {
		if round == 1 {
			anyOK := false
			for _, res := range results {
				anyOK = anyOK || res.OK()
			}
			if !anyOK {
				break
			}
		}
		snapshot := append([]int(nil), pending...)
		for _, ti := range snapshot {
			refs := r.pages[ti].refs
			var found *PageRegistration

			type cand struct {
				inl     int
				H       [3][3]float64
				method  string
				refArea float64
				hasArea bool
				tpl     *pageTemplate
			}
			var cands []cand
			if round == 0 {
				for _, tpl := range refs {
					for qi, qf := range quadFeats {
						if taken[qi] {
							continue
						}
						m := r.match(tpl, qf.kp, qf.desc, coarseReprojPx, 0, false, imaging.Identity3(), false)
						if m.OK && m.Inliers >= minCoarseInliers {
							cands = append(cands, cand{m.Inliers, m.H, fmt.Sprintf("quad%d", qi),
								contourArea(quads[qi]), true, tpl})
						}
					}
					best := -1
					for _, c := range cands {
						if c.inl > best {
							best = c.inl
						}
					}
					if best < goodCoarseInliers {
						m := r.match(tpl, kp, desc, coarseReprojPx, 0, false, imaging.Identity3(), false)
						if m.OK && m.Inliers >= minCoarseInliers {
							cands = append(cands, cand{m.Inliers, m.H, "global", 0, false, tpl})
						}
					}
					best = -1
					for _, c := range cands {
						if c.inl > best {
							best = c.inl
						}
					}
					if len(cands) > 0 && best >= goodCoarseInliers {
						break
					}
				}
				sort.SliceStable(cands, func(a, b int) bool { return cands[a].inl > cands[b].inl })
				limit := maxCoarseCandidates
				if limit > len(cands) {
					limit = len(cands)
				}
				for _, c := range cands[:limit] {
					if f := r.accept(gray, c.tpl, c.H, refineRadiusPx, c.refArea, c.hasArea, false); f != nil {
						f.Method, f.Ref = c.method, refIndex(refs, c.tpl)
						found = f
						break
					}
				}
			}

			if found == nil {
				for tj, other := range results {
					if tj == ti || !other.OK() {
						continue
					}
					dy := float64(ti-tj) * float64(r.pageH) * (1 + chainGapFrac)
					shift := [3][3]float64{{1, 0, 0}, {0, 1, -dy}, {0, 0, 1}}
					prior := imaging.MulH(shift, *other.H)
					for _, tpl := range refs {
						if f := r.accept(gray, tpl, prior, chainRadiusPx, 0, false, true); f != nil {
							f.Method, f.Ref = "chain:"+other.Name, refIndex(refs, tpl)
							found = f
							break
						}
					}
					if found != nil {
						break
					}
				}
			}

			if found == nil && round == 0 {
				for qi, q := range quads {
					if taken[qi] {
						continue
					}
					prior, ok := imaging.SolvePerspective4(
						[4]imaging.Point{imaging.OrderPoints(q)[0], imaging.OrderPoints(q)[1],
							imaging.OrderPoints(q)[2], imaging.OrderPoints(q)[3]},
						refs[0].corners)
					if !ok {
						continue
					}
					for _, tpl := range refs {
						if f := r.accept(gray, tpl, prior, chainRadiusPx, contourArea(q), true, true); f != nil {
							f.Method, f.Ref = fmt.Sprintf("quadprior%d", qi), refIndex(refs, tpl)
							found = f
							break
						}
					}
					if found != nil {
						break
					}
				}
			}

			if found == nil {
				continue
			}
			found.Name = r.pages[ti].name
			results[ti] = *found
			pending = removeInt(pending, ti)
			switch {
			case strings.HasPrefix(found.Method, "quadprior"):
				var idx int
				fmt.Sscanf(found.Method[len("quadprior"):], "%d", &idx)
				taken[idx] = true
			case strings.HasPrefix(found.Method, "quad"):
				var idx int
				fmt.Sscanf(found.Method[len("quad"):], "%d", &idx)
				taken[idx] = true
			}
		}
		if len(pending) == 0 {
			break
		}
	}
	return results
}

func refIndex(refs []*pageTemplate, tpl *pageTemplate) int {
	for i, t := range refs {
		if t == tpl {
			return i
		}
	}
	return 0
}

func removeInt(s []int, v int) []int {
	out := s[:0]
	for _, x := range s {
		if x != v {
			out = append(out, x)
		}
	}
	return out
}

// NativeScale is PageRegistrar.native_scale: the output scale (<=1) at which no
// registered page is upsampled, floored at 0.25. 1.0 when nothing registered.
func (r *PageRegistrar) NativeScale(regs []PageRegistration) float64 {
	scale := 1.0
	for _, reg := range regs {
		if !reg.OK() {
			continue
		}
		q := imaging.OrderPoints(reg.Quad)
		wNative := 0.5 * (dist(q[1], q[0]) + dist(q[2], q[3]))
		s := wNative / float64(r.pageW)
		if s < scale {
			scale = s
		}
	}
	if scale < 0.25 {
		return 0.25
	}
	return scale
}

func (r *PageRegistrar) OutSize(scale float64) (int, int) {
	w := int(math.Round(float64(r.outW) * scale))
	h := int(math.Round(float64(r.outH) * scale))
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	return w, h
}

// WarpQuad warps a photo quad (e.g. from Borders) into the canonical page frame,
// matching PageRegistrar.warp_quad.
func (r *PageRegistrar) WarpQuad(img imaging.Image, quad []imaging.Point, scale float64) imaging.Image {
	m := float64(r.margin)
	src := imaging.OrderPoints(quad)
	dstPts := [4]imaging.Point{
		{X: m, Y: m},
		{X: m + float64(r.pageW), Y: m},
		{X: m + float64(r.pageW), Y: m + float64(r.pageH)},
		{X: m, Y: m + float64(r.pageH)},
	}
	for i := range dstPts {
		dstPts[i].X *= scale
		dstPts[i].Y *= scale
	}
	M, ok := imaging.SolvePerspective4([4]imaging.Point{src[0], src[1], src[2], src[3]}, dstPts)
	if !ok {
		return img.Clone()
	}
	w, h := r.OutSize(scale)
	return imaging.WarpByHomography(img, M, w, h, true)
}

// WarpPage warps the photo to the canonical page frame at scale via reg.H, matching
// PageRegistrar.warp_page.
func (r *PageRegistrar) WarpPage(img imaging.Image, reg PageRegistration, scale float64) imaging.Image {
	m := float64(r.margin)
	shift := [3][3]float64{{1, 0, m}, {0, 1, m}, {0, 0, 1}}
	s := [3][3]float64{{scale, 0, 0}, {0, scale, 0}, {0, 0, 1}}
	M := imaging.MulH(s, imaging.MulH(shift, *reg.H))
	w, h := r.OutSize(scale)
	return imaging.WarpByHomography(img, M, w, h, true)
}

// PageQuads is PageRegistrar.page_quads: page quads from Borders contours, straight
// lines fitted to each (line_quads=true, the default).
func (r *PageRegistrar) PageQuads(segments [][]imaging.Point, imgH, imgW int) ([][]imaging.Point, []QuadFitInfo) {
	var quads [][]imaging.Point
	var infos []QuadFitInfo
	for _, s := range segments {
		var q []imaging.Point
		var info QuadFitInfo
		if r.LineQuads {
			q, info = FitQuadLines(s, imgH, imgW, float64(r.pageH)/float64(r.pageW), true)
		} else {
			q = imaging.ExtractQuad(s)
			info = QuadFitInfo{Method: "polygon"}
		}
		if q != nil {
			quads = append(quads, imaging.OrderPoints(q))
			infos = append(infos, info)
		}
	}
	return quads, infos
}

// QuadIoU is PageRegistrar.quad_iou (a plain function here, matching its @staticmethod).
func QuadIoU(a, b []imaging.Point) float64 {
	return quadIoU(a, b)
}

// Straighten is PageRegistrar.straighten: first the homography (line_refine), then the
// bend map (line_dewarp) on the result. CONSUMES page (closes it if a correction
// replaces it, returns it as-is otherwise) - the caller must not touch page again,
// only the returned Image.
func (r *PageRegistrar) Straighten(page imaging.Image, scale float64) (imaging.Image, StraightenInfo) {
	inset := int(math.Round(float64(r.margin) * scale))
	info := StraightenInfo{Reason: "disabled"}
	cur := page
	if r.LineRefine {
		gray := imaging.ToGray(cur)
		Hm, refInfo := RefineByLines(gray, inset)
		gray.Close()
		info = refInfo
		if Hm != nil {
			next := ApplyRefinement(cur, *Hm)
			cur.Close()
			cur = next
		}
	}
	if r.LineDewarp {
		gray := imaging.ToGray(cur)
		v, dInfo := DewarpByLines(gray, inset)
		gray.Close()
		if v != nil {
			next := ApplyDewarp(cur, v)
			cur.Close()
			cur = next
		}
		info.Applied = info.Applied || dInfo.Applied
	}
	return cur, info
}

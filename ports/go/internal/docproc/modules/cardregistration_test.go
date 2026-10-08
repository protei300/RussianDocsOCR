package modules

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Mirrors tests/test_card_registration.py of the reference. The blank itself, put into a
// photo in perspective, is the card; no models, but the templates of the repository.

var stsTypes = []string{"STS_1996", "STSBACK_1996", "STS_2019", "STSBACK_2019"}

func repoRoot(t *testing.T) string {
	t.Helper()
	if root := os.Getenv("RDOCS_REPO_ROOT"); root != "" {
		return root
	}
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "..", "..", "..")
	if _, err := os.Stat(filepath.Join(root, "document_processing", "pipeline_modules")); err != nil {
		t.Skip("repository root not found; set RDOCS_REPO_ROOT")
	}
	return root
}

func TestEveryStsTypeHasTemplates(t *testing.T) {
	root := repoRoot(t)
	for _, doc := range stsTypes {
		reg, err := NewPageRegistrar(root, doc, 6000)
		if err != nil {
			t.Fatalf("%s: %v", doc, err)
		}
		if names := reg.PageNames(); len(names) != 1 || names[0] != "card" {
			t.Errorf("%s: pages %v", doc, names)
		}
		for _, tpl := range reg.pages[0].refs {
			if len(tpl.kp) < 500 {
				t.Errorf("%s/%s: only %d features on the printed blank", doc, tpl.name, len(tpl.kp))
			}
		}
		reg.Close()
	}
}

func TestACardInPerspectiveIsFoundToAFewPixels(t *testing.T) {
	root := repoRoot(t)
	// the first reference of each type, as the template json names it
	files := map[string]string{"STS_1996": "sts_1996_old.jpg", "STSBACK_1996": "stsback_1996_old.jpg",
		"STS_2019": "sts_2019_new.jpg", "STSBACK_2019": "stsback_2019_new.jpg"}
	dir := filepath.Join(root, "document_processing", "pipeline_modules", "page_registration", "templates")
	for _, doc := range stsTypes {
		reg, err := NewPageRegistrar(root, doc, 6000)
		if err != nil {
			t.Fatal(err)
		}
		card, err := imaging.LoadRGB(filepath.Join(dir, files[doc]))
		if err != nil {
			t.Fatal(err)
		}
		w, h := float64(card.Width()), float64(card.Height())
		corners := [4]imaging.Point{{X: 260, Y: 180}, {X: 1320, Y: 240}, {X: 1380, Y: 1700}, {X: 210, Y: 1640}}
		M, ok := imaging.SolvePerspective4([4]imaging.Point{{X: 0, Y: 0}, {X: w, Y: 0}, {X: w, Y: h}, {X: 0, Y: h}}, corners)
		if !ok {
			t.Fatal("no perspective")
		}
		photo := imaging.WarpByHomographyFill(card, M, 1600, 1900, [3]uint8{90, 110, 130})
		regs := reg.Register(photo, nil)
		if len(regs) != 1 || !regs[0].OK() || regs[0].Inliers < 40 {
			t.Fatalf("%s: not found (%+v)", doc, regs)
		}
		var worst float64
		for _, q := range regs[0].Quad {
			best := math.Inf(1)
			for _, c := range corners {
				best = math.Min(best, math.Hypot(q.X-c.X, q.Y-c.Y))
			}
			worst = math.Max(worst, best)
		}
		if worst >= 4.0 {
			t.Errorf("%s: card corners off by %.1f px", doc, worst)
		}
		photo.Close()
		card.Close()
		reg.Close()
	}
}

// The registered quad and the native scale are float32 values, like the reference's arrays.
func TestRegisteredQuadAndScaleAreFloat32(t *testing.T) {
	root := repoRoot(t)
	reg, err := NewPageRegistrar(root, "STS_2019", 6000)
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()
	H := [3][3]float64{{1.0003, 0.0002, 3.3}, {0, 0.9998, 4.1}, {1e-7, 0, 1}}
	q := reg.pages[0].refs[0].quadInImage(H)
	for _, p := range q {
		if p.X != float64(float32(p.X)) || p.Y != float64(float32(p.Y)) {
			t.Fatalf("quad is not float32: %v", q)
		}
	}
	s := reg.NativeScale([]PageRegistration{{H: &H, Quad: q}})
	if s != float64(float32(s)) {
		t.Errorf("scale %v is not a float32 value", s)
	}
}

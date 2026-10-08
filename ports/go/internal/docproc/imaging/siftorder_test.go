package imaging

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"gocv.io/x/gocv"
)

// Mirrors tests/test_sift_order.py of the reference: template matching does not depend on the
// order OpenCV hands keypoints over. OpenCV orders SIFT keypoints with std::sort and cuts its
// nfeatures budget with std::nth_element; the order of ties depends on the C++ library it was
// built with, and MAGSAC samples by index. orderKeypoints sorts the keypoints itself and cuts the
// budget from that order. No OpenCV call here.

func siftKeypoints() []gocv.KeyPoint {
	rng := rand.New(rand.NewSource(0))
	var kps []gocv.KeyPoint
	for i := 0; i < 300; i++ {
		kps = append(kps, gocv.KeyPoint{X: rng.Float64() * 500, Y: rng.Float64() * 500, Size: 3,
			Response: rng.Float64()})
	}
	// ties on response, broken only by position
	for x := 0; x < 20; x++ {
		kps = append(kps, gocv.KeyPoint{X: float64(x), Y: 10, Size: 3, Response: 0.5})
	}
	return kps
}

// handed returns kps in a given shuffled order, remembering where each came from (the row of a
// descriptor table that belongs to it).
func handed(kps []gocv.KeyPoint, seed int64) (shuffled []gocv.KeyPoint, origin []int) {
	origin = rand.New(rand.NewSource(seed)).Perm(len(kps))
	for _, i := range origin {
		shuffled = append(shuffled, kps[i])
	}
	return shuffled, origin
}

func signature(kps []gocv.KeyPoint, order []int, origin []int) [][3]float64 {
	out := make([][3]float64, len(order))
	for i, j := range order {
		out[i] = [3]float64{kps[j].X, kps[j].Y, float64(origin[j])}
	}
	return out
}

func TestTheOrderHandedOverDoesNotMatter(t *testing.T) {
	kps := siftKeypoints()
	var first [][3]float64
	for seed := int64(0); seed < 5; seed++ {
		shuffled, origin := handed(kps, seed)
		got := signature(shuffled, orderKeypoints(shuffled, 100), origin)
		if seed == 0 {
			first = got
			continue
		}
		if !reflect.DeepEqual(got, first) {
			t.Fatalf("seed %d: another selection or order", seed)
		}
	}
}

func TestTheBudgetKeepsTheStrongestAndDescriptorsFollowTheirKeypoints(t *testing.T) {
	kps := siftKeypoints()
	// a descriptor table: row i belongs to kps[i]
	desc := make([][]float32, len(kps))
	for i := range desc {
		desc[i] = []float32{float32(i), float32(4 * i)}
	}
	shuffled, origin := handed(kps, 1)
	order := orderKeypoints(shuffled, 50)

	if len(order) != 50 {
		t.Fatalf("kept %d", len(order))
	}
	prev := math.Inf(1)
	for _, j := range order {
		if r := shuffled[j].Response; r > prev {
			t.Fatal("not strongest first")
		} else {
			prev = r
		}
	}
	// the 50 strongest of all
	all := make([]float64, len(kps))
	for i, k := range kps {
		all[i] = k.Response
	}
	cut := 0.0
	for _, r := range all {
		stronger := 0
		for _, o := range all {
			if o > r {
				stronger++
			}
		}
		if stronger == 49 {
			cut = r
		}
	}
	if prev < cut {
		t.Errorf("the weakest kept (%v) is weaker than the 50th strongest (%v)", prev, cut)
	}
	// the descriptor row that goes with each kept keypoint is the one of its origin
	for _, j := range order {
		row := desc[origin[j]]
		if row[0] != float32(origin[j]) || kps[origin[j]].X != shuffled[j].X {
			t.Fatal("a descriptor does not follow its keypoint")
		}
	}
}

// Ties are broken the way np.lexsort does: response descending, then y, x, size, angle, octave.
func TestTiesAreBrokenByPositionThenSizeAngleOctave(t *testing.T) {
	kps := []gocv.KeyPoint{
		{X: 5, Y: 2, Size: 1, Angle: 0, Octave: 0, Response: 1},
		{X: 3, Y: 2, Size: 1, Angle: 0, Octave: 0, Response: 1}, // same y, smaller x: before the first
		{X: 9, Y: 1, Size: 1, Angle: 0, Octave: 0, Response: 1}, // smaller y: first of the three
		{X: 3, Y: 2, Size: 2, Angle: 0, Octave: 0, Response: 1}, // same place, larger size
		{X: 3, Y: 2, Size: 2, Angle: 7, Octave: 0, Response: 1},
		{X: 3, Y: 2, Size: 2, Angle: 7, Octave: 3, Response: 1},
		{X: 0, Y: 0, Size: 1, Response: 2}, // stronger than all: before them
		{X: 0, Y: 0, Size: 1, Response: 0.1},
	}
	want := []int{6, 2, 1, 3, 4, 5, 0, 7}
	if got := orderKeypoints(kps, 0); !reflect.DeepEqual(got, want) {
		t.Errorf("order %v, want %v", got, want)
	}
}

// cv2.findHomography(src, dst, 0): the least-squares homography of all the pairs.
func TestLeastSquaresHomographyRecoversAnExactMap(t *testing.T) {
	m := [3][3]float64{{1.05, 0.02, 12}, {-0.01, 0.97, -5}, {1e-5, -2e-5, 1}}
	var src, dst []Point
	for x := 0; x <= 400; x += 100 {
		for y := 0; y <= 300; y += 75 {
			w := m[2][0]*float64(x) + m[2][1]*float64(y) + m[2][2]
			src = append(src, Point{float64(x), float64(y)})
			dst = append(dst, Point{(m[0][0]*float64(x) + m[0][1]*float64(y) + m[0][2]) / w,
				(m[1][0]*float64(x) + m[1][1]*float64(y) + m[1][2]) / w})
		}
	}
	H, ok := FindHomographyLeastSquares(src, dst)
	if !ok {
		t.Fatal("no homography")
	}
	for i := range src {
		p := TransformPoint(H, src[i])
		if math.Hypot(p.X-dst[i].X, p.Y-dst[i].Y) > 1e-3 {
			t.Fatalf("pair %d off by %v", i, math.Hypot(p.X-dst[i].X, p.Y-dst[i].Y))
		}
	}
}

package imaging

import "math"

// Pure 3x3 homography algebra: no OpenCV call, no cgo. Kept here (not in modules,
// which must not import gocv) simply because Point lives here too. Used by page
// registration to avoid a gocv Mat round trip for every point transform - these are
// plain float64 formulas, identical to what cv2.perspectiveTransform /
// cv2.getPerspectiveTransform compute.

// Identity3 is the 3x3 identity matrix.
func Identity3() [3][3]float64 {
	return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
}

// MulH multiplies two homographies: (a * b) applied to a point p means a(b(p)).
func MulH(a, b [3][3]float64) [3][3]float64 {
	var out [3][3]float64
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += a[r][k] * b[k][c]
			}
			out[r][c] = s
		}
	}
	return out
}

// InvertH inverts a 3x3 matrix by cofactor expansion. ok is false for a singular
// matrix (determinant ~0), mirroring what np.linalg.inv would raise on.
func InvertH(m [3][3]float64) (inv [3][3]float64, ok bool) {
	a, b, c := m[0][0], m[0][1], m[0][2]
	d, e, f := m[1][0], m[1][1], m[1][2]
	g, h, i := m[2][0], m[2][1], m[2][2]

	A := e*i - f*h
	B := -(d*i - f*g)
	C := d*h - e*g
	det := a*A + b*B + c*C
	if math.Abs(det) < 1e-12 {
		return inv, false
	}
	invDet := 1.0 / det
	inv[0][0] = A * invDet
	inv[1][0] = B * invDet
	inv[2][0] = C * invDet
	inv[0][1] = -(b*i - c*h) * invDet
	inv[1][1] = (a*i - c*g) * invDet
	inv[2][1] = -(a*h - b*g) * invDet
	inv[0][2] = (b*f - c*e) * invDet
	inv[1][2] = -(a*f - c*d) * invDet
	inv[2][2] = (a*e - b*d) * invDet
	return inv, true
}

// TransformPoint applies a homography to one point, matching cv2.perspectiveTransform
// (division by the homogeneous w).
func TransformPoint(H [3][3]float64, p Point) Point {
	x := H[0][0]*p.X + H[0][1]*p.Y + H[0][2]
	y := H[1][0]*p.X + H[1][1]*p.Y + H[1][2]
	w := H[2][0]*p.X + H[2][1]*p.Y + H[2][2]
	return Point{X: x / w, Y: y / w}
}

// TransformPoints applies a homography to every point.
func TransformPoints(H [3][3]float64, pts []Point) []Point {
	out := make([]Point, len(pts))
	for i, p := range pts {
		out[i] = TransformPoint(H, p)
	}
	return out
}

// SolvePerspective4 computes the exact homography mapping src[i] -> dst[i] for four
// point correspondences (the general position case getPerspectiveTransform assumes),
// by direct linear transform: solving the 8x8 system for h0..h7 with h8 = 1.
// Matches cv2.getPerspectiveTransform's algorithm and result to solver precision.
func SolvePerspective4(src, dst [4]Point) ([3][3]float64, bool) {
	var A [8][8]float64
	var b [8]float64
	for i := 0; i < 4; i++ {
		x, y := src[i].X, src[i].Y
		X, Y := dst[i].X, dst[i].Y
		r0 := 2 * i
		A[r0] = [8]float64{x, y, 1, 0, 0, 0, -x * X, -y * X}
		b[r0] = X
		r1 := 2*i + 1
		A[r1] = [8]float64{0, 0, 0, x, y, 1, -x * Y, -y * Y}
		b[r1] = Y
	}
	h, ok := solveLinear8(A, b)
	if !ok {
		return [3][3]float64{}, false
	}
	return [3][3]float64{
		{h[0], h[1], h[2]},
		{h[3], h[4], h[5]},
		{h[6], h[7], 1},
	}, true
}

// solveLinear8 solves an 8x8 linear system by Gaussian elimination with partial
// pivoting.
func solveLinear8(A [8][8]float64, b [8]float64) ([8]float64, bool) {
	const n = 8
	var x [8]float64
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(A[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(A[r][col]); v > best {
				piv, best = r, v
			}
		}
		if best < 1e-12 {
			return x, false
		}
		if piv != col {
			A[col], A[piv] = A[piv], A[col]
			b[col], b[piv] = b[piv], b[col]
		}
		for r := col + 1; r < n; r++ {
			f := A[r][col] / A[col][col]
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				A[r][c] -= f * A[col][c]
			}
			b[r] -= f * b[col]
		}
	}
	for r := n - 1; r >= 0; r-- {
		s := b[r]
		for c := r + 1; c < n; c++ {
			s -= A[r][c] * x[c]
		}
		x[r] = s / A[r][r]
	}
	return x, true
}

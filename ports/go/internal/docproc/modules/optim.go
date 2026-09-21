package modules

import "math"

// LevenbergMarquardt minimises sum(residuals(x)^2) over x, for the small (4-parameter)
// problems line_refine.py solves with scipy.optimize.least_squares. Per the task's
// coordinator: a small hand-written LM is enough here - the Jacobian is by finite
// differences (central difference, step 1e-6), damping updates by x10 on a rejected
// step and /10 on an accepted one (classic Marquardt scaling: the diagonal of J^T J is
// scaled by (1+lambda), not a flat addition), and the loop stops on a tiny step norm or
// after maxEvals residual evaluations.
//
// This is NOT scipy's trust-region-reflective algorithm, so the same starting point is
// not guaranteed to land on the same optimum on a non-convex residual (the IRLS loop
// around this one is itself only piecewise-convex) - see the task log for the measured
// effect on the two internal-passport conformance cases this feeds.
func LevenbergMarquardt(x0 []float64, residuals func([]float64) []float64, maxEvals int) []float64 {
	n := len(x0)
	x := append([]float64(nil), x0...)
	r := residuals(x)
	evals := 1
	lambda := 1e-3
	const h = 1e-6

	for evals < maxEvals {
		m := len(r)
		J := make([][]float64, m)
		for i := range J {
			J[i] = make([]float64, n)
		}
		for j := 0; j < n && evals < maxEvals; j++ {
			xh := append([]float64(nil), x...)
			xh[j] += h
			rh := residuals(xh)
			evals++
			for i := 0; i < m; i++ {
				J[i][j] = (rh[i] - r[i]) / h
			}
		}

		JTJ := make([][]float64, n)
		JTr := make([]float64, n)
		for a := 0; a < n; a++ {
			JTJ[a] = make([]float64, n)
			for b := 0; b < n; b++ {
				var s float64
				for i := 0; i < m; i++ {
					s += J[i][a] * J[i][b]
				}
				JTJ[a][b] = s
			}
			var s float64
			for i := 0; i < m; i++ {
				s += J[i][a] * r[i]
			}
			JTr[a] = s
		}

		improved := false
		for evals < maxEvals {
			A := make([][]float64, n)
			for i := range A {
				A[i] = append([]float64(nil), JTJ[i]...)
				A[i][i] *= 1 + lambda
			}
			b := make([]float64, n)
			for i := range b {
				b[i] = -JTr[i]
			}
			delta, ok := solveLinearN(A, b)
			if !ok {
				lambda *= 10
				if lambda > 1e12 {
					return x
				}
				continue
			}
			xNew := make([]float64, n)
			for i := range xNew {
				xNew[i] = x[i] + delta[i]
			}
			rNew := residuals(xNew)
			evals++
			if sumSq(rNew) < sumSq(r) {
				x, r = xNew, rNew
				lambda /= 10
				improved = true
				if normVec(delta) < 1e-10 {
					return x
				}
				break
			}
			lambda *= 10
			if lambda > 1e12 {
				return x
			}
		}
		if !improved {
			return x
		}
	}
	return x
}

func sumSq(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return s
}

func normVec(v []float64) float64 { return math.Sqrt(sumSq(v)) }

// solveLinearN solves an n x n linear system by Gaussian elimination with partial
// pivoting. ok is false for a (near-)singular matrix.
func solveLinearN(a [][]float64, b []float64) ([]float64, bool) {
	n := len(b)
	A := make([][]float64, n)
	for i := range A {
		A[i] = append([]float64(nil), a[i]...)
	}
	bb := append([]float64(nil), b...)
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(A[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(A[r][col]); v > best {
				piv, best = r, v
			}
		}
		if best < 1e-15 {
			return nil, false
		}
		if piv != col {
			A[col], A[piv] = A[piv], A[col]
			bb[col], bb[piv] = bb[piv], bb[col]
		}
		for r := col + 1; r < n; r++ {
			f := A[r][col] / A[col][col]
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				A[r][c] -= f * A[col][c]
			}
			bb[r] -= f * bb[col]
		}
	}
	x := make([]float64, n)
	for r := n - 1; r >= 0; r-- {
		s := bb[r]
		for c := r + 1; c < n; c++ {
			s -= A[r][c] * x[c]
		}
		x[r] = s / A[r][r]
	}
	return x, true
}

// wmedian is the weighted median (dates.py-adjacent helper, used by both
// line_refine.refine_by_lines and line_dewarp.dewarp_by_lines to report the residual
// tilt before/after): the smallest value whose cumulative weight reaches half the
// total, matching np.searchsorted(cumsum(w), 0.5*total) on values sorted ascending.
func wmedian(values, weights []float64) float64 {
	n := len(values)
	if n == 0 {
		return 0
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sortInts := idx
	// simple insertion-free sort via sort.Slice equivalent (avoid extra import churn)
	for i := 1; i < n; i++ {
		j := i
		for j > 0 && values[sortInts[j-1]] > values[sortInts[j]] {
			sortInts[j-1], sortInts[j] = sortInts[j], sortInts[j-1]
			j--
		}
	}
	total := 0.0
	for _, w := range weights {
		total += w
	}
	half := 0.5 * total
	cum := 0.0
	for _, i := range sortInts {
		cum += weights[i]
		if cum >= half {
			return values[i]
		}
	}
	return values[sortInts[n-1]]
}

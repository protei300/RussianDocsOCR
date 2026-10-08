package imaging

// Microsoft's std::nth_element and std::partition, reproduced step for step.
//
// WHY THIS EXISTS. OpenCV's SIFT keeps its best `nfeatures` keypoints with
// KeyPointsFilter::retainBest, which is std::nth_element followed by std::partition over the
// keypoints sorted by position. The SET it keeps is the same in every standard library (all
// keypoints of the boundary response stay), but the ORDER of the survivors is whatever the
// library's nth_element leaves behind - and that is implementation-defined. The reference's
// wheels are built with MSVC; this port's OpenCV is built with GCC (libstdc++). The order
// matters: page registration matches the template's keypoints to the photo's in list order and
// hands the matches to MAGSAC, whose random sampling then draws different minimal sets, lands
// on a different homography, and on a card (STS) that flips the choice between the template
// canvas and the Borders canvas (measured on STS_2019: 1670 vs 1760 coarse inliers from the
// SAME 2668 matches, the first keypoint of the list differing).
//
// So the order is made the reference's: the raw keypoints are taken from OpenCV in the order
// it produces before retainBest (position sorted, library-independent), cut with the routines
// below, and only then given descriptors (SiftDetector). Verified against the reference's own
// output on 8779 raw keypoints cut to 6000: identical lists, element for element
// (stlselect_test.go keeps a small vector of the same kind).
//
// The algorithm is MSVC STL's introselect: median of three (Tukey's ninther above 40
// elements) around a "fat pivot" partition, insertion sort for ranges of at most 32.

const isortMax = 32

type selElem struct {
	resp float64
	id   int
}

// greater is KeypointResponseGreaterThanThreshold: a.response > b.response.
func greater(a, b selElem) bool { return a.resp > b.resp }

func med3(a []selElem, f, m, l int) {
	if greater(a[m], a[f]) {
		a[m], a[f] = a[f], a[m]
	}
	if greater(a[l], a[m]) {
		a[l], a[m] = a[m], a[l]
		if greater(a[m], a[f]) {
			a[m], a[f] = a[f], a[m]
		}
	}
}

func guessMedian(a []selElem, f, mid, l int) {
	cnt := l - f
	if 40 < cnt { // Tukey's ninther
		step := (cnt + 1) >> 3
		two := step << 1
		med3(a, f, f+step, f+two)
		med3(a, mid-step, mid, mid+step)
		med3(a, l-1-two, l-1-step, l-1)
		med3(a, f+step, mid, l-1-step)
	} else {
		med3(a, f, mid, l-1)
	}
}

// partitionByMedianGuess partitions [f, l) around a guessed median; the returned [pf, pl) is
// the "fat pivot" of elements equal to it.
func partitionByMedianGuess(a []selElem, f, l int) (int, int) {
	mid := f + ((l - f) >> 1)
	guessMedian(a, f, mid, l)
	pf := mid
	pl := pf + 1
	for f < pf && !greater(a[pf-1], a[pf]) && !greater(a[pf], a[pf-1]) {
		pf--
	}
	for pl < l && !greater(a[pl], a[pf]) && !greater(a[pf], a[pl]) {
		pl++
	}
	gf, gl := pl, pf
	for {
		for gf < l {
			if greater(a[pf], a[gf]) {
				gf++
				continue
			} else if greater(a[gf], a[pf]) {
				break
			} else {
				if pl != gf {
					a[pl], a[gf] = a[gf], a[pl]
				}
				pl++
				gf++
			}
		}
		for f < gl {
			if greater(a[gl-1], a[pf]) {
				gl--
				continue
			} else if greater(a[pf], a[gl-1]) {
				break
			} else {
				pf--
				if pf != gl-1 {
					a[pf], a[gl-1] = a[gl-1], a[pf]
				}
				gl--
			}
		}
		if gl == f && gf == l {
			return pf, pl
		}
		if gl == f { // no room at bottom, rotate pivot upwards
			if pl != gf {
				a[pf], a[pl] = a[pl], a[pf]
			}
			pl++
			a[pf], a[gf] = a[gf], a[pf]
			pf++
			gf++
		} else if gf == l { // no room at top, rotate pivot downwards
			gl--
			pf--
			if gl != pf {
				a[gl], a[pf] = a[pf], a[gl]
			}
			pl--
			a[pf], a[pl] = a[pl], a[pf]
		} else {
			gl--
			a[gf], a[gl] = a[gl], a[gf]
			gf++
		}
	}
}

func insertionSort(a []selElem, f, l int) {
	for i := f + 1; i < l; i++ {
		v := a[i]
		j := i
		if greater(v, a[f]) {
			copy(a[f+1:i+1], a[f:i])
			a[f] = v
		} else {
			for greater(v, a[j-1]) {
				a[j] = a[j-1]
				j--
			}
			a[j] = v
		}
	}
}

// nthElement is std::nth_element(a.begin(), a.begin()+nth, a.end(), greater).
func nthElement(a []selElem, nth int) {
	f, l := 0, len(a)
	if nth == l {
		return
	}
	for isortMax < l-f {
		pf, pl := partitionByMedianGuess(a, f, l)
		if pl <= nth {
			f = pl
		} else if pf <= nth {
			return // nth inside the fat pivot
		} else {
			l = pf
		}
	}
	insertionSort(a, f, l)
}

// partitionGE is std::partition(a.begin()+first, a.end(), resp >= limit) for random access
// iterators; it returns the index one past the last kept element.
func partitionGE(a []selElem, first int, limit float64) int {
	last := len(a)
	pred := func(e selElem) bool { return e.resp >= limit }
	for {
		for {
			if first == last {
				return first
			} else if pred(a[first]) {
				first++
			} else {
				break
			}
		}
		for {
			last--
			if first == last {
				return first
			} else if !pred(a[last]) {
				continue
			}
			break
		}
		a[first], a[last] = a[last], a[first]
		first++
	}
}

// retainBestOrder is KeyPointsFilter::retainBest on a list of responses: the indices of the
// kept keypoints, in the order the MSVC standard library leaves them. n >= len keeps all in
// the given order.
func retainBestOrder(resp []float64, n int) []int {
	a := make([]selElem, len(resp))
	for i, r := range resp {
		a[i] = selElem{r, i}
	}
	if n >= 0 && len(a) > n {
		if n == 0 {
			return nil
		}
		nthElement(a, n-1)
		amb := a[n-1].resp
		end := partitionGE(a, n, amb)
		a = a[:end]
	}
	out := make([]int, len(a))
	for i, e := range a {
		out[i] = e.id
	}
	return out
}

package postprocess

import (
	"fmt"
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Metric is nearest-centroid classification over an embedding — the DocTypeAngles
// document-type head. Port of MetricPostprocessing (postprocessing.py:204-254).
//
// Python builds a sklearn NearestNeighbors index over the centroids and calls
// radius_neighbors. With NINE centroids that is a linear scan, so sklearn is a
// convenience here and not an algorithm; reproducing it needs no dependency. What DOES
// have to be reproduced exactly is the pair of gates, because they behave differently:
//
//  1. **The radius filter.** sklearn's cosine distance is `1 - cosine_similarity`, so
//     `radius=1` means "only centroids with positive similarity". A vector with no
//     centroid inside the radius yields ('NONE', +Inf, 0.0) — and note the 0.0
//     threshold is a sentinel that DocTypeAngles checks before dividing by it.
//  2. **The per-class threshold.** Even the nearest centroid is rejected when
//     `dist >= max_distance[i]`, returning 'NONE' but reporting the real distance and
//     threshold.
//
// Dropping either gate turns "unknown document" into a confident wrong answer.
type Metric struct {
	metric  string
	radius  float64
	labels  []string
	centers [][]float32
	maxDist []float64
}

// NewMetric loads the centroid table from a .npz.
//
// The archive holds three row-aligned arrays: `labels` (dtype <U64, fixed-width
// UTF-32LE — naive byte slicing yields "I\0\0\0N\0\0\0T…"), `centers` (float32,
// [n, dim]) and `max_distance` (float32, [n]).
func NewMetric(npzPath, metric string) (*Metric, error) {
	var radius float64
	switch metric {
	case "Cosine", "cosine":
		radius = 1
	case "Euclidean", "euclidean":
		radius = 10
	default:
		return nil, fmt.Errorf("postprocess: unsupported metric %q", metric)
	}

	blob, err := tensor.LoadNPZ(npzPath)
	if err != nil {
		return nil, fmt.Errorf("postprocess: centers: %w", err)
	}
	labelsArr, ok := blob["labels"]
	if !ok {
		return nil, fmt.Errorf("postprocess: %s has no 'labels'", npzPath)
	}
	centersArr, ok := blob["centers"]
	if !ok {
		return nil, fmt.Errorf("postprocess: %s has no 'centers'", npzPath)
	}
	maxArr, ok := blob["max_distance"]
	if !ok {
		return nil, fmt.Errorf("postprocess: %s has no 'max_distance'", npzPath)
	}

	labels := labelsArr.Strings
	if len(centersArr.Shape) != 2 || centersArr.Shape[0] != len(labels) {
		return nil, fmt.Errorf("postprocess: centers %v does not align with %d labels",
			centersArr.Shape, len(labels))
	}
	dim := centersArr.Shape[1]
	centers := make([][]float32, len(labels))
	for i := range labels {
		centers[i] = centersArr.F32[i*dim : (i+1)*dim]
	}
	maxDist := make([]float64, len(labels))
	for i := range labels {
		maxDist[i] = float64(maxArr.F32[i])
	}

	return &Metric{metric: metric, radius: radius, labels: labels,
		centers: centers, maxDist: maxDist}, nil
}

// Labels exposes the class names, in centroid order.
func (m *Metric) Labels() []string { return m.labels }

func (m *Metric) Apply(out *tensor.Array, _ Context) (Result, error) {
	vec, err := out.AsFloat32()
	if err != nil {
		return nil, fmt.Errorf("postprocess: metric input: %w", err)
	}
	if len(m.centers) == 0 {
		return nil, fmt.Errorf("postprocess: no centroids loaded")
	}
	if len(vec) != len(m.centers[0]) {
		return nil, fmt.Errorf("postprocess: embedding has %d dims, centroids have %d",
			len(vec), len(m.centers[0]))
	}

	// Nearest centroid WITHIN the radius. Ties keep the lower index, matching
	// sort_results=True on a stable ordering.
	bestIdx, bestDist := -1, math.Inf(1)
	for i, c := range m.centers {
		var d float64
		if m.radius == 1 {
			d = tensor.CosineDistance(vec, c)
		} else {
			d = tensor.EuclideanDistance(vec, c)
		}
		if d > m.radius {
			continue // outside the radius: not a neighbour at all
		}
		if d < bestDist {
			bestIdx, bestDist = i, d
		}
	}

	if bestIdx < 0 {
		// No centroid within the radius. The 0.0 threshold is a SENTINEL, not a real
		// threshold: DocTypeAngles tests `threshold > 0` before dividing by it, and
		// the distance is +Inf so the division would otherwise be nonsense.
		return MetricResult{Label: "NONE", Distance: math.Inf(1), Threshold: 0}, nil
	}

	threshold := m.maxDist[bestIdx]
	if bestDist < threshold {
		return MetricResult{Label: m.labels[bestIdx], Distance: bestDist, Threshold: threshold}, nil
	}
	// Nearest, but not near enough: report 'NONE' while still returning the measured
	// distance and the threshold it failed.
	return MetricResult{Label: "NONE", Distance: bestDist, Threshold: threshold}, nil
}

package postprocess

import (
	"fmt"
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// OcrProbs is greedy CTC decoding with per-step alphabet masking.
// Port of OCRProbsPostprocessing (postprocessing.py:139-201).
//
// Input is a softmax matrix [batch, T, C] (or [T, C]) where class 0 is the CTC blank and
// class i maps to alphabet[i-1]. `allowed` is the set of characters this document may
// emit; characters outside it are SUBSTITUTED, not dropped.
type OcrProbs struct {
	// alphabet is indexed by RUNE, not byte. This is the trap the spike was built to
	// catch: the Cyrillic alphabet is multi-byte UTF-8, so `alphabet[idx-1]` on a Go
	// string yields a byte in the middle of a character and the output is mojibake with
	// certainty, not by chance. Stored pre-split so the mistake cannot be made later.
	alphabet   []rune
	blankIndex int
	// allowedIdx is nil when no mask applies; then decoding is a plain argmax.
	allowedIdx    map[int]bool
	disallowedIdx map[int]bool
}

func NewOcrProbs(alphabet string, allowed map[rune]bool, blankIndex int) (*OcrProbs, error) {
	if alphabet == "" {
		return nil, fmt.Errorf("postprocess: OCRProbs needs an Alphabet")
	}
	p := &OcrProbs{alphabet: []rune(alphabet), blankIndex: blankIndex}
	if allowed == nil {
		return p, nil
	}
	p.allowedIdx = make(map[int]bool, len(p.alphabet))
	p.disallowedIdx = make(map[int]bool)
	for i, ch := range p.alphabet {
		// Class indices are 1-based over the alphabet, because 0 is the blank.
		if allowed[ch] {
			p.allowedIdx[i+1] = true
		} else {
			p.disallowedIdx[i+1] = true
		}
	}
	return p, nil
}

func (p *OcrProbs) Apply(out *tensor.Array, _ Context) (Result, error) {
	text, err := p.Decode(out)
	if err != nil {
		return nil, err
	}
	return TextResult{Text: text}, nil
}

// Decode collapses the probability matrix to a string.
func (p *OcrProbs) Decode(out *tensor.Array) (string, error) {
	data, err := out.AsFloat32()
	if err != nil {
		return "", err
	}
	shape := out.Shape
	if len(shape) == 3 {
		// Only the first batch element, as the reference's `p[0]` does.
		shape = shape[1:]
	}
	if len(shape) != 2 {
		return "", fmt.Errorf("postprocess: OCRProbs expects [T,C] or [1,T,C], got %v", out.Shape)
	}
	steps, classes := shape[0], shape[1]
	if classes > len(p.alphabet)+1 {
		return "", fmt.Errorf("postprocess: %d classes exceeds alphabet of %d plus blank",
			classes, len(p.alphabet))
	}

	indices := make([]int, steps)
	masking := len(p.disallowedIdx) > 0
	for t := 0; t < steps; t++ {
		row := data[t*classes : (t+1)*classes]
		best := argmaxFirst(row)
		if !masking || best == p.blankIndex || p.allowedIdx[best] {
			indices[t] = best
			continue
		}
		// A disallowed character won. Pick the best allowed NON-BLANK class instead, so
		// a diacritic or near-lookalike becomes its plain counterpart (Î -> I, І -> И)
		// rather than vanishing.
		//
		// Blank is excluded deliberately, and this is the whole reason the masking is
		// written as -inf rather than as zeroing the disallowed columns: when the model
		// is very confident about the diacritic, zeroing lets BLANK win and the
		// character is silently deleted.
		bestAllowed, bestScore := -1, math.Inf(-1)
		for c := 0; c < classes; c++ {
			if c == p.blankIndex || p.disallowedIdx[c] {
				continue
			}
			// Strict >, so the FIRST maximum wins — see argmaxFirst.
			if v := float64(row[c]); v > bestScore {
				bestAllowed, bestScore = c, v
			}
		}
		if bestAllowed < 0 {
			// Every class masked away. Cannot happen with the shipped configs (the
			// specials alone keep dozens allowed), but falling back to the unmasked
			// argmax beats emitting a random class.
			bestAllowed = best
		}
		indices[t] = bestAllowed
	}

	// Greedy CTC collapse: drop repeats, then drop blanks.
	var buf []rune
	prev := -1
	for _, idx := range indices {
		if idx != prev && idx != p.blankIndex {
			buf = append(buf, p.alphabet[idx-1])
		}
		prev = idx
	}
	return string(buf), nil
}

// argmaxFirst returns the index of the FIRST maximum, matching np.argmax.
//
// Strict `>` is load-bearing: on a tie a `>=` comparison returns the LAST maximum, which
// selects a different class, which changes a character in the output. Ties are reachable
// because these are quantised softmax outputs (CONVENTIONS §6.2).
func argmaxFirst(row []float32) int {
	best, bestScore := 0, float32(math.Inf(-1))
	for i, v := range row {
		if v > bestScore {
			best, bestScore = i, v
		}
	}
	return best
}

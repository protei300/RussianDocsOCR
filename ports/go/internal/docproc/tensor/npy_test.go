package tensor

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
)

// Round-trip through our own reader and writer. Necessary but NOT sufficient: a
// reader and writer that share a misunderstanding agree with each other perfectly.
// The authoritative check is that NumPy loads what we write, which the conformance
// harness performs on every run — `prepare.npy` is compared against a golden digest
// produced by numpy.
func TestRoundTrip(t *testing.T) {
	cases := []*Array{
		{Dtype: Float32, Shape: []int{3, 4, 5}, F32: seqF32(60)},
		{Dtype: Float64, Shape: []int{2, 3}, F64: []float64{1, -2.5, 3, 4, 5, 6}},
		{Dtype: Uint8, Shape: []int{7, 11}, U8: seqU8(77)},
		{Dtype: Int64, Shape: []int{5}, I64: []int64{-3, -1, 0, 1, 1 << 40}},
		// 1-D: numpy writes a 1-tuple shape with a trailing comma, "(4,)", which a
		// naive parser drops or mis-splits.
		{Dtype: Float32, Shape: []int{4}, F32: []float32{1.5, -0.25, 0, 3.75}},
		// Fixed-width unicode, the dtype centers.npz uses for its labels.
		{Dtype: "<U16", Shape: []int{2}, Strings: []string{"INTPASSPORT_2011", "DL_2011"}},
	}

	for _, want := range cases {
		var buf bytes.Buffer
		if err := Write(&buf, want); err != nil {
			t.Fatalf("write %s%v: %v", want.Dtype, want.Shape, err)
		}
		got, err := Read(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("read %s%v: %v", want.Dtype, want.Shape, err)
		}
		if got.Dtype != want.Dtype {
			t.Errorf("dtype: got %s want %s", got.Dtype, want.Dtype)
		}
		if !equalInts(got.Shape, want.Shape) {
			t.Errorf("shape: got %v want %v", got.Shape, want.Shape)
		}
		if !equalArrays(got, want) {
			t.Errorf("%s%v: values differ", want.Dtype, want.Shape)
		}
	}
}

// The header must be padded so the DATA starts on a 64-byte boundary and the header
// ends with '\n'. Get this wrong and our own reader still accepts the file while
// NumPy rejects it — the failure appears on the far side of the comparison, which is
// the worst place for it.
func TestHeaderIs64ByteAligned(t *testing.T) {
	for _, shape := range [][]int{{1}, {3, 4}, {1, 32, 137, 3}, {9, 1100}} {
		a := &Array{Dtype: Float32, Shape: shape, F32: make([]float32, product(shape))}
		var buf bytes.Buffer
		if err := Write(&buf, a); err != nil {
			t.Fatal(err)
		}
		raw := buf.Bytes()
		headerLen := int(raw[8]) | int(raw[9])<<8
		total := 10 + headerLen
		if total%64 != 0 {
			t.Errorf("shape %v: data begins at %d, not a multiple of 64", shape, total)
		}
		if raw[total-1] != '\n' {
			t.Errorf("shape %v: header does not end with newline", shape)
		}
	}
}

func TestRejectsUnsupportedForms(t *testing.T) {
	// Fortran order must be an error, never a best-effort read: a column-major
	// payload read as row-major is the archetypal "right shape, wrong numbers" bug.
	header := "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }"
	blob := buildNpy(header, make([]byte, 16))
	if _, err := Read(bytes.NewReader(blob)); err == nil {
		t.Error("fortran_order=True was accepted")
	}

	// Version 2.0 uses a 4-byte header length; refuse rather than half-support.
	v2 := buildNpy("{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }", make([]byte, 4))
	v2[6] = 2
	if _, err := Read(bytes.NewReader(v2)); err == nil {
		t.Error("version 2.0 was accepted")
	}
}

// centers.npz is the reason this reader exists in production at all: DocTypeAngles
// loads its centroids from it. The labels are '<U64' — fixed-width UTF-32LE,
// NUL-padded — and naive byte slicing yields "I\0\0\0N\0\0\0T..." instead of
// "INTPASSPORT_2011".
func TestReadsCentersNpz(t *testing.T) {
	root, err := config.ModelsRoot()
	if err != nil {
		t.Skipf("no repository root: %v", err)
	}
	path := filepath.Join(root, "document_processing", "models", "DocTypeAngles",
		"ONNX", "resources", "centers.npz")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("centers.npz not present: %v", err)
	}

	blob, err := LoadNPZ(path)
	if err != nil {
		t.Fatalf("LoadNPZ: %v", err)
	}
	for _, key := range []string{"labels", "centers", "max_distance"} {
		if _, ok := blob[key]; !ok {
			t.Fatalf("centers.npz has no %q; got %v", key, keysOf(blob))
		}
	}

	labels := blob["labels"]
	if len(labels.Strings) == 0 {
		t.Fatal("labels decoded to nothing")
	}
	for i, s := range labels.Strings {
		if s == "" {
			t.Errorf("label[%d] is empty", i)
		}
		// The tell-tale of a byte-wise decode: NULs surviving into the string.
		if bytes.ContainsRune([]byte(s), 0) {
			t.Errorf("label[%d]=%q contains NUL — UTF-32LE decode is wrong", i, s)
		}
	}

	centers := blob["centers"]
	if len(centers.Shape) != 2 || centers.Shape[0] != len(labels.Strings) {
		t.Errorf("centers %v does not align with %d labels", centers.Shape, len(labels.Strings))
	}
	if got := len(blob["max_distance"].F32); got != len(labels.Strings) {
		t.Errorf("max_distance has %d entries, labels %d", got, len(labels.Strings))
	}
}

// ---------------------------------------------------------------- helpers ----

func buildNpy(header string, body []byte) []byte {
	pad := 64 - (10+len(header)+1)%64
	if pad == 64 {
		pad = 0
	}
	full := header
	for i := 0; i < pad; i++ {
		full += " "
	}
	full += "\n"
	out := []byte{0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
		byte(len(full) & 0xFF), byte(len(full) >> 8)}
	out = append(out, []byte(full)...)
	return append(out, body...)
}

func seqF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i) * 0.5
	}
	return out
}

func seqU8(n int) []uint8 {
	out := make([]uint8, n)
	for i := range out {
		out[i] = uint8(i * 3 % 256)
	}
	return out
}

func product(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalArrays(a, b *Array) bool {
	switch a.Dtype {
	case Float32:
		if len(a.F32) != len(b.F32) {
			return false
		}
		for i := range a.F32 {
			if a.F32[i] != b.F32[i] {
				return false
			}
		}
	case Float64:
		if len(a.F64) != len(b.F64) {
			return false
		}
		for i := range a.F64 {
			if math.Float64bits(a.F64[i]) != math.Float64bits(b.F64[i]) {
				return false
			}
		}
	case Uint8:
		return bytes.Equal(a.U8, b.U8)
	case Int64:
		if len(a.I64) != len(b.I64) {
			return false
		}
		for i := range a.I64 {
			if a.I64[i] != b.I64[i] {
				return false
			}
		}
	default:
		if len(a.Strings) != len(b.Strings) {
			return false
		}
		for i := range a.Strings {
			if a.Strings[i] != b.Strings[i] {
				return false
			}
		}
	}
	return true
}

func keysOf[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

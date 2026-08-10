// Package tensor holds the array types the pipeline passes between stages, and the
// strict subset of NumPy's .npy / .npz formats needed to exchange them with the
// Python reference.
//
// The .npy reader is not only a test fixture: DocTypeAngles loads its centroids from
// models/DocTypeAngles/ONNX/resources/centers.npz, which is a zip of three .npy
// members, so production needs it regardless. The normative description of the
// supported subset is conformance/spec/npy-subset.md.
//
// Why .npy at all for the harness, rather than JSON numbers or a flat float32 blob
// with a JSON sidecar describing the shape:
//
//   - The judging side is Python, and there np.load/np.save is zero code. Any
//     custom format puts bug surface on the side that has to be TRUSTED.
//   - It is self-describing. dtype, shape and memory order travel with the
//     bytes, so a transposed array fails loudly instead of "passing". That is
//     exactly the failure mode that would cost a day here: NHWC vs NCHW, or
//     [1,T,C] vs [T,C].
//   - The production port needs this reader anyway. DocTypeAngles loads its
//     centroids from models/DocTypeAngles/ONNX/resources/centers.npz, which is a
//     zip of three .npy members. Writing it during the spike is free and
//     de-risks the port.
//
// Deliberately NOT supported: pickled object arrays (a code-execution vector --
// the Python side already moved off centers.pkl for this reason), big-endian,
// Fortran order, structured dtypes, and .npy v2/v3 headers. Every one of those
// is an error rather than a best-effort guess, because a silent
// misinterpretation here would be scored as a numerical divergence in a model
// three stages downstream.
package tensor

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// DType is the NumPy dtype string, kept as the literal wire spelling ("<f4")
// rather than a Go enum so that what we assert is what NumPy actually wrote.
type DType string

const (
	Float32 DType = "<f4"
	Float64 DType = "<f8"
	Uint8   DType = "|u1"
	Int64   DType = "<i8"
	// Unicode64 is fixed-width UTF-32LE, 64 code points = 256 bytes per row,
	// NUL-padded. centers.npz stores its 9 class labels this way. Naive byte
	// slicing yields "I\0\0\0N\0\0\0T..." -- see R20 in the plan.
	Unicode64 DType = "<U64"
)

// Array is one .npy payload. Exactly one of the typed slices is populated,
// selected by Dtype; Shape is always set.
//
// A struct with parallel optional fields rather than an interface or generics:
// the concrete element type is not known until the header is parsed, so any
// generic design ends in a runtime type switch anyway -- and this shape
// transliterates directly into a C# record and a Kotlin data class.
type Array struct {
	Dtype   DType
	Shape   []int
	F32     []float32
	F64     []float64
	U8      []uint8
	I64     []int64
	Strings []string
}

// Count returns the number of elements implied by Shape. A zero-length Shape is
// a NumPy scalar, which has exactly one element -- not zero.
func (a *Array) Count() int {
	n := 1
	for _, d := range a.Shape {
		n *= d
	}
	return n
}

// AsFloat64 widens whatever is stored into float64 for comparison and printing.
// Comparison happens in Python, so this is only for Go-side sanity output; never
// use it to feed a model, because float32 math must stay float32 (see the
// "float32 throughout" rule in CONVENTIONS).
func (a *Array) AsFloat64() ([]float64, error) {
	switch a.Dtype {
	case Float32:
		out := make([]float64, len(a.F32))
		for i, v := range a.F32 {
			out[i] = float64(v)
		}
		return out, nil
	case Float64:
		return a.F64, nil
	case Uint8:
		out := make([]float64, len(a.U8))
		for i, v := range a.U8 {
			out[i] = float64(v)
		}
		return out, nil
	case Int64:
		out := make([]float64, len(a.I64))
		for i, v := range a.I64 {
			out[i] = float64(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("npy: dtype %q is not numeric", a.Dtype)
}

var magic = []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}

// The header dict is a Python literal, not JSON: single quotes, True/False, and
// a trailing comma in one-element tuples ("(3,)"). Parsing it with three
// regexps is honest about that; using a JSON decoder would be a lie that works
// until it doesn't.
var (
	reDescr   = regexp.MustCompile(`'descr'\s*:\s*'([^']*)'`)
	reFortran = regexp.MustCompile(`'fortran_order'\s*:\s*(True|False)`)
	reShape   = regexp.MustCompile(`'shape'\s*:\s*\(([^)]*)\)`)
)

// Read parses one .npy stream.
func Read(r io.Reader) (*Array, error) {
	head := make([]byte, 10)
	if _, err := io.ReadFull(r, head); err != nil {
		return nil, fmt.Errorf("npy: short magic: %w", err)
	}
	if !bytes.Equal(head[:6], magic) {
		return nil, fmt.Errorf("npy: bad magic %v", head[:6])
	}
	major := head[6]
	if major != 1 {
		// v2 uses a 4-byte header length, v3 adds UTF-8 names. Nothing the
		// reference writes needs them, so refuse rather than half-support.
		return nil, fmt.Errorf("npy: unsupported version %d.%d", head[6], head[7])
	}
	hlen := int(binary.LittleEndian.Uint16(head[8:10]))
	hdr := make([]byte, hlen)
	if _, err := io.ReadFull(r, hdr); err != nil {
		return nil, fmt.Errorf("npy: short header: %w", err)
	}
	dict := string(hdr)

	m := reFortran.FindStringSubmatch(dict)
	if m == nil {
		return nil, fmt.Errorf("npy: no fortran_order in header %q", dict)
	}
	if m[1] != "False" {
		// Assert rather than assume: a column-major payload read as row-major
		// is the single most plausible "wrong numbers, right shape" bug.
		return nil, fmt.Errorf("npy: fortran_order=True is not supported")
	}

	m = reDescr.FindStringSubmatch(dict)
	if m == nil {
		return nil, fmt.Errorf("npy: no descr in header %q", dict)
	}
	descr := m[1]

	m = reShape.FindStringSubmatch(dict)
	if m == nil {
		return nil, fmt.Errorf("npy: no shape in header %q", dict)
	}
	shape, err := parseShape(m[1])
	if err != nil {
		return nil, err
	}

	a := &Array{Dtype: DType(descr), Shape: shape}
	n := a.Count()

	body, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("npy: read body: %w", err)
	}

	switch DType(descr) {
	case Float32:
		if len(body) < n*4 {
			return nil, shortBody(descr, n*4, len(body))
		}
		a.F32 = make([]float32, n)
		for i := range a.F32 {
			a.F32[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[i*4:]))
		}
	case Float64:
		if len(body) < n*8 {
			return nil, shortBody(descr, n*8, len(body))
		}
		a.F64 = make([]float64, n)
		for i := range a.F64 {
			a.F64[i] = math.Float64frombits(binary.LittleEndian.Uint64(body[i*8:]))
		}
	case Int64:
		if len(body) < n*8 {
			return nil, shortBody(descr, n*8, len(body))
		}
		a.I64 = make([]int64, n)
		for i := range a.I64 {
			a.I64[i] = int64(binary.LittleEndian.Uint64(body[i*8:]))
		}
	case Uint8:
		if len(body) < n {
			return nil, shortBody(descr, n, len(body))
		}
		a.U8 = make([]uint8, n)
		copy(a.U8, body[:n])
	default:
		// Fixed-width unicode, e.g. "<U64". Width is in CODE POINTS; each is 4
		// bytes of UTF-32LE, NUL-padded on the right.
		if !strings.HasPrefix(descr, "<U") && !strings.HasPrefix(descr, "|U") {
			return nil, fmt.Errorf("npy: unsupported dtype %q", descr)
		}
		w, err := strconv.Atoi(descr[2:])
		if err != nil {
			return nil, fmt.Errorf("npy: bad unicode width in %q", descr)
		}
		stride := w * 4
		if len(body) < n*stride {
			return nil, shortBody(descr, n*stride, len(body))
		}
		a.Strings = make([]string, n)
		for i := 0; i < n; i++ {
			a.Strings[i] = decodeUTF32LE(body[i*stride : (i+1)*stride])
		}
	}
	return a, nil
}

func shortBody(descr string, want, got int) error {
	return fmt.Errorf("npy: dtype %s wants %d body bytes, got %d", descr, want, got)
}

// decodeUTF32LE turns one fixed-width NumPy unicode cell into a Go string,
// dropping the NUL padding. Surrogate pairs cannot appear in UTF-32, so the
// runes are taken directly; a code point that is out of range or in the
// surrogate block becomes U+FFFD so malformed input stays visible rather than
// silently truncating the label.
func decodeUTF32LE(cell []byte) string {
	runes := make([]rune, 0, len(cell)/4)
	for i := 0; i+4 <= len(cell); i += 4 {
		cp := binary.LittleEndian.Uint32(cell[i : i+4])
		if cp == 0 {
			break // NUL padding: the rest of the cell is padding too.
		}
		if cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF) {
			runes = append(runes, unicode.ReplacementChar)
			continue
		}
		runes = append(runes, rune(cp))
	}
	return string(runes)
}

func parseShape(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []int{}, nil // NumPy scalar: shape ()
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue // trailing comma of a 1-tuple: "(3,)"
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("npy: bad shape component %q", p)
		}
		out = append(out, v)
	}
	return out, nil
}

// Write emits .npy v1.0.
//
// The header must be padded so that the DATA starts on a 64-byte boundary and
// the header itself ends with '\n'. Getting that padding wrong produces a file
// that this reader accepts and NumPy rejects -- so the arithmetic is spelled out
// rather than hidden in a helper.
func Write(w io.Writer, a *Array) error {
	dict := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }",
		a.Dtype, shapeTuple(a.Shape))
	const preamble = 10 // magic(6) + version(2) + header length(2)
	pad := 64 - (preamble+len(dict)+1)%64
	if pad == 64 {
		pad = 0
	}
	full := dict + strings.Repeat(" ", pad) + "\n"

	if _, err := w.Write(magic); err != nil {
		return err
	}
	if _, err := w.Write([]byte{1, 0}); err != nil {
		return err
	}
	var l [2]byte
	binary.LittleEndian.PutUint16(l[:], uint16(len(full)))
	if _, err := w.Write(l[:]); err != nil {
		return err
	}
	if _, err := w.Write([]byte(full)); err != nil {
		return err
	}

	buf := new(bytes.Buffer)
	switch a.Dtype {
	case Float32:
		for _, v := range a.F32 {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			buf.Write(b[:])
		}
	case Float64:
		for _, v := range a.F64 {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
			buf.Write(b[:])
		}
	case Int64:
		for _, v := range a.I64 {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], uint64(v))
			buf.Write(b[:])
		}
	case Uint8:
		buf.Write(a.U8)
	default:
		if !strings.HasPrefix(string(a.Dtype), "<U") && !strings.HasPrefix(string(a.Dtype), "|U") {
			return fmt.Errorf("npy: cannot write dtype %q", a.Dtype)
		}
		width, err := strconv.Atoi(string(a.Dtype)[2:])
		if err != nil {
			return fmt.Errorf("npy: bad unicode width in %q", a.Dtype)
		}
		for _, s := range a.Strings {
			runes := []rune(s)
			if len(runes) > width {
				return fmt.Errorf("npy: string %q exceeds width %d", s, width)
			}
			for _, r := range runes {
				var b [4]byte
				binary.LittleEndian.PutUint32(b[:], uint32(r))
				buf.Write(b[:])
			}
			// NUL-pad the remainder of the fixed-width cell.
			buf.Write(make([]byte, (width-len(runes))*4))
		}
	}
	_, err := w.Write(buf.Bytes())
	return err
}

func shapeTuple(shape []int) string {
	if len(shape) == 0 {
		return ""
	}
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = strconv.Itoa(d)
	}
	if len(shape) == 1 {
		return parts[0] + "," // NumPy writes 1-tuples as "(3,)"
	}
	return strings.Join(parts, ", ")
}

// Load reads a .npy file from disk.
func Load(path string) (*Array, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	a, err := Read(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	return a, nil
}

// Save writes a .npy file, creating parent directories as needed.
//
// filepath.Join everywhere and never a literal separator: the spike has to run
// on Linux as well as this Windows box, and the shipped model.json files already
// contain Windows-style paths that only bite on Linux.
func Save(path string, a *Array) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Write(f, a); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Float32Of builds a float32 array, checking that the data length matches the
// shape -- a mismatch here is a caller bug that would otherwise surface as a
// truncated tensor in a model.
func Float32Of(shape []int, data []float32) (*Array, error) {
	a := &Array{Dtype: Float32, Shape: shape, F32: data}
	if got, want := len(data), a.Count(); got != want {
		return nil, fmt.Errorf("npy: %d float32 values for shape %v (want %d)", got, shape, want)
	}
	return a, nil
}

// Uint8Of builds a uint8 array with the same length check.
func Uint8Of(shape []int, data []uint8) (*Array, error) {
	a := &Array{Dtype: Uint8, Shape: shape, U8: data}
	if got, want := len(data), a.Count(); got != want {
		return nil, fmt.Errorf("npy: %d uint8 values for shape %v (want %d)", got, shape, want)
	}
	return a, nil
}

// Int64Of builds an int64 array with the same length check.
func Int64Of(shape []int, data []int64) (*Array, error) {
	a := &Array{Dtype: Int64, Shape: shape, I64: data}
	if got, want := len(data), a.Count(); got != want {
		return nil, fmt.Errorf("npy: %d int64 values for shape %v (want %d)", got, shape, want)
	}
	return a, nil
}

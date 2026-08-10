package imaging

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Interpolation mirrors the cv2 flags the pipeline actually uses. String constants
// are avoided here because these never appear in model.json — only the tag names do
// (CONVENTIONS §1).
type Interpolation int

const (
	InterLinear  Interpolation = Interpolation(gocv.InterpolationLinear)
	InterArea    Interpolation = Interpolation(gocv.InterpolationArea)
	InterNearest Interpolation = Interpolation(gocv.InterpolationNearestNeighbor)
)

// DecodeRGB loads image bytes and converts to RGB.
//
// This is BasePreprocessing.__call__ (processing/preprocessing.py:48-53) and the
// first half of Pipeline._prepare_image: imdecode(IMREAD_COLOR) gives BGR, and the
// pipeline works in RGB.
//
// Decoding goes through OpenCV, never through Go's image/jpeg. Measured in the
// spike: OpenCV and the standard library disagree by up to 14 LSB on 58-83 % of
// pixels, because libjpeg-turbo's IDCT differs from Go's — enough that "1e-3 on
// numeric outputs" becomes unreachable before inference even starts. Format is
// sniffed from the content, not from a file extension: tests/images/OCRv2 contains
// a file named .png that is not one, and cv2 never cared because it sniffs too.
func DecodeRGB(data []byte) (Image, error) {
	bgr, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return Image{}, fmt.Errorf("imaging: decode: %w", err)
	}
	defer bgr.Close()
	if bgr.Empty() {
		return Image{}, fmt.Errorf("imaging: decoded an empty image (%d bytes)", len(data))
	}
	rgb := gocv.NewMat()
	gocv.CvtColor(bgr, &rgb, gocv.ColorBGRToRGB)
	return Image{mat: rgb}, nil
}

// DecodeSize returns the pixel dimensions of an encoded image, or ok=false if it cannot be
// decoded at all.
//
// It still performs a FULL decode, on purpose: the caller uses this to reject an undecodable
// upload with an immediate, actionable error instead of letting it become a mysterious failed
// job later, and only a real decode proves decodability. What it skips is the BGR->RGB
// conversion, which DecodeRGB owes the pipeline and which a caller wanting two integers does
// not: that conversion is a second full pass over the image, and on a 4032x3024 phone photo
// it is ~36 MB of pointless copying on every upload. Measured on the upload path before this
// existed: ~72 ms per file in Go against ~22 ms in the reference, which does no conversion
// here either (repositories/artifacts.py::decode_dimensions calls imdecode and reads .shape).
func DecodeSize(data []byte) (w, h int, ok bool) {
	mat, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return 0, 0, false
	}
	defer mat.Close()
	if mat.Empty() {
		return 0, 0, false
	}
	return mat.Cols(), mat.Rows(), true
}

// LoadRGB reads a file and decodes it to RGB.
func LoadRGB(path string) (Image, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Image{}, fmt.Errorf("imaging: %w", err)
	}
	return DecodeRGB(data)
}

// Resize scales to an exact (width, height).
//
// Argument order follows cv2.resize's dsize=(w, h), NOT numpy's (h, w). The shipped
// model input sizes are square, which hides an axis swap — the conformance suite
// therefore includes a deliberately non-square resize.
func Resize(src Image, width, height int, interp Interpolation) Image {
	dst := gocv.NewMat()
	gocv.Resize(src.mat, &dst, image.Pt(width, height), 0, 0, gocv.InterpolationFlags(interp))
	return Image{mat: dst}
}

// FitToLongestSide is Pipeline._prepare_image's resize step
// (pipeline/pipeline.py:1107-1111):
//
//	ratio        = max(max(h, w) / imgSize, 1)
//	new_h, new_w = int(h // ratio), int(w // ratio)
//	resize(dsize=(new_w, new_h), INTER_LINEAR)
//
// Two details that must not be "cleaned up":
//
//   - It only ever shrinks. The max(..., 1) means an image smaller than imgSize is
//     left alone, and a port that scales up changes every downstream coordinate.
//   - `h // ratio` is Python's FLOAT floor division, which is **not**
//     `math.Floor(h / ratio)`. CPython computes it via `fmod`, and the two disagree
//     in the last bit on some inputs: a 2999x1777 image resizes to width 1499 in
//     Python and 1500 with the naive formula — a one-pixel-different canvas, and
//     therefore every downstream box shifted. See tensor.FloorDiv, which reproduces
//     CPython's algorithm, and the pinned cases in crop_test.go.
//
// Note the consequence, which is counter-intuitive enough to state: the longest side
// is NOT guaranteed to come out equal to imgSize.
func FitToLongestSide(src Image, imgSize int) Image {
	h, w := src.Height(), src.Width()
	longest := h
	if w > longest {
		longest = w
	}
	ratio := float64(longest) / float64(imgSize)
	if ratio < 1 {
		ratio = 1
	}
	newH := tensor.FloorDivInt(float64(h), ratio)
	newW := tensor.FloorDivInt(float64(w), ratio)
	return Resize(src, newW, newH, InterLinear)
}

// ToBGR flips channel order. The v2 OCR models were trained on OpenCV BGR patches
// while the pipeline works in RGB, so OCRv2Preprocessing performs this flip; it is
// not redundant.
func ToBGR(src Image) Image {
	dst := gocv.NewMat()
	gocv.CvtColor(src.mat, &dst, gocv.ColorRGBToBGR)
	return Image{mat: dst}
}

// ToGray converts RGB to single-channel grey (the deskewer's first step).
func ToGray(src Image) Image {
	dst := gocv.NewMat()
	gocv.CvtColor(src.mat, &dst, gocv.ColorRGBToGray)
	return Image{mat: dst}
}

// Rotate90CCW rotates counter-clockwise. DocTypeAngles applies it angle/90 times to
// bring a document upright.
func Rotate90CCW(src Image) Image {
	dst := gocv.NewMat()
	gocv.Rotate(src.mat, &dst, gocv.Rotate90CounterClockwise)
	return Image{mat: dst}
}

// CopyMakeBorderConstant pads with a constant colour.
//
// The colour is (r, g, b) in the image's own channel order — the YOLO letterbox uses
// (114, 114, 114), where the order does not matter, but a future caller's might.
func CopyMakeBorderConstant(src Image, top, bottom, left, right int, r, g, b uint8) Image {
	dst := gocv.NewMat()
	gocv.CopyMakeBorder(src.mat, &dst, top, bottom, left, right, gocv.BorderConstant,
		color.RGBA{R: r, G: g, B: b, A: 0})
	return Image{mat: dst}
}

// ResizeArea downscales with AREA AVERAGING rather than the InterLinear the pipeline uses
// everywhere else.
//
// A thumbnail is the one place in this project that shrinks by a large factor, and linear
// sampling at 1/10 scale reads one pixel in ten and aliases badly while area averaging reads
// all of them. Not a parity concern — thumbnails are never compared to anything.
func ResizeArea(src Image, width, height int) Image {
	return Resize(src, width, height, InterArea)
}

// WritePNGFromRGB encodes an RGB image to a PNG file.
//
// The conversion is the point: this package's images are RGB and every OpenCV encoder
// expects BGR. Skipping it swaps red and blue in every stored document, and on a passport
// the result looks plausible enough to ship unnoticed.
//
// Written via a temp file whose name KEEPS THE REAL EXTENSION. The encoder is chosen from
// the suffix, so writing to "canvas.png.tmp" fails with "could not find a writer for the
// specified extension" — a trap the reference hit and documented.
func WritePNGFromRGB(path string, rgb Image) error {
	return writeEncoded(path, rgb, ".png", []int{
		int(gocv.IMWritePngCompression), 3,
	})
}

// WriteJPEGFromRGB encodes an RGB image to a JPEG file at the given quality.
func WriteJPEGFromRGB(path string, rgb Image, quality int) error {
	return writeEncoded(path, rgb, ".jpg", []int{
		int(gocv.IMWriteJpegQuality), quality,
	})
}

func writeEncoded(path string, rgb Image, ext string, params []int) error {
	bgr := gocv.NewMat()
	defer bgr.Close()
	gocv.CvtColor(rgb.mat, &bgr, gocv.ColorRGBToBGR)

	dir, base := filepath.Split(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	tmp := filepath.Join(dir, stem+".tmp"+ext)

	if ok := gocv.IMWriteWithParams(tmp, bgr, params); !ok {
		_ = os.Remove(tmp)
		return fmt.Errorf("imaging: encode %s failed", tmp)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("imaging: rename %s: %w", tmp, err)
	}
	return nil
}

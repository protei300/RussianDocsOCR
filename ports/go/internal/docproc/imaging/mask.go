package imaging

import (
	"encoding/binary"
	"image"
	"math"

	"gocv.io/x/gocv"
)

// FloatMask is a single-channel float32 image, used for the segmentation masks between
// the sigmoid and the final threshold.
//
// Held as a Go slice rather than a gocv.Mat so the per-pixel work (the matmul, the
// box zeroing) happens without a cgo call per pixel — at 160x160 proto resolution and
// then 650k pixels after upscaling, per-pixel cgo would dominate the runtime. It
// crosses into OpenCV exactly twice: once to resize, once to threshold.
type FloatMask struct {
	data []float32
	h, w int
}

// NewFloatMask wraps an existing float32 buffer in row-major order. The buffer is not
// copied.
func NewFloatMask(data []float32, h, w int) FloatMask {
	return FloatMask{data: data, h: h, w: w}
}

func (m FloatMask) Height() int { return m.h }
func (m FloatMask) Width() int  { return m.w }

// Crop returns rows [top,bottom) and columns [left,right).
func (m FloatMask) Crop(top, bottom, left, right int) FloatMask {
	h, w := bottom-top, right-left
	out := make([]float32, h*w)
	for y := 0; y < h; y++ {
		copy(out[y*w:(y+1)*w], m.data[(top+y)*m.w+left:(top+y)*m.w+left+w])
	}
	return FloatMask{data: out, h: h, w: w}
}

// Resize scales to an exact size with bilinear interpolation.
//
// The reference writes `cv2.resize(mask, (w, h), cv2.INTER_LINEAR)`, where the third
// POSITIONAL argument of cv2.resize is `dst`, not `interpolation` — so the interpolation
// actually used is the default, which is INTER_LINEAR. The outcome is the same either
// way, which is why the slip is invisible; matched here explicitly rather than copied
// blindly.
func (m FloatMask) Resize(width, height int) FloatMask {
	src, err := gocv.NewMatFromBytes(m.h, m.w, gocv.MatTypeCV32F, float32sToBytes(m.data))
	if err != nil {
		// The dimensions come from our own buffer, so this cannot fail on valid input;
		// returning the mask unchanged keeps the caller's contract rather than
		// propagating an impossible error through five layers.
		return m
	}
	defer src.Close()
	dst := gocv.NewMat()
	defer dst.Close()
	gocv.Resize(src, &dst, image.Pt(width, height), 0, 0, gocv.InterpolationLinear)
	return FloatMask{data: bytesToFloat32s(dst.ToBytes()), h: height, w: width}
}

// ZeroOutsideBox zeroes everything outside the given rectangle.
//
// The comparisons are STRICT — a pixel survives only when x1 < col < x2 and
// y1 < row < y2 — matching clip_boxes' `(r > x1) * (r < x2)`. Using >= would keep one
// extra row and column, which shifts the contour and therefore the rectified crop.
func (m FloatMask) ZeroOutsideBox(x1, y1, x2, y2 float64) {
	for y := 0; y < m.h; y++ {
		insideRow := float64(y) > y1 && float64(y) < y2
		for x := 0; x < m.w; x++ {
			if !insideRow || !(float64(x) > x1 && float64(x) < x2) {
				m.data[y*m.w+x] = 0
			}
		}
	}
}

// Threshold produces an 8-bit 0/255 mask, ready for contour finding.
//
// Strictly greater than the filter, matching `np.where(masks > mask_filter, 1, 0) * 255`.
func (m FloatMask) Threshold(filter float64) Image {
	buf := make([]byte, m.h*m.w)
	for i, v := range m.data {
		if float64(v) > filter {
			buf[i] = 255
		}
	}
	mat, err := gocv.NewMatFromBytes(m.h, m.w, gocv.MatTypeCV8U, buf)
	if err != nil {
		return Image{mat: gocv.NewMatWithSize(m.h, m.w, gocv.MatTypeCV8U)}
	}
	return Image{mat: mat}
}

func float32sToBytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}

func bytesToFloat32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

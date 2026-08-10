package repo

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// Binary artifacts: the uploaded original, the rendered canvas, the thumbnail.
//
// **THIS LAYER STAYS ON THE FILESYSTEM EVEN AFTER A SQL MIGRATION.** Multi-megabyte PNGs do
// not belong in a database — in a real deployment this file grows an S3 implementation, not a
// BLOB column. That is why it is separate from documents.go rather than folded into it.

// magic identifies the formats accepted, keyed by the bytes that actually identify them.
//
// SNIFFED rather than trusting the client's Content-Type, which is attacker-controlled and
// routinely wrong even when it is not.
var magic = []struct {
	prefix []byte
	ext    string
	media  string
}{
	{[]byte{0xff, 0xd8, 0xff}, ".jpg", "image/jpeg"},
	{[]byte("\x89PNG\r\n\x1a\n"), ".png", "image/png"},
	{[]byte("BM"), ".bmp", "image/bmp"},
	{[]byte("II*\x00"), ".tif", "image/tiff"},
	{[]byte("MM\x00*"), ".tif", "image/tiff"},
}

// SniffImage returns the extension and media type for a supported image.
//
// WEBP needs a two-part check — 'RIFF' at 0 and 'WEBP' at 8 — which is why it is not in the
// table.
func SniffImage(data []byte) (ext, media string, ok bool) {
	for _, m := range magic {
		if bytes.HasPrefix(data, m.prefix) {
			return m.ext, m.media, true
		}
	}
	if len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) &&
		bytes.Equal(data[8:12], []byte("WEBP")) {
		return ".webp", "image/webp", true
	}
	return "", "", false
}

// IsPDF is detected separately so the error can say WHY. Users will try PDFs.
func IsPDF(data []byte) bool { return bytes.HasPrefix(data, []byte("%PDF")) }

// DocDir returns the artifact directory, creating it.
func DocDir(db store.DocumentStore, id int) (string, error) {
	dir := db.DocDir(id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("repo: create %s: %w", dir, err)
	}
	return dir, nil
}

// SaveOriginal stores the upload byte-for-byte under a FIXED name.
//
// The client's filename is kept on the record for display only and never touches the
// filesystem — so it cannot be a path-traversal vector no matter what it contains.
func SaveOriginal(db store.DocumentStore, id int, data []byte, ext string) (string, error) {
	dir, err := DocDir(db, id)
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "original"+ext)
	if err := store.AtomicWriteBytes(path, data); err != nil {
		return "", err
	}
	return path, nil
}

// DecodeDimensions returns the upload's width and height, or false if it cannot be decoded.
//
// Done SYNCHRONOUSLY at upload time so an undecodable file becomes an immediate, actionable
// 422 instead of a mysterious failed job minutes later.
// DecodeSize rather than DecodeRGB: the colour conversion DecodeRGB owes the pipeline is a
// second full pass over the image, and nothing here reads a pixel. It was measurable, not
// theoretical — ~72 ms per upload against ~22 ms in the reference.
func DecodeDimensions(data []byte) (w, h int, ok bool) {
	return imaging.DecodeSize(data)
}

// SaveCanvas writes the corrected canvas as PNG and returns its dimensions.
//
// The canvas is RGB and the encoder expects BGR. Skipping the conversion swaps red and blue
// in every displayed document — and the result looks plausible enough on a passport that it
// can ship unnoticed. Hence the explicit conversion here and the regression test asserting a
// known-red pixel stays red.
func SaveCanvas(db store.DocumentStore, id int, rgb imaging.Image) (string, int, int, error) {
	dir, err := DocDir(db, id)
	if err != nil {
		return "", 0, 0, err
	}
	path := filepath.Join(dir, "canvas.png")
	if err := imaging.WritePNGFromRGB(path, rgb); err != nil {
		return "", 0, 0, err
	}
	return path, rgb.Width(), rgb.Height(), nil
}

// SaveThumbnail writes a small JPEG for the list page.
//
// Without it the log page pulls full canvases for every visible row on each three-second
// poll — megabytes per refresh for images rendered at 56 px wide.
func SaveThumbnail(db store.DocumentStore, id int, rgb imaging.Image, width int) (string, error) {
	dir, err := DocDir(db, id)
	if err != nil {
		return "", err
	}
	if width <= 0 {
		width = 96
	}
	height := (rgb.Height()*width + rgb.Width()/2) / rgb.Width()
	if height < 1 {
		height = 1
	}
	small := imaging.ResizeArea(rgb, width, height)
	defer small.Close()

	path := filepath.Join(dir, "thumb.jpg")
	if err := imaging.WriteJPEGFromRGB(path, small, 80); err != nil {
		return "", err
	}
	return path, nil
}

// OpenArtifact returns the path and media type for "original", "canvas" or "thumb".
func OpenArtifact(db store.DocumentStore, id int, kind string) (path, media string, ok bool) {
	dir := db.DocDir(id)
	switch kind {
	case "canvas":
		// PNG for anything this service rendered; JPEG for the pre-computed seed
		// fixtures, which trade exactness for a committable repository footprint.
		for _, c := range []struct{ name, media string }{
			{"canvas.png", "image/png"}, {"canvas.jpg", "image/jpeg"},
		} {
			p := filepath.Join(dir, c.name)
			if isFile(p) {
				return p, c.media, true
			}
		}
		return "", "", false

	case "thumb":
		p := filepath.Join(dir, "thumb.jpg")
		if isFile(p) {
			return p, "image/jpeg", true
		}
		// Falls back to the full canvas rather than 404ing: a missing thumbnail is a
		// performance problem, not a missing document.
		return OpenArtifact(db, id, "canvas")

	case "original":
		matches, _ := filepath.Glob(filepath.Join(dir, "original.*"))
		sort.Strings(matches)
		for _, candidate := range matches {
			if filepath.Ext(candidate) == ".tmp" {
				continue
			}
			head := make([]byte, 16)
			f, err := os.Open(candidate)
			if err != nil {
				continue
			}
			n, _ := f.Read(head)
			_ = f.Close()
			if _, m, ok := SniffImage(head[:n]); ok {
				return candidate, m, true
			}
			return candidate, "application/octet-stream", true
		}
	}
	return "", "", false
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

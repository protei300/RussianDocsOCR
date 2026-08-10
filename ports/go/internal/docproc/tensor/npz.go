package tensor

import (
	"archive/zip"
	"fmt"
	"strings"
)

// LoadNPZ reads a NumPy .npz archive, which is simply a zip whose members are
// .npy files named "<key>.npy".
//
// The production port needs this for exactly one file:
// models/DocTypeAngles/ONNX/resources/centers.npz, holding `labels` (<U64),
// `centers` (float32 9x1100) and `max_distance` (float32 9). It replaced an
// earlier centers.pkl, which was both a code-execution vector and fragile across
// numpy versions -- so this reader deliberately supports only the pickle-free
// form and would fail loudly on anything else.
//
// Uncompressed (np.savez) and deflated (np.savez_compressed) members both work:
// archive/zip handles the stored and deflate methods transparently.
func LoadNPZ(path string) (map[string]*Array, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("npz: open %s: %w", path, err)
	}
	defer zr.Close()

	out := make(map[string]*Array, len(zr.File))
	for _, f := range zr.File {
		key := strings.TrimSuffix(f.Name, ".npy")
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("npz: open member %s: %w", f.Name, err)
		}
		a, err := Read(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("npz: member %s: %w", f.Name, err)
		}
		out[key] = a
	}
	return out, nil
}

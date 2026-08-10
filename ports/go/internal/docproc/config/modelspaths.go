// Package config resolves the library's two data files: the model-directory map and
// the OCR alphabet table.
//
// Both are shared, unmodified, with the Python reference and with every other port —
// they live in document_processing/config/ and are never copied into ports/ (there is
// a CI guard against that: 215 MB of models times four ports is not something to put
// in git history).
package config

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// utf8BOM is written as an escape, never as a literal character: Go rejects a BOM in
// the middle of a source file, so embedding one here would fail to compile. A fitting
// demonstration of why DEVIATIONS D-10 exists.
var utf8BOM = string(rune(0xFEFF))

// ModelsRoot resolves the directory holding document_processing/.
//
// RDOCS_MODELS_ROOT wins if set; otherwise the repository root is located by walking
// up from the executable and then from the working directory, looking for the marker
// directory. Mirrors config/__init__.py's ROOT resolution, and exists for the same
// reason: the CLI is invoked from several places and a cwd-relative path silently
// picks up the wrong models.
func ModelsRoot() (string, error) {
	if v := os.Getenv("RDOCS_MODELS_ROOT"); v != "" {
		if ok, _ := isLibraryRoot(v); ok {
			return v, nil
		}
		return "", fmt.Errorf("config: RDOCS_MODELS_ROOT=%q has no document_processing/models", v)
	}

	var starts []string
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}
	for _, start := range starts {
		dir := start
		for i := 0; i < 8; i++ {
			if ok, _ := isLibraryRoot(dir); ok {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return "", fmt.Errorf("config: could not locate document_processing/models; " +
		"set RDOCS_MODELS_ROOT to the repository root")
}

func isLibraryRoot(dir string) (bool, error) {
	info, err := os.Stat(filepath.Join(dir, "document_processing", "models"))
	if err != nil {
		return false, err
	}
	return info.IsDir(), nil
}

// ModelPaths maps a module name to its model directory, e.g.
// "DocDetector" -> "<root>/document_processing/models/Borders".
type ModelPaths struct {
	root  string
	paths map[string]string
}

// LoadModelPaths reads document_processing/config/models_path.yaml.
//
// A hand-written parser rather than a YAML dependency: the file is fourteen lines of
// flat `key: value` with no nesting, lists or quoting. Adding a YAML library to four
// languages to read that would be a poor trade, and this is the same decision each
// port should make.
func LoadModelPaths(root string) (*ModelPaths, error) {
	path := filepath.Join(root, "document_processing", "config", "models_path.yaml")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	defer f.Close()

	out := make(map[string]string)
	sc := bufio.NewScanner(f)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		// A UTF-8 BOM on the first line would otherwise become part of the first
		// key, and every lookup would then miss. See DEVIATIONS D-10.
		if line == 1 {
			text = strings.TrimPrefix(text, utf8BOM)
		}
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		key, value, ok := strings.Cut(text, ":")
		if !ok {
			return nil, fmt.Errorf("config: %s:%d: cannot parse %q",
				filepath.Base(path), line, text)
		}
		out[strings.TrimSpace(key)] = strings.TrimSpace(value)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("config: %s: %w", filepath.Base(path), err)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("config: %s has no entries", filepath.Base(path))
	}
	return &ModelPaths{root: root, paths: out}, nil
}

// Dir returns the artifact directory for a module, for the given format subfolder
// (always "ONNX" today).
//
// The stored values are Windows-style (`models\Borders`) because that is what ships.
// On Linux a backslash is an ordinary filename character, so without this
// normalisation DocTypeAngles dies at construction — only inside a container, never
// on a Windows dev box. Python normalises in code rather than re-shipping the data,
// and every port must do the same (CONVENTIONS §2).
func (m *ModelPaths) Dir(module, format string) (string, error) {
	rel, ok := m.paths[module]
	if !ok {
		return "", fmt.Errorf("config: models_path.yaml has no entry for %q", module)
	}
	rel = filepath.FromSlash(strings.ReplaceAll(rel, `\`, "/"))
	return filepath.Join(m.root, "document_processing", rel, format), nil
}

// Modules lists the registered module names.
func (m *ModelPaths) Modules() []string {
	out := make([]string, 0, len(m.paths))
	for k := range m.paths {
		out = append(out, k)
	}
	return out
}

// NormalizeRelPath applies the same backslash rule to any path taken from a data
// file — notably model.json's `"Centers": "resources\\centers.npz"`.
func NormalizeRelPath(p string) string {
	return filepath.FromSlash(strings.ReplaceAll(p, `\`, "/"))
}

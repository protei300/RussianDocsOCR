package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// alphabetTable mirrors document_processing/config/ocr_alphabets.json.
//
// This file constrains only which characters a decode step may EMIT. The model's own
// full alphabet lives in its model.json; the two are different things and conflating
// them silently disables masking.
type alphabetTable struct {
	Description       string                       `json:"description"`
	Specials          string                       `json:"specials"`
	DefaultCountry    map[string]string            `json:"default_country"`
	LettersPerCountry map[string]map[string]string `json:"letters_per_country"`
}

var (
	alphabetOnce  sync.Once
	alphabetData  *alphabetTable
	alphabetError error
)

// LoadAlphabets reads and caches the alphabet table.
//
// Cached because `allowed_charset` is consulted once per OCR call and the file never
// changes at runtime — matching the lru_cache on the Python side.
func LoadAlphabets(root string) (*alphabetTable, error) {
	alphabetOnce.Do(func() {
		path := filepath.Join(root, "document_processing", "config", "ocr_alphabets.json")
		raw, err := os.ReadFile(path)
		if err != nil {
			alphabetError = fmt.Errorf("config: %w", err)
			return
		}
		var t alphabetTable
		// No BOM tolerance on purpose: encoding/json rejects one, and a config that
		// has acquired a BOM is corrupt rather than merely unusual (D-10).
		if err := json.Unmarshal(raw, &t); err != nil {
			alphabetError = fmt.Errorf("config: %s: %w", filepath.Base(path), err)
			return
		}
		if t.Specials == "" || len(t.LettersPerCountry) == 0 {
			alphabetError = fmt.Errorf("config: %s is missing specials or letters_per_country",
				filepath.Base(path))
			return
		}
		alphabetData = &t
	})
	return alphabetData, alphabetError
}

// DefaultCountry returns the country whose alphabet a script uses when none is given
// (cyrillic -> RUS, latin -> USA).
//
// An unknown script is an error, not a silent fallback: Python raises KeyError here
// deliberately, because guessing an alphabet would corrupt text rather than fail.
func DefaultCountry(root, script string) (string, error) {
	t, err := LoadAlphabets(root)
	if err != nil {
		return "", err
	}
	country, ok := t.DefaultCountry[script]
	if !ok {
		return "", fmt.Errorf("config: no default country for script %q", script)
	}
	return country, nil
}

// AllowedCharset resolves the set of characters a decode step may emit: the country's
// letters plus the shared specials (digits and ASCII punctuation, always allowed).
//
// Pass an empty country to use the script's default. Returned as a rune set because
// the caller masks per class index and needs membership tests, not order.
//
// Note the Cyrillic alphabets are UPPERCASE only — the printed-document models were
// trained that way. A handwriting model would need a case-sensitive entry, which is
// why this table is keyed by script AND country rather than hardcoded.
func AllowedCharset(root, script, country string) (map[rune]bool, error) {
	t, err := LoadAlphabets(root)
	if err != nil {
		return nil, err
	}
	if country == "" {
		if country, err = DefaultCountry(root, script); err != nil {
			return nil, err
		}
	}
	byCountry, ok := t.LettersPerCountry[script]
	if !ok {
		return nil, fmt.Errorf("config: unknown script %q", script)
	}
	letters, ok := byCountry[country]
	if !ok {
		return nil, fmt.Errorf("config: script %q has no country %q", script, country)
	}

	set := make(map[rune]bool, len(letters)+len(t.Specials))
	// []rune, never bytes: the Cyrillic alphabets are multi-byte UTF-8 and byte
	// iteration would add fragments of characters to the set (CONVENTIONS §6).
	for _, r := range letters {
		set[r] = true
	}
	for _, r := range t.Specials {
		set[r] = true
	}
	return set, nil
}

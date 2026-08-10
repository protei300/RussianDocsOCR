package repo

import (
	"crypto/subtle"
	"sort"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// DefaultKeyID is reserved for the environment key so it never collides with a stored one.
const DefaultKeyID = 0

// The environment-provided default key is SYNTHESISED AT STARTUP, NOT STORED.
//
// That keeps one awkward case honest: runtime-created keys live in the ephemeral store and
// vanish on restart, so if the default were merely stored too, a restart could leave the API
// with no working credential at all. Deriving it from the environment on every boot means it
// is always present — and deleting it is refused rather than silently undone by the next
// restart.
func defaultKey(cfg auth.Config) (*model.ApiKey, bool, error) {
	raw, generated, err := auth.ResolveDefaultKey(cfg)
	if err != nil {
		return nil, false, err
	}
	label := "Default (environment)"
	if generated {
		label = "Default (generated at startup)"
	}
	return &model.ApiKey{
		ID:        DefaultKeyID,
		Label:     label,
		Prefix:    auth.Prefix(raw),
		KeyHash:   auth.HashApiKey(raw),
		IsDefault: true,
	}, generated, nil
}

// AllApiKeys returns every usable key, default first.
func AllApiKeys(db store.DocumentStore, cfg auth.Config) ([]*model.ApiKey, error) {
	def, _, err := defaultKey(cfg)
	if err != nil {
		return nil, err
	}
	stored := db.AllApiKeys()
	sort.SliceStable(stored, func(a, b int) bool {
		if stored[a].CreatedAt.Set && stored[b].CreatedAt.Set {
			return stored[a].CreatedAt.Time.Before(stored[b].CreatedAt.Time)
		}
		return stored[a].ID < stored[b].ID
	})
	return append([]*model.ApiKey{def}, stored...), nil
}

// CreateApiKey mints a key. Returns the record and the PLAINTEXT, which is shown once.
func CreateApiKey(db store.DocumentStore, label string) (*model.ApiKey, string, error) {
	raw, err := auth.GenerateApiKey()
	if err != nil {
		return nil, "", err
	}
	name := strings.TrimSpace(label)
	if name == "" {
		name = "Unnamed key"
	}
	id := db.NextApiKeyID()
	if id == DefaultKeyID {
		// Never shadow the environment key: id 0 is reserved, and a stored key holding it
		// would make the default unreachable through the API.
		id = 1
	}
	rec := &model.ApiKey{
		ID:        id,
		Label:     name,
		Prefix:    auth.Prefix(raw),
		KeyHash:   auth.HashApiKey(raw),
		IsDefault: false,
		CreatedAt: model.At(model.UtcNow()),
	}
	if _, err := db.PutApiKey(rec); err != nil {
		return nil, "", err
	}
	return rec, raw, nil
}

func DeleteApiKey(db store.DocumentStore, id int) (bool, error) { return db.DropApiKey(id) }

// VerifyApiKey matches a presented key against every known hash, in CONSTANT TIME.
//
// A constant-time compare per candidate rather than a map lookup: `==` on a digest leaks
// how much of it matched through timing. Returning early on a match is fine — what must not
// vary with the secret is the comparison itself; the NUMBER of configured keys is not
// secret, and the list is tiny anyway.
func VerifyApiKey(db store.DocumentStore, cfg auth.Config, candidate string) (*model.ApiKey, error) {
	if candidate == "" {
		return nil, nil
	}
	digest := auth.HashApiKey(candidate)
	keys, err := AllApiKeys(db, cfg)
	if err != nil {
		return nil, err
	}
	for _, key := range keys {
		if subtle.ConstantTimeCompare([]byte(digest), []byte(key.KeyHash)) == 1 {
			return key, nil
		}
	}
	return nil, nil
}

// TouchApiKey records last use. The environment key is not persisted, so it is skipped.
func TouchApiKey(db store.DocumentStore, key *model.ApiKey) {
	if key.IsDefault {
		return
	}
	key.LastUsedAt = model.At(model.UtcNow())
	_, _ = db.PutApiKey(key)
}

// PublicApiKeys is the list for the UI.
//
// The GENERATED default is returned IN FULL: it exists only in this process's memory, so
// masking it would make it unusable — the operator would have no way to learn a key the
// service invented. A key supplied via DEFAULT_API_KEY stays masked, because whoever set it
// already has it and echoing a configured secret back into a browser is gratuitous.
func PublicApiKeys(db store.DocumentStore, cfg auth.Config) ([]map[string]any, error) {
	raw, generated, err := auth.ResolveDefaultKey(cfg)
	if err != nil {
		return nil, err
	}
	keys, err := AllApiKeys(db, cfg)
	if err != nil {
		return nil, err
	}
	out := make([]map[string]any, 0, len(keys))
	for _, key := range keys {
		entry := key.Public()
		if key.IsDefault {
			entry["is_generated"] = generated
			if generated {
				entry["key"] = raw
			}
		}
		out = append(out, entry)
	}
	return out, nil
}

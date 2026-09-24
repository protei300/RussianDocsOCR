package repo

import (
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// The action log: who did what, to which object, when.
//
// Thin on purpose — the rules that matter live in model.AuditEntry (no personal data) and in
// FileStore.AppendAudit (append a line, never let a logging failure break the action being
// logged).
//
// The actor is a STRING rather than a user so the PIN path can use it too: in PIN mode there is
// no account, and the actor is the literal "pin". One log covers both modes instead of a second
// mechanism that exists in only one of them.
//
// Port of service/repositories/audit.py.

// Audit records one action. TargetID is a string on disk and on the wire; pass an id formatted
// with strconv.Itoa, never a filename.
func Audit(db store.DocumentStore, action, actor, targetType, targetID, detail string) model.AuditEntry {
	if actor == "" {
		actor = "anonymous"
	}
	return db.AppendAudit(model.AuditEntry{
		Action:     action,
		Actor:      actor,
		TargetType: targetType,
		TargetID:   targetID,
		Detail:     detail,
		At:         model.At(model.StampNow()),
	})
}

// RecentAudit returns the newest `limit` entries matching the filters, newest first.
func RecentAudit(db store.DocumentStore, limit int, action, actor string) []model.AuditEntry {
	return db.RecentAudit(limit, action, actor)
}

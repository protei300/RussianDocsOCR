package net.russiandocs.service.repositories

import net.russiandocs.service.model.AuditEntry
import net.russiandocs.service.store.DocumentStore

/**
 * The action log: who did what, to which object, when.
 *
 * Thin on purpose — the rules that matter live in [AuditEntry] (no personal data) and in the store's
 * `appendAudit` (append a line, never let a logging failure break the action being logged).
 *
 * The actor is a STRING rather than a [net.russiandocs.service.model.User] so the PIN path uses the same
 * log: in PIN mode there is no account and the actor is the literal `pin`.
 *
 * Port of `service/repositories/audit.py`.
 */
public object Audit {

    public fun record(
        db: DocumentStore,
        action: String,
        actor: String = "",
        targetType: String = "",
        targetId: Any? = null,
        detail: String = "",
    ): AuditEntry = db.appendAudit(AuditEntry(
        id = 0,                                 // assigned under the store's lock
        action = action,
        actor = actor.ifEmpty { "anonymous" },
        targetType = targetType,
        targetId = targetId?.toString() ?: "",
        detail = detail,
    ))

    public fun recent(db: DocumentStore, limit: Int, action: String, actor: String): List<AuditEntry> =
        db.recentAudit(limit, action, actor)
}

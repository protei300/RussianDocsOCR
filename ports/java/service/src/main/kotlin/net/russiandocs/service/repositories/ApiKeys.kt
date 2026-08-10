package net.russiandocs.service.repositories

import java.security.MessageDigest
import java.time.Instant
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.model.ApiKey
import net.russiandocs.service.model.Timestamps
import net.russiandocs.service.store.DocumentStore

/**
 * The repository for API keys. Signatures copied from `service/repositories/api_keys.py`.
 */
public object ApiKeys {

    /** Reserved for the environment key so it never collides with a stored one. */
    public const val DEFAULT_KEY_ID: Int = 0

    /**
     * The environment-provided default key is SYNTHESISED AT STARTUP, NOT STORED.
     *
     * That keeps one awkward case honest: runtime-created keys live in the ephemeral store and vanish on
     * restart, so if the default were merely stored too, a restart could leave the API with no working
     * credential at all. Deriving it from the environment on every boot means it is always present — and
     * deleting it is refused rather than silently undone by the next restart.
     */
    private fun defaultKey(cfg: Tokens.Config): Pair<ApiKey, Boolean> {
        val (raw, generated) = Tokens.resolveDefaultKey(cfg)
        return ApiKey(
            id = DEFAULT_KEY_ID,
            label = if (generated) "Default (generated at startup)" else "Default (environment)",
            prefix = Tokens.prefix(raw),
            keyHash = Tokens.hashApiKey(raw),
            isDefault = true,
        ) to generated
    }

    /** Every usable key, default first. */
    public fun all(db: DocumentStore, cfg: Tokens.Config): List<ApiKey> {
        val (def, _) = defaultKey(cfg)
        // sortedWith is stable, which matters: two keys created in the same clock tick must not swap
        // places between requests.
        val stored = db.allApiKeys()
            .sortedWith(compareBy({ it.createdAt ?: Instant.MAX }, { it.id }))
        return listOf(def) + stored
    }

    /** Mints a key. Returns the record and the PLAINTEXT, which is shown once. */
    public fun create(db: DocumentStore, label: String): Pair<ApiKey, String> {
        val raw = Tokens.generateApiKey()
        val name = label.trim().ifEmpty { "Unnamed key" }
        var id = db.nextApiKeyId()
        if (id == DEFAULT_KEY_ID) {
            // Never shadow the environment key: id 0 is reserved, and a stored key holding it would make
            // the default unreachable through the API.
            id = 1
        }
        val record = ApiKey(
            id = id,
            label = name,
            prefix = Tokens.prefix(raw),
            keyHash = Tokens.hashApiKey(raw),
            isDefault = false,
            createdAt = Timestamps.now(),
        )
        db.putApiKey(record)
        return record to raw
    }

    public fun delete(db: DocumentStore, id: Int): Boolean = db.dropApiKey(id)

    /**
     * Matches a presented key against every known hash, in CONSTANT TIME.
     *
     * A constant-time compare per candidate rather than a map lookup: ordinary equality on a digest leaks
     * how much of it matched through timing. Returning early on a match is fine — what must not vary with
     * the secret is the comparison itself; the NUMBER of configured keys is not secret, and the list is
     * tiny anyway.
     */
    public fun verify(db: DocumentStore, cfg: Tokens.Config, candidate: String): ApiKey? {
        if (candidate.isEmpty()) {
            return null
        }
        val digest = Tokens.hashApiKey(candidate).toByteArray(Charsets.UTF_8)
        for (key in all(db, cfg)) {
            if (MessageDigest.isEqual(digest, key.keyHash.toByteArray(Charsets.UTF_8))) {
                return key
            }
        }
        return null
    }

    /** Records last use. The environment key is not persisted, so it is skipped. */
    public fun touch(db: DocumentStore, key: ApiKey) {
        if (key.isDefault) {
            return
        }
        db.putApiKey(key.copy(lastUsedAt = Timestamps.now()))
    }

    /**
     * The list for the UI.
     *
     * **The GENERATED default is returned IN FULL**: it exists only in this process's memory, so masking
     * it would make it unusable — the operator would have no way to learn a key the service invented. A
     * key supplied via `DEFAULT_API_KEY` stays masked, because whoever set it already has it and echoing a
     * configured secret back into a browser is gratuitous.
     */
    public fun public(db: DocumentStore, cfg: Tokens.Config): List<Map<String, Any?>> {
        val (raw, generated) = Tokens.resolveDefaultKey(cfg)
        return all(db, cfg).map { key ->
            val entry = LinkedHashMap(key.public())
            if (key.isDefault) {
                entry["is_generated"] = generated
                if (generated) {
                    entry["key"] = raw
                }
            }
            entry
        }
    }
}

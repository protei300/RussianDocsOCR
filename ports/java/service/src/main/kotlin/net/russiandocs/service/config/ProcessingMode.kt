package net.russiandocs.service.config

/**
 * Which processing mode is in force: documents stored, or recognised in memory only.
 *
 * `PROCESSING_MODE=stored` (the default) is the service as it always was. `PROCESSING_MODE=memory` is the mode
 * for deployments that must not keep personal data at all: the document routes, the background worker and the
 * seeding are off, and `POST /recognize` is the only way a document gets in — decoded, recognised and answered
 * inside the request, with nothing written.
 *
 * **Resolution never fails, and a wrong value fails toward privacy** — the opposite of `AUTH_MODE`, and the
 * difference is deliberate: somebody who typed `memroy` asked for documents not to be stored, and quietly
 * storing them would break the one promise the setting exists to make.
 *
 * Configuration only, never a runtime setting: a switch on a web page would let an admin turn storage back on
 * for a service that promised not to store.
 *
 * Port of `service/core/processing.py`; the contract is ports/MEMORY.md §1.
 */
public object ProcessingMode {
    public const val STORED: String = "stored"
    public const val MEMORY: String = "memory"
    private val MODES = listOf(STORED, MEMORY)

    /** The effective mode, and why it is not what was configured (null when it is). */
    public data class Resolved(val mode: String, val reason: String?) {
        /** True when documents are stored — the document routes, the worker and seeding are on. */
        val storing: Boolean get() = mode == STORED
    }

    /** The ONE reading of `PROCESSING_MODE`. Nothing else looks at the raw value. */
    public fun resolve(raw: String?): Resolved {
        val value = (raw ?: "").trim().lowercase()
        if (value.isEmpty()) {
            return Resolved(STORED, null)
        }
        if (value in MODES) {
            return Resolved(value, null)
        }
        return Resolved(MEMORY,
            "PROCESSING_MODE='$value' is not one of ${MODES.joinToString(", ")} — using 'memory' (documents " +
                "are not stored), the safe reading of a mistyped privacy setting")
    }
}

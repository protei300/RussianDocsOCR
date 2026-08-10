package net.russiandocs.service.errors

/**
 * The seven service error kinds — one per genuinely different caller reaction.
 *
 * D-02 licenses exceptions here where Go returns `(T, error)`. What must NOT change is the TAXONOMY: seven
 * kinds, mapped to seven status codes in exactly one place, so a handler never picks a status code. That is
 * what keeps 401-versus-403 and 409-versus-400 consistent across a dozen endpoints.
 */
public enum class ErrorKind {
    /** The lease could not be acquired in time. Transient — the job returns to the queue. */
    PIPELINE_BUSY,

    /** The upload decoded to nothing. NOT transient: retrying a corrupt JPEG is pointless. */
    IMAGE_UNREADABLE,

    /** Models are still loading, or failed to load. Transient. */
    RUNTIME_NOT_READY,

    NOT_FOUND,
    UNAUTHORIZED,
    CONFLICT,
    BAD_REQUEST,
}

/**
 * A service error carrying its kind and a CLIENT-FACING message.
 *
 * The message is separate from the kind on purpose. The Go port shipped a real defect by folding the two
 * together — wrapping with `fmt.Errorf("%w: msg", sentinel)` put the sentinel's own name into the response
 * body, so a 409 read as `"conflict: The default key ..."`. Here [message] is what the user reads and [kind]
 * is what selects the status.
 */
public class ServiceException(
    public val kind: ErrorKind,
    message: String,
) : RuntimeException(message) {

    /**
     * Whether retrying could plausibly succeed.
     *
     * **An UNKNOWN error counts as non-transient**, and that default is deliberate: an unknown error retried
     * forever stops the queue and leaves nothing in the log explaining why. Only busy and not-ready are
     * retried.
     */
    public val transient: Boolean
        get() = kind == ErrorKind.PIPELINE_BUSY || kind == ErrorKind.RUNTIME_NOT_READY

    public companion object {
        public fun notFound(message: String = "Not found"): ServiceException =
            ServiceException(ErrorKind.NOT_FOUND, message)

        public fun unauthorized(message: String = "Not authenticated"): ServiceException =
            ServiceException(ErrorKind.UNAUTHORIZED, message)

        public fun conflict(message: String): ServiceException =
            ServiceException(ErrorKind.CONFLICT, message)

        public fun badRequest(message: String): ServiceException =
            ServiceException(ErrorKind.BAD_REQUEST, message)

        public fun busy(message: String): ServiceException =
            ServiceException(ErrorKind.PIPELINE_BUSY, message)

        public fun notReady(message: String): ServiceException =
            ServiceException(ErrorKind.RUNTIME_NOT_READY, message)

        public fun unreadable(message: String): ServiceException =
            ServiceException(ErrorKind.IMAGE_UNREADABLE, message)
    }
}

/**
 * Whether an arbitrary throwable should be retried.
 *
 * Separate from [ServiceException.transient] because the worker also sees exceptions from the library and the
 * filesystem, which are not ServiceExceptions at all.
 */
public object Transience {
    public fun isTransient(error: Throwable): Boolean = when (error) {
        is ServiceException -> error.transient
        // An I/O error is worth one retry: a locked file or a full pipe can clear.
        is java.io.IOException -> true
        is java.util.concurrent.TimeoutException -> true
        // Everything else — including a decode failure and a bug — is not.
        else -> false
    }
}

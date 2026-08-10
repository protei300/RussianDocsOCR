namespace RussianDocs.Service.Errors;

/// <summary>
/// The seven service error kinds — one per genuinely different caller reaction.
///
/// <para>
/// D-02 licenses exceptions here where Go returns <c>(T, error)</c>. What must NOT change is the
/// TAXONOMY: seven kinds, mapped to seven status codes in exactly one place, so a handler never picks
/// a status code. That is what keeps 401-versus-403 and 409-versus-400 consistent across a dozen
/// endpoints.
/// </para>
/// </summary>
public enum ErrorKind
{
    /// <summary>The lease could not be acquired in time. Transient — the job returns to the queue.</summary>
    PipelineBusy,

    /// <summary>The upload decoded to nothing. NOT transient: retrying a corrupt JPEG is pointless.</summary>
    ImageUnreadable,

    /// <summary>Models are still loading, or failed to load. Transient.</summary>
    RuntimeNotReady,

    NotFound,
    Unauthorized,
    Conflict,
    BadRequest,
}

/// <summary>
/// A service error carrying its kind and a CLIENT-FACING message.
///
/// <para>
/// The message is separate from the kind on purpose. The Go port shipped a real defect by folding the
/// two together — wrapping with <c>fmt.Errorf("%w: msg", sentinel)</c> put the sentinel's own name
/// into the response body, so a 409 read as <c>"conflict: The default key ..."</c>. Here
/// <see cref="Exception.Message"/> is what the user reads and <see cref="Kind"/> is what selects the
/// status.
/// </para>
/// </summary>
public sealed class ServiceException(ErrorKind kind, string message)
    : Exception(message)
{
    public ErrorKind Kind { get; } = kind;

    /// <summary>
    /// Whether retrying could plausibly succeed.
    ///
    /// <para>
    /// **An UNKNOWN error counts as non-transient**, and that default is deliberate: an unknown error
    /// retried forever stops the queue and leaves nothing in the log explaining why. Only busy and
    /// not-ready are retried.
    /// </para>
    /// </summary>
    public bool Transient => Kind is ErrorKind.PipelineBusy or ErrorKind.RuntimeNotReady;

    public static ServiceException NotFound(string message = "Not found") =>
        new(ErrorKind.NotFound, message);

    public static ServiceException Unauthorized(string message = "Not authenticated") =>
        new(ErrorKind.Unauthorized, message);

    public static ServiceException Conflict(string message) => new(ErrorKind.Conflict, message);

    public static ServiceException BadRequest(string message) => new(ErrorKind.BadRequest, message);

    public static ServiceException Busy(string message) => new(ErrorKind.PipelineBusy, message);

    public static ServiceException NotReady(string message) => new(ErrorKind.RuntimeNotReady, message);

    public static ServiceException Unreadable(string message) =>
        new(ErrorKind.ImageUnreadable, message);
}

/// <summary>
/// Whether an arbitrary exception should be retried.
///
/// <para>
/// Separate from <see cref="ServiceException.Transient"/> because the worker also sees exceptions from
/// the library and the filesystem, which are not ServiceExceptions at all.
/// </para>
/// </summary>
public static class Transience
{
    public static bool IsTransient(Exception error) => error switch
    {
        ServiceException service => service.Transient,
        // An I/O error is worth one retry: a locked file or a full pipe can clear.
        IOException => true,
        TimeoutException => true,
        // Everything else — including a decode failure and a bug — is not.
        _ => false,
    };
}

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// The one parallel primitive. Every concurrent group in the pipeline goes through it.
/// </summary>
public static class Group
{
    /// <summary>
    /// Runs a fixed set of tasks concurrently and collects their results POSITIONALLY.
    ///
    /// <para>
    /// The shape is mandated by CONVENTIONS §3 and must be preserved across ports: one launch per
    /// member in the reference's source order, one join, one deterministic collection indexed by
    /// position.
    /// </para>
    ///
    /// <para>
    /// **Why positional and not a completion-ordered collection:** the reference collects with
    /// <c>futures[i].result()</c>, which is ordered by construction. A version that appends as tasks
    /// finish returns results in completion order, which varies with load — and that reorders boxes,
    /// reorders words, and changes the joined field string. It is an exact-match conformance failure
    /// with no float anywhere near it, and it only appears under concurrency, which is the worst way
    /// to find a bug.
    /// </para>
    ///
    /// <para>
    /// **On failure the PARTIAL results are still returned**, deliberately. When <typeparamref
    /// name="T"/> owns an unmanaged resource — the word-splitting group's does — a task that
    /// succeeded before a sibling failed has already allocated, and swallowing its result makes
    /// cleanup impossible for the caller. Python did not need this because its GC collected the
    /// abandoned futures; a port that discards them leaks. Callers that ignore results on error stay
    /// correct, so this cannot hurt the quality group, whose T is a plain verdict.
    /// </para>
    /// </summary>
    public static (T?[] Results, Exception? Error) Run<T>(int limit, IReadOnlyList<Func<T>> tasks)
    {
        var results = new T?[tasks.Count];
        var running = new Task[tasks.Count];

        // A limit of 0 means unlimited. The word-splitting group passes min(8, n) to match the
        // reference's ThreadPoolExecutor(max_workers=8); the quality group has four members and
        // passes 0.
        using var gate = limit > 0 ? new SemaphoreSlim(limit) : null;

        for (int i = 0; i < tasks.Count; i++)
        {
            int index = i; // captured per iteration, or every task writes to the last slot
            Func<T> task = tasks[i];
            running[i] = Task.Run(() =>
            {
                gate?.Wait();
                try
                {
                    // Written BY INDEX into a pre-sized array: no append, no lock, no ordering
                    // question.
                    results[index] = task();
                }
                finally
                {
                    gate?.Release();
                }
            });
        }

        try
        {
            Task.WhenAll(running).GetAwaiter().GetResult();
            return (results, null);
        }
        catch (Exception ex)
        {
            // The FIRST error wins, matching errgroup in Go and the reference's own
            // aggregate-then-raise-first. `results` is returned regardless — see above.
            Exception first = running.Select(t => t.Exception)
                                    .OfType<AggregateException>()
                                    .Select(a => a.Flatten().InnerExceptions[0])
                                    .FirstOrDefault() ?? ex;
            return (results, first);
        }
    }

    /// <summary>
    /// <c>min(cap, n)</c>, spelled out so the three ports read alike rather than each inlining it.
    /// </summary>
    public static int MinLimit(int cap, int n) => Math.Min(cap, n);
}

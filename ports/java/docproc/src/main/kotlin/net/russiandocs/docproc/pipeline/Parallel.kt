package net.russiandocs.docproc.pipeline

import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * The one sanctioned fan-out/join, and the shape is normative rather than a matter of taste.
 *
 * **Obligatory form, identical in all four ports:** one launch per member IN THE ORDER OF THE PYTHON
 * SOURCE, one join, one deterministic POSITIONAL collection, one timing entry for the group as a whole.
 *
 * The positional part is the whole point. The reference collects `futures[i].result()`, which is
 * positional and therefore deterministic; a version that appends results as they finish reorders under
 * load and changes the order of boxes, of words, and of the joined field text. That is an exact-match
 * failure with no float anywhere in it — the hardest kind to attribute, because the code looks right and
 * fails intermittently.
 *
 * ### Why not coroutines
 *
 * The plan named `coroutineScope` + `awaitAll`, and this uses an [ExecutorService] instead. The reason is
 * concrete rather than stylistic: every member of this group blocks inside NATIVE code — ONNX Runtime and
 * OpenCV — for tens of milliseconds. A coroutine that blocks its thread gains nothing over a thread and
 * needs `Dispatchers.IO` plus `runBlocking` at the boundary to be called from ordinary code, since the
 * pipeline is not suspending and must not become so (the reference is synchronous, and every port keeps
 * the call graph comparable). That is a dependency and two concepts bought for no benefit.
 *
 * `invokeAll` returns futures **in the order the tasks were submitted**, which gives the positional
 * collection for free and by construction rather than by discipline.
 */
public object Parallel {

    /**
     * Runs every task and returns the results POSITIONALLY.
     *
     * The pool is created and shut down per call. That sounds wasteful and is not measurable here — the
     * group runs once per document against tasks taking tens of milliseconds — and it buys the property
     * that matters: no thread outlives the call, so a pipeline instance holds no background state and
     * closing it cannot leave a pool running. A shared pool would also have to be sized, and sizing it is
     * how Python ended up with 32 threads racing for one lease.
     */
    public fun <T> run(tasks: List<() -> T>): List<T> {
        if (tasks.isEmpty()) {
            return emptyList()
        }
        if (tasks.size == 1) {
            // No pool for a single task: it would be pure overhead, and the semantics are identical.
            return listOf(tasks[0]())
        }

        val pool: ExecutorService = Executors.newFixedThreadPool(tasks.size) { runnable ->
            Thread(runnable, "rdocs-parallel").apply {
                // Daemon, so a task wedged in native code cannot keep the JVM alive after main returns.
                // The pipeline's own timeout is the mechanism that gives up on it; this only ensures the
                // process can still exit.
                isDaemon = true
            }
        }
        try {
            val futures: List<Future<T>> = pool.invokeAll(tasks.map { Callable(it) })
            // Positional, by construction: invokeAll's list is in submission order.
            return futures.map { future ->
                try {
                    future.get()
                } catch (e: java.util.concurrent.ExecutionException) {
                    // Unwrapped, so the caller sees the real failure rather than a wrapper naming the
                    // executor. The first failure wins; the rest have already been collected.
                    throw e.cause ?: e
                }
            }
        } finally {
            pool.shutdown()
        }
    }
}

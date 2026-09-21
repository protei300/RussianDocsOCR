package net.russiandocs.docproc

import net.russiandocs.docproc.tensors.LevenbergMarquardt
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Pins the solver against two textbook problems with a KNOWN answer, independent of anything in
 * `document_processing` — the numerical correctness of the algorithm itself, not its use inside
 * `line_refine.py`'s page-straightening fit (that comparison belongs to a conformance run against
 * the reference's own numbers, not a unit test with no golden to check against).
 */
class LevenbergMarquardtTests {

    /**
     * The Rosenbrock residuals `[10*(x2 - x1^2), 1 - x1]`: minimising their sum of squares has the
     * unique global minimum at (1, 1), a standard nonlinear least-squares test problem precisely
     * because the curved valley defeats a naive gradient step and rewards a real Levenberg-Marquardt
     * damping schedule.
     */
    @Test
    fun convergesOnRosenbrock() {
        val result = LevenbergMarquardt.solve(
            initial = doubleArrayOf(-1.2, 1.0),
            residuals = { p -> doubleArrayOf(10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]) },
        )
        assertTrue(abs(result.params[0] - 1.0) < 1e-6, "x1 should converge to 1, got ${result.params[0]}")
        assertTrue(abs(result.params[1] - 1.0) < 1e-6, "x2 should converge to 1, got ${result.params[1]}")
        assertTrue(result.cost < 1e-12, "cost should be ~0 at the optimum, got ${result.cost}")
    }

    /**
     * A genuinely nonlinear fit — `y = a*exp(b*x)` — recovered from noise-free synthetic samples.
     * Closer in shape to the page-straightening model (nonlinear in its parameters, evaluated at
     * several measurement points) than Rosenbrock is, and a different failure mode: an exponential's
     * Jacobian is nowhere near constant, so this also exercises the per-parameter finite-difference
     * step scaling.
     */
    @Test
    fun recoversExponentialFit() {
        val trueA = 2.5
        val trueB = 0.3
        val xs = doubleArrayOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        val ys = DoubleArray(xs.size) { trueA * exp(trueB * xs[it]) }

        val result = LevenbergMarquardt.solve(
            initial = doubleArrayOf(1.0, 0.1),
            residuals = { p -> DoubleArray(xs.size) { i -> p[0] * exp(p[1] * xs[i]) - ys[i] } },
        )
        assertTrue(abs(result.params[0] - trueA) < 1e-5, "a should recover to $trueA, got ${result.params[0]}")
        assertTrue(abs(result.params[1] - trueB) < 1e-5, "b should recover to $trueB, got ${result.params[1]}")
    }

    /** Already at the optimum: zero residuals, zero iterations of real work, no divide-by-zero. */
    @Test
    fun staysPutAtTheOptimum() {
        val result = LevenbergMarquardt.solve(
            initial = doubleArrayOf(1.0, 1.0),
            residuals = { p -> doubleArrayOf(10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]) },
        )
        assertTrue(result.cost < 1e-20, "should recognise it is already at the minimum")
    }
}

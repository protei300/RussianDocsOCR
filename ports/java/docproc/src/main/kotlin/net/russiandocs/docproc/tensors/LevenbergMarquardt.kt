package net.russiandocs.docproc.tensors

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * A small, dependency-free Levenberg-Marquardt least-squares solver with a forward-difference
 * Jacobian.
 *
 * Written for one call site: `page_registration/line_refine.py::_fit` (the page-straightening IRLS
 * fit — pipeline.py's `_register_pages` → `PageRegistrar.straighten`), which uses
 * `scipy.optimize.least_squares` on a 4-parameter model. There is no SciPy on the JVM, and the
 * user's own instruction for this port is to replace it with a hand-written LM solver rather than
 * hand-translate SciPy's trust-region internals (`trf`, the default `least_squares` uses) — LM is
 * the textbook method for exactly this shape of problem: a small number of parameters, a smooth
 * residual, no bounds.
 *
 * **This is NOT a byte-for-byte port of `trf`.** It is a different (classic, textbook) algorithm
 * that solves the SAME minimisation problem. For a smooth, well-conditioned residual with a unique
 * local minimum near the start point — which `_fit`'s zero-initialised 4-parameter model is — both
 * methods converge to the same optimum to the precision the reference actually uses (its own results
 * are rounded to 2-4 decimal places before they reach anything downstream). Where this matters, the
 * caller MUST measure its own residual before/after against the reference's numbers rather than
 * trust this docstring — see the deviation note this solver's first caller carries.
 */
public object LevenbergMarquardt {

    /** [params] is the converged (or best-found) parameter vector; [cost] is `0.5 * sum(residuals^2)`. */
    public class Result(
        public val params: DoubleArray,
        public val cost: Double,
        public val iterations: Int,
        public val converged: Boolean,
    )

    /**
     * Minimises `0.5 * sum(residuals(params)^2)` starting from [initial].
     *
     * Classic Levenberg-Marquardt: at each step, solve `(JᵀJ + λ·diag(JᵀJ)) δ = -Jᵀr` for the step
     * δ; accept it and shrink λ if the cost drops, otherwise grow λ (move toward gradient descent)
     * and retry. [maxIterations] bounds the number of ACCEPTED steps, mirroring `max_nfev` loosely —
     * it is a step budget, not a function-evaluation budget, which is a deliberate simplification for
     * a solver this small.
     *
     * The Jacobian is estimated by forward differences, one residual evaluation per parameter per
     * accepted step — `scipy.optimize.least_squares`'s own default when no analytic Jacobian is
     * given, and the reference's `_fit` does not supply one either.
     */
    public fun solve(
        initial: DoubleArray,
        residuals: (DoubleArray) -> DoubleArray,
        maxIterations: Int = 100,
        costTol: Double = 1e-10,
        stepTol: Double = 1e-10,
        finiteDiffRelStep: Double = SQRT_EPS,
    ): Result {
        var params = initial.copyOf()
        var r = residuals(params)
        var cost = halfSumSquares(r)
        var lambda = 1e-3
        var iterations = 0
        var converged = false

        while (iterations < maxIterations) {
            val jac = forwardDifferenceJacobian(params, r, finiteDiffRelStep, residuals)
            val jtJ = normalMatrix(jac)
            val jtR = normalVector(jac, r)

            var accepted = false
            var attempt = 0
            // A handful of λ retries per step: each retry only re-solves a tiny (n<=4) linear system,
            // not the Jacobian, so this stays cheap even in the worst case.
            while (!accepted && attempt < 30) {
                val damped = Array(jtJ.size) { i ->
                    DoubleArray(jtJ.size) { j -> jtJ[i][j] + if (i == j) lambda * jtJ[i][i] else 0.0 }
                }
                val rhs = DoubleArray(jtR.size) { -jtR[it] }
                val delta = solveSymmetric(damped, rhs)
                if (delta == null) {
                    lambda *= 10.0
                    attempt++
                    continue
                }
                val candidate = DoubleArray(params.size) { params[it] + delta[it] }
                val candidateR = residuals(candidate)
                val candidateCost = halfSumSquares(candidateR)

                if (candidateCost < cost) {
                    val stepNorm = norm(delta)
                    val paramNorm = norm(params)
                    val relativeGain = if (cost > 0) (cost - candidateCost) / cost else 0.0
                    params = candidate
                    r = candidateR
                    cost = candidateCost
                    lambda = max(lambda / 3.0, 1e-12)
                    accepted = true
                    if (relativeGain < costTol || stepNorm < stepTol * (paramNorm + stepTol)) {
                        converged = true
                    }
                } else {
                    lambda *= 10.0
                }
                attempt++
            }

            iterations++
            if (!accepted || converged) {
                break
            }
        }

        return Result(params, cost, iterations, converged)
    }

    /** `sqrt(machine epsilon)` — the classic finite-difference step scale (SciPy's own default). */
    private const val SQRT_EPS: Double = 1.4901161193847656e-08

    private fun halfSumSquares(v: DoubleArray): Double {
        var s = 0.0
        for (x in v) s += x * x
        return 0.5 * s
    }

    private fun norm(v: DoubleArray): Double {
        var s = 0.0
        for (x in v) s += x * x
        return sqrt(s)
    }

    /**
     * `J[k][i] = d residuals[k] / d params[i]`, by forward differences with a step scaled to each
     * parameter's own magnitude — `h_i = relStep * max(1, |params[i]|)` — so a near-zero parameter
     * still gets a usable (not vanishing) step.
     */
    private fun forwardDifferenceJacobian(
        params: DoubleArray,
        r0: DoubleArray,
        relStep: Double,
        residuals: (DoubleArray) -> DoubleArray,
    ): Array<DoubleArray> {
        val m = r0.size
        val n = params.size
        val jac = Array(m) { DoubleArray(n) }
        for (i in 0 until n) {
            val step = relStep * max(1.0, abs(params[i]))
            val perturbed = params.copyOf()
            perturbed[i] += step
            val ri = residuals(perturbed)
            for (k in 0 until m) {
                jac[k][i] = (ri[k] - r0[k]) / step
            }
        }
        return jac
    }

    private fun normalMatrix(jac: Array<DoubleArray>): Array<DoubleArray> {
        val m = jac.size
        val n = if (m > 0) jac[0].size else 0
        val out = Array(n) { DoubleArray(n) }
        for (i in 0 until n) {
            for (j in i until n) {
                var s = 0.0
                for (k in 0 until m) s += jac[k][i] * jac[k][j]
                out[i][j] = s
                out[j][i] = s
            }
        }
        return out
    }

    private fun normalVector(jac: Array<DoubleArray>, r: DoubleArray): DoubleArray {
        val m = jac.size
        val n = if (m > 0) jac[0].size else 0
        val out = DoubleArray(n)
        for (i in 0 until n) {
            var s = 0.0
            for (k in 0 until m) s += jac[k][i] * r[k]
            out[i] = s
        }
        return out
    }

    /**
     * Solves `a x = b` for a small square (here n<=4) matrix by Gaussian elimination with partial
     * pivoting. Returns null on a singular (or too close to singular) matrix, which the caller treats
     * as "grow the damping and retry" — exactly the situation the Levenberg-Marquardt damping term
     * exists to avoid in the first place, so this path is a safety net rather than a common case.
     */
    private fun solveSymmetric(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = b.size
        val m = Array(n) { i -> DoubleArray(n + 1) { j -> if (j < n) a[i][j] else b[i] } }
        for (col in 0 until n) {
            var pivot = col
            for (row in col + 1 until n) {
                if (abs(m[row][col]) > abs(m[pivot][col])) pivot = row
            }
            if (abs(m[pivot][col]) < 1e-15) {
                return null
            }
            if (pivot != col) {
                val tmp = m[col]
                m[col] = m[pivot]
                m[pivot] = tmp
            }
            for (row in 0 until n) {
                if (row == col) continue
                val factor = m[row][col] / m[col][col]
                for (c in col..n) {
                    m[row][c] -= factor * m[col][c]
                }
            }
        }
        return DoubleArray(n) { m[it][n] / m[it][it] }
    }
}

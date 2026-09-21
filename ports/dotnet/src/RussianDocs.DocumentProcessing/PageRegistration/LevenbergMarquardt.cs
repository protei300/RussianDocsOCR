namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// A small, dependency-free Levenberg-Marquardt least-squares solver with a finite-difference
/// Jacobian.
///
/// <para>
/// <b>Why this exists, and what it is not.</b> The reference's <c>line_refine._fit</c>
/// (page_registration/line_refine.py:204-217) calls <c>scipy.optimize.least_squares(_residuals, x,
/// args=(...), max_nfev=100)</c> to fit a 4-parameter page-straightening model (rotation, shear, two
/// perspective terms) inside 3 rounds of IRLS. SciPy's own default there is Trust-Region-Reflective
/// (TRF), not classic Levenberg-Marquardt — the two are different algorithms and this port does not
/// attempt to reproduce TRF's internals bit-for-bit (that would mean reimplementing SciPy, not
/// porting the pipeline). What actually needs porting is the SAME PROBLEM CLASS: a small (here, 4
/// parameter), smooth, unconstrained nonlinear least-squares fit, robustified by the IRLS loop that
/// wraps it — and LM is the standard general-purpose solver for exactly that, well understood and
/// easy to verify independently of SciPy's specific trust-region bookkeeping. <c>page_registration/</c>
/// itself is not ported yet (a separate, much larger milestone — SIFT+MAGSAC template matching,
/// <c>quad_fit.py</c>, <c>line_dewarp.py</c>, the template assets); this solver is a self-contained
/// building block for that future work, usable on its own and testable without any of it.
/// </para>
///
/// <para>
/// <b>Algorithm.</b> Classic damped Gauss-Newton (Levenberg-Marquardt, Marquardt's diagonal
/// scaling): at each step, solve <c>(JᵀJ + λ·diag(JᵀJ)) δ = -Jᵀr</c> for the step <c>δ</c>; accept it
/// and shrink <c>λ</c> when the cost decreases, otherwise grow <c>λ</c> and retry from the same
/// point. The Jacobian is estimated by forward finite differences, one residual evaluation per
/// parameter per accepted or rejected step.
/// </para>
/// </summary>
public static class LevenbergMarquardt
{
    /// <summary>The outcome of a solve: the best parameters found and why the loop stopped.</summary>
    public sealed record Result(double[] X, double Cost, int Iterations, bool Converged);

    /// <summary>
    /// Minimises <c>0.5 * sum(residuals(x)^2)</c> over <paramref name="x0"/>.
    /// </summary>
    /// <param name="residuals">
    /// Maps parameters to a residual vector (any length — <c>line_refine._residuals</c> concatenates
    /// per-measurement deviations with a prior term, exactly the shape this expects).
    /// </param>
    /// <param name="x0">Starting point. Not mutated; the result carries its own copy.</param>
    /// <param name="maxIterations">Outer iteration cap — accepted steps, not Jacobian evaluations.</param>
    /// <param name="costTolerance">Stop when a step reduces the cost by less than this (relative).</param>
    /// <param name="stepTolerance">Stop when the step norm is below this (relative to the parameter norm).</param>
    /// <param name="gradientTolerance">Stop when the max-norm of <c>Jᵀr</c> falls below this.</param>
    /// <param name="finiteDifferenceStep">
    /// Relative forward-difference step for the Jacobian: the actual step on parameter <c>j</c> is
    /// <c>finiteDifferenceStep * max(1, |x_j|)</c>, so a parameter near zero still gets a sane
    /// absolute step. SciPy's own default for <c>'2-point'</c> Jacobians is
    /// <c>sqrt(machine epsilon) ~= 1.49e-8</c>; this defaults to the same value.
    /// </param>
    public static Result Solve(
        Func<double[], double[]> residuals,
        double[] x0,
        int maxIterations = 100,
        double costTolerance = 1e-10,
        double stepTolerance = 1e-10,
        double gradientTolerance = 1e-10,
        double finiteDifferenceStep = 1.4901161193847656e-8)
    {
        ArgumentNullException.ThrowIfNull(residuals);
        ArgumentNullException.ThrowIfNull(x0);
        int n = x0.Length;
        double[] x = (double[])x0.Clone();

        double[] r = residuals(x);
        double cost = SumSquares(r);
        bool converged = false;
        int iterations = 0;

        // Marquardt's own scaling: the damping term is lambda * diag(J^T J), not lambda * I, so a
        // parameter whose Jacobian column is naturally small (weak sensitivity) is not damped as
        // hard as one with a large column norm. Starting value is conventional (Nocedal & Wright,
        // "Numerical Optimization", ch. 10) and self-corrects within the first few steps regardless.
        double lambda = 1e-3;

        for (; iterations < maxIterations; iterations++)
        {
            double[,] jac = FiniteDifferenceJacobian(residuals, x, r, finiteDifferenceStep);
            double[,] jtj = Gram(jac, n);
            double[] jtr = JacobianTransposeTimes(jac, r, n);

            double gradNorm = 0.0;
            for (int j = 0; j < n; j++)
            {
                gradNorm = Math.Max(gradNorm, Math.Abs(jtr[j]));
            }
            if (gradNorm < gradientTolerance)
            {
                converged = true;
                break;
            }

            // Try increasingly damped steps from the SAME point until one actually reduces the
            // cost, or the damping has grown enough that a further increase cannot plausibly help
            // (guards against spinning forever on a degenerate Jacobian).
            bool improved = false;
            for (int attempt = 0; attempt < 30; attempt++)
            {
                double[,] normal = (double[,])jtj.Clone();
                for (int j = 0; j < n; j++)
                {
                    normal[j, j] += lambda * Math.Max(jtj[j, j], 1e-12);
                }
                double[] negJtr = new double[n];
                for (int j = 0; j < n; j++)
                {
                    negJtr[j] = -jtr[j];
                }

                if (!SolveLinearSystem(normal, negJtr, out double[] delta))
                {
                    // Singular normal equations at this damping: more damping makes them better
                    // conditioned, so grow and retry rather than giving up.
                    lambda *= 10.0;
                    continue;
                }

                double[] candidate = new double[n];
                double stepNorm = 0.0, xNorm = 0.0;
                for (int j = 0; j < n; j++)
                {
                    candidate[j] = x[j] + delta[j];
                    stepNorm += delta[j] * delta[j];
                    xNorm += x[j] * x[j];
                }
                stepNorm = Math.Sqrt(stepNorm);

                double[] candidateResiduals = residuals(candidate);
                double candidateCost = SumSquares(candidateResiduals);

                if (candidateCost < cost)
                {
                    bool costConverged = cost > 0 && (cost - candidateCost) < costTolerance * cost;
                    bool stepConverged = stepNorm < stepTolerance * (Math.Sqrt(xNorm) + stepTolerance);

                    x = candidate;
                    r = candidateResiduals;
                    cost = candidateCost;
                    lambda = Math.Max(lambda * 0.3, 1e-12);
                    improved = true;

                    if (costConverged || stepConverged)
                    {
                        converged = true;
                    }
                    break;
                }

                lambda *= 10.0;
            }

            if (converged || !improved)
            {
                break;
            }
        }

        return new Result(x, cost, iterations, converged);
    }

    private static double SumSquares(double[] v)
    {
        double s = 0.0;
        foreach (double d in v)
        {
            s += d * d;
        }
        return 0.5 * s;
    }

    /// <summary>
    /// Forward-difference Jacobian: column <c>j</c> is <c>(residuals(x + h*e_j) - r0) / h</c>. One
    /// extra residual evaluation per parameter — cheap here because the model has only 4 of them.
    /// </summary>
    private static double[,] FiniteDifferenceJacobian(Func<double[], double[]> residuals,
        double[] x, double[] r0, double relativeStep)
    {
        int n = x.Length, m = r0.Length;
        var jac = new double[m, n];
        double[] perturbed = (double[])x.Clone();
        for (int j = 0; j < n; j++)
        {
            double h = relativeStep * Math.Max(1.0, Math.Abs(x[j]));
            double original = perturbed[j];
            perturbed[j] = original + h;
            double[] rh = residuals(perturbed);
            perturbed[j] = original;

            // Recompute the actual step: `original + h` can lose precision to rounding for a large
            // |original|, and the Jacobian must use the step that was ACTUALLY taken, not the
            // nominal one (standard finite-difference hygiene).
            double actualH = perturbed[j] + h - perturbed[j];
            if (actualH == 0.0)
            {
                actualH = h;
            }
            for (int i = 0; i < m; i++)
            {
                jac[i, j] = (rh[i] - r0[i]) / actualH;
            }
        }
        return jac;
    }

    private static double[,] Gram(double[,] jac, int n)
    {
        int m = jac.GetLength(0);
        var jtj = new double[n, n];
        for (int a = 0; a < n; a++)
        {
            for (int b = a; b < n; b++)
            {
                double sum = 0.0;
                for (int i = 0; i < m; i++)
                {
                    sum += jac[i, a] * jac[i, b];
                }
                jtj[a, b] = sum;
                jtj[b, a] = sum;
            }
        }
        return jtj;
    }

    private static double[] JacobianTransposeTimes(double[,] jac, double[] r, int n)
    {
        int m = r.Length;
        var jtr = new double[n];
        for (int a = 0; a < n; a++)
        {
            double sum = 0.0;
            for (int i = 0; i < m; i++)
            {
                sum += jac[i, a] * r[i];
            }
            jtr[a] = sum;
        }
        return jtr;
    }

    /// <summary>
    /// Gaussian elimination with partial pivoting for a small dense SPD-ish system (the normal
    /// equations are symmetric and, once damped, positive definite in every case this solver is
    /// meant for — 4 parameters at a time). Returns false on a singular pivot rather than throwing:
    /// the caller's response (increase the damping and retry) is a normal part of the LM loop, not
    /// an error.
    /// </summary>
    private static bool SolveLinearSystem(double[,] a, double[] b, out double[] x)
    {
        int n = b.Length;
        var m = (double[,])a.Clone();
        var v = (double[])b.Clone();
        x = new double[n];

        for (int col = 0; col < n; col++)
        {
            int pivotRow = col;
            double pivotValue = Math.Abs(m[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(m[row, col]) > pivotValue)
                {
                    pivotRow = row;
                    pivotValue = Math.Abs(m[row, col]);
                }
            }
            if (pivotValue < 1e-14)
            {
                return false;
            }
            if (pivotRow != col)
            {
                for (int k = 0; k < n; k++)
                {
                    (m[col, k], m[pivotRow, k]) = (m[pivotRow, k], m[col, k]);
                }
                (v[col], v[pivotRow]) = (v[pivotRow], v[col]);
            }

            for (int row = col + 1; row < n; row++)
            {
                double factor = m[row, col] / m[col, col];
                if (factor == 0.0)
                {
                    continue;
                }
                for (int k = col; k < n; k++)
                {
                    m[row, k] -= factor * m[col, k];
                }
                v[row] -= factor * v[col];
            }
        }

        for (int row = n - 1; row >= 0; row--)
        {
            double sum = v[row];
            for (int k = row + 1; k < n; k++)
            {
                sum -= m[row, k] * x[k];
            }
            if (Math.Abs(m[row, row]) < 1e-14)
            {
                return false;
            }
            x[row] = sum / m[row, row];
        }
        return true;
    }
}

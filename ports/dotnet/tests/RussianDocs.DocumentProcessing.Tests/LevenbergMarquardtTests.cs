using RussianDocs.DocumentProcessing.PageRegistration;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// <see cref="LevenbergMarquardt"/> — the general nonlinear least-squares solver written for the
/// future <c>line_refine.py</c> port (4-parameter page-straightening fit). Verified here against
/// problems with a KNOWN answer, independent of page registration itself, which is not ported yet.
/// </summary>
[TestFixture]
public class LevenbergMarquardtTests
{
    /// <summary>
    /// A plain linear least-squares problem (fit <c>y = a*x + b</c>) has a closed-form answer via
    /// the normal equations — the cheapest possible ground truth for a general nonlinear solver.
    /// </summary>
    [Test]
    public void Solve_RecoversAKnownLinearFit()
    {
        double[] xs = [0, 1, 2, 3, 4, 5];
        double[] ys = [1.1, 2.9, 5.05, 6.9, 9.1, 11.0]; // approx y = 2x + 1, with noise

        double[] Residuals(double[] p) =>
            [.. xs.Zip(ys, (x, y) => (p[0] * x + p[1]) - y)];

        LevenbergMarquardt.Result result = LevenbergMarquardt.Solve(Residuals, [0.0, 0.0]);

        Assert.Multiple(() =>
        {
            Assert.That(result.Converged, Is.True);
            Assert.That(result.X[0], Is.EqualTo(2.0).Within(0.05), "slope");
            Assert.That(result.X[1], Is.EqualTo(1.0).Within(0.1), "intercept");
        });
    }

    /// <summary>
    /// A genuinely nonlinear fit (exponential decay, three parameters) generated from known
    /// parameters plus tiny noise — the solver must find its way back to them from a starting point
    /// that is not already close.
    /// </summary>
    [Test]
    public void Solve_RecoversKnownExponentialDecayParameters()
    {
        const double trueA = 3.0, trueK = 0.5, trueC = 0.2;
        double[] xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        double[] noise = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.005, 0.0];
        double[] ys = [.. xs.Select((x, i) =>
            trueA * Math.Exp(-trueK * x) + trueC + noise[i])];

        double[] Residuals(double[] p) =>
            [.. xs.Zip(ys, (x, y) => (p[0] * Math.Exp(-p[1] * x) + p[2]) - y)];

        LevenbergMarquardt.Result result = LevenbergMarquardt.Solve(
            Residuals, [1.0, 1.0, 0.0], maxIterations: 200);

        Assert.Multiple(() =>
        {
            Assert.That(result.Converged, Is.True);
            Assert.That(result.X[0], Is.EqualTo(trueA).Within(0.1));
            Assert.That(result.X[1], Is.EqualTo(trueK).Within(0.05));
            Assert.That(result.X[2], Is.EqualTo(trueC).Within(0.1));
        });
    }

    /// <summary>Zero residuals everywhere means the starting point is already optimal.</summary>
    [Test]
    public void Solve_StaysPutWhenAlreadyExact()
    {
        double[] Residuals(double[] p) => [p[0] - 5.0, p[1] + 2.0];

        LevenbergMarquardt.Result result = LevenbergMarquardt.Solve(Residuals, [5.0, -2.0]);

        Assert.Multiple(() =>
        {
            Assert.That(result.X[0], Is.EqualTo(5.0).Within(1e-6));
            Assert.That(result.X[1], Is.EqualTo(-2.0).Within(1e-6));
            Assert.That(result.Cost, Is.LessThan(1e-12));
        });
    }

    /// <summary>
    /// A shape close to the real use case: <c>line_refine._residuals</c> concatenates per-measurement
    /// deviations with a 3-element prior on params[1:] (page_registration/line_refine.py:198-201).
    /// Mirrors that shape with a toy 4-parameter model to pin that mixed-length residual vectors work.
    /// </summary>
    [Test]
    public void Solve_HandlesAConcatenatedMeasurementPlusPriorResidualLikeLineRefine()
    {
        // "Measurements": 4 params should drive four values to specific targets; the trailing 3
        // entries are a prior pulling params[1..3] toward zero, exactly as `_residuals` appends
        // `params[1:] / PRIOR_SCALE`.
        double[] targets = [1.0, 2.0, -1.0, 0.5];

        double[] Residuals(double[] p)
        {
            var measurement = new double[4];
            for (int i = 0; i < 4; i++)
            {
                measurement[i] = p[i] - targets[i];
            }
            double[] prior = [p[1] / 0.05, p[2] / 0.03, p[3] / 0.03];
            return [.. measurement, .. prior];
        }

        LevenbergMarquardt.Result result = LevenbergMarquardt.Solve(Residuals, [0.0, 0.0, 0.0, 0.0]);

        // The prior pulls params 1-3 toward zero, so they land BETWEEN zero and their target,
        // not exactly on it — this asserts the solver actually balances both terms rather than
        // ignoring one of them.
        Assert.Multiple(() =>
        {
            Assert.That(result.Converged, Is.True);
            Assert.That(result.X[0], Is.EqualTo(1.0).Within(0.01), "unpenalised param hits its target");
            Assert.That(result.X[1], Is.GreaterThan(0.0).And.LessThan(2.0), "penalised param is pulled toward 0");
        });
    }
}

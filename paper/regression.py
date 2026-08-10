"""
Significance and effect-size analysis for a variance-median relationship
across populations, with errors-in-variables handled by resampling.

Setting
-------
For each unit i:
  * x_i = sample variance of a quantity over population (i, 1)
    (optionally log-transformed; recommended)
  * y_i = sample median of the same quantity over a DIFFERENT population (i, 2)
The underlying distribution is not assumed normal, and the two populations
per unit are assumed disjoint (so the sampling errors of x_i and y_i are
independent given the truths).

Pipeline (see analyze() for the one-call entry point)
-----------------------------------------------------
1. Point estimates and their uncertainties by *within-population*
   nonparametric bootstrap (no normal-theory formulas needed; this captures
   the skewed, kurtosis-driven sampling error of a variance, and works for
   the median, which has no simple closed-form SE at all).
   Optionally work with log-variance on the x side, whose sampling
   distribution is far closer to Gaussian (recommended; monotone-invariant
   for "does the median increase with the variance" questions).
   NOTE on medians: the bootstrap distribution of a sample median is
   discrete (it only takes observed data values), so for small populations
   it can look lumpy.  The pipeline is robust to this, but if populations
   are tiny (< ~20), consider more within-population bootstrap replicates.
2. York (1966/2004) errors-in-variables fit = Gaussian ML for a line with
   per-point sigma_x, sigma_y.  Gives a de-attenuated slope, analytic
   standard errors, and a reduced chi-squared diagnostic for intrinsic
   scatter / mis-stated uncertainties.
3. Significance: exact-style permutation test that
     - permutes each y-side unit (y_i, sigma_y_i) as a glued pair against
       the x-side units (preserving exchangeability of units),
     - uses the *studentized* York slope T = b / se(b) as the statistic
       (Chung & Romano: studentization keeps the permutation test
       asymptotically valid under heteroscedastic, non-identical units,
       while remaining exact under exchangeability),
     - optionally restricts permutations within user-supplied strata
       (e.g. bins of population size) if precision plausibly tracks x.
4. Effect size: nested nonparametric bootstrap CI for the slope.
   Synthetic datasets are generated around the *fitted* line at the York
   adjusted (estimated true) positions, perturbed by draws from each
   point's CENTERED within-population bootstrap distribution -- i.e. the
   corrected parametric bootstrap, with the Gaussian noise model replaced
   by the empirical sampling distribution of each point.  This avoids the
   classic double-counting error of re-perturbing observed values.

Only numpy is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# 1. Within-population bootstrap for point estimates and uncertainties
# ----------------------------------------------------------------------


@dataclass
class PointEstimates:
    x: np.ndarray  # (log-)variances of populations (i, 1)
    y: np.ndarray  # medians of populations (i, 2)
    sx: np.ndarray  # bootstrap SE of x_i
    sy: np.ndarray  # bootstrap SE of y_i
    x_boot: np.ndarray  # (n_points, n_boot) bootstrap replicates of x_i
    y_boot: np.ndarray  # (n_points, n_boot) bootstrap replicates of y_i
    log_x: bool

    @property
    def x_boot_centered(self) -> np.ndarray:
        """Centered replicates = empirical sampling-error distribution of x_i."""
        return self.x_boot - self.x[:, None]

    @property
    def y_boot_centered(self) -> np.ndarray:
        return self.y_boot - self.y[:, None]


def bootstrap_point_estimates(
    populations_x: Sequence[np.ndarray],
    populations_y: Sequence[np.ndarray],
    n_boot: int = 2000,
    log_x: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> PointEstimates:
    """
    populations_x[i]: raw samples of population (i, 1)
                      -> x_i = var (ddof=1), or log(var) if log_x=True
    populations_y[i]: raw samples of population (i, 2)
                      -> y_i = median
    """
    if len(populations_x) != len(populations_y):
        raise ValueError("populations_x and populations_y must have equal length")
    rng = np.random.default_rng() if rng is None else rng

    n = len(populations_x)
    x = np.empty(n)
    y = np.empty(n)
    x_boot = np.empty((n, n_boot))
    y_boot = np.empty((n, n_boot))

    for i, (px, py) in enumerate(zip(populations_x, populations_y)):
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        if px.size < 3 or py.size < 2:
            raise ValueError(
                f"population {i} too small (need >=3 for variance, >=2 for median)"
            )

        v = px.var(ddof=1)
        if log_x and v <= 0:
            raise ValueError(
                f"population (i={i}, 1) has zero variance; cannot take log"
            )
        x[i] = np.log(v) if log_x else v
        y[i] = np.median(py)

        # bootstrap the (log-)variance
        idx = rng.integers(0, px.size, size=(n_boot, px.size))
        vb = px[idx].var(axis=1, ddof=1)
        if log_x:
            # guard against pathological zero-variance resamples of tiny populations
            tiny = np.finfo(float).tiny
            vb = np.maximum(vb, tiny)
            x_boot[i] = np.log(vb)
        else:
            x_boot[i] = vb

        # bootstrap the median
        idy = rng.integers(0, py.size, size=(n_boot, py.size))
        y_boot[i] = np.median(py[idy], axis=1)

    sx = x_boot.std(axis=1, ddof=1)
    sy = y_boot.std(axis=1, ddof=1)

    if np.any(sy == 0):
        bad = np.flatnonzero(sy == 0)
        raise ValueError(
            f"bootstrap SE of the median is exactly 0 for population(s) {bad.tolist()} "
            "(degenerate/tiny sample); York weights would be infinite. "
            "Increase n_boot or inspect these populations."
        )

    return PointEstimates(
        x=x, y=y, sx=sx, sy=sy, x_boot=x_boot, y_boot=y_boot, log_x=log_x
    )


# ----------------------------------------------------------------------
# 2. York (2004) errors-in-variables line fit
# ----------------------------------------------------------------------


@dataclass
class YorkResult:
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    chi2: float
    chi2_reduced: float
    x_adjusted: np.ndarray  # ML estimates of the true x positions
    n_iter: int
    converged: bool
    tau: float = 0.0  # intrinsic-scatter SD folded into the y-side weights
    chi2_reduced_no_scatter: float = np.nan  # diagnostic: reduced chi^2 at tau = 0

    @property
    def t(self) -> float:
        """Studentized slope."""
        return self.slope / self.slope_se


def york_fit(
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    r: float | np.ndarray = 0.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> YorkResult:
    """
    Straight-line fit with per-point errors in x and y (and optional per-point
    x-y error correlation r), following York, Evensen, Martinez & Delgado,
    Am. J. Phys. 72, 367 (2004).  Identical to Gaussian ML for the
    errors-in-variables model.
    """
    x, y, sx, sy = (np.asarray(a, dtype=float) for a in (x, y, sx, sy))
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 points")
    r = np.broadcast_to(np.asarray(r, dtype=float), x.shape)

    wx = 1.0 / sx**2
    wy = 1.0 / sy**2
    alpha = np.sqrt(wx * wy)

    # initial slope from OLS
    b = np.polyfit(x, y, 1)[0]

    converged = False
    for it in range(1, max_iter + 1):
        W = wx * wy / (wx + b**2 * wy - 2.0 * b * r * alpha)
        Xbar = np.sum(W * x) / np.sum(W)
        Ybar = np.sum(W * y) / np.sum(W)
        U = x - Xbar
        V = y - Ybar
        beta = W * (U / wy + b * V / wx - (b * U + V) * r / alpha)
        b_new = np.sum(W * beta * V) / np.sum(W * beta * U)
        if abs(b_new - b) <= tol * (abs(b) + tol):
            b = b_new
            converged = True
            break
        b = b_new

    W = wx * wy / (wx + b**2 * wy - 2.0 * b * r * alpha)
    Xbar = np.sum(W * x) / np.sum(W)
    Ybar = np.sum(W * y) / np.sum(W)
    a = Ybar - b * Xbar

    U = x - Xbar
    V = y - Ybar
    beta = W * (U / wy + b * V / wx - (b * U + V) * r / alpha)
    x_adj = Xbar + beta
    xbar_adj = np.sum(W * x_adj) / np.sum(W)
    u = x_adj - xbar_adj

    var_b = 1.0 / np.sum(W * u**2)
    var_a = 1.0 / np.sum(W) + xbar_adj**2 * var_b

    chi2 = float(np.sum(W * (y - b * x - a) ** 2))
    dof = n - 2
    return YorkResult(
        slope=float(b),
        intercept=float(a),
        slope_se=float(np.sqrt(var_b)),
        intercept_se=float(np.sqrt(var_a)),
        chi2=chi2,
        chi2_reduced=chi2 / dof,
        x_adjusted=x_adj,
        n_iter=it,
        converged=converged,
    )


def york_fit_scatter(
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    r: float | np.ndarray = 0.0,
    tol: float = 1e-10,
) -> YorkResult:
    """
    York fit with an intrinsic-scatter term: the y-side weight of each point
    becomes 1 / (sy_i^2 + tau^2 + b^2 sx_i^2), with tau^2 >= 0 calibrated so
    that the reduced chi-squared equals 1 (the profile / method-of-moments
    estimate, analogous to REML scale estimation).

    This repairs the two intrinsic-scatter failure modes at once:
      * weights: precisely-measured points can no longer hijack the fit,
        because tau^2 puts a floor under every point's effective variance;
      * studentization: T = slope/se is computed from the corrected weights,
        restoring (approximate) pivotality for the permutation test, and the
        analytic SE and bootstrap CI are no longer optimistic.

    If the data show no excess scatter (reduced chi^2 <= 1 at tau = 0), this
    reduces exactly to york_fit.  The SE treats tau^2 as known (plug-in); the
    uncertainty in tau^2 itself is *not* propagated -- that is the remaining
    advantage of the fully Bayesian treatment (linmix).

    Only r = 0 (independent x/y measurement errors) is supported here, which
    is the disjoint-two-populations design; a nonzero r would need the
    cross-term rescaled against the inflated sy.
    """
    if np.any(np.asarray(r) != 0.0):
        raise NotImplementedError("york_fit_scatter supports r = 0 only")
    x, y, sx, sy = (np.asarray(a, dtype=float) for a in (x, y, sx, sy))

    base = york_fit(x, y, sx, sy)
    base.chi2_reduced_no_scatter = base.chi2_reduced
    if base.chi2_reduced <= 1.0:
        return base  # tau = 0

    def fit_at(tau2: float) -> YorkResult:
        return york_fit(x, y, sx, np.sqrt(sy**2 + tau2))

    # bracket: chi2_reduced is monotone decreasing in tau^2
    lo = 0.0
    hi = max(float(np.var(y, ddof=1)), float(np.max(sy) ** 2), 1e-30)
    while fit_at(hi).chi2_reduced > 1.0:
        hi *= 4.0
        if hi > 1e30:
            raise RuntimeError("failed to bracket tau^2 (chi^2 never reaches 1)")

    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if fit_at(mid).chi2_reduced > 1.0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= tol * hi:
            break

    result = fit_at(hi)
    result.tau = float(np.sqrt(hi))
    result.chi2_reduced_no_scatter = base.chi2_reduced
    return result


# ----------------------------------------------------------------------
# 3. Glued, studentized permutation test (optionally stratified)
# ----------------------------------------------------------------------


def _permutation_within_strata(
    n: int, strata: Optional[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Random permutation of 0..n-1; if strata given, permute only within strata."""
    if strata is None:
        return rng.permutation(n)
    perm = np.arange(n)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        perm[idx] = idx[rng.permutation(idx.size)]
    return perm


def studentized_permutation_test(
    est: PointEstimates,
    n_perm: int = 4999,
    strata: Optional[Sequence] = None,
    intrinsic_scatter: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Exact-style permutation test of H0: no association between the true
    (log-)variances and the true medians.

    Each y-side unit (y_i, sy_i) is permuted as a glued pair against the
    x-side units (x_i, sx_i); the statistic is the studentized York slope.
    With intrinsic_scatter=True the entire statistic -- including the
    profiling of tau^2 -- is recomputed on every permuted dataset, which is
    the correct discipline (the statistic must be the same function of the
    data on real and shuffled datasets alike).
    p-value uses the standard add-one correction (never exactly zero).
    """
    rng = np.random.default_rng() if rng is None else rng
    strata_arr = None if strata is None else np.asarray(strata)
    fitter = york_fit_scatter if intrinsic_scatter else york_fit

    obs = fitter(est.x, est.y, est.sx, est.sy)
    t_obs = obs.t

    t_perm = np.empty(n_perm)
    n_failed = 0
    for k in range(n_perm):
        p = _permutation_within_strata(est.x.size, strata_arr, rng)
        try:
            fit = fitter(est.x, est.y[p], est.sx, est.sy[p])
            t_perm[k] = fit.t
        except Exception:
            t_perm[k] = np.nan
            n_failed += 1

    valid = t_perm[np.isfinite(t_perm)]
    p_two = (1 + np.sum(np.abs(valid) >= abs(t_obs))) / (valid.size + 1)
    p_greater = (1 + np.sum(valid >= t_obs)) / (valid.size + 1)
    p_less = (1 + np.sum(valid <= t_obs)) / (valid.size + 1)

    return {
        "t_observed": t_obs,
        "p_two_sided": float(p_two),
        "p_slope_positive": float(p_greater),  # small => evidence slope > 0
        "p_slope_negative": float(p_less),
        "t_null_distribution": valid,
        "n_permutations_used": int(valid.size),
        "n_failed_fits": int(n_failed),
        "observed_fit": obs,
    }


# ----------------------------------------------------------------------
# 4. Nested nonparametric bootstrap CI for the slope
# ----------------------------------------------------------------------


def nested_bootstrap_slope_ci(
    est: PointEstimates,
    fit: Optional[YorkResult] = None,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    intrinsic_scatter: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Corrected parametric-style bootstrap with empirical noise:

      truth proxy:  (x_adj_i, a + b * x_adj_i)   -- points ON the fitted line
      scatter:      one draw from N(0, tau^2) per point (if the fit found
                    intrinsic scatter) -- the variance component whose
                    omission made the interval too narrow,
      noise:        one draw from point i's CENTERED within-population
                    bootstrap distribution (empirical sampling error),
                    independently for x and y sides.

    Each synthetic dataset therefore carries exactly ONE dose of sampling
    noise, like the real one (no double counting), and the noise inherits
    the true skewness/kurtosis (and, for the median, discreteness) of each
    estimator.  Each synthetic dataset is refit with the same York
    estimator.  Percentile and basic intervals are returned; they agree
    when the bootstrap distribution is symmetric.
    """
    rng = np.random.default_rng() if rng is None else rng
    fitter = york_fit_scatter if intrinsic_scatter else york_fit
    if fit is None:
        fit = fitter(est.x, est.y, est.sx, est.sy)

    x_true = fit.x_adjusted
    y_true = fit.intercept + fit.slope * x_true
    dx = est.x_boot_centered
    dy = est.y_boot_centered
    n_points, n_rep = dx.shape

    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    n_failed = 0
    for k in range(n_boot):
        jx = rng.integers(0, n_rep, size=n_points)
        jy = rng.integers(0, n_rep, size=n_points)
        xs = x_true + dx[np.arange(n_points), jx]
        ys = y_true + dy[np.arange(n_points), jy]
        if fit.tau > 0:
            ys = ys + rng.normal(0.0, fit.tau, size=n_points)
        try:
            f = fitter(xs, ys, est.sx, est.sy)
            slopes[k] = f.slope
            intercepts[k] = f.intercept
        except Exception:
            slopes[k] = np.nan
            intercepts[k] = np.nan
            n_failed += 1

    ok = np.isfinite(slopes) & np.isfinite(intercepts)
    slopes, intercepts = slopes[ok], intercepts[ok]
    alpha = 1.0 - ci_level
    lo_p, hi_p = np.quantile(slopes, [alpha / 2, 1 - alpha / 2])
    # basic (reflected) interval
    lo_b, hi_b = 2 * fit.slope - hi_p, 2 * fit.slope - lo_p

    return {
        "slope": fit.slope,
        "slope_se_analytic": fit.slope_se,
        "slope_se_bootstrap": float(slopes.std(ddof=1)),
        "ci_percentile": (float(lo_p), float(hi_p)),
        "ci_basic": (float(lo_b), float(hi_b)),
        "ci_level": ci_level,
        "bootstrap_slopes": slopes,
        "bootstrap_intercepts": intercepts,
        "n_failed_fits": int(n_failed),
    }


# ----------------------------------------------------------------------
# 5. One-call entry point
# ----------------------------------------------------------------------


@dataclass
class AnalysisResult:
    estimates: PointEstimates
    fit: YorkResult
    permutation: dict
    bootstrap: dict
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        f = self.fit
        p = self.permutation
        b = self.bootstrap
        xname = "log-variance" if self.estimates.log_x else "variance"
        lines = [
            f"Errors-in-variables fit of median on {xname} (York/ML):",
            f"  slope      = {f.slope:.5g}  (analytic SE {f.slope_se:.3g}, "
            f"bootstrap SE {b['slope_se_bootstrap']:.3g})",
            f"  intercept  = {f.intercept:.5g}  (SE {f.intercept_se:.3g})",
            f"  intrinsic scatter tau = {f.tau:.5g}"
            + ("  (none required)" if f.tau == 0 else "  (profiled so chi^2_red = 1)"),
            f"  reduced chi^2 without scatter term = "
            f"{(f.chi2_reduced_no_scatter if np.isfinite(f.chi2_reduced_no_scatter) else f.chi2_reduced):.3g}"
            f"  (n-2 = {f.x_adjusted.size - 2} dof)",
            "",
            f"Permutation test (glued units, studentized slope, "
            f"{p['n_permutations_used']} permutations):",
            f"  T = slope/SE = {p['t_observed']:.3g}",
            f"  two-sided p  = {p['p_two_sided']:.4g}",
            "",
            f"Nested bootstrap {100 * b['ci_level']:.0f}% CI for slope:",
            f"  percentile: [{b['ci_percentile'][0]:.5g}, {b['ci_percentile'][1]:.5g}]",
            f"  basic:      [{b['ci_basic'][0]:.5g}, {b['ci_basic'][1]:.5g}]",
        ]
        if self.warnings:
            lines += [""] + [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)

    def plot(self, ax=None, show_scatter_band: bool = True):
        """Data with error bars, best-fit line, and bootstrap CI band.
        Returns (fig, ax).  See plot_fit for details."""
        return plot_fit(self, ax=ax, show_scatter_band=show_scatter_band)


def plot_fit(
    result: "AnalysisResult", ax=None, n_grid: int = 200, show_scatter_band: bool = True
):
    """
    Plot the (x_i, y_i) points with their bootstrap-estimated error bars, the
    York best-fit line, and a shaded confidence band for the LINE, obtained
    pointwise from the nested-bootstrap (slope, intercept) replicates: at each
    grid x, the band spans the [alpha/2, 1-alpha/2] quantiles of a_k + b_k*x.
    The band therefore inherits everything the bootstrap knows about --
    empirical measurement noise, de-attenuation, and intrinsic scatter's
    effect on the line's uncertainty.

    If the fit found intrinsic scatter (tau > 0) and show_scatter_band is
    True, dashed lines at (best fit) +/- tau are drawn as well: that is the
    expected spread of TRUE points around the line, i.e. a population band,
    not to be confused with the confidence band for the line itself.

    Returns (fig, ax).  Requires matplotlib (imported lazily so the rest of
    the module stays numpy-only).
    """
    import matplotlib.pyplot as plt

    est = result.estimates
    fit = result.fit
    boot = result.bootstrap
    ci = boot["ci_level"]
    alpha = 1.0 - ci

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    # confidence band for the line from bootstrap (slope, intercept) pairs
    pad = 0.05 * (est.x.max() - est.x.min() or 1.0)
    grid = np.linspace(est.x.min() - pad, est.x.max() + pad, n_grid)
    lines = (
        boot["bootstrap_intercepts"][:, None]
        + boot["bootstrap_slopes"][:, None] * grid[None, :]
    )
    lo, hi = np.quantile(lines, [alpha / 2, 1 - alpha / 2], axis=0)
    ax.fill_between(
        grid,
        lo,
        hi,
        alpha=0.25,
        linewidth=0,
        label=f"{100 * ci:.0f}% CI of the line (nested bootstrap)",
    )

    # best-fit line
    ax.plot(
        grid,
        fit.intercept + fit.slope * grid,
        lw=2,
        label=(f"York fit: slope = {fit.slope:.3g} $\\pm$ {fit.slope_se:.2g}"),
    )

    # optional intrinsic-scatter (population) band
    if show_scatter_band and fit.tau > 0:
        for s, lab in (
            (+1, f"$\\pm\\tau$ intrinsic scatter ({fit.tau:.3g})"),
            (-1, None),
        ):
            ax.plot(
                grid,
                fit.intercept + fit.slope * grid + s * fit.tau,
                ls="--",
                lw=1,
                color="gray",
                label=lab,
            )

    # data with error bars
    ax.errorbar(
        est.x,
        est.y,
        xerr=est.sx,
        yerr=est.sy,
        fmt="o",
        ms=5,
        capsize=2,
        lw=1,
        zorder=3,
        label="data",
    )

    ax.set_xlabel(
        "log-variance of population (i, 1)"
        if est.log_x
        else "variance of population (i, 1)"
    )
    ax.set_ylabel("median of population (i, 2)")
    p = result.permutation["p_two_sided"]
    ax.set_title(
        f"permutation p = {p:.4g}   "
        f"(reduced $\\chi^2$ without scatter = "
        f"{(fit.chi2_reduced_no_scatter if np.isfinite(fit.chi2_reduced_no_scatter) else fit.chi2_reduced):.2f})"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig, ax


def analyze(
    populations_x: Sequence[np.ndarray],
    populations_y: Sequence[np.ndarray],
    log_x: bool = True,
    intrinsic_scatter: bool = True,
    n_boot_within: int = 2000,
    n_perm: int = 4999,
    n_boot_slope: int = 2000,
    ci_level: float = 0.95,
    strata: Optional[Sequence] = None,
    seed: Optional[int] = None,
) -> AnalysisResult:
    """
    Full pipeline.  populations_x[i] / populations_y[i] are the raw samples of
    populations (i,1) and (i,2); x_i is the (log-)variance of the former and
    y_i the median of the latter.  Set strata (e.g. bins of population size)
    if per-point precision plausibly tracks x.  Returns an AnalysisResult;
    call .summary() for a readable report.
    """
    rng = np.random.default_rng(seed)

    est = bootstrap_point_estimates(
        populations_x, populations_y, n_boot=n_boot_within, log_x=log_x, rng=rng
    )
    fitter = york_fit_scatter if intrinsic_scatter else york_fit
    fit = fitter(est.x, est.y, est.sx, est.sy)
    perm = studentized_permutation_test(
        est, n_perm=n_perm, strata=strata, intrinsic_scatter=intrinsic_scatter, rng=rng
    )
    boot = nested_bootstrap_slope_ci(
        est,
        fit=fit,
        n_boot=n_boot_slope,
        ci_level=ci_level,
        intrinsic_scatter=intrinsic_scatter,
        rng=rng,
    )

    chi2_diag = (
        fit.chi2_reduced_no_scatter
        if np.isfinite(fit.chi2_reduced_no_scatter)
        else fit.chi2_reduced
    )

    warnings = []
    if chi2_diag > 1.5 and not intrinsic_scatter:
        warnings.append(
            f"reduced chi^2 = {chi2_diag:.2f} >> 1: scatter exceeds the "
            "bootstrap-estimated uncertainties; there is likely intrinsic scatter. "
            "The permutation p-value remains valid, but York's analytic SE (and "
            "hence the bootstrap CI, which assumes points lie on a line) is "
            "optimistic -- rerun with intrinsic_scatter=True."
        )
    if chi2_diag > 1.5 and intrinsic_scatter:
        warnings.append(
            f"excess scatter detected (reduced chi^2 = {chi2_diag:.2f} without the "
            f"scatter term) and absorbed into tau = {fit.tau:.4g}. SEs, T, and the "
            "bootstrap CI include it; note tau^2 is a plug-in estimate (its own "
            "uncertainty is not propagated -- linmix would do that), and part of "
            "tau may be curvature rather than true scatter if the relation is "
            "monotone but nonlinear."
        )
    if chi2_diag < 0.5:
        warnings.append(
            f"reduced chi^2 = {chi2_diag:.2f} << 1: points hug the line far "
            "more tightly than the estimated uncertainties allow; the uncertainty "
            "estimates are probably overstated, or the x/y errors are not "
            "independent (shared samples, shared normalization?)."
        )
    if not fit.converged:
        warnings.append("York fit did not converge on the observed data.")
    small = min(min(len(p) for p in populations_x), min(len(p) for p in populations_y))
    if small < 20:
        warnings.append(
            f"smallest population has {small} samples: the bootstrap distribution "
            "of a median is discrete and can be lumpy at this size; results are "
            "still valid but consider raising n_boot_within."
        )
    if perm["n_failed_fits"] > 0 or boot["n_failed_fits"] > 0:
        warnings.append(
            f"{perm['n_failed_fits']} permutation fits and {boot['n_failed_fits']} "
            "bootstrap fits failed and were dropped."
        )

    return AnalysisResult(
        estimates=est, fit=fit, permutation=perm, bootstrap=boot, warnings=warnings
    )


# ----------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)

    def make_data(related: bool, n_units: int = 25):
        """Gamma-distributed quantity.  When `related` is True, populations
        (i,1) and (i,2) share the same shape parameter, so the true variance
        of the former and the true median of the latter co-vary; otherwise
        they are independent."""
        pops_x, pops_y = [], []
        for _ in range(n_units):
            k1 = rng.uniform(2.0, 20.0)  # shape of population (i,1)
            k2 = k1 if related else rng.uniform(2.0, 20.0)
            scale = 1.5
            n1 = rng.integers(30, 200)
            n2 = rng.integers(30, 200)
            pops_x.append(rng.gamma(k1, scale, size=n1))
            pops_y.append(rng.gamma(k2, scale, size=n2))
        return pops_x, pops_y

    print("=== Case 1: true variance-median relationship (shared gamma shape) ===")
    px, py = make_data(related=True)
    res = analyze(
        px, py, log_x=True, seed=1, n_boot_within=1000, n_perm=999, n_boot_slope=500
    )
    print(res.summary())
    fig, _ = res.plot()
    fig.savefig("fit_related.png", dpi=150)

    print("\n=== Case 2: null (independent populations) ===")
    px, py = make_data(related=False)
    res0 = analyze(
        px, py, log_x=True, seed=2, n_boot_within=1000, n_perm=999, n_boot_slope=500
    )
    print(res0.summary())
    fig0, _ = res0.plot()
    fig0.savefig("fit_null.png", dpi=150)

import warnings

import numpy as np
import pandas as pd
from scipy import stats

try:
    import qp

    QP_INSTALLED = True
except Exception:  # pragma: no cover - qp missing or unimportable in this env
    QP_INSTALLED = False

__all__ = ["QP_INSTALLED", "DEFAULT_ZMAX", "DEFAULT_QUANTS", "package_pdfs", "predictions_mean_std"]

# The p(z) is a Student-t in log(1 + z), truncated to [0, DEFAULT_ZMAX] and renormalised (its mean in z is infinite
# without the upper cut). It is stored as a qp interp_irregular ensemble: each object has its own grid, its quantiles at
# the levels DEFAULT_QUANTS (0, 1 and the normal CDF of 29 evenly spaced points in [-3, 3], denser in the tails), so
# the grid is fine where the p(z) is narrow whatever its width. The values on the grid are the mean densities between
# consecutive quantiles, so qp's CDF (a running sum of grid step x value, linearly interpolated) passes through the
# exact quantiles: cdf, ppf, rvs and the PIT are exact at the grid points and cannot overshoot. Both arrays are
# float32: 31 levels take 248 bytes per object. pdf() interpolates the values linearly, which is only approximately
# consistent with that CDF: for n(z), bin the CDF or draws rather than averaging pdf() on a grid.
# qp.quant, which stores the quantiles alone, is not used: its quadratic CDF spline overshoots in the tails (PITs of
# catastrophic outliers fall outside [0, 1] and qp's PIT drops them) and it fails on tied float32 quantiles.
DEFAULT_ZMAX = 3.0
DEFAULT_QUANTS = np.concatenate([[0.0], stats.norm.cdf(np.linspace(-3, 3, 29)), [1.0]])
_MIN_STEP = np.float32(1e-6)  # minimum grid step in z, so float32-tied quantiles do not give infinite densities
# the finer levels the p(z) mean and std are integrated over, as E[z^k] = int_0^1 ppf(q)^k dq
_MOMENT_QUANTS = np.concatenate([[0.0], stats.norm.cdf(np.linspace(-6, 6, 601)), [1.0]])
_CHUNK = 20000  # objects at a time, to bound the memory of the moment integrals


def _ppf(q, mu, scale, nu, zmax):
    """
    Quantiles in z, of shape (len(mu), len(q)), of a Student-t in y = log(1 + z) with location mu and scale scale,
    truncated to z in [0, zmax]. The truncation is applied in the survival function for objects below the middle of
    the range and in the CDF for those above: for an object far below (above) the range, [0, zmax] is in its upper
    (lower) tail, where the survival function (CDF) is small and stays resolved in float64.
    """
    y_lo, y_hi = 0.0, np.log1p(zmax)
    lower = mu < 0.5 * (y_lo + y_hi)
    y = np.empty((mu.size, q.size))
    m, s = mu[lower, None], scale[lower, None]
    s_lo, s_hi = stats.t.sf(y_lo, nu, m, s), stats.t.sf(y_hi, nu, m, s)
    y[lower] = stats.t.isf(s_lo - q * (s_lo - s_hi), nu, m, s)
    m, s = mu[~lower, None], scale[~lower, None]
    c_lo, c_hi = stats.t.cdf(y_lo, nu, m, s), stats.t.cdf(y_hi, nu, m, s)
    y[~lower] = stats.t.ppf(c_lo + q * (c_hi - c_lo), nu, m, s)
    return np.clip(np.expm1(y), 0.0, zmax)


def _mode(mu, scale, nu, zmax):
    """
    Mode in z of the Student-t in log(1 + z): p(z) = t(y) / (1 + z) peaks where d/dy [log t(y) - y] = 0, i.e. at
    y = mu + d with d^2 + (nu + 1) d + nu scale^2 = 0 (the root near 0), clipped to [0, zmax].
    """
    d = 0.5 * (np.sqrt((nu + 1) ** 2 - 4 * nu * scale**2) - (nu + 1))
    return np.clip(np.expm1(mu + d), 0.0, zmax)


def _mean_std(mu, scale, nu, zmax):
    """Mean and standard deviation in z of the truncated p(z), integrating its quantile function."""
    mean, std = np.empty(mu.size), np.empty(mu.size)
    for i in range(0, mu.size, _CHUNK):
        z = _ppf(_MOMENT_QUANTS, mu[i : i + _CHUNK], scale[i : i + _CHUNK], nu, zmax)
        m1 = np.trapezoid(z, _MOMENT_QUANTS, axis=1)
        m2 = np.trapezoid(z**2, _MOMENT_QUANTS, axis=1)
        mean[i : i + _CHUNK], std[i : i + _CHUNK] = m1, np.sqrt(np.maximum(m2 - m1**2, 0.0))
    return mean, std


def package_pdfs(mu, scale, nu, ids=None, zmax=DEFAULT_ZMAX, quants=DEFAULT_QUANTS):
    """
    Package per-object Student-t p(z)s: location mu and scale in log(1 + z), nu degrees of freedom, truncated to
    [0, zmax] (see the top of this module).

    If ``qp`` is installed, returns a ``qp.Ensemble`` (``qp.interp_irregular``, float32) on a per-object grid at the
    quantile levels quants, with ``object_id``, ``zmode``, ``zmean`` and ``zstd`` as ancillary data: the mode, mean
    and standard deviation of the exact p(z). Use these rather than the ensemble's ``mean()``/``std()``, which qp
    cannot compute reliably for these ensembles. If ``qp`` is not installed, falls back to a ``pandas.DataFrame``
    with "object_id", "zmode", "mean", "std" columns.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    scale = np.asarray(scale, dtype=float).ravel()
    ids = np.arange(len(mu)) if ids is None else np.asarray(ids)
    zmode = _mode(mu, scale, nu, zmax)
    mean, std = _mean_std(mu, scale, nu, zmax)

    if not QP_INSTALLED:
        warnings.warn(
            "qp is not installed; returning a pandas DataFrame of id/zmode/mean/std instead "
            "of a qp Ensemble. Install qp-prob to get p(z) ensembles.",
            stacklevel=2,
        )
        return pd.DataFrame({"object_id": ids, "zmode": zmode, "mean": mean, "std": std})

    quants = np.asarray(quants, dtype=float)
    xvals = np.empty((mu.size, quants.size), dtype=np.float32)
    for i in range(0, mu.size, _CHUNK):
        xvals[i : i + _CHUNK] = _ppf(quants, mu[i : i + _CHUNK], scale[i : i + _CHUNK], nu, zmax)
    for j in range(1, quants.size):  # strictly increasing after the float32 rounding
        xvals[:, j] = np.maximum(xvals[:, j], xvals[:, j - 1] + _MIN_STEP)
    # mean density between consecutive grid points, from the rounded grid, so that the running sum of step x value
    # (qp's CDF) is the quantile level at every grid point; the first value only sets pdf() on the first step
    yvals = np.empty_like(xvals)
    yvals[:, 1:] = np.diff(quants) / np.diff(xvals.astype(float), axis=1)
    yvals[:, 0] = yvals[:, 1]
    ancil = {"object_id": ids, "zmode": zmode, "zmean": mean, "zstd": std}
    return qp.Ensemble(qp.interp_irregular, data={"xvals": xvals, "yvals": yvals}, ancil=ancil)


def predictions_mean_std(result):
    """
    Extract per-object predicted redshift mean and std from the output of :func:`package_pdfs` (a qp Ensemble or the
    fallback DataFrame). For other qp ensembles, without "zmean"/"zstd" in their ancil, uses their mean() and std().

    Returns
    -------
    y_pred_mean, y_pred_std : np.ndarray
        Arrays of shape (N, 1).
    """
    if isinstance(result, pd.DataFrame):
        mean, std = result["mean"], result["std"]
    elif result.ancil is not None and "zmean" in result.ancil and "zstd" in result.ancil:
        mean, std = result.ancil["zmean"], result.ancil["zstd"]
    else:
        mean, std = result.mean(), result.std()
    return np.asarray(mean).reshape(-1, 1), np.asarray(std).reshape(-1, 1)

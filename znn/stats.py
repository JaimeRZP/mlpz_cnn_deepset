import numpy as np
from scipy.stats import sigmaclip

__all__ = [
    "biweight_location",
    "biweight_midvariance",
    "biweight_scale",
    "get_biweight_mean_sigma_outlier",
]


def _mad(x, M):
    """Median absolute deviation about M (unscaled)."""
    return np.median(np.abs(x - M))


def biweight_location(data, c=6.0, ignore_nan=False):
    """Robust estimate of the centre. Analogue of np.mean."""
    x = np.asarray(data, dtype=float).ravel()
    if ignore_nan:
        x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan

    M = np.median(x)
    mad = _mad(x, M)
    if mad == 0 or not np.isfinite(mad):
        return M

    u = (x - M) / (c * mad)
    mask = np.abs(u) < 1
    if not mask.any():
        return M

    u = u[mask]
    w = (1.0 - u**2) ** 2
    return M + np.sum((x[mask] - M) * w) / np.sum(w)


def biweight_midvariance(data, c=9.0, ignore_nan=False, modify_sample_size=False):
    """Robust estimate of the variance."""
    x = np.asarray(data, dtype=float).ravel()
    if ignore_nan:
        x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan

    M = np.median(x)
    mad = _mad(x, M)
    if mad == 0 or not np.isfinite(mad):
        return 0.0

    u = (x - M) / (c * mad)
    mask = np.abs(u) < 1
    u = u[mask]

    n = mask.sum() if modify_sample_size else x.size

    f1 = np.sum((x[mask] - M) ** 2 * (1.0 - u**2) ** 4)
    f2 = np.abs(np.sum((1.0 - u**2) * (1.0 - 5.0 * u**2))) ** 2
    if f2 == 0:
        return 0.0
    return n * f1 / f2


def biweight_scale(data, c=9.0, ignore_nan=False, modify_sample_size=False):
    """Robust estimate of the scatter. Analogue of np.std."""
    return np.sqrt(biweight_midvariance(data, c=c, ignore_nan=ignore_nan, modify_sample_size=modify_sample_size))


def get_biweight_mean_sigma_outlier(subset, abs_out_thresh=0.2):
    """
    Point-estimate statistics of e.g. dz = (z_pred - z_ref) / (1 + z_ref). subset is 3-sigma clipped until
    convergence, and the mean and std are the biweight location and scale of the clipped values.
    Returns (mean, mean error, std, outlier rate, absolute outlier rate). The outlier rate is the fraction of all of
    subset with |value| > 3 np.std(clipped); the absolute outlier rate is the fraction with |value| > abs_out_thresh.
    """
    subset_clip, _, _ = sigmaclip(subset, low=3, high=3)  # iterates until no more values are clipped

    mean = biweight_location(subset_clip)
    std = biweight_scale(subset_clip)
    outlier_rate = np.sum(np.abs(subset) > 3 * np.std(subset_clip)) / len(subset)
    abs_outlier_rate = np.sum(np.abs(subset) > abs_out_thresh) / len(subset)

    return (
        mean,
        std / np.sqrt(len(subset_clip)),
        std,
        outlier_rate,
        abs_outlier_rate,
    )
